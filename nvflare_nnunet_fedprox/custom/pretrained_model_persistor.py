"""
Pretrained Model Persistor for Server Initialization
===================================================
Loads pretrained nnU-Net weights on the server side for Round 0.
Saves aggregated models in nnU-Net compatible format for direct inference.
"""

import os
import torch
import numpy as np
from typing import Dict
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class PretrainedModelPersistor(PTFileModelPersistor):
    """
    Server-side persistor that initializes global model from pretrained weights.
    
    This ensures:
    1. Server broadcasts pretrained weights in Round 0
    2. All clients start from the same w_global (pretrained)
    3. FedProx proximal term starts small and increases during training
    """
    
    def __init__(self, pretrained_path: str, 
                 keep_last_n_rounds: int = 5, save_every_n_rounds: int = 10):
        """
        Args:
            pretrained_path: Path to pretrained checkpoint (e.g., checkpoint_final.pth)
            keep_last_n_rounds: Keep only last N round checkpoints (default: 5)
                               Set to -1 to keep all rounds
            save_every_n_rounds: Save round checkpoint every N rounds (default: 10)
                                Set to 1 to save every round
        """
        # PTFileModelPersistor doesn't accept save_name in newer versions
        # Just call with model (will be set later)
        from .model_wrapper import DummyModel
        dummy_model = DummyModel()
        super().__init__(model=dummy_model)
        
        self.pretrained_path = pretrained_path
        self._loaded_pretrained = False
        self.keep_last_n_rounds = keep_last_n_rounds
        self.save_every_n_rounds = save_every_n_rounds
        
        # Track best model for validation-based saving
        self.best_val_dice = -1.0
        self.best_round = -1
    
    def load_model(self, fl_ctx: FLContext):
        """
        Load model from pretrained checkpoint for Round 0.
        
        For Round 1+, loads the aggregated model from previous rounds.
        """
        # Get current round from FL context (compatible with both POC and Simulator)
        current_round = fl_ctx.get_prop('current_round', 0)
        if current_round is None:
            current_round = 0
        
        if current_round == 0 and not self._loaded_pretrained:
            # Round 0: Load pretrained weights
            self.logger.info(f"[Round 0] Loading pretrained weights from: {self.pretrained_path}")
            
            try:
                # PyTorch 2.6+ requires weights_only=False for nnU-Net checkpoints
                checkpoint = torch.load(self.pretrained_path, map_location='cpu', weights_only=False)
                
                # nnU-Net checkpoint structure
                if 'network_weights' in checkpoint:
                    torch_state_dict = checkpoint['network_weights']
                elif 'state_dict' in checkpoint:
                    torch_state_dict = checkpoint['state_dict']
                else:
                    torch_state_dict = checkpoint
                
                # ### CRITICAL: Convert PyTorch tensors to NumPy arrays ###
                # NVFLARE expects Dict[str, np.ndarray] for transmission
                numpy_state_dict = {}
                for name, param in torch_state_dict.items():
                    if isinstance(param, torch.Tensor):
                        numpy_state_dict[name] = param.cpu().numpy()
                    elif isinstance(param, np.ndarray):
                        numpy_state_dict[name] = param
                    else:
                        self.logger.warning(f"Unknown parameter type for {name}: {type(param)}")
                        numpy_state_dict[name] = np.array(param)
                
                # Store the numpy version for NVFLARE transmission
                self._pretrained_state = numpy_state_dict
                self.logger.info(f"[Round 0] Loaded pretrained weights with {len(numpy_state_dict)} parameters")
                self.logger.info(f"[Round 0] Converted PyTorch tensors to NumPy arrays for NVFLARE transmission")
                
                # CRITICAL: Wrap in ModelLearnable for NVFLARE compatibility
                from nvflare.app_common.abstract.model import make_model_learnable
                model_learnable = make_model_learnable(numpy_state_dict, {})
                
                self._loaded_pretrained = True
                return model_learnable
                
            except Exception as e:
                self.logger.error(f"[Round 0] Failed to load pretrained weights: {e}")
                self.logger.error("Server will use default initialization")
                return super().load_model(fl_ctx)
        else:
            # Round 1+: Normal loading from saved aggregated model
            return super().load_model(fl_ctx)
    
    def get_model_inventory(self, fl_ctx: FLContext):
        """
        Get model inventory - returns ModelLearnable for NVFLARE compatibility.
        
        For Round 0, wraps pretrained weights in ModelLearnable.
        For Round 1+, uses default behavior.
        """
        # Get current round from FL context
        current_round = fl_ctx.get_prop('current_round', 0)
        if current_round is None:
            current_round = 0
            
        if current_round == 0 and hasattr(self, '_pretrained_state') and self._pretrained_state is not None:
            # Round 0: Wrap pretrained weights in ModelLearnable
            from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
            
            # Create ModelLearnable from numpy state dict
            model_learnable = make_model_learnable(self._pretrained_state, {})
            return model_learnable
        
        # Round 1+: Use default behavior
        return super().get_model_inventory(fl_ctx)
    
    def save_model(self, ml_model: Dict, fl_ctx: FLContext):
        """
        Save aggregated model in NVFLARE format only.
        
        This creates two NVFLARE format files:
        1. FL_global_model.pt - Latest model (always updated, default PTFileModelPersistor behavior)
        2. FL_global_model_best.pt - Best validation performance model (only when improved)
        
        Use convert_fl_to_nnunet.py to convert to nnU-Net format when needed.
        
        Args:
            ml_model: Model state (Dict[str, np.ndarray] from aggregator)
            fl_ctx: FL context
        """
        # 1. Save latest model in NVFLARE format (default behavior)
        # This saves to: workspace/.../app_server/FL_global_model.pt
        super().save_model(ml_model, fl_ctx)
        
        # 2. Save best model in NVFLARE format (if validation improved)
        try:
            engine = fl_ctx.get_engine()
            # Get round number (compatible with both Simulator and POC modes)
            if hasattr(engine, 'get_current_round_number'):
                current_round = engine.get_current_round_number()
            elif hasattr(engine, 'current_round'):
                current_round = engine.current_round
            else:
                current_round = fl_ctx.get_prop('current_round', 0)
            
            # ### Get app_dir using multiple fallback methods ###
            app_dir = None
            
            # Method 1: Try engine.run_dir (Simulator mode) - CRITICAL FIX
            # Simulator path structure: workspace/server/simulate_job/app_server
            if hasattr(engine, 'run_dir'):
                try:
                    # run_dir gives us: workspace/server
                    # We need to append: simulate_job/app_server
                    base_dir = engine.run_dir
                    self.logger.debug(f"[Model Save] engine.run_dir = {base_dir}")
                    
                    # Check for simulate_job subdirectory (Simulator mode)
                    simulate_path = os.path.join(base_dir, 'simulate_job', 'app_server')
                    if os.path.exists(simulate_path):
                        app_dir = simulate_path
                        self.logger.info(f"[Model Save] Found Simulator path: {app_dir}")
                    else:
                        # Fallback: POC mode might use server/app_server directly
                        poc_path = os.path.join(base_dir, 'app_server')
                        if os.path.exists(poc_path):
                            app_dir = poc_path
                            self.logger.info(f"[Model Save] Found POC path: {app_dir}")
                except Exception as e:
                    self.logger.warning(f"[Model Save] engine.run_dir method failed: {e}")
            
            # Method 2: Try engine.get_workspace() (Simulator mode returns full path)
            if not app_dir and hasattr(engine, 'get_workspace'):
                try:
                    workspace = engine.get_workspace()
                    if hasattr(workspace, 'get_app_dir'):
                        # In Simulator mode, get_app_dir() returns full path including simulate_job
                        # e.g., workspace/server/simulate_job/app_server (not server/server/app_server)
                        candidate_path = workspace.get_app_dir('server')
                        self.logger.debug(f"[Model Save] workspace.get_app_dir returned: {candidate_path}")
                        # Check if path exists
                        if os.path.exists(candidate_path):
                            app_dir = candidate_path
                            self.logger.info(f"[Model Save] Found path from workspace.get_app_dir: {app_dir}")
                        else:
                            # Path doesn't exist, try simulate_job subdirectory
                            if 'server/server' in candidate_path:
                                # Fix double 'server' issue: replace server/server with server/simulate_job
                                fixed_path = candidate_path.replace('server/server', 'server/simulate_job')
                                if os.path.exists(fixed_path):
                                    app_dir = fixed_path
                                    self.logger.info(f"[Model Save] Fixed path (server/server → server/simulate_job): {app_dir}")
                except Exception as e:
                    self.logger.debug(f"[Model Save] workspace.get_app_dir failed: {e}")
            
            # Method 3: Try fl_ctx workspace (POC mode)
            if not app_dir:
                try:
                    workspace = fl_ctx.get_prop('workspace_object', None)
                    if workspace and hasattr(workspace, 'get_app_dir'):
                        app_dir = workspace.get_app_dir('server')
                        self.logger.info(f"[Model Save] Got from fl_ctx workspace: {app_dir}")
                except Exception as e:
                    self.logger.debug(f"[Model Save] fl_ctx workspace failed: {e}")
            
            # Fallback: Use hardcoded path (last resort)
            if not app_dir:
                app_dir = '/root/autodl-tmp/workspace_poc/server/simulate_job/app_server'
                self.logger.warning(f"[Model Save] Using hardcoded fallback: {app_dir}")
            
            if not os.path.exists(app_dir):
                self.logger.error(f"[Model Save] app_dir does not exist: {app_dir}")
                self.logger.error(f"[Model Save] Please check workspace structure")
                return
            
            
            current_val_dice = -1.0
            try:
                # Method 1: Get from FLContext (set by IntimeModelSelector)
                validation_result = fl_ctx.get_prop('validation_result', None)
                if validation_result and isinstance(validation_result, dict):
                    current_val_dice = validation_result.get('val_dice', -1.0)
                    self.logger.info(f"[Round {current_round}] Got val_dice from FLContext: {current_val_dice:.4f}")
                
                # Method 2: Check model_selector's best model info
                if current_val_dice == -1.0:
                    model_selector = engine.get_component('model_selector')
                    if model_selector and hasattr(model_selector, '_best_model_info'):
                        best_info = model_selector._best_model_info
                        if best_info and 'metrics' in best_info:
                            current_val_dice = best_info['metrics'].get('val_dice', -1.0)
                            self.logger.info(f"[Round {current_round}] Got val_dice from model_selector: {current_val_dice:.4f}")
                
                # Method 3: Fallback - manually aggregate from client metrics (if available)
                if current_val_dice == -1.0:
                    client_metrics = fl_ctx.get_prop('client_validation_metrics', {})
                    if client_metrics:
                        total_weight = 0.0
                        weighted_sum = 0.0
                        for site_name, metrics in client_metrics.items():
                            if 'val_dice' in metrics and 'weight' in metrics:
                                weighted_sum += metrics['val_dice'] * metrics['weight']
                                total_weight += metrics['weight']
                        if total_weight > 0:
                            current_val_dice = weighted_sum / total_weight
                            self.logger.info(f"[Round {current_round}] Computed val_dice from client metrics: {current_val_dice:.4f}")
                
                if current_val_dice > 0:
                    self.logger.info(f"[Round {current_round}] ✅ Global validation Dice: {current_val_dice:.4f} (weighted avg of all clients)")
                else:
                    self.logger.warning(f"[Round {current_round}] ⚠️ Could not retrieve global validation Dice (will use -1.0)")
            except Exception as e:
                self.logger.warning(f"[Round {current_round}] Error retrieving val_dice: {e}")
            
            # ### Save best model in NVFLARE format ###
            if current_val_dice > self.best_val_dice:
                self.best_val_dice = current_val_dice
                self.best_round = current_round
                
                # Save best model as NVFLARE format (same location as latest model)
                best_model_path = os.path.join(app_dir, 'FL_global_model_best.pt')
                
                # Create ModelLearnable for NVFLARE format
                from nvflare.app_common.abstract.model import make_model_learnable
                model_learnable = make_model_learnable(ml_model, {})
                
                # Save using torch
                torch.save(model_learnable, best_model_path)
                
                self.logger.info(f"[Round {current_round}] 🏆 NEW BEST MODEL!")
                self.logger.info(f"[Round {current_round}]    Dice: {current_val_dice:.4f}")
                self.logger.info(f"[Round {current_round}]    Saved: {best_model_path}")
            else:
                self.logger.info(f"[Round {current_round}] Best model remains: Round {self.best_round}, Dice: {self.best_val_dice:.4f}")
            
            self.logger.info(f"[Round {current_round}] ✅ Saved 2 NVFLARE format models:")
            self.logger.info(f"[Round {current_round}]    Latest: FL_global_model.pt")
            self.logger.info(f"[Round {current_round}]    Best:   FL_global_model_best.pt")
            
        except Exception as e:
            self.logger.error(f"[Round {current_round}] Failed to save best model: {e}")
            self.logger.error("Latest NVFLARE model is saved, but best model may be missing")