"""
nnU-Net FedProx Executor for NVIDIA FLARE
==========================================
This executor integrates nnU-Net V2 training with FedProx algorithm in a federated learning setting.

Key Features:
- Loads pretrained nnU-Net weights before training
- Implements FedProx proximal term in the loss function
- Compatible with NVIDIA FLARE's federated learning workflow
"""

import os
import sys
import copy
import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

# nnU-Net imports
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager

# Custom FedProx Trainer
from .nnUNetTrainerFedProx import nnUNetTrainerFedProx


class nnUNetFedProxExecutor(Executor):
    """
    Custom Executor for nnU-Net training with FedProx in federated learning.
    
    This executor:
    1. Loads pretrained nnU-Net weights at initialization
    2. Receives global model from server
    3. Performs local training with FedProx proximal term
    4. Returns updated model to server
    """
    
    def __init__(
        self,
        ### TODO: USER_MODIFY ### 
        dataset_name_or_id: str = "Dataset015_ThinNormalAndAbnormalPortalVeins", 
        configuration: str = "3d_fullres",  
        fold: int = 4,  
        trainer_name: str = "nnUNetTrainer",  
        plans_identifier: str = "nnUNetResEncUNetMPlans",  
        pretrained_weights: str = "/root/autodl-tmp/data/nnUNet_results/Dataset015_ThinNormalAndAbnormalPortalVeins/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_4/checkpoint_final.pth",
        num_epochs_per_round: int = 3,  
        fedprox_mu: float = 0.01,  
        device: str = "cuda",  
        nnunet_preprocessed: str = None,  
        nnunet_results: str = None,  
    ):
        """
        Initialize the nnU-Net FedProx Executor.
        
        Args:
            dataset_name_or_id: nnU-Net dataset identifier
            configuration: nnU-Net configuration (e.g., '2d', '3d_fullres')
            fold: Cross-validation fold number
            trainer_name: Name of the nnUNetTrainer class to use
            plans_identifier: Plans file identifier
            pretrained_weights: Path to pretrained .pth weights file
            num_epochs_per_round: Number of epochs to train per FL round
            fedprox_mu: FedProx proximal term coefficient (μ)
            device: Training device ('cuda' or 'cpu')
            nnunet_preprocessed: Override for nnUNet_preprocessed path
            nnunet_results: Override for nnUNet_results path
        """
        super().__init__()
        
        # ### TODO: USER_MODIFY ### 
        if nnunet_preprocessed:
            os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
        if nnunet_results:
            os.environ['nnUNet_results'] = nnunet_results
        
        # Configuration parameters
        self.dataset_name_or_id = dataset_name_or_id
        self.configuration = configuration
        self.fold = fold
        self.trainer_name = trainer_name
        self.plans_identifier = plans_identifier
        self.pretrained_weights = pretrained_weights
        self.num_epochs_per_round = num_epochs_per_round
        self.fedprox_mu = fedprox_mu
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Internal state
        self.trainer = None
        self.global_model_weights = None  # Store global model for FedProx
        self.current_round = 0
        
        # Store init info for logging after parent class is fully initialized
        self._init_logged = False
        self._init_info = {
            'dataset': dataset_name_or_id,
            'configuration': configuration,
            'pretrained': pretrained_weights,
            'fedprox_mu': fedprox_mu,
            'device': str(self.device)
        }
    
    def _initialize_trainer(self, fl_ctx: FLContext):
        """
        Initialize nnU-Net FedProx trainer and load pretrained weights.
        
        Note: Dataset is automatically determined based on site name:
        - site-1 → Dataset014_ThinAbnormalPortalVeins
        - site-2 → Dataset018_AbnormalPortalVeins
        
        Args:
            fl_ctx: FLContext to get site information
        """
        if self.trainer is not None:
            return
        
        self.logger.info(f"[nnUNet-FedProx] Initializing nnU-Net FedProx trainer...")
        
        # ### AUTO-DETECT DATASET BASED ON SITE NAME ###
        # This allows using a generic client config while supporting different datasets per site
        if self.dataset_name_or_id == 'PLACEHOLDER_WILL_BE_DETERMINED_BY_SITE':
            # Get site name from FL context
            from nvflare.apis.fl_constant import FLContextKey
            site_name = fl_ctx.get_identity_name()  # Returns site name like "site-1", "site-2"
            
            # Map site names to datasets
            dataset_mapping = {
                'site-1': 'Dataset015_ThinNormalAndAbnormalPortalVeins',
                'site-2': 'Dataset018_NormalAndAbnormalPortalVeins',
            }
            
            if site_name in dataset_mapping:
                self.dataset_name_or_id = dataset_mapping[site_name]
                self.logger.info(f"[nnUNet-FedProx] Auto-detected dataset for {site_name}: {self.dataset_name_or_id}")
            else:
                self.logger.warning(f"[nnUNet-FedProx] Unknown site '{site_name}', using default Dataset015")
                self.dataset_name_or_id = 'Dataset015_ThinNormalAndAbnormalPortalVeins'
        
        # Load plans.json from preprocessed dataset folder
        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        from nnunetv2.paths import nnUNet_preprocessed
        
        preprocessed_folder = join(nnUNet_preprocessed, self.dataset_name_or_id)
        plans_file = join(preprocessed_folder, f'{self.plans_identifier}.json')
        
        if not os.path.exists(plans_file):
            # Fallback: try default plans.json name
            plans_file = join(preprocessed_folder, 'nnUNetPlans.json')
        
        self.logger.info(f"[nnUNet-FedProx] Loading plans from: {plans_file}")
        plans_dict = load_json(plans_file)
        
        # Load dataset.json
        dataset_json_file = join(preprocessed_folder, 'dataset.json')
        dataset_json = load_json(dataset_json_file) if os.path.exists(dataset_json_file) else None
        
        # ### CRITICAL CHANGE: Use custom FedProx Trainer instead of base Trainer ###
        # This properly extends nnU-Net rather than using monkey patching
        self.trainer = nnUNetTrainerFedProx(
            plans=plans_dict,  # Pass plans dict, not string identifier
            configuration=self.configuration,
            fold=self.fold,
            dataset_json=dataset_json  # Pass actual dataset_json dict
        )
        
        # Set FedProx hyperparameter μ
        self.trainer.set_fedprox_mu(self.fedprox_mu)
        
        # Initialize trainer (load plans, dataset, etc.)
        self.trainer.initialize()
        
        # Load pretrained weights
        if self.pretrained_weights and os.path.exists(self.pretrained_weights):
            self.logger.info(f"[nnUNet-FedProx] Loading pretrained weights from: {self.pretrained_weights}")
            # PyTorch 2.6+ requires weights_only=False for nnU-Net checkpoints
            checkpoint = torch.load(self.pretrained_weights, map_location=self.device, weights_only=False)
            
            # Extract network weights
            if isinstance(checkpoint, dict) and 'network_weights' in checkpoint:
                network_weights = checkpoint['network_weights']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                network_weights = checkpoint['state_dict']
            else:
                network_weights = checkpoint
            
            # Handle torch.compile() key name prefix issue
            # Check if model is compiled (OptimizedModule with _orig_mod prefix)
            model_state_dict = self.trainer.network.state_dict()
            first_model_key = next(iter(model_state_dict.keys()))
            first_weight_key = next(iter(network_weights.keys()))
            
            # If model has _orig_mod prefix but weights don't, add prefix
            if first_model_key.startswith('_orig_mod.') and not first_weight_key.startswith('_orig_mod.'):
                self.logger.info("[nnUNet-FedProx] Adding '_orig_mod.' prefix to pretrained weights for compiled model")
                network_weights = {'_orig_mod.' + k: v for k, v in network_weights.items()}
            # If weights have _orig_mod prefix but model doesn't, remove prefix
            elif first_weight_key.startswith('_orig_mod.') and not first_model_key.startswith('_orig_mod.'):
                self.logger.info("[nnUNet-FedProx] Removing '_orig_mod.' prefix from pretrained weights")
                network_weights = {k.replace('_orig_mod.', ''): v for k, v in network_weights.items()}
            
            # Load state dict
            self.trainer.network.load_state_dict(network_weights)
            
            self.logger.info(f"[nnUNet-FedProx] Pretrained weights loaded successfully!")
        else:
            self.logger.warning(f"[nnUNet-FedProx] Pretrained weights not found at {self.pretrained_weights}, using random initialization")
        
        # Store initial weights for FedProx (will be updated with global model later)
        self.global_model_weights = copy.deepcopy(self._get_model_state())
    
    def _get_model_state(self) -> Dict[str, np.ndarray]:
        """
        Extract model weights as numpy arrays for federated learning.
        
        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        state_dict = {}
        for name, param in self.trainer.network.state_dict().items():
            state_dict[name] = param.cpu().numpy()
        return state_dict
    
    def _set_model_state(self, state_dict: Dict[str, np.ndarray]):
        """
        Load model weights from numpy arrays.
        
        Args:
            state_dict: Dictionary mapping parameter names to numpy arrays
        """
        torch_state_dict = {}
        for name, value in state_dict.items():
            torch_state_dict[name] = torch.from_numpy(value).to(self.device)
        self.trainer.network.load_state_dict(torch_state_dict)
    
    def _compute_fedprox_loss(self, original_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute FedProx loss = original_loss + (μ/2) * ||w - w_global||^2
        
        NOTE: This method is now DEPRECATED and not used.
        FedProx loss computation is handled inside nnUNetTrainerFedProx.train_step()
        
        Args:
            original_loss: Original nnU-Net loss (e.g., Dice + CE)
        
        Returns:
            Loss with FedProx proximal term added
        """
        # This method is kept for backward compatibility but is not used
        # The actual FedProx loss is computed in nnUNetTrainerFedProx.train_step()
        return original_loss
    
    def _train_with_fedprox(self, num_epochs: int):
        """
        Execute local training with FedProx proximal term.
        
        This method uses the custom nnUNetTrainerFedProx which properly
        extends nnU-Net's training loop instead of monkey patching.
        
        Args:
            num_epochs: Number of epochs to train
        """
        self.logger.info(f"[nnUNet-FedProx] Starting local training for {num_epochs} epochs with FedProx (μ={self.fedprox_mu})...")
        
        # ### CRITICAL: Set global model weights for FedProx ###
        # The custom trainer needs to know the global model to compute proximal term
        if self.global_model_weights is not None:
            # Convert numpy arrays back to tensors for the trainer
            torch_global_weights = {}
            for name, value in self.global_model_weights.items():
                if isinstance(value, np.ndarray):
                    torch_global_weights[name] = torch.from_numpy(value)
                else:
                    torch_global_weights[name] = value
            
            self.trainer.set_global_model_weights(torch_global_weights)
        
        # Calculate epoch range for this FL round
        start_epoch = self.current_round * self.num_epochs_per_round
        end_epoch = start_epoch + num_epochs
        
        # Set trainer's epoch configuration
        original_num_epochs = self.trainer.num_epochs
        self.trainer.num_epochs = end_epoch
        self.trainer.current_epoch = start_epoch
        
        try:
            # Run nnU-Net training
            # The FedProx loss is automatically computed in nnUNetTrainerFedProx.train_step()
            self.trainer.run_training()
            
            self.logger.info(f"[nnUNet-FedProx] Local training completed (epochs {start_epoch} to {end_epoch})")
        
        finally:
            # Restore original configuration
            self.trainer.num_epochs = original_num_epochs
    
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Main execution method called by NVIDIA FLARE.
        
        Args:
            task_name: Name of the task (e.g., 'train', 'validate')
            shareable: Data received from server (e.g., global model)
            fl_ctx: Federated learning context
            abort_signal: Signal to check for abortion
        
        Returns:
            Shareable containing the updated model or results
        """
        try:
            # Log initialization info on first execution (using standard logger)
            if not self._init_logged:
                self.logger.info(f"[nnUNet-FedProx] nnUNet FedProx Executor initialized with:")
                for key, value in self._init_info.items():
                    self.logger.info(f"[nnUNet-FedProx]   - {key}: {value}")
                self._init_logged = True
            
            # Initialize trainer on first call
            if self.trainer is None:
                self._initialize_trainer(fl_ctx)
            
            if task_name == AppConstants.TASK_TRAIN:
                return self._execute_train(shareable, fl_ctx, abort_signal)
            elif task_name == AppConstants.TASK_VALIDATE:
                return self._execute_validate(shareable, fl_ctx, abort_signal)
            else:
                self.logger.error(f"[nnUNet-FedProx] Unknown task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.logger.exception(f"[nnUNet-FedProx] Error during execution: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
    
    def _execute_train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Execute training task.
        
        Args:
            shareable: Contains global model from server
            fl_ctx: FL context
            abort_signal: Abort signal
        
        Returns:
            Shareable with updated local model
        """
        self.logger.info(f"[nnUNet-FedProx] === Training Round {self.current_round} ===")
        
        # Extract global model from shareable
        dxo = from_shareable(shareable)
        if dxo.data_kind != DataKind.WEIGHTS:
            self.logger.error(f"[nnUNet-FedProx] Expected WEIGHTS data kind, got {dxo.data_kind}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        
        global_weights = dxo.data
        
        
        if self.current_round == 0:
            self.global_model_weights = copy.deepcopy(global_weights)
            self.logger.info(f"[nnUNet-FedProx] Round 0: Saved server's global model as FedProx reference (w_global)")
            self.logger.info(f"[nnUNet-FedProx] Round 0: Keeping local pretrained weights for training (w_local)")
            self.logger.info(f"[nnUNet-FedProx] Round 0: FedProx will compute ||w_local - w_global|| where:")
            self.logger.info(f"[nnUNet-FedProx]          w_local  = pretrained weights (starting point)")
            self.logger.info(f"[nnUNet-FedProx]          w_global = server's initial model (FedProx reference)")
        else:
            self.logger.info(f"[nnUNet-FedProx] Updating local model with global weights from server...")
            self._set_model_state(global_weights)
            
            # Store global weights for FedProx computation
            self.global_model_weights = copy.deepcopy(global_weights)
        
        # Check for abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        # Perform local training with FedProx
        self._train_with_fedprox(num_epochs=self.num_epochs_per_round)
        
        # Check for abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        # Extract updated weights
        updated_weights = self._get_model_state()
        
        # Create DXO and shareable
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=updated_weights)
        dxo.set_meta_prop(key="current_round", value=self.current_round)

        num_train_samples = 1.0  # Default fallback
        try:
            from batchgenerators.utilities.file_and_folder_operations import join, load_json
            from nnunetv2.paths import nnUNet_preprocessed
            
            # Debug: Log paths for troubleshooting
            self.logger.info(f"[nnUNet-FedProx] nnUNet_preprocessed path: {nnUNet_preprocessed}")
            self.logger.info(f"[nnUNet-FedProx] Dataset: {self.dataset_name_or_id}, Fold: {self.fold}")
            
            preprocessed_folder = join(nnUNet_preprocessed, self.dataset_name_or_id)
            splits_file = join(preprocessed_folder, 'splits_final.json')
            
            self.logger.info(f"[nnUNet-FedProx] Looking for splits file at: {splits_file}")
            
            if os.path.exists(splits_file):
                splits = load_json(splits_file)
                self.logger.info(f"[nnUNet-FedProx] Loaded splits_final.json, type: {type(splits)}, length: {len(splits) if isinstance(splits, list) else 'N/A'}")
                
                # splits is a list of dicts, each dict has 'train' and 'val' keys
                if isinstance(splits, list) and self.fold < len(splits):
                    fold_data = splits[self.fold]
                    self.logger.info(f"[nnUNet-FedProx] Fold {self.fold} data keys: {list(fold_data.keys())}")
                    
                    if 'train' in fold_data:
                        num_train_samples = len(fold_data['train'])
                        self.logger.info(f"[nnUNet-FedProx] ✅ Read from splits_final.json: {num_train_samples} training samples for fold {self.fold}")
                    else:
                        self.logger.warning(f"[nnUNet-FedProx] 'train' key not found in fold {self.fold} data, keys: {list(fold_data.keys())}")
                else:
                    self.logger.warning(f"[nnUNet-FedProx] Invalid splits structure (type={type(splits)}) or fold {self.fold} out of range (total folds={len(splits) if isinstance(splits, list) else 'N/A'})")
            else:
                self.logger.warning(f"[nnUNet-FedProx] splits_final.json NOT FOUND at {splits_file}")
                self.logger.warning(f"[nnUNet-FedProx] Preprocessed folder exists: {os.path.exists(preprocessed_folder)}")
                if os.path.exists(preprocessed_folder):
                    try:
                        files = os.listdir(preprocessed_folder)
                        self.logger.info(f"[nnUNet-FedProx] Files in preprocessed folder: {files[:10]}")  # First 10 files
                    except:
                        pass
        except Exception as e:
            import traceback
            self.logger.error(f"[nnUNet-FedProx] Error reading splits_final.json: {e}")
            self.logger.error(f"[nnUNet-FedProx] Traceback: {traceback.format_exc()}")
            self.logger.warning(f"[nnUNet-FedProx] Using default aggregation weight 1.0")
        
        # Set aggregation weight for weighted averaging using NVFLARE standard MetaKey
        # InTimeAccumulateWeightedAggregator with weigh_by_local_iter=True reads NUM_STEPS_CURRENT_ROUND
        dxo.set_meta_prop(key=MetaKey.NUM_STEPS_CURRENT_ROUND, value=float(num_train_samples))
        self.logger.info(f"[nnUNet-FedProx] Setting NUM_STEPS_CURRENT_ROUND={num_train_samples} (training samples for weighted aggregation)")
        

        try:
            if hasattr(self.trainer, 'logger') and self.trainer.logger is not None:
                logger = self.trainer.logger
                if hasattr(logger, 'my_fantastic_logging'):
                    log_dict = logger.my_fantastic_logging
                    if 'ema_fg_dice' in log_dict and len(log_dict['ema_fg_dice']) > 0:
                        val_dice = float(log_dict['ema_fg_dice'][-1])
                        dxo.set_meta_prop(key="val_dice", value=val_dice)
                        self.logger.info(f"[nnUNet-FedProx] Sending validation dice: {val_dice:.4f}")
        except Exception as e:
            self.logger.debug(f"[nnUNet-FedProx] Could not extract validation metric: {e}")
        
        # Increment round counter
        self.current_round += 1
        
        self.logger.info(f"[nnUNet-FedProx] Training completed. Sending updated weights to server (aggregation weight: {num_train_samples}).")
        
        return dxo.to_shareable()
    
    def _execute_validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        Execute validation task.
        
        Args:
            shareable: Contains model to validate
            fl_ctx: FL context
            abort_signal: Abort signal
        
        Returns:
            Shareable with validation metrics
        """
        self.logger.info(f"[nnUNet-FedProx] Executing validation...")
        
        # Extract model from shareable
        dxo = from_shareable(shareable)
        if dxo.data_kind != DataKind.WEIGHTS:
            self.logger.error(f"[nnUNet-FedProx] Expected WEIGHTS data kind, got {dxo.data_kind}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        
        # Update model
        self._set_model_state(dxo.data)
        
        # Run nnU-Net validation
        # ### TODO: USER_MODIFY ### - Implement validation logic based on your needs
        # This is a placeholder - nnU-Net has validation in training loop
        # You may need to call trainer.validate() or implement custom validation
        
        metrics = {
            "val_loss": 0.0,  # Placeholder
            "val_dice": 0.0,  # Placeholder
        }
        
        dxo_result = DXO(data_kind=DataKind.METRICS, data=metrics)
        
        return dxo_result.to_shareable()
