"""
nnUNetTrainerFedProx - FedProx Algorithm for nnU-Net V2
========================================================
This is a custom nnU-Net Trainer that implements the FedProx algorithm
by adding a proximal term to the loss function.

Key Features:
- Inherits from nnUNetTrainer (official extension method)
- Adds proximal term: (μ/2) * ||w - w_global||²
- Compatible with nnU-Net's training pipeline
- No monkey patching - clean OOP design

Reference:
- FedProx Paper: https://arxiv.org/abs/1812.06127
- nnU-Net Extension Guide: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md
"""

import torch
import numpy as np
from typing import Union, Tuple, List
from torch import nn
from copy import deepcopy

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFedProx(nnUNetTrainer):
    """
    nnU-Net Trainer with FedProx algorithm.
    
    FedProx adds a proximal term to the loss function to prevent local models
    from deviating too far from the global model, which helps in federated
    learning scenarios with heterogeneous data.
    
    Loss = Original_Loss + (μ/2) * ||w - w_global||²
    
    Usage:
        In federated learning, the global model weights should be set via
        `set_global_model_weights()` at the beginning of each local training round.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict):
        """
        Initialize FedProx Trainer.
        
        Args:
            plans: nnU-Net plans dictionary
            configuration: Configuration name (e.g., '3d_fullres')
            fold: Cross-validation fold
            dataset_json: Dataset JSON dictionary
        
        Note:
            Device is automatically managed by nnUNetTrainer parent class (defaults to CUDA if available).
        """
        # Initialize parent class (will use default device: torch.device('cuda'))
        super().__init__(plans, configuration, fold, dataset_json)
        
        # FedProx specific parameters
        self.fedprox_mu = 0.01  # Default μ value, can be modified
        self.global_model_weights = None  # Will store global model weights
        
        # Flag to enable/disable FedProx (useful for debugging)
        self.enable_fedprox = True
        
        # Flag to avoid redundant dataloader initialization 
        self._fedprox_training_initialized = False
        
        # Register FedProx-specific logging keys in nnU-Net logger
        # This must be done AFTER super().__init__() which creates self.logger
        if hasattr(self, 'logger') and self.logger is not None:
            # Initialize FedProx logging keys as empty lists
            if 'fedprox_proximal_term' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['fedprox_proximal_term'] = []
            if 'train_losses_original' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['train_losses_original'] = []
        
        print(f"nnUNetTrainerFedProx initialized with μ = {self.fedprox_mu}")
    
    def set_fedprox_mu(self, mu: float):
        """
        Set the FedProx hyperparameter μ.
        
        Args:
            mu: Proximal term coefficient (typically 0.001 - 0.1)
                - Larger μ: stronger constraint (more similar to global model)
                - Smaller μ: weaker constraint (more freedom for local training)
        """
        assert mu >= 0, "μ must be non-negative"
        self.fedprox_mu = mu
        print(f"FedProx μ set to {self.fedprox_mu}")
    
    def set_global_model_weights(self, global_weights: dict):
        """
        Set the global model weights for FedProx proximal term computation.
        
        This should be called at the beginning of each federated learning round,
        after receiving the global model from the server.
        
        Args:
            global_weights: Dictionary of global model state dict
                           Can be obtained from: model.state_dict()
        """
        # Store global weights on the same device as the model
        self.global_model_weights = {}
        for name, param in global_weights.items():
            if isinstance(param, torch.Tensor):
                self.global_model_weights[name] = param.clone().detach().to(self.device)
            elif isinstance(param, np.ndarray):
                self.global_model_weights[name] = torch.from_numpy(param).to(self.device)
            else:
                self.global_model_weights[name] = param
        
        print(f"Global model weights set for FedProx (μ={self.fedprox_mu})")
    
    def compute_fedprox_loss(self) -> torch.Tensor:
        """
        Compute the FedProx proximal term: (μ/2) * ||w - w_global||²
        
        This measures the L2 distance between current local weights and global weights.
        
        Returns:
            Proximal term as a scalar tensor
        """
        if not self.enable_fedprox or self.global_model_weights is None or self.fedprox_mu == 0:
            # If FedProx is disabled or no global weights, return 0
            return torch.tensor(0.0, device=self.device)
        
        proximal_term = torch.tensor(0.0, device=self.device)
        
        # Compute L2 norm of the difference between current and global weights
        for name, param in self.network.named_parameters():
            if param.requires_grad and name in self.global_model_weights:
                global_param = self.global_model_weights[name]
                # ||w - w_global||²
                proximal_term += torch.sum((param - global_param) ** 2)
        
        # Apply μ/2 coefficient
        proximal_term = (self.fedprox_mu / 2.0) * proximal_term
        
        return proximal_term
    
    def train_step(self, batch: dict) -> dict:
        """
        Override train_step to add FedProx proximal term to the loss.
        
        This is the core modification for FedProx algorithm.
        
        Args:
            batch: Training batch dictionary with 'data' and 'target'
        
        Returns:
            Dictionary with 'loss' and optionally 'fedprox_term'
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass and compute original nnU-Net loss
        from nnunetv2.utilities.helpers import dummy_context
        from torch import autocast
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            
            # Original nnU-Net loss (Dice + CE)
            original_loss = self.loss(output, target)
            
            # Compute FedProx proximal term
            proximal_term = self.compute_fedprox_loss()
            
            # Total FedProx loss
            total_loss = original_loss + proximal_term

        # Backward pass (same as original nnUNetTrainer)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        # Return both total loss and proximal term for logging
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'original_loss': original_loss.detach().cpu().numpy(),
            'fedprox_term': proximal_term.detach().cpu().numpy()
        }
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        Override to log FedProx-specific metrics.
        
        Args:
            train_outputs: List of outputs from train_step
        """
        from nnunetv2.utilities.collate_outputs import collate_outputs
        from torch import distributed as dist
        
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            # Gather losses from all workers
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            
            # Also gather FedProx terms
            if 'fedprox_term' in outputs:
                fedprox_tr = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(fedprox_tr, outputs['fedprox_term'])
                fedprox_here = np.vstack(fedprox_tr).mean()
            else:
                fedprox_here = 0.0
        else:
            loss_here = np.mean(outputs['loss'])
            fedprox_here = np.mean(outputs.get('fedprox_term', [0.0]))

        # Log total loss
        self.logger.log('train_losses', loss_here, self.current_epoch)
        
        # Log FedProx proximal term separately
        if self.enable_fedprox and self.fedprox_mu > 0:
            self.logger.log('fedprox_proximal_term', fedprox_here, self.current_epoch)
            
            # Also log original loss if available
            if 'original_loss' in outputs:
                original_loss_here = np.mean(outputs['original_loss'])
                self.logger.log('train_losses_original', original_loss_here, self.current_epoch)
        
        # Print progress
        if self.local_rank == 0:
            print(f"Epoch {self.current_epoch}: "
                  f"Total Loss = {loss_here:.4f}, "
                  f"FedProx Term = {fedprox_here:.6f} (μ={self.fedprox_mu})")
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Override to save FedProx-specific parameters in checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        # Call parent's save method first
        super().save_checkpoint(filename)
        
        # Note: FedProx parameters (mu, global_weights) are typically managed
        # by the federated learning framework, so we don't save them in the checkpoint.
        # If needed, you can extend this method to save them.
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Override to load FedProx-specific parameters from checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        # Call parent's load method
        super().load_checkpoint(filename)
        
        # FedProx parameters will be set by the federated learning framework
        # after loading the checkpoint
