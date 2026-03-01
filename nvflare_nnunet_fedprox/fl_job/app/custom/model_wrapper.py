"""
Dummy Model Wrapper for Server-side Persistence
================================================
Since nnU-Net models are dynamically created based on dataset plans,
we use a simple wrapper on the server side for model persistence.
"""

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    """
    Lightweight placeholder model for server-side persistence.
    
    The actual nnU-Net architecture is initialized on the client side.
    Server only needs this for the persistor component to work.
    """
    
    def __init__(self):
        super(DummyModel, self).__init__()
        # Minimal dummy parameter
        self.dummy = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return x
