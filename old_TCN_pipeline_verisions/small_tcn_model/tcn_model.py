"""
Temporal Convolutional Network (TCN) for detecting bow shock crossing events.

Based on the TCN architecture from "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" by Bai et al. (2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution, weight normalization, and residual connection."""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the right side of the input."""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequence classification.
    
    Args:
        num_inputs: Number of input features (e.g., energy bins)
        num_channels: List of channel sizes for each layer
        kernel_size: Size of convolutional kernel
        dropout: Dropout probability
        num_classes: Number of output classes (default: 1 for binary classification)
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 1,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size
            
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]
        
        self.network = nn.Sequential(*layers)
        
        # Output layer for binary classification
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, num_inputs)
        
        Returns:
            Output tensor of shape (batch, sequence_length, num_classes)
        """
        # TCN expects (batch, channels, sequence_length)
        x = x.transpose(1, 2)  # (batch, num_inputs, sequence_length)
        
        # Pass through TCN layers
        y = self.network(x)  # (batch, num_channels[-1], sequence_length)
        
        # Transpose back to (batch, sequence_length, num_channels[-1])
        y = y.transpose(1, 2)
        
        # Apply output layer
        output = self.fc(y)  # (batch, sequence_length, num_classes)
        
        return output


class BowShockTCN(nn.Module):
    """
    TCN model specifically for bow shock crossing event detection.
    
    This model takes spectrogram data (time, energy) and predicts
    whether a bow shock crossing event occurs at each time step.
    """
    
    def __init__(
        self,
        num_energy_bins: int = 63,
        num_channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        if num_channels is None:
            # Default architecture: progressively increase channels
            num_channels = [64, 128, 256, 128]
        
        self.tcn = TCN(
            num_inputs=num_energy_bins,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_classes=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, num_energy_bins)
        
        Returns:
            Output tensor of shape (batch, sequence_length, 1) with logits
        """
        return self.tcn(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, num_energy_bins)
        
        Returns:
            Probability tensor of shape (batch, sequence_length, 1)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, num_energy_bins)
            threshold: Classification threshold
        
        Returns:
            Binary predictions of shape (batch, sequence_length, 1)
        """
        proba = self.predict_proba(x)
        return (proba > threshold).long()

