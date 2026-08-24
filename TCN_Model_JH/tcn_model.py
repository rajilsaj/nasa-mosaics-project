"""
Temporal Convolutional Network (TCN) for detecting bow shock crossing events.

Based on: "An Empirical Evaluation of Generic Convolutional and Recurrent
Networks for Sequence Modeling" - Bai et al. (2018).

Architecture overview
---------------------
Input: (batch, time_steps, energy_bins)
  -> transpose to (batch, energy_bins, time_steps)  -- Conv1d expects channels first
  -> TemporalBlock x N  -- each block doubles the dilation, widening the receptive field
  -> transpose back to (batch, time_steps, channels)
  -> Dropout -> Linear(channels -> 1)  -- one logit per timestep
Output: (batch, time_steps, 1)  raw logits -- pass through sigmoid for probabilities

Dilated convolutions let each layer "see" further back in time without
using a very large kernel. With 4 layers and kernel_size=3:
  layer 0: dilation 1 -> receptive field  3 timesteps
  layer 1: dilation 2 -> receptive field  7 timesteps
  layer 2: dilation 4 -> receptive field 15 timesteps
  layer 3: dilation 8 -> receptive field 31 timesteps
At 8-second cadence, 31 timesteps = ~4 minutes of context per output step.
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Helper -- trim the extra padding added by causal convolution
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """
    Remove the right-side padding that causal convolutions add.

    Causal means the output at time t only depends on inputs at t and earlier --
    the model cannot "cheat" by looking at future timesteps.
    We achieve this by padding (kernel_size - 1) * dilation on the left,
    then trimming the same amount from the right here.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


# ---------------------------------------------------------------------------
# One TCN building block
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """
    A single TCN residual block.

    Structure
    ---------
    Input
      -> Conv1d (dilated, causal) -> Chomp -> ReLU -> Dropout
      -> Conv1d (dilated, causal) -> Chomp -> ReLU -> Dropout
      -> + residual (1x1 conv if channel count changes, else identity)
      -> ReLU
    Output

    The residual connection (skip connection) lets gradients flow directly
    back through the network during training, which helps avoid the vanishing
    gradient problem in deeper stacks.

    Args:
        n_inputs    : number of input channels
        n_outputs   : number of output channels
        kernel_size : convolution kernel width
        dilation    : dilation factor -- controls how far back the block looks
        dropout     : fraction of neurons randomly zeroed during training
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Padding chosen so that after Chomp, output length == input length
        padding = (kernel_size - 1) * dilation

        self.block = nn.Sequential(
            nn.Conv1d(n_inputs,  n_outputs, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual path: 1x1 conv aligns channel counts when they differ,
        # otherwise passes the input straight through unchanged
        self.residual = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.residual(x))


# ---------------------------------------------------------------------------
# Full TCN model
# ---------------------------------------------------------------------------

class BowShockTCN(nn.Module):
    """
    TCN for binary bow shock / magnetopause crossing detection.

    Each layer in the stack doubles the dilation, so later layers capture
    longer-range patterns in the plasma data.

    Args:
        num_energy_bins : number of input features per timestep (ELS energy bins)
        num_channels    : output channels for each TCN layer, e.g. [64, 128, 128, 64]
                          len(num_channels) == number of layers in the stack
        kernel_size     : convolution kernel width (default 3)
        dropout         : dropout probability (0.2-0.4 recommended)

    Input  shape: (batch, time_steps, num_energy_bins)
    Output shape: (batch, time_steps, 1)  -- raw logits
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
            # Simpler progression than original [64, 128, 256, 128].
            # Fewer parameters = less capacity to memorise noise = less overfitting.
            num_channels = [32, 64, 128, 64, 32]   

        # Build the stack of TemporalBlocks
        layers = []
        in_channels = num_energy_bins
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i          # 1, 2, 4, 8 ... grows the receptive field
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        # One output logit per timestep.
        # Extra Dropout here acts as a final regularisation gate before the prediction.
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time_steps, energy_bins)
        Returns:
            logits: (batch, time_steps, 1)
        """
        # Conv1d expects (batch, channels, time) -- transpose in, then back out
        x = x.transpose(1, 2)          # -> (batch, energy_bins, time_steps)
        x = self.tcn(x)                # -> (batch, num_channels[-1], time_steps)
        x = x.transpose(1, 2)          # -> (batch, time_steps, num_channels[-1])
        return self.output_layer(x)    # -> (batch, time_steps, 1)
