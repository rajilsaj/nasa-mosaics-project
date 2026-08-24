"""
Export a trained BowShockTCN checkpoint to ONNX, in a form that survives
SNPE conversion.
 
Background
----------
The trained model applies its final `nn.Linear` head to a (batch, time,
channels) tensor, relying on PyTorch broadcasting the linear layer across
the time dimension. Exported as-is, that becomes a `Transpose` feeding a
rank-3 `Gemm`/`MatMul` right at the TCN -> dense boundary -- a pattern
SNPE's ONNX importer does not reliably support (its FullyConnected op
expects 2D input, and its Transpose support is weak outside image-style
layouts).
 
This script exports a wrapper model (`BowShockTCNExport`) that swaps that
Linear head for a mathematically identical 1x1 Conv1d applied directly in
(batch, channels, time) layout -- the layout the TCN stack already
produces -- so no transpose sits between the conv stack and the final
layer. No retraining is required: the Conv1d's weights are copied
straight from the trained Linear layer, and a sanity check confirms the
copy reproduces the original model's output exactly before anything is
exported.
 
Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pt --output checkpoints/best_model.onnx
"""
 
import argparse
from pathlib import Path
 
import torch
import torch.nn as nn
 
from tcn_model import BowShockTCN
 
 
# ---------------------------------------------------------------------------
# Model loading (mirrors inference.py's load_model)
# ---------------------------------------------------------------------------
 
def load_model(checkpoint_path: Path, num_energy_bins: int) -> BowShockTCN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
 
    model = BowShockTCN(
        num_energy_bins = saved_args.get("num_energy_bins", num_energy_bins),
        num_channels    = saved_args.get("num_channels", [32, 64, 128, 64, 32]),
        dropout         = saved_args.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
 
 
# ---------------------------------------------------------------------------
# SNPE-safe export wrapper
# ---------------------------------------------------------------------------
 
class BowShockTCNExport(nn.Module):
    """
    Export-only wrapper around a trained BowShockTCN.
 
    Replaces the per-timestep nn.Linear head with a mathematically
    equivalent 1x1 Conv1d applied while the tensor is still in
    (batch, channels, time) layout, instead of transposing to
    (batch, time, channels) first. This removes the Transpose -> Linear
    (3D) pattern at the conv/dense boundary that trips up SNPE's ONNX
    converter, without changing what the model computes.
 
    Output layout matches the original model, (batch, time_steps, 1),
    unless strip_output_transpose=True, in which case the trailing
    transpose is left out of the traced graph entirely (output becomes
    (batch, 1, time_steps)) for maximum SNPE op compatibility -- do the
    reshape in application code instead.
    """
 
    def __init__(self, trained_model: BowShockTCN, strip_output_transpose: bool = False):
        super().__init__()
        self.tcn = trained_model.tcn
        self.strip_output_transpose = strip_output_transpose
 
        dropout_p = 0.0
        linear = None
        for layer in trained_model.output_layer:
            if isinstance(layer, nn.Dropout):
                dropout_p = layer.p
            elif isinstance(layer, nn.Linear):
                linear = layer
        if linear is None:
            raise ValueError("Expected an nn.Linear inside output_layer, found none")
 
        conv = nn.Conv1d(linear.in_features, linear.out_features, kernel_size=1)
        with torch.no_grad():
            conv.weight.copy_(linear.weight.unsqueeze(-1))  # (out, in) -> (out, in, 1)
            conv.bias.copy_(linear.bias)
 
        # Dropout is a no-op in eval mode; kept only for architectural parity.
        self.output_layer = nn.Sequential(nn.Dropout(dropout_p), conv)
        self.eval()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (batch, time, energy_bins) -> (batch, energy_bins, time)
        x = self.tcn(x)                # -> (batch, num_channels[-1], time)
        x = self.output_layer(x)       # -> (batch, 1, time) -- 1x1 Conv1d, no transpose needed
        if self.strip_output_transpose:
            return x                   # (batch, 1, time_steps)
        return x.transpose(1, 2)       # (batch, time_steps, 1) -- matches original model's output shape
 
 
def sanity_check_export_model(
    trained_model: BowShockTCN,
    export_model: BowShockTCNExport,
    num_energy_bins: int,
    time_steps: int,
):
    """Confirm the Conv1d weight-copy reproduces the original model's output exactly."""
    x = torch.randn(2, time_steps, num_energy_bins)
    with torch.no_grad():
        original_out = trained_model(x)
        export_out = export_model(x)
        if export_model.strip_output_transpose:
            export_out = export_out.transpose(1, 2)
    max_diff = (original_out - export_out).abs().max().item()
    if max_diff > 1e-5:
        raise RuntimeError(
            f"Export wrapper does not match original model output (max diff={max_diff:.2e}). "
            "Check the Linear -> Conv1d weight copy."
        )
    print(f"Sanity check passed: export wrapper matches original model (max diff={max_diff:.2e}).")
 
 
# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
 
def export_onnx(
    model: nn.Module,
    output_path: Path,
    num_energy_bins: int,
    time_steps: int,
    opset: int,
    dynamic_axes: bool,
    output_time_axis: int,
):
    dummy_input = torch.randn(1, time_steps, num_energy_bins)
 
    axes = None
    if dynamic_axes:
        axes = {
            "input":  {0: "batch", 1: "time_steps"},
            "logits": {0: "batch", output_time_axis: "time_steps"},
        }
 
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=opset,
        dynamo=False,  # SNPE targets low opsets (e.g. 11); the legacy TorchScript-based
                        # exporter honors that directly. Newer torch defaults to the dynamo
                        # exporter, which targets much higher opsets and needs onnxscript.
    )
    print(f"Saved ONNX model to {output_path}")
 
 
def verify_onnx(model: nn.Module, output_path: Path, num_energy_bins: int, time_steps: int):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("onnx / onnxruntime not installed -- skipping verification")
        return
 
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
 
    dummy_input = torch.randn(1, time_steps, num_energy_bins)
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
 
    session = ort.InferenceSession(str(output_path))
    ort_out = session.run(None, {"input": dummy_input.numpy()})[0]
 
    max_diff = abs(torch_out - ort_out).max()
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX output diverges from PyTorch output (max diff={max_diff:.2e})")
    print(f"ONNX model check + inference passed (max diff vs PyTorch={max_diff:.2e}).")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Export a BowShockTCN checkpoint to ONNX (SNPE-safe)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output", default=None, help="Output .onnx path (defaults to checkpoint name)")
    parser.add_argument("--num-energy-bins", type=int, default=63, help="Fallback if not in checkpoint args")
    parser.add_argument("--time-steps", type=int, default=128, help="Sequence length used to trace the model")
    parser.add_argument(
        "--opset", type=int, default=11,
        help="ONNX opset. Kept low (11) by default for SNPE ONNX-importer compatibility; "
             "raise it if you are not targeting SNPE.",
    )
    parser.add_argument("--no-dynamic-axes", action="store_true", help="Fix batch/time_steps size in the graph")
    parser.add_argument(
        "--strip-output-transpose", action="store_true",
        help="Omit the final transpose from the graph; output is (batch, 1, time_steps) instead of "
             "(batch, time_steps, 1). Do the reshape outside the model. Maximizes SNPE op compatibility.",
    )
    args = parser.parse_args()
 
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")
 
    if output_path.is_dir():
        output_path = output_path / checkpoint_path.with_suffix(".onnx").name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path.resolve()}")
 
    trained_model = load_model(checkpoint_path, args.num_energy_bins)
    num_energy_bins = trained_model.tcn[0].block[0].in_channels
    print(f"Loaded model from {checkpoint_path} (energy_bins={num_energy_bins})")
 
    export_model = BowShockTCNExport(trained_model, strip_output_transpose=args.strip_output_transpose)
    sanity_check_export_model(trained_model, export_model, num_energy_bins, args.time_steps)
 
    output_time_axis = 2 if args.strip_output_transpose else 1
 
    export_onnx(
        export_model,
        output_path,
        num_energy_bins=num_energy_bins,
        time_steps=args.time_steps,
        opset=args.opset,
        dynamic_axes=not args.no_dynamic_axes,
        output_time_axis=output_time_axis,
    )
    verify_onnx(export_model, output_path, num_energy_bins, args.time_steps)
 
 
if __name__ == "__main__":
    main()