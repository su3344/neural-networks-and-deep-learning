"""gpu_benchmark.py
~~~~~~~~~~~~~~~~~~

Benchmark feedforward neural network inference on available GPU hardware.
Useful for comparing inference throughput and latency across GPU
architectures such as the NVIDIA V100 and GeForce RTX 5090.

The network architecture mirrors the one described in the book:
  - Input layer:  784 neurons  (28×28 MNIST pixels)
  - Hidden layer: 100 neurons
  - Output layer:  10 neurons  (digit classes 0-9)

Usage (requires PyTorch):
    python gpu_benchmark.py [--batch-sizes 1 32 128 512] [--warmup 50]
                            [--runs 200] [--device cuda]

If no CUDA device is available the benchmark falls back to CPU.
"""

import argparse
import time

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class FeedforwardNet(nn.Module):
    """Simple sigmoid feedforward network matching the book's architecture."""

    def __init__(self, sizes):
        """
        Parameters
        ----------
        sizes : list[int]
            Number of neurons in each layer, e.g. [784, 100, 10].
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _device_info(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        return (
            f"{props.name}  |  "
            f"SM count: {props.multi_processor_count}  |  "
            f"VRAM: {total_gb:.1f} GB  |  "
            f"Compute capability: {props.major}.{props.minor}"
        )
    return "CPU"


def benchmark(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    input_dim: int,
    warmup_runs: int,
    timed_runs: int,
) -> dict:
    """Run inference benchmark and return timing statistics.

    Parameters
    ----------
    model       : the network (already moved to *device*)
    device      : torch.device to benchmark on
    batch_size  : number of samples per forward pass
    input_dim   : number of input features (784 for MNIST)
    warmup_runs : number of forward passes to discard (GPU warm-up)
    timed_runs  : number of forward passes to measure

    Returns
    -------
    dict with keys: batch_size, throughput_sps, latency_ms_mean,
                    latency_ms_p50, latency_ms_p99
    """
    model.eval()
    x = torch.randn(batch_size, input_dim, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - t0)

    latencies_ms = [l * 1000 for l in latencies]
    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p50_ms = latencies_ms[int(len(latencies_ms) * 0.50)]
    p99_ms = latencies_ms[int(len(latencies_ms) * 0.99)]
    throughput = batch_size / (mean_ms / 1000)

    return {
        "batch_size": batch_size,
        "throughput_sps": throughput,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50_ms,
        "latency_ms_p99": p99_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark neural network inference on GPU (V100 / RTX 5090) or CPU."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 32, 128, 512, 1024],
        metavar="N",
        help="Batch sizes to benchmark (default: 1 32 128 512 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        metavar="N",
        help="Number of warm-up forward passes before timing (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        metavar="N",
        help="Number of timed forward passes per batch size (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        metavar="DEVICE",
        help='PyTorch device string, e.g. "cuda", "cuda:0", "cpu". '
             "Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[784, 100, 10],
        metavar="N",
        help="Layer sizes for the network (default: 784 100 10)",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("Neural Network GPU Inference Benchmark")
    print("=" * 70)
    print(f"Device     : {_device_info(device)}")
    print(f"Network    : {' → '.join(str(s) for s in args.sizes)}")
    print(f"Warm-up    : {args.warmup} runs")
    print(f"Timed runs : {args.runs} per batch size")
    print("=" * 70)

    model = FeedforwardNet(args.sizes).to(device)

    header = (
        f"{'Batch':>6}  {'Throughput (sps)':>18}  "
        f"{'Mean (ms)':>10}  {'P50 (ms)':>10}  {'P99 (ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    for bs in args.batch_sizes:
        result = benchmark(
            model=model,
            device=device,
            batch_size=bs,
            input_dim=args.sizes[0],
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )
        print(
            f"{result['batch_size']:>6}  "
            f"{result['throughput_sps']:>18,.0f}  "
            f"{result['latency_ms_mean']:>10.4f}  "
            f"{result['latency_ms_p50']:>10.4f}  "
            f"{result['latency_ms_p99']:>10.4f}"
        )

    print("=" * 70)
    print("Columns: Batch = batch size | sps = samples per second")
    print("         Mean/P50/P99 = forward-pass latency percentiles in ms")


if __name__ == "__main__":
    main()
