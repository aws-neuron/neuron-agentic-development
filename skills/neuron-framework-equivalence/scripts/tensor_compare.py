# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Three-tensor and two-tensor comparison utility for the equivalence framework.

Implements the R-ratio metric from the design doc:

    R_{l,k} = ||T̃^{port,16}_{l,k} - T^{ref,32}_{l,k}||_F
              / (||T^{ref,16}_{l,k} - T^{ref,32}_{l,k}||_F + ε)

Where:
  - T^{ref,32} = reference module in FP32 (ground truth)
  - T^{ref,16} = reference module in BF16 (precision baseline)
  - T̃^{port,16} = ported module in BF16 (target under test)

Supports:
  - 3-tensor comparison (compare_3tensors) with R-ratio
  - 2-tensor comparison (compare_2tensors) for pairwise checks
  - QQ plot + histogram visualization for error distribution analysis
  - CLI mode for comparing pre-saved .pt tensor files
  - Self-test mode for verifying API usage
"""

import torch
import numpy as np
from typing import Dict, Optional, Union
from argparse import ArgumentParser
import os


# Machine epsilon constants
EPSILON_FP32 = 2**-23
EPSILON_BF16 = 2**-7


def _to_np64(t):
    """Convert a tensor to flattened numpy float64."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return t.reshape(-1).astype(np.float64)


# ── Visualization ──

def _visualize_differences_one_series(
    diff, output_folder, figure_format="png", output_prefix="",
):
    """Visualize a single error series as a histogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_folder, exist_ok=True)

    title = (f"min {np.min(diff):.4f}, max {np.max(diff):.4f}, "
             f"mean {np.mean(diff):.4f}, std {np.std(diff):.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diff.flatten(), bins=100, alpha=0.75, edgecolor='black')
    ax.set_xlabel('Difference')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_folder}/{output_prefix}_histogram.{figure_format}",
        format=figure_format, dpi=150, bbox_inches='tight',
    )
    plt.close(fig)


def _visualize_differences_two_series(
    elem_diff1: np.ndarray,
    elem_diff2: np.ndarray,
    output_folder,
    output_prefix,
    figure_format="png",
    num_quantiles=10000,
):
    """Plot overlaid histogram and QQ plot for two error series.

    The QQ plot is the primary diagnostic: points on the 45-degree line
    indicate matching distributions (PASS). Systematic deviations indicate
    a distributional shift pointing to a specific implementation error.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_folder, exist_ok=True)

    title = (
        f"series 1: min {np.min(elem_diff1):.4f}, max {np.max(elem_diff1):.4f}, "
        f"mean {np.mean(elem_diff1):.4f}, std {np.std(elem_diff1):.4f}\n"
        f"series 2: min {np.min(elem_diff2):.4f}, max {np.max(elem_diff2):.4f}, "
        f"mean {np.mean(elem_diff2):.4f}, std {np.std(elem_diff2):.4f}"
    )
    print(f"num elements in each series: {elem_diff1.flatten().size}, "
          f"{elem_diff2.flatten().size}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    if elem_diff1.flatten().size != elem_diff2.flatten().size:
        raise ValueError(
            f"Both series must have the same number of elements, "
            f"got {elem_diff1.flatten().size} and {elem_diff2.flatten().size}"
        )

    n_bins = min(1000, int(np.sqrt(len(elem_diff1.flatten()))))

    # Histogram subplot
    ax1.hist(elem_diff1.flatten(), bins=n_bins, alpha=0.75, color='blue',
             label='Error series 1', edgecolor=None, density=True)
    ax1.hist(elem_diff2.flatten(), bins=n_bins, alpha=0.75, color='red',
             label='Error series 2', edgecolor=None, density=True)
    ax1.set_xlabel('Difference')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set x-axis range for histogram
    mean_val = np.mean(elem_diff1)
    std_val = np.max((np.std(elem_diff1), np.std(elem_diff2)))
    ax1.set_xlim([mean_val - 5 * std_val, mean_val + 5 * std_val])

    # QQ-Plot subplot
    n = min(len(elem_diff1), len(elem_diff2), num_quantiles)
    p = np.linspace(0, 1, n)
    sorted_diff1 = np.sort(elem_diff1.flatten())
    sorted_diff2 = np.sort(elem_diff2.flatten())
    q1 = np.quantile(sorted_diff1, p)
    q2 = np.quantile(sorted_diff2, p)

    ax2.scatter(q1, q2, c='green', s=5, alpha=0.6, label='QQ-Plot')

    # Add 45-degree reference line
    min_val = min(q1.min(), q2.min())
    max_val = max(q1.max(), q2.max())
    ref_line = np.linspace(min_val, max_val, 100)
    ax2.plot(ref_line, ref_line, 'r--', label='45-degree line')

    ax2.set_xlabel('Quantiles Series 1')
    ax2.set_ylabel('Quantiles Series 2')
    ax2.set_title('QQ-Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()

    plt.savefig(
        f"{output_folder}/{output_prefix}_qqplot_hist.{figure_format}",
        format=figure_format, dpi=150, bbox_inches='tight',
    )
    plt.close(fig)


# ── Core comparison functions ──

def compare_3tensors(
    ref_fp32: Union[torch.Tensor, np.ndarray],
    ref_bf16: Union[torch.Tensor, np.ndarray],
    port_bf16: Union[torch.Tensor, np.ndarray],
    visualize: bool = False,
    figure_format: str = "png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
) -> Dict[str, float]:
    """Three-tensor comparison returning the full metric suite.

    Args:
        ref_fp32:  Reference output in FP32 (ground truth)
        ref_bf16:  Reference output in BF16 (precision baseline)
        port_bf16: Ported output in BF16 (target under test)
        visualize: If True, generate QQ plot + histogram
        figure_format: Image format (png, pdf, svg, jpg)
        output_folder: Directory for visualization output
        output_prefix: Filename prefix for plots

    Returns:
        Dict with 12 metrics (6 for baseline _2_1, 6 for target _3_1)
        plus the r_ratio.
    """
    t1 = _to_np64(ref_fp32)
    t2 = _to_np64(ref_bf16)
    t3 = _to_np64(port_bf16)

    diff_2_1 = t2 - t1
    diff_3_1 = t3 - t1

    norm_t1 = np.linalg.norm(t1)

    fro_2_1 = np.linalg.norm(diff_2_1) / (norm_t1 + 1e-30)
    fro_3_1 = np.linalg.norm(diff_3_1) / (norm_t1 + 1e-30)
    inf_2_1 = np.max(np.abs(diff_2_1)) / (np.max(np.abs(t1)) + 1e-30)
    inf_3_1 = np.max(np.abs(diff_3_1)) / (np.max(np.abs(t1)) + 1e-30)

    rel_2_1 = np.where(t1 != 0, np.abs(diff_2_1) / np.abs(t1), 0)
    rel_3_1 = np.where(t1 != 0, np.abs(diff_3_1) / np.abs(t1), 0)

    # R-ratio: R = ||port - ref_fp32||_F / (||ref_bf16 - ref_fp32||_F + ε)
    numerator = np.linalg.norm(diff_3_1)
    denominator = np.linalg.norm(diff_2_1)
    r_ratio = numerator / (denominator + 1e-12)

    if visualize:
        if output_folder is None:
            raise ValueError("output_folder must be specified when visualize=True")
        _visualize_differences_two_series(
            diff_2_1, diff_3_1,
            output_folder=output_folder,
            output_prefix=output_prefix + "_diff",
            figure_format=figure_format,
        )
        _visualize_differences_two_series(
            rel_2_1, rel_3_1,
            output_folder=output_folder,
            output_prefix=output_prefix + "_rel_diff",
            figure_format=figure_format,
        )

    return {
        # Baseline (ref_bf16 vs ref_fp32)
        "abs_max_elem_diff_2_1": float(np.max(np.abs(diff_2_1))),
        "abs_mean_elem_diff_2_1": float(np.mean(np.abs(diff_2_1))),
        "rel_max_elem_diff_2_1": float(np.max(rel_2_1)),
        "rel_mean_elem_diff_2_1": float(np.mean(rel_2_1)),
        "rel_matrix_fro_norm_dist_2_1": float(fro_2_1),
        "rel_matrix_inf_norm_dist_2_1": float(inf_2_1),
        # Target (port_bf16 vs ref_fp32)
        "abs_max_elem_diff_3_1": float(np.max(np.abs(diff_3_1))),
        "abs_mean_elem_diff_3_1": float(np.mean(np.abs(diff_3_1))),
        "rel_max_elem_diff_3_1": float(np.max(rel_3_1)),
        "rel_mean_elem_diff_3_1": float(np.mean(rel_3_1)),
        "rel_matrix_fro_norm_dist_3_1": float(fro_3_1),
        "rel_matrix_inf_norm_dist_3_1": float(inf_3_1),
        # R-ratio
        "r_ratio": float(r_ratio),
    }


def compare_2tensors(
    tensor1: Union[torch.Tensor, np.ndarray],
    tensor2: Union[torch.Tensor, np.ndarray],
    visualize: bool = False,
    figure_format: str = "png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
) -> Dict[str, float]:
    """Pairwise comparison of two tensors (no FP32 ground truth needed).

    Useful for:
      - Perturbation-based baseline (Method 2 in equiv-concept)
      - Quick sanity checks between any two outputs
      - Comparing TP=1 vs TP=N outputs

    Args:
        tensor1: Reference tensor
        tensor2: Tensor to compare against reference

    Returns:
        Dict with 6 metrics (abs/rel max/mean, fro norm, inf norm).
    """
    t1 = _to_np64(tensor1)
    t2 = _to_np64(tensor2)

    diff = t1 - t2
    abs_diff = np.abs(diff)

    fro_norm_err = np.linalg.norm(abs_diff) / (np.linalg.norm(t1) + 1e-30)
    inf_norm_err = np.max(abs_diff) / (np.max(np.abs(t1)) + 1e-30)

    elem_wise_rel_err = np.where(t1 != 0, abs_diff / np.abs(t1), 0)

    if visualize:
        if output_folder is None:
            raise ValueError("output_folder must be specified when visualize=True")
        _visualize_differences_one_series(
            diff=diff, output_folder=output_folder,
            figure_format=figure_format, output_prefix=output_prefix + "_diff",
        )
        _visualize_differences_one_series(
            diff=elem_wise_rel_err, output_folder=output_folder,
            figure_format=figure_format, output_prefix=output_prefix + "_rel_diff",
        )

    return {
        "abs_max_elem_diff": float(np.max(abs_diff)),
        "abs_mean_elem_diff": float(np.mean(abs_diff)),
        "rel_max_elem_diff": float(np.max(elem_wise_rel_err)),
        "rel_mean_elem_diff": float(np.mean(elem_wise_rel_err)),
        "rel_matrix_fro_norm_dist": float(fro_norm_err),
        "rel_matrix_inf_norm_dist": float(inf_norm_err),
    }


def compute_r_ratio(
    ref_fp32: Union[torch.Tensor, np.ndarray],
    ref_bf16: Union[torch.Tensor, np.ndarray],
    port_bf16: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute just the R-ratio (shorthand for quick checks)."""
    result = compare_3tensors(ref_fp32, ref_bf16, port_bf16)
    return result["r_ratio"]


def check_3tensor_result(
    result: Dict[str, float],
    component_name: str,
    tolerance_ratio: float = 1.2,
    verbose: bool = True,
) -> bool:
    """Check whether the target's error is within tolerance of the baseline.

    Returns True if PASS (r_ratio < tolerance_ratio).
    """
    r_ratio = result["r_ratio"]
    passed = r_ratio < tolerance_ratio

    if verbose:
        tag = "PASS" if passed else "FAIL"
        print(f"\n{'=' * 60}")
        print(f"  [{tag}] {component_name}")
        print(f"{'=' * 60}")
        print(f"  Baseline (ref_bf16 vs ref_fp32):")
        print(f"    rel_fro_norm  = {result['rel_matrix_fro_norm_dist_2_1']:.6e}")
        print(f"    abs_max_diff  = {result['abs_max_elem_diff_2_1']:.6e}")
        print(f"  Target (port_bf16 vs ref_fp32):")
        print(f"    rel_fro_norm  = {result['rel_matrix_fro_norm_dist_3_1']:.6e}")
        print(f"    abs_max_diff  = {result['abs_max_elem_diff_3_1']:.6e}")
        print(f"  R-ratio = {r_ratio:.4f}  (threshold={tolerance_ratio})")
        print(f"{'=' * 60}")

    return passed


# ── Batch comparison helpers ──

def compare_tensors_from_files(
    list_of_tensor_files: list,
    visualize: bool = False,
    figure_format: str = "png",
    output_folder: Optional[str] = None,
    output_prefix: str = "",
):
    """Load .pt files and compare them (2 or 3 files)."""
    if len(list_of_tensor_files) not in [2, 3]:
        raise ValueError(f"Must provide 2 or 3 tensor files, got {len(list_of_tensor_files)}")

    tensors = [torch.load(f, map_location="cpu", weights_only=True) for f in list_of_tensor_files]

    if len(tensors) == 3:
        result = compare_3tensors(
            tensors[0], tensors[1], tensors[2],
            visualize=visualize, figure_format=figure_format,
            output_folder=output_folder, output_prefix=output_prefix,
        )
    else:
        result = compare_2tensors(
            tensors[0], tensors[1],
            visualize=visualize, figure_format=figure_format,
            output_folder=output_folder, output_prefix=output_prefix,
        )

    print(f"Comparison results: {result}")
    return result


# ── CLI ──

def _test_api_usage():
    """Self-test: generate random tensors and run both comparison modes."""
    t1 = torch.randn(100, 100, dtype=torch.float32)
    t2 = t1 + torch.randn(100, 100, dtype=torch.float32) * 0.01
    t3 = t1 + torch.randn(100, 100, dtype=torch.float32) * 0.02

    print("=== 2-tensor comparison ===")
    r2 = compare_2tensors(t1, t2, visualize=True,
                          output_folder="output", output_prefix="self_test_2t")
    print(r2)

    print("\n=== 3-tensor comparison ===")
    r3 = compare_3tensors(t1, t2, t3, visualize=True,
                          output_folder="output", output_prefix="self_test_3t")
    print(r3)
    print(f"\nR-ratio: {r3['r_ratio']:.4f}")
    print(f"PASS: {r3['r_ratio'] < 1.2}")


def parse_args():
    parser = ArgumentParser(description="Tensor comparison utility")
    parser.add_argument("--tensor_file1", type=str, required=True,
                        help="Path to first tensor file (.pt)")
    parser.add_argument("--tensor_file2", type=str, required=True,
                        help="Path to second tensor file (.pt)")
    parser.add_argument("--tensor_file3", type=str, default=None,
                        help="Path to third tensor file (optional, for 3-tensor comparison)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate QQ plot + histogram")
    parser.add_argument("--figure_format", type=str, default="png",
                        choices=["png", "pdf", "svg", "jpg"])
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Directory for visualization output")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Filename prefix for output files")
    parser.add_argument("--basic_self_test", action="store_true",
                        help="Run self-test with random tensors")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.basic_self_test:
        _test_api_usage()
        exit(0)

    tensor_files = [args.tensor_file1, args.tensor_file2]
    if args.tensor_file3:
        tensor_files.append(args.tensor_file3)

    compare_tensors_from_files(
        list_of_tensor_files=tensor_files,
        visualize=args.visualize,
        figure_format=args.figure_format,
        output_folder=args.output_folder,
        output_prefix=args.output_prefix,
    )
