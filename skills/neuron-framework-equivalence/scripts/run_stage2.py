#!/usr/bin/env python3
"""
Stage 2: Component Test Runner.

Imports and executes test_*.py files in a tests directory, captures results.
Supports both:
  - R-ratio format (tensor_compare.py: "R-ratio = X.XXXX (threshold=Y.YYYY)")
  - BC/σ-ratio format (assert_close_three_way: "✓ PASS name" / "✗ FAIL name")

Usage:
    python3 scripts/run_stage2.py --tests-dir /path/to/tests --tau-r 1.2
"""

import argparse
import importlib
import importlib.util
import io
import json
import re
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Component Test Runner")
    parser.add_argument("--tests-dir", required=True)
    parser.add_argument("--tau-r", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["NXD_CPU_MODE"] = "1"

    # Redirect logging to stdout so we can capture assert_close_three_way output
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(message)s", force=True)

    from pathlib import Path
    tests_path = Path(args.tests_dir)
    test_files = sorted(tests_path.glob("test_*.py"))

    if not test_files:
        print(f"No test_*.py files found in {args.tests_dir}")
        return 1

    if str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))

    # Old format: R-ratio
    r_pattern = re.compile(r"R-ratio\s*=\s*([\d.]+)\s*\(threshold=([\d.]+)\)")
    old_pass_pattern = re.compile(r"\[PASS\]\s+(.+)")
    old_fail_pattern = re.compile(r"\[FAIL\]\s+(.+)")

    # New format: BC/σ-ratio from assert_close_three_way
    bc_pass_pattern = re.compile(r"✓ PASS\s+(.+)")
    bc_fail_pattern = re.compile(r"✗ FAIL\s+(.+)")
    sigma_pattern = re.compile(r"σ-ratio:\s*([\d.]+)\s+BC:\s*([\d.]+)")

    results = []
    total = passed = failed = 0

    for test_file in test_files:
        try:
            mod_name = f"_eq_{test_file.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, str(test_file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            name = test_file.stem.split("_", 2)[-1]
            print(f"\n  ✗ {name}: IMPORT ERROR — {e}")
            results.append({"component": name, "r_ratio": None, "bc": None,
                           "sigma_ratio": None, "passed": False, "error": str(e)})
            total += 1; failed += 1
            continue

        test_fns = [(n, getattr(module, n)) for n in sorted(dir(module))
                    if n.startswith("test_") and callable(getattr(module, n))]

        for fn_name, fn in test_fns:
            total += 1
            name = fn_name.replace("test_", "")

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            captured = io.StringIO()
            sys.stdout = captured
            sys.stderr = captured

            # Also capture logging output
            import logging
            log_handler = logging.StreamHandler(captured)
            log_handler.setLevel(logging.DEBUG)
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            ok = False
            err = None
            try:
                fn()
                ok = True
            except AssertionError as e:
                err = str(e)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                root_logger.removeHandler(log_handler)

            output = captured.getvalue()
            # Also check the error message for BC/σ patterns (assert_close_three_way puts summary there)
            if err:
                output += "\n" + err
            print(output, end="")

            # Try new BC/σ format first
            bc_pass_match = bc_pass_pattern.search(output)
            bc_fail_match = bc_fail_pattern.search(output)
            sigma_match = sigma_pattern.search(output)

            r_ratio = None
            bc = None
            sigma_ratio = None
            test_passed = False

            if bc_pass_match or bc_fail_match:
                # BC/σ-ratio format
                if bc_pass_match:
                    name = bc_pass_match.group(1).strip()
                elif bc_fail_match:
                    name = bc_fail_match.group(1).strip()

                if sigma_match:
                    sigma_ratio = float(sigma_match.group(1))
                    bc = float(sigma_match.group(2))

                test_passed = ok and bc_pass_match is not None
            else:
                # Old R-ratio format
                r_match = r_pattern.search(output)
                r_ratio = float(r_match.group(1)) if r_match else None
                threshold = float(r_match.group(2)) if r_match else args.tau_r

                name_match = old_pass_pattern.search(output) or old_fail_pattern.search(output)
                if name_match:
                    name = name_match.group(1).strip()

                test_passed = ok and (r_ratio is not None and r_ratio < threshold)

            if test_passed:
                passed += 1
            else:
                failed += 1
            if err:
                print(f"  Error: {err}")

            results.append({
                "component": name,
                "r_ratio": r_ratio,
                "bc": bc,
                "sigma_ratio": sigma_ratio,
                "passed": test_passed,
            })

    print(f"\n{'=' * 60}")
    print(f"  Stage 2 Summary: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    for r in results:
        tag = "✓" if r["passed"] else "✗"
        if r.get("bc") is not None:
            metric = f"BC={r['bc']:.4f} σ={r['sigma_ratio']:.3f}"
        elif r.get("r_ratio") is not None:
            metric = f"R={r['r_ratio']:.4f}"
        else:
            metric = "ERROR"
        print(f"  {tag} {r['component']}: {metric}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"results": results, "passed": passed, "failed": failed, "total": total},
                      f, indent=2, default=str)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
