"""Best-effort profiler diagnostics without constructing a model."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path


def probe(serve_config, *, version=importlib.metadata.version,
          load=importlib.import_module):
    # None means unverified, not unsupported. Versions are provenance only;
    # distribution metadata and private Python entry points can differ by installation.
    report = {"supported": None, "versions": {}, "errors": [], "warnings": []}
    for package in ("vllm", "vllm-neuron", "libtorch-neuronx-lite"):
        try:
            report["versions"][package] = version(package)
        except (ImportError, OSError, ValueError) as error:
            report["versions"][package] = None
            report["warnings"].append(f"{package} version unavailable: {error}")
    try:
        module = load("vllm_neuron.vllm.worker.neuron_profiler")
        config_class = load("vllm.config").ProfilerConfig
        config_class(**serve_config["profiler_config"])
        module.NeuronProfilerConfig(
            serve_config["additional_config"]["neuron_profiler"]
        )
        report["supported"] = True
    except (ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        report["warnings"].append(
            f"Profiler interface unverified: {type(error).__name__}: {error}. "
            "Use the installed serving interface and check the resulting traces."
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_spec")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        serve_config = json.loads(Path(args.run_spec).read_text())["serve_config"]
        if not isinstance(serve_config, dict):
            raise ValueError("serve_config must be an object")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"cannot read run specification: {error}")
    report = probe(serve_config)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    # Success means a diagnostic was produced, not that a hardware capture was verified.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
