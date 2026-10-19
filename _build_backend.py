"""Custom build backend: copies repo-root artifacts into the package before building."""
import runpy
import shutil
from pathlib import Path
from setuptools.build_meta import *  # noqa: F401,F403
from setuptools.build_meta import build_wheel as _build_wheel, build_sdist as _build_sdist

_ROOT = Path(__file__).parent
_PKG_ARTIFACTS = _ROOT / "src" / "neuron_agentic_development" / "artifacts"
# The build backend runs before the package is installed.
_LAYOUT = runpy.run_path(str(_PKG_ARTIFACTS.parent / "artifact_layout.py"))
ARTIFACT_DIRS = _LAYOUT["ARTIFACT_DIRS"]
IGNORE_PATTERNS = _LAYOUT["IGNORE_PATTERNS"]


def _bundle():
    _PKG_ARTIFACTS.mkdir(exist_ok=True)
    for d in ARTIFACT_DIRS:
        src = _ROOT / d
        dst = _PKG_ARTIFACTS / d
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
            )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _bundle()
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _bundle()
    return _build_sdist(sdist_directory, config_settings)
