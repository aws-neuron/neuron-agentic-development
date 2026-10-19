"""Shared artifact layout and copy exclusions for builds and deployment."""

ARTIFACT_DIRS = ("agent-sops", "agents", "context", "evals", "hooks", "skills", "tools")
IGNORE_PATTERNS = ("__pycache__", "*.py[co]")
