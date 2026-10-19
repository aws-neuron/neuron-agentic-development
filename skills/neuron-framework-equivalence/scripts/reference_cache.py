"""Bounded exact-token references; target correctness is never cached."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def checkpoint_identity(model_path, revision=None):
    root = Path(model_path)
    if root.is_dir():
        if revision:
            raise ValueError("a local checkpoint uses its content identity, not an HF revision")
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            files.append([path.relative_to(root).as_posix(), checksum.hexdigest()])
        if not files:
            raise ValueError("checkpoint directory is empty")
        return "sha256:" + digest(files)
    if root.exists():
        raise ValueError("local checkpoint must be a directory")
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a remote HF checkpoint requires an immutable 40-hex model revision")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", str(model_path)):
        raise ValueError("remote model-path must be an HF repository ID")
    return f"{model_path}@{revision}"


def validate_tokens(rows, count, length):
    if (not isinstance(rows, list) or len(rows) != count or
            any(not isinstance(row, list) or len(row) != length or
                any(type(token) is not int or token < 0 for token in row) for row in rows)):
        raise ValueError(f"expected {count} token sequences of exactly {length} tokens")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".reference-", delete=False) as stream:
            name = stream.name
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


class ReferenceCache:
    def __init__(self, root, max_entries=16, max_bytes=64 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("reference cache bounds must be positive")
        self.root = Path(root)
        self.max_entries, self.max_bytes = max_entries, max_bytes

    def get_or_create(self, identity, generate):
        required = {"checkpoint", "prompt_token_ids", "code_revision", "runtime",
                    "precision", "method", "input_tokens", "output_tokens"}
        if not required <= identity.keys():
            raise ValueError(f"reference identity requires {sorted(required)}")
        for field in required - {"prompt_token_ids", "input_tokens", "output_tokens"}:
            if not identity[field]:
                raise ValueError(f"reference identity requires {field}")
        for field in ("input_tokens", "output_tokens"):
            if type(identity[field]) is not int or identity[field] < 1:
                raise ValueError(f"{field} must be positive")
        prompts = identity["prompt_token_ids"]
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("reference prompts cannot be empty")
        validate_tokens(prompts, len(prompts), identity["input_tokens"])
        key = digest(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise RuntimeError(
                    f"reference cache busy at {self.root}; reconcile/resume the existing producer"
                ) from error
            for leftover in self.root.glob(".reference-*"):
                leftover.unlink()
            path = self.root / f"{key}.json"
            if path.is_file():
                if path.stat().st_size > self.max_bytes:
                    raise ValueError("cached reference exceeds byte limit")
                saved = json.loads(path.read_text())
                if saved["identity"] != identity or saved["sha256"] != digest(saved["tokens"]):
                    raise ValueError("cached reference identity or content digest mismatch")
                validate_tokens(saved["tokens"], len(prompts), identity["output_tokens"])
                os.utime(path, None)
                return saved["tokens"], {"key": key, "hit": True}
            tokens = generate()
            validate_tokens(tokens, len(prompts), identity["output_tokens"])
            document = {"identity": identity, "tokens": tokens, "sha256": digest(tokens)}
            data = json.dumps(document, separators=(",", ":")).encode()
            if len(data) > self.max_bytes:
                raise ValueError("reference exceeds cache byte limit")
            entries = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
            size = sum(p.stat().st_size for p in entries)
            while entries and (len(entries) >= self.max_entries or
                               size + len(data) > self.max_bytes):
                oldest = entries.pop(0)
                size -= oldest.stat().st_size
                oldest.unlink()
            write_json(path, document)
            return tokens, {"key": key, "hit": False}
