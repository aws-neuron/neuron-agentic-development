"""Linux process identities and atomic ownership records."""

import contextlib
import fcntl
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    sid: int
    start_ticks: int
    state: str

    @classmethod
    def read(cls, pid: int):
        try:
            # Only start time and state require procfs; the command can contain ')'.
            fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
            return cls(pid, os.getpgid(pid), os.getsid(pid), int(fields[19]), fields[0])
        except (OSError, ValueError, IndexError):
            return None

    @property
    def is_alive(self) -> bool:
        return self.state not in ("Z", "X")

    def matches(self, recorded: dict) -> bool:
        return self.is_alive and all(
            getattr(self, field) == recorded[field]
            for field in ("pid", "pgid", "sid", "start_ticks")
        )


def group_members(pgid: int) -> dict:
    members = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            pid = int(path.name)
            identity = ProcessIdentity.read(pid)
            if identity and identity.pgid == pgid:
                members[str(pid)] = asdict(identity)
    return members


def new_group_record(leader: ProcessIdentity) -> dict:
    return {
        "pid": leader.pid,
        "pgid": leader.pgid,
        "boot_id": _boot_id(),
        "members": {str(leader.pid): asdict(leader)},
    }


def assert_group_owned(doc: dict) -> None:
    """Require a live recorded member of the original session, even if its leader exited."""
    pgid = int(doc["pgid"])
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError(f"refusing caller/init process group {pgid}")
    if not doc.get("boot_id") or not doc.get("members"):
        raise ValueError("runfile lacks process identity; refusing unverified ownership")
    if doc["boot_id"] != _boot_id():
        raise ValueError("runfile belongs to another system boot")
    for pid, recorded in doc["members"].items():
        current = ProcessIdentity.read(int(pid))
        if (
            current
            and current.pgid == pgid
            and current.sid == int(doc["pid"])
            and current.matches(recorded)
        ):
            return
    raise ValueError("recorded processes exited or their identities changed")


def snapshot_group(doc: dict) -> None:
    assert_group_owned(doc)
    doc["members"] = group_members(int(doc["pgid"]))


@contextlib.contextmanager
def runfile_lock(path: str):
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def write_runfile(path: str, doc: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target.parent, prefix=target.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(doc, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def remember_members(path: str, doc: dict) -> None:
    with runfile_lock(path):
        current = json.loads(Path(path).read_text())
        if current.get("launch_id") != doc["launch_id"]:
            raise ValueError("runfile now describes another launch")
        snapshot_group(current)
        write_runfile(path, current)
        doc.update(current)
