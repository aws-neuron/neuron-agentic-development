"""Launch an isolated process group and verify its service readiness.

The runfile records process identities and the initial log offset. Teardown refuses
stale or unverifiable identities. The existing launch/wait/teardown/status commands
are shared by the optimization and profiling skill entry points.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

from .process import (
    ProcessIdentity, assert_group_owned, group_members, new_group_record,
    remember_members, runfile_lock, snapshot_group, write_runfile,
)
from .server_readiness import LaunchLog, owns_health_endpoint


# ----------------------------------------------------------------------------- helpers
def _now() -> float:
    return time.time()


def _pgid_alive(pgid: int) -> bool:
    """True if any process in the group is still alive (signal 0 probes without killing)."""
    try:
        os.killpg(pgid, 0)
        members = group_members(pgid)
        return not members or any(ProcessIdentity(**p).is_alive for p in members.values())
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned such that we cannot signal — treat as alive so we do not
        # falsely report it gone.
        return True


def _caller_pgid() -> int:
    """The process group of THIS interpreter — the group teardown must NEVER signal."""
    return os.getpgrp()


def _reap_group(pgid: int) -> None:
    """Best-effort reap of any zombies in the group that are OUR children.

    In production teardown runs in a different process than the launcher, so the server
    processes are children of init and init reaps them — this is a no-op (ChildProcessError,
    caught). It only matters when the caller happens to be the parent (e.g. the selftest),
    where an unreaped zombie would otherwise keep the group id 'alive' to os.killpg(pgid, 0).
    """
    try:
        while True:
            wpid, _ = os.waitpid(-pgid, os.WNOHANG)
            if wpid == 0:
                break
    except ChildProcessError:
        pass


def _load_runfile(path: str) -> dict:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class LaunchSpec:
    runfile: Path
    log: Path
    command: tuple[str, ...]

    def __post_init__(self):
        if not self.command:
            raise ValueError("launch requires a command")


def _abort_launch(proc) -> None:
    """Terminate the failed launch and bound the wait even for an uninterruptible child."""
    try:
        if os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"process {proc.pid} did not exit within 5s after SIGKILL") from error


# ------------------------------------------------------------------------------- launch
def launch_process(spec: LaunchSpec) -> dict:
    with runfile_lock(spec.runfile):
        return _launch_process(spec)


def _launch_process(spec: LaunchSpec) -> dict:
    if spec.runfile.exists():
        previous = _load_runfile(spec.runfile)
        if _pgid_alive(int(previous["pgid"])):
            try:
                assert_group_owned(previous)
            except ValueError:
                pass
            else:
                raise ValueError("runfile still names a live process group; use another runfile")
    spec.log.parent.mkdir(parents=True, exist_ok=True)
    with spec.log.open("ab", buffering=0) as log_fh:
        log_stat = os.fstat(log_fh.fileno())
        # setsid creates a new session and process-group leader, isolated from this caller.
        proc = subprocess.Popen(
            spec.command, stdout=log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
        )
    try:
        identity = ProcessIdentity.read(proc.pid)
        if identity is None or not identity.is_alive:
            raise ValueError(f"process {proc.pid} exited during launch; inspect {spec.log}")
        if identity.pgid == _caller_pgid():
            raise ValueError("child shares the caller's process group; isolation failed")
        doc = {
            **new_group_record(identity),
            "cmd": list(spec.command),
            "log": str(spec.log.resolve()),
            "launcher_pgid": _caller_pgid(),
            "start_ts": _now(),
            "launch_id": uuid.uuid4().hex,
            "log_device": log_stat.st_dev,
            "log_inode": log_stat.st_ino,
            "log_offset": log_stat.st_size,
        }
        snapshot_group(doc)
        write_runfile(spec.runfile, doc)
    except Exception:
        _abort_launch(proc)
        raise
    return doc


def cmd_launch(args) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    spec = LaunchSpec(Path(args.runfile), Path(args.log or Path(args.runfile).with_suffix(".log")),
                      tuple(command))
    doc = launch_process(spec)
    print(f"safe_serve: launched pid={doc['pid']} pgid={doc['pgid']} (isolated; caller pgid="
          f"{_caller_pgid()})")
    print(f"  log:     {doc['log']}")
    print(f"  runfile: {os.path.abspath(args.runfile)}")
    print(f"  teardown: python3 -m neuron_agentic_development.server_lifecycle teardown --runfile "
          f"{os.path.abspath(args.runfile)}")
    return 0


# --------------------------------------------------------------------------------- wait
def _health_ok(url: str, doc: dict, timeout=5) -> bool:
    try:
        if not owns_health_endpoint(doc, url):
            return False
        with urllib.request.urlopen(url, timeout=timeout) as response:
            # Named bounds also work on Python 3.10, before HTTPStatus.is_success.
            return HTTPStatus.OK <= response.status < HTTPStatus.MULTIPLE_CHOICES and (
                response.geturl() == url or owns_health_endpoint(doc, response.geturl())
            )
    except (OSError, urllib.error.URLError):
        return False


@dataclass
class WaitProgress:
    start_ts: float
    log: Path | None
    report_every: float
    last_report: float = float("-inf")

    @property
    def log_age(self) -> float:
        # Clamp to this launch: launch_process opens the log with "ab", so a retry into
        # an existing run directory inherits the previous run's mtime until the new server
        # writes its first byte, and --stale can fire immediately. The decode-isolation
        # fallback relaunches into the same directory by design.
        if not (self.log and self.log.exists()):
            return -1
        return _now() - max(self.log.stat().st_mtime, self.start_ts)

    def report(self) -> None:
        now = time.monotonic()
        if now - self.last_report >= self.report_every:
            self.last_report = now
            print(f"safe_serve: waiting... elapsed={_now() - self.start_ts:.0f}s "
                  f"log_stale={self.log_age:.0f}s")


def cmd_wait(args) -> int:
    if not args.health and not args.ready_pattern:
        raise ValueError("wait requires --health URL and/or --ready-pattern")
    doc = _load_runfile(args.runfile)
    pid = int(doc["pid"])
    logf = args.log or doc.get("log")
    assert_group_owned(doc)
    deadline = time.monotonic() + args.timeout
    progress = WaitProgress(doc.get("start_ts", _now()), Path(logf) if logf else None,
                            args.report_every)
    reader = LaunchLog(doc, logf) if args.ready_pattern else None
    while True:
        assert_group_owned(doc)
        # Server process died?
        identity = ProcessIdentity.read(pid)
        if identity is None or not identity.is_alive:
            raise ValueError(f"pid {pid} exited before ready")
        # Ready by health endpoint?
        if args.health and _health_ok(args.health, doc, min(5, max(.001, deadline - time.monotonic()))):
            remember_members(args.runfile, doc)
            print(f"safe_serve: READY (health {args.health}) after "
                  f"{_now() - doc.get('start_ts', _now()):.0f}s from launch.")
            return 0
        # Ready by log pattern?
        if args.ready_pattern and reader.contains(args.ready_pattern):
            remember_members(args.runfile, doc)
            print(f"safe_serve: READY (fresh log pattern {args.ready_pattern!r}).")
            return 0
        if args.stale > 0 and progress.log_age > args.stale:
            raise ValueError(f"no log progress for {progress.log_age:.0f}s (> --stale {args.stale}s)")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"TIMEOUT after {args.timeout}s")
        progress.report()
        time.sleep(min(args.poll, remaining))


# ----------------------------------------------------------------------------- teardown
def _owned_group_alive(doc: dict) -> bool:
    pgid = int(doc["pgid"])
    if pgid <= 1 or pgid == _caller_pgid():
        raise ValueError(f"refusing caller/init process group {pgid}")
    if not _pgid_alive(pgid):
        return False
    try:
        assert_group_owned(doc)
    except ValueError:
        if _pgid_alive(pgid):
            raise
        return False
    return True


def _teardown_group(args, doc: dict) -> None:
    pgid = int(doc["pgid"])
    if not _owned_group_alive(doc):
        return
    try:
        snapshot_group(doc)
    except ValueError:
        if _pgid_alive(pgid):
            raise
        return
    write_runfile(args.runfile, doc)

    for sig, timeout in ((signal.SIGTERM, args.grace), (signal.SIGKILL, 15)):
        if timeout <= 0:
            continue
        if not _owned_group_alive(doc):
            return
        print(f"safe_serve: {sig.name} -> group {pgid}, wait {timeout}s ...")
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while True:
            _reap_group(pgid)
            if not _pgid_alive(pgid):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.5, remaining))
    raise ValueError(f"group {pgid} still alive 15s after SIGKILL")


def cmd_teardown(args) -> int:
    try:
        with runfile_lock(args.runfile):
            doc = _load_runfile(args.runfile)
            error = None
            try:
                _teardown_group(args, doc)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                print(f"safe_serve teardown: REFUSED — {error}", file=sys.stderr)
            receipt = {
                "status": "complete" if error is None else "failed",
                "observed_at": _now(),
                "process": process_status(doc),
            }
            if error is not None:
                receipt["error"] = error
            if error is not None or doc.get("cleanup", {}).get("status") != "complete":
                doc["cleanup"] = receipt
                write_runfile(args.runfile, doc)
            return int(error is not None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"safe_serve teardown: could not record cleanup — {error}", file=sys.stderr)
        return 1


# ------------------------------------------------------------------------------- status
def process_status(doc: dict) -> dict:
    """Observe process ownership; terminal state says nothing about measured results."""
    status = {
        "launch_id": doc.get("launch_id"),
        "pid": int(doc["pid"]),
        "pgid": int(doc["pgid"]),
        "state": "unverified",
        "leader_alive": False,
    }
    try:
        status["state"] = "live" if _owned_group_alive(doc) else "terminal"
        if status["state"] == "live":
            leader = ProcessIdentity.read(status["pid"])
            recorded = doc["members"].get(str(status["pid"]))
            status["leader_alive"] = bool(leader and recorded and leader.matches(recorded))
    except (OSError, ValueError, KeyError, TypeError) as error:
        status["state"] = "unverified"
        status["error"] = str(error)
    return status


def cmd_status(args) -> int:
    doc = _load_runfile(args.runfile)
    status = {**process_status(doc), "cleanup": doc.get("cleanup")}
    if getattr(args, "json", False):
        print(json.dumps(status, sort_keys=True))
    else:
        print(f"runfile : {os.path.abspath(args.runfile)}")
        print(f"cmd     : {' '.join(doc.get('cmd', []))}")
        print(f"log     : {doc.get('log')}")
        print(f"pid     : {status['pid']} (leader_alive={status['leader_alive']})")
        print(f"pgid    : {status['pgid']} ({status['state']})")
        print(f"cleanup : {json.dumps(status['cleanup'], sort_keys=True)}")
        if status.get("error"):
            print(f"error   : {status['error']}")
    return int(status["state"] == "unverified")


# ----------------------------------------------------------------------------- selftest
def cmd_selftest(args) -> int:
    """Prove isolation + the self-guard with a throwaway `sleep`, no devices touched."""
    failures = []

    def expect(cond, label):
        print(("  PASS  " if cond else "  FAIL  ") + label)
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory() as tmp:
        rf = str(Path(tmp) / "run.json")
        # launch a harmless long sleep
        ns = argparse.Namespace(runfile=rf, log=str(Path(tmp) / "s.log"),
                                command=["sleep", "120"])
        rc = cmd_launch(ns)
        expect(rc == 0, "launch returns 0")
        doc = _load_runfile(rf)
        pid, pgid = doc["pid"], doc["pgid"]
        expect(pgid != _caller_pgid(), "child pgid is isolated from the caller's pgid")
        expect(pgid == pid, "setsid child is its own group leader (pgid == pid)")
        expect(_pgid_alive(pgid), "child group is alive after launch")

        # self-guard: refuse to tear down a runfile that names the CALLER's own pgid
        guard_rf = str(Path(tmp) / "self.json")
        write_runfile(guard_rf, {"pid": os.getpid(), "pgid": _caller_pgid(),
                                 "cmd": ["python", "self"], "log": ""})
        rc_guard = cmd_teardown(argparse.Namespace(runfile=guard_rf, grace=0))
        expect(rc_guard == 1, "teardown REFUSES a pgid equal to the caller's group")
        expect(_pgid_alive(_caller_pgid()), "caller's own group survived the refused teardown")

        # real teardown of the isolated child
        rc_td = cmd_teardown(argparse.Namespace(runfile=rf, grace=0))
        expect(rc_td == 0, "teardown of the isolated group returns 0")
        expect(not _pgid_alive(pgid), "child group is gone after teardown")

    print(f"\nsafe_serve selftest: {'ALL PASSED' if not failures else str(len(failures)) + ' FAILED'}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("launch", help="start a command in its own isolated session (setsid).")
    p.add_argument("--runfile", required=True, help="JSON runfile to record pid/pgid/cmd/log.")
    p.add_argument("--log", default="", help="log file for the command's stdout+stderr.")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="command and its arguments; an optional `--` separates launcher options.")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("wait", help="poll until ready (health URL and/or log pattern).")
    p.add_argument("--runfile", required=True)
    p.add_argument("--health", default="", help="health URL that returns 2xx when ready.")
    p.add_argument("--ready-pattern", default="", help="substring in the log that means ready.")
    p.add_argument("--log", default="", help="log file (defaults to the runfile's).")
    p.add_argument("--timeout", type=float, default=1500, help="overall readiness timeout (s).")
    p.add_argument("--stale", type=float, default=900,
                   help="fail if the log has not changed for this many seconds (0 disables).")
    p.add_argument("--poll", type=float, default=5, help="poll interval (s).")
    p.add_argument("--report-every", type=float, default=60, help="progress line cadence (s).")
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("teardown", help="SIGKILL the recorded process group (self-guarded).")
    p.add_argument("--runfile", required=True)
    p.add_argument("--grace", type=float, default=0,
                   help="if >0, SIGTERM then wait this many seconds before SIGKILL. "
                        "Default 0 = immediate SIGKILL (the kill-after-capture lever).")
    p.set_defaults(func=cmd_teardown)

    p = sub.add_parser("status", help="show whether the recorded run is alive (read-only).")
    p.add_argument("--runfile", required=True)
    p.add_argument("--json", action="store_true", help="emit process state and cleanup receipt.")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("selftest", help="prove isolation + self-guard (throwaway sleep).")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    try:
        for name in ("timeout", "stale", "grace", "poll", "report_every"):
            value = getattr(args, name, None)
            if value is not None and (
                not math.isfinite(value) or value < 0 or
                (name in ("poll", "report_every") and value == 0)
            ):
                raise ValueError(f"--{name.replace('_', '-')} has an invalid duration")
        return args.func(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"safe_serve: REFUSED — {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
