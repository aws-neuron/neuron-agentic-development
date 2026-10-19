"""Readiness evidence must come from this launch's log and local server."""

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .process import assert_group_owned, group_members

TCP_LISTEN = "0A"  # Linux /proc/net/tcp state for a listening socket.
TCP_TABLES = {socket.AF_INET: "/proc/net/tcp", socket.AF_INET6: "/proc/net/tcp6"}


class LaunchLog:
    """Scan appended bytes once; compiler logs can grow throughout a long readiness wait.

    Byte offsets preserve the launch boundary and partial UTF-8 sequences across polls.
    Only the possible prefix of the readiness marker is retained between chunks.
    """
    def __init__(self, doc: dict, path: str):
        self.path = Path(path)
        self.offset = int(doc["log_offset"])
        self.identity = (doc["log_device"], doc["log_inode"])
        self.tail = b""

    def contains(self, pattern: str) -> bool:
        needle = pattern.encode()
        with self.path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino) != self.identity or stat.st_size < self.offset:
                raise ValueError("launch log was replaced or truncated")
            stream.seek(self.offset)
            remaining = stat.st_size - self.offset
            while remaining:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    raise ValueError("launch log was truncated while reading")
                text = self.tail + chunk
                found = needle in text
                self.tail = text[-max(len(needle) - 1, 0):] if len(needle) > 1 else b""
                self.offset += len(chunk)
                remaining -= len(chunk)
                if found:
                    return True
            return False


def _local_addresses(host: str, port: int) -> set:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    local = set()
    for family, kind, proto, _, address in addresses:
        try:
            with socket.socket(family, kind, proto) as probe:
                # Binding port zero checks locality without connecting to the endpoint.
                probe.bind((address[0], 0, *address[2:]))
        except OSError:
            continue
        packed = socket.inet_pton(family, address[0])
        proc_address = "".join(
            f"{int.from_bytes(packed[i:i + 4], sys.byteorder):08X}"
            for i in range(0, len(packed), 4)
        )
        local.add((family, proc_address))
    if not local:
        raise ValueError("health endpoint has no local address")
    return local


def owns_health_endpoint(doc: dict, url: str) -> bool:
    endpoint = urlsplit(url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("health endpoint must be a local HTTP(S) URL")
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    addresses = _local_addresses(endpoint.hostname, port)
    assert_group_owned(doc)
    sockets = set()
    for pid in group_members(int(doc["pgid"])):
        try:
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith("socket:["):
                        sockets.add(target[8:-1])
                except OSError:
                    continue
        except OSError:
            continue
    listeners = set()
    for family in {kind for kind, _ in addresses}:
        table = TCP_TABLES[family]
        local_addresses = {address for kind, address in addresses if kind == family}
        wildcard = "0" * (8 if family == socket.AF_INET else 32)
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for row in rows:
            fields = row.split()
            address, port_hex = fields[1].split(":")
            if (
                fields[3] == TCP_LISTEN
                and int(port_hex, 16) == port
                # Another local interface may have an unrelated listener on this port.
                and (address == wildcard or address in local_addresses)
            ):
                listeners.add(fields[9])
    # Refuse ambiguity when another group also listens on this port/interface family.
    return bool(listeners) and listeners.issubset(sockets)
