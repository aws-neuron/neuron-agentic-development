"""Resource grants without pretending that a local CLI enforces hardware access."""

from pathlib import Path

from .contracts import ContractError


def prepare_resources(capability_spec, request, project_root):
    resources = request.get("resources", {})
    allowed = {"devices", "external_reservation_id", "workspace"}
    unknown = sorted(set(resources) - allowed)
    if unknown:
        raise ContractError(f"resources has unknown fields: {', '.join(unknown)}")

    grant = {"hardware_enforcement": "not_required"}
    if capability_spec["hardware"]:
        devices = resources.get("devices")
        if (
            not isinstance(devices, list)
            or not devices
            or any(isinstance(device, bool) or not isinstance(device, int) for device in devices)
            or any(device < 0 for device in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ContractError(
                "hardware capabilities require a unique list of non-negative devices"
            )
        grant["devices"] = sorted(devices)
        reservation = resources.get("external_reservation_id")
        if not isinstance(reservation, str) or not reservation:
            raise ContractError(
                "hardware capabilities require an external_reservation_id"
            )
        grant["external_reservation_id"] = reservation
        grant["hardware_enforcement"] = "declared_external_reservation"

    if capability_spec["source_write"]:
        workspace = resources.get("workspace")
        if not isinstance(workspace, str) or not workspace:
            raise ContractError("source-writing capabilities require resources.workspace")
        path = Path(workspace)
        if not path.is_absolute():
            path = Path(project_root) / path
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError:
            raise ContractError(f"workspace does not exist: {path}")
        root = Path(project_root).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError:
            raise ContractError("workspace must be inside project_root")
        if not path.is_dir():
            raise ContractError("workspace must be a directory")
        if not (path / ".git").exists():
            raise ContractError("workspace must be a Git checkout or worktree")
        grant["workspace"] = str(path)
    return grant
