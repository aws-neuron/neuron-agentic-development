"""Typed input contract for artifact admission."""

from dataclasses import asdict, dataclass, field

from .contracts import ContractError, EvidenceClass, _object, _string


@dataclass
class Artifact:
    path: str
    type: str
    evidence_class: EvidenceClass
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        _string(self.path, "path")
        _string(self.type, "type")
        _object(self.metadata, "metadata")
        try:
            self.evidence_class = EvidenceClass(self.evidence_class)
        except (ValueError, TypeError) as error:
            raise ContractError("evidence_class is invalid") from error

    @classmethod
    def from_dict(cls, document, name):
        _object(document, name)
        try:
            return cls(**document)
        except (TypeError, ContractError) as error:
            raise ContractError(f"{name}: {error}") from error

    def admitted(self, path, digest, size):
        return {
            **asdict(self),
            "path": str(path),
            "evidence_class": self.evidence_class.value,
            "sha256": digest,
            "size_bytes": size,
        }
