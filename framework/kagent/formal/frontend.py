"""FrontEnd / Spec interfaces and the tri-state Verdict."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(Enum):
    PROVED = "PROVED"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class FormalResult:
    verdict: Verdict
    detail: str = ""
    witness: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict.value, "detail": self.detail,
                "witness": self.witness, "artifacts": self.artifacts}


class FrontEnd:
    """Parses a kernel in some IR into a form check.py can query."""

    name = "abstract"

    def parse(self, source: str) -> Any:  # -> a module/dag
        raise NotImplementedError

    def writers(self, dag: Any, buffer: str, linear_index: int):
        """All (thread, program) pairs that write the queried element."""
        raise NotImplementedError

    def stored_value(self, dag: Any, writer) -> Any:
        """Expression tree of what `writer` stores."""
        raise NotImplementedError


class Spec:
    """Reference math for one output element, in the same abstraction as the
    candidate (same uninterpreted exp/rsqrt, same named load variables)."""

    name = "abstract"

    def expected(self, element: dict) -> tuple[Any, set]:
        """Returns (expected expression, expected multiset of loads)."""
        raise NotImplementedError