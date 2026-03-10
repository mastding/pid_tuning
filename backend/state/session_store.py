from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any


class SessionStore(MutableMapping[str, Any]):
    """Thin session-scoped state wrapper used by the multi-agent workflow."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)

