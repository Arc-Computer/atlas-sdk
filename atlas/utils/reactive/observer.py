# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.observer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from atlas.utils.reactive.base.observer_base import ObserverBase

logger = logging.getLogger(__name__)

_T_in_contra = TypeVar("_T_in_contra", contravariant=True)
_T = TypeVar("_T")

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Observer(ObserverBase[_T_in_contra]):
    def __init__(
        self,
        on_next: OnNext | None = None,
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> None:
        self._on_next = on_next
        self._on_error = on_error
        self._on_complete = on_complete
        self._stopped = False

    def on_next(self, value: _T) -> None:
        if self._stopped or self._on_next is None:
            return
        try:
            self._on_next(value)
        except Exception as exc:
            self.on_error(exc)

    def on_error(self, exc: Exception) -> None:
        if self._stopped:
            return
        if self._on_error:
            try:
                self._on_error(exc)
            except Exception as error:
                logger.exception("Observer on_error callback raised an exception", exc_info=error)

    def on_complete(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._on_complete:
            try:
                self._on_complete()
            except Exception as error:
                logger.exception("Observer on_complete callback raised an exception", exc_info=error)
