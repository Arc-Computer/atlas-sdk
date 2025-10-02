# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.observable."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from atlas.utils.reactive.base.observable_base import ObservableBase
from atlas.utils.reactive.base.observer_base import ObserverBase
from atlas.utils.reactive.observer import Observer
from atlas.utils.reactive.subscription import Subscription

_T_out_co = TypeVar("_T_out_co", covariant=True)
_T = TypeVar("_T")

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Observable(ObservableBase[_T_out_co]):
    __slots__ = ()

    def _subscribe_core(self, observer: ObserverBase) -> Subscription:
        raise NotImplementedError

    def subscribe(
        self,
        on_next: ObserverBase[_T_out_co] | OnNext[_T_out_co] | None = None,
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> Subscription:
        if isinstance(on_next, ObserverBase):
            return self._subscribe_core(on_next)
        return self._subscribe_core(Observer(on_next, on_error, on_complete))
