# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.base.observable_base."""

from __future__ import annotations

import typing
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import Generic
from typing import TypeVar

from atlas.utils.reactive.base.observer_base import ObserverBase
from atlas.utils.reactive.subscription import Subscription

_T_out_co = TypeVar("_T_out_co", covariant=True)
_T = TypeVar("_T")

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class ObservableBase(Generic[_T_out_co], ABC):
    @typing.overload
    def subscribe(self, on_next: ObserverBase[_T_out_co]) -> Subscription:
        ...

    @typing.overload
    def subscribe(
        self,
        on_next: OnNext[_T_out_co] | None = None,
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> Subscription:
        ...

    @abstractmethod
    def subscribe(
        self,
        on_next: ObserverBase[_T_out_co] | OnNext[_T_out_co] | None = None,
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> Subscription:
        raise NotImplementedError
