# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.base.subject_base."""

from __future__ import annotations

import typing
from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

from atlas.utils.reactive.base.observable_base import ObservableBase
from atlas.utils.reactive.base.observer_base import ObserverBase

if typing.TYPE_CHECKING:
    from atlas.utils.reactive.subscription import Subscription

T = TypeVar("T")

OnNext = Callable[[T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class SubjectBase(ObserverBase[T], ObservableBase[T]):
    @abstractmethod
    def _unsubscribe_observer(self, observer: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def subscribe(
        self,
        on_next: ObserverBase[T] | OnNext[T] | None = None,
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> "Subscription":
        raise NotImplementedError

    @abstractmethod
    def on_next(self, value: T) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_error(self, exc: Exception) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_complete(self) -> None:
        raise NotImplementedError
