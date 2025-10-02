# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.subject."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from atlas.utils.reactive.base.subject_base import SubjectBase
from atlas.utils.reactive.observable import Observable
from atlas.utils.reactive.observer import Observer
from atlas.utils.reactive.subscription import Subscription

T = TypeVar("T")

OnNext = Callable[[T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Subject(Observable[T], Observer[T], SubjectBase[T]):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._closed = False
        self._error: Exception | None = None
        self._observers: list[Observer[T]] = []
        self._disposed = False

    def _subscribe_core(self, observer: Observer[T]) -> Subscription:
        with self._lock:
            if self._disposed:
                return Subscription(self, None)
            self._observers.append(observer)
            return Subscription(self, observer)

    def on_next(self, value: T) -> None:
        with self._lock:
            if self._closed or self._disposed:
                return
            current_observers = list(self._observers)
        for observer in current_observers:
            observer.on_next(value)

    def on_error(self, exc: Exception) -> None:
        with self._lock:
            if self._closed or self._disposed:
                return
            current_observers = list(self._observers)
        for observer in current_observers:
            observer.on_error(exc)

    def on_complete(self) -> None:
        with self._lock:
            if self._closed or self._disposed:
                return
            current_observers = list(self._observers)
            self.dispose()
        for observer in current_observers:
            observer.on_complete()

    def _unsubscribe_observer(self, observer: Observer[T]) -> None:
        with self._lock:
            if not self._disposed and observer in self._observers:
                self._observers.remove(observer)

    def dispose(self) -> None:
        with self._lock:
            if not self._disposed:
                self._disposed = True
                self._observers.clear()
                self._closed = True
                self._error = None
