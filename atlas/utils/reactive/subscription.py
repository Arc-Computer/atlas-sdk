# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.subscription."""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Generic
from typing import TypeVar

if typing.TYPE_CHECKING:
    from atlas.utils.reactive.base.subject_base import SubjectBase

_T = TypeVar("_T")

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Subscription(Generic[_T]):
    def __init__(self, subject: "SubjectBase", observer: object | None):
        self._subject = subject
        self._observer = observer
        self._unsubscribed = False

    def unsubscribe(self) -> None:
        if not self._unsubscribed and self._observer is not None:
            self._subject._unsubscribe_observer(self._observer)
            self._observer = None
            self._unsubscribed = True
