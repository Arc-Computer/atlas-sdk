# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.utils.reactive.base.observer_base."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

_T_in_contra = TypeVar("_T_in_contra", contravariant=True)


class ObserverBase(Generic[_T_in_contra], ABC):
    @abstractmethod
    def on_next(self, value: _T_in_contra) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_error(self, exc: Exception) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_complete(self) -> None:
        raise NotImplementedError
