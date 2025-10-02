# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.data_models.invocation_node."""

from __future__ import annotations

from pydantic import BaseModel


class InvocationNode(BaseModel):
    function_id: str
    function_name: str
    parent_id: str | None = None
    parent_name: str | None = None
