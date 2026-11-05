# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Shared host fixtures; no CUDA execution or numerical qualification."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass, field
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch


def definitions(path, names, **values):
    """Execute unmodified host definitions with explicit dependency boundaries."""
    nodes = [
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names
    ]
    assert {n.name for n in nodes} == set(names), (path, names)
    module = ModuleType("_kimi_host_fixture")
    module.__dict__.update(
        torch=torch, dataclass=dataclass, field=field, hashlib=hashlib, json=json
    )
    module.__dict__.update(values)
    body = [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes]
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(
            compile(ast.fix_missing_locations(ast.Module(body, [])), str(path), "exec"),
            module.__dict__,
        )
    return module


def resource_fields(**overrides):
    """Fresh ResourceSet fields, shared by real-resource and host-only fixtures."""
    fields = {
        name: {}
        for name in (
            "attention",
            "feed_forward",
            "moe_front_weights",
            "before",
            "fused_before",
            "boxes",
            "scratch",
            "workspaces",
            "runtime_handles",
            "payload_modules",
            "memory_before",
            "memory_after",
        )
    }
    fields.update(
        model=NS(), layers=(), final=None, compilation_inventory=[], deadline_ns=1
    )
    return fields | overrides


def hidden_buffers(rows, tensor=torch.zeros):
    return {
        name: tensor((rows, 7168))
        for name in (
            "prefix",
            "blocks",
            "hidden.0",
            "hidden.1",
            "hidden.out",
            "mla.projected",
            "kda.projected",
            "attention.local",
        )
    }


def execution_resources(
    rows, *, layers=range(93), mla_layers=(0,), tensor=torch.zeros, **overrides
):
    """Host executor plumbing, with independent recording mocks for every layer.

    Tests supply state views and assert call arguments/order explicitly. These
    providers do not implement model arithmetic or claim device correctness.
    """
    layers = tuple(layers)
    attention = {
        i: NS(
            kind="mla" if i in mla_layers else "kda",
            projection=NS(run=Mock()),
            core=NS(run=Mock(), run_verify=Mock()),
            output=NS(run=Mock()),
            cache=object(),
            caches=(object(), object()),
            intermediates=(object(), object()),
        )
        for i in layers
    }
    return NS(
        **resource_fields(
            layers=tuple(NS(layer_idx=i) for i in layers),
            attention=attention,
            scratch=hidden_buffers(rows, tensor),
            before={i: NS(run=Mock()) for i in layers},
            feed_forward={
                i: NS(
                    run=Mock(
                        side_effect=lambda x, out, consume=None: (
                            out if consume is None else consume(out)
                        )
                    )
                )
                for i in layers
            },
            final=NS(run=Mock()),
            binding_fingerprint="bound",
            tokens_per_request=1,
            widths=(rows,),
            aux_capture_layers=(),
            prefetch=None,
        )
        | overrides
    )


def inventory_fixture(path, tensor_type, *, kind="provider"):
    """Exercise the real ownership walk with metadata-only tensors/handles."""
    stream, event = type("Stream", (), {}), type("Event", (), {})
    module = definitions(
        path,
        ("_owned_resource_inventory",),
        torch=NS(Tensor=tensor_type, cuda=NS(Stream=stream, Event=event)),
        _tensor_row=lambda kind, name, value: dict(
            kind=kind, name=name, data_ptr=value.data_ptr()
        ),
        _handle_row=lambda kind, name, value: dict(
            kind=kind, name=name, data_ptr=id(value)
        ),
        _resource_kind=lambda name: kind,
    )
    return module, stream, event
