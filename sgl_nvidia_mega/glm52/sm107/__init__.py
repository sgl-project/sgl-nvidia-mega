# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import traceback

TARGET = "sglang.srt.model_executor.model_runner"
DRAFT_TARGET = "sglang.srt.speculative.eagle_worker_v2"

SGLANG_READS = {
    "srt/model_executor/model_runner.py": ("ModelRunner", "load_model", "init_cuda_graphs", "is_draft_worker"),
}


def _guard(fn, code):
    """A failure exits the process instead of serving stock."""
    try:
        fn()
    except BaseException:
        traceback.print_exc()
        os._exit(code)


def _wrap(module):
    cls = module.ModelRunner
    if getattr(cls, "_glm5mk", False):
        return
    load_model, init_graphs = cls.load_model, cls.init_cuda_graphs

    def load_model_w(self, *a, **k):
        out = load_model(self, *a, **k)
        from . import draft, target

        _guard(lambda: (draft.on_draft_loaded if getattr(self, "is_draft_worker", False) else target.on_model_loaded)(self), 71)
        return out

    def init_graphs_w(self, *a, **k):
        from . import draft, target

        _guard(lambda: (draft.before_draft_graphs if getattr(self, "is_draft_worker", False) else target.before_graphs)(self), 72)
        return init_graphs(self, *a, **k)

    cls.load_model, cls.init_cuda_graphs, cls._glm5mk = load_model_w, init_graphs_w, True


def _wrap_draft(module):
    from . import draft

    _guard(lambda: draft._wrap_worker(module), 73)


class _Loader(importlib.abc.Loader):
    def __init__(self, orig, wrap):
        self.orig, self.wrap = orig, wrap

    def create_module(self, spec):
        f = getattr(self.orig, "create_module", None)
        return f(spec) if f else None

    def exec_module(self, module):
        self.orig.exec_module(module)
        self.wrap(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, targets):
        self.targets = dict(targets)

    def find_spec(self, name, path, target=None):
        if name not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("glm52.sm107: cannot locate " + name)
        spec.loader = _Loader(spec.loader, self.targets.pop(name))
        if not self.targets:
            sys.meta_path.remove(self)
        return spec


def install() -> bool:
    if os.environ.get("GLM5MK") != "1":
        return False
    if getattr(install, "_done", False):
        return True
    todo = {}
    for name, fn in ((TARGET, _wrap), (DRAFT_TARGET, _wrap_draft)):
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            todo[name] = fn
    if todo:
        sys.meta_path.insert(0, _Finder(todo))
    install._done = True
    return True
