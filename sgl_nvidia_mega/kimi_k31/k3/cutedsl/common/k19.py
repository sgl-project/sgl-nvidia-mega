# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K-19 trigger fences (default off): ``K31MK_K19_FENCE``, a "+"-joined subset of ``SITES``.

K-19 is the PDL visibility gap by the PTX letter: a dependent grid's accesses before its own
``griddepcontrol.wait`` are ordered after a primary's memory operations only through a release fence
before the primary's ``griddepcontrol.launch_dependents`` (PTX ISA 9.7.15.14), chained over every trigger
from a grid that waited up to the reader. Launch order on the served paths is already structural (each
sublayer has a wait-then-trigger gate grid); these knobs only add ``fence.acq_rel.gpu`` right before the
trigger at each named site, so the chain holds by the letter too. They change no value (bitwise class).

Sites (file: the trigger the fence precedes):
- ``oproj_producer``: oproj/producer.py ``OprojProducer.compute`` (split producer: trigger after its wait;
  built through ``oproj.prepared.producer``; the fused o-projection's own producer is not affected);
- ``oproj_umma``: oproj/producer.py ``OprojUmma.kernel`` (tcgen05 producer above 8 rows);
- ``front_gemm``: gemm/ws_gemm.py ``WsGemm.kernel`` for the MoE front's ``Fp32FrontGemm``
  (``moe.front.compile_publish``; every batch it compiles);
- ``rubin64``: rubin/moe/experts_rubin64.py ``RubinGroupedExpertsKernel.body`` (``compile_rubin64``);
- ``tail9``: the published / two-shot ``WaveMoeTail`` at 9-128 rows (its ``tfence`` option; every trigger
  of the kernel: entry with ``early``, the TMEM-grant point with ``early`` + ``umma``, else after the output).
  Rows <= 8 keep ``K31MK_MOE_TAIL_WAVE8`` "tfence";
- ``rubin_experts``: rubin/moe/experts_rubin.py ``RubinExpertsKernel.kernel`` (the 8-row experts,
  ``compile_rubin``; the c=1 site).
KdaSplit keeps its own knob, ``K31MK_KDA_TRIGGER_FENCE=1``.

Engagement: the first time a site is built with its fence in a process, one line
``K31MK_K19_FENCE engaged site=<site> pid=<pid>`` goes to stderr (once per site per process, also on a
persistent kernel-cache hit: the line comes from the kernel's construction, and the env is part of that
cache's key). Read when the kernel objects are built."""

import os
import sys

ENV = "K31MK_K19_FENCE"
SITES = ("oproj_producer", "oproj_umma", "front_gemm", "rubin64", "tail9", "rubin_experts")
_engaged: set = set()


def sites():
    """The sites ``K31MK_K19_FENCE`` names (a frozenset; empty when unset). Unknown names raise."""
    names = frozenset(n for n in os.environ.get(ENV, "").split("+") if n.strip())
    bad = sorted(names - set(SITES))
    if bad:
        raise ValueError(f"{ENV}: unknown sites {bad}; known: {'+'.join(SITES)}")
    return names


def fence(site):
    """True iff ``site`` is fenced; logs the site's engagement line the first time it is."""
    if site not in SITES:
        raise ValueError(f"k19.fence: unknown site {site!r}")
    on = site in sites()
    if on and site not in _engaged:
        _engaged.add(site)
        print(f"{ENV} engaged site={site} pid={os.getpid()}", file=sys.stderr, flush=True)
    return on
