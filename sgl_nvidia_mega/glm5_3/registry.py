# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Which kernel serves each step of a fused forward, per step mode and width.

A **candidate** (``Candidate``) claims some steps of one step group (``GROUPS``)
atomically and states its coverage (``covers``: a decline reason or None) and its
contract tokens (``provides`` / ``requires``). Candidates of one **family** share its
build: ``FAMILIES[family](model, derived, profile)`` returns an object with
``compile(widths)`` and ``provider(candidate, rows, mode)`` (the step-group object that
overrides the candidate's steps), and optionally ``step_reasons(model,
forward_batch)`` (host facts read at the first capture of a width) and
``error_report()`` (the record of a timed-out cross-GPU wait). Family modules declare
module-level ``CANDIDATES`` and ``FAMILIES``; ``units.PREFERENCE`` orders the
candidates of every step.

Composition at one (mode, rows):

1. each step takes the first candidate in its preference list that is registered,
   switched on (``units.enabled``), preferred here (``Pref.when``) and covers the
   width; a candidate that claims several steps is taken only if it is first for
   every one of them;
2. to a fixed point: a taken candidate one of whose ``requires`` is not provided by
   another taken candidate is passed over, and step 1 runs again without it (a bare
   group name, e.g. "mlp", is provided by any taken candidate of that group);
3. the steps nobody claims run the stock body; a group whose steps all come from one
   provider object is that object, else a copy of the stock group with each claimed
   step bound to its provider's method;
4. a provider whose token a taken consumer requires gets ``wire(token)``.

A test hook, ``SGLANG_NVIDIA_MEGAKERNEL_GLM_FORCE_BUILD_ERROR="family[@rank],..."``,
fails a family's build (``ll.jit`` reads the same list for its JIT modules).

No silent fallback. A candidate whose ``covers`` gives a reason (or that is passed
over) is a **decline**: the step runs stock at that width, logged once per width on
rank 0 ("stock at N rows: ..."). A family whose build or compile raises, or returns
None, for a width a switched-on candidate covers is a **failure**: the reasons of
every rank are gathered and every rank raises, naming the units to switch off. A
provider may still raise ``Decline`` for a fact only the build knows; declines are
agreed across ranks before the width is composed again.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from sgl_nvidia_mega.glm5_3.profile import Profile

logger = logging.getLogger(__name__)

# The step modes (SGLang's Step.mode / the draft spec's modes)
MODES = ("target_verify", "decode", "draft_extend")

# The attention steps, in call order (attention.step.AttentionStep)
ATTENTION_STEPS = (
    "forward",
    "qkv_a",
    "prologue",
    "index_k",
    "index_q",
    "indexer",
    "q_absorb",
    "rope_cache",
    "sparse_mla",
    "uv_proj",
)
# group -> its steps, in call order. The target's groups are plan.STEPS; the
# draft's are the NextN layer's sections (plan section A.3: D1a front, D1b eh_proj,
# D2 attention, D3 MoE, D5 top-1).
GROUPS: Dict[str, Tuple[str, ...]] = {
    "attention": ATTENTION_STEPS,
    "hops": ("pre_attention", "post_attention", "final"),
    "mlp": ("forward",),
    "draft_front": ("front",),
    "draft_eh": ("eh_proj",),
    "draft_attention": ATTENTION_STEPS,
    "draft_mlp": ("forward",),
    "draft_top1": ("top1",),
}
TARGET_GROUPS = ("attention", "hops", "mlp")
DRAFT_GROUPS = tuple(g for g in GROUPS if g.startswith("draft_"))

Covers = Callable[[Profile, int, str], Optional[str]]


@dataclass(frozen=True)
class Candidate:
    """A kernel unit that can serve some steps of one group."""

    name: str  # "ll.attention", "ll.sparse_mla_fd", "draft.front", ...
    group: str  # a GROUPS key
    steps: Tuple[str, ...]  # the group steps it claims (taken for all or none)
    family: str  # built once per model and shared by its candidates
    covers: Covers  # (profile, rows, mode) -> None, or the decline reason
    # contract tokens, e.g. "mlp:records", "hops:dnorm", "mlp:deferred_finalize"
    provides: Tuple[str, ...] = ()
    # a bare group name keeps its legacy meaning: any taken candidate of that group
    requires: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"candidate {self.name}: unknown group {self.group!r}")
        unknown = [s for s in self.steps if s not in GROUPS[self.group]]
        if not self.steps or unknown:
            raise ValueError(
                f"candidate {self.name}: steps {self.steps} are not steps of "
                f"{self.group} {GROUPS[self.group]}"
            )

    @property
    def key(self) -> Tuple[str, str]:
        return (self.group, self.name)


class Decline(Exception):
    """Raised by ``provider``: the family cannot serve this width after all, for a fact
    only its build knows (e.g. a per-rank cap agreed at build). The width is composed
    again without the candidate on every rank."""


class DeclinedFamily:
    """A built family that serves no width (every provider declines with ``reason``)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def compile(self, widths) -> None:
        pass

    def provider(self, candidate: Candidate, rows: int, mode: str):
        raise Decline(self.reason)


@dataclass
class Composition:
    """The candidates taken at one (mode, rows), and why others were passed over."""

    mode: str
    rows: int
    owners: Dict[Tuple[str, str], Candidate]  # (group, step) -> its candidate
    # (group, name) -> why it was passed over, for the candidates reached in a
    # preference list before one was taken (or the list ran out)
    passed: Dict[Tuple[str, str], str] = field(default_factory=dict)

    @property
    def chosen(self) -> Tuple[Candidate, ...]:
        return tuple(dict.fromkeys(self.owners.values()))

    def all_stock(self) -> bool:
        return not self.owners

    def stock_reasons(
        self, candidates: Mapping[Tuple[str, str], Candidate]
    ) -> Dict[str, str]:
        """name -> reason, for the passed-over candidates that claim a step left stock."""
        out: Dict[str, str] = {}
        for key, why in self.passed.items():
            c = candidates.get(key)
            if c is None:
                continue
            if any((c.group, s) not in self.owners for s in c.steps):
                out.setdefault(c.name, why)
        return out


@dataclass(frozen=True)
class Pref:
    """One entry of a step's preference list (units.PREFERENCE)."""

    name: str
    # (profile, rows, mode) -> whether the entry applies (a TP / row predicate)
    when: Optional[Callable[[Profile, int, str], bool]] = None
    note: str = ""  # the measurement behind the order

    def applies(self, profile: Profile, rows: int, mode: str) -> bool:
        return self.when is None or bool(self.when(profile, rows, mode))


def _always(candidate: Candidate) -> bool:
    return True


def select(
    profile: Profile,
    rows: int,
    mode: str,
    *,
    candidates: Iterable[Candidate],
    preference: Mapping[Tuple[str, str], Sequence[Pref]],
    groups: Sequence[str],
    allowed: Callable[[Candidate], bool] = _always,
    excluded: Optional[Mapping[Tuple[str, str], str]] = None,
) -> Composition:
    """The composition at (``mode``, ``rows``): steps 1 and 2 of the module docstring.
    ``excluded``: (group, name) -> reason, candidates already passed over here (e.g.
    a provider's ``Decline``)."""
    by_key = {c.key: c for c in candidates if c.group in groups}
    passed: Dict[Tuple[str, str], str] = dict(excluded or {})
    covers_memo: Dict[Tuple[str, str], Optional[str]] = {}

    def reason(c: Candidate) -> Optional[str]:
        if c.key not in covers_memo:
            if not allowed(c):
                covers_memo[c.key] = (
                    "switched off (not in SGLANG_NVIDIA_MEGAKERNEL_SLOTS)"
                )
            else:
                covers_memo[c.key] = c.covers(profile, rows, mode)
        return covers_memo[c.key]

    while True:
        first: Dict[Tuple[str, str], Candidate] = {}
        for g in groups:
            for s in GROUPS[g]:
                for pref in preference.get((g, s), ()):
                    c = by_key.get((g, pref.name))
                    if c is None or s not in c.steps or c.key in passed:
                        continue
                    if not pref.applies(profile, rows, mode):
                        continue
                    why = reason(c)
                    if why:
                        passed[c.key] = why
                        continue
                    first[(g, s)] = c
                    break
        changed = False
        # multi-step candidates: first for every step they claim, or none
        for c in dict.fromkeys(first.values()):
            lost = [s for s in c.steps if first.get((c.group, s)) is not c]
            if lost:
                other = first.get((c.group, lost[0]))
                passed[c.key] = f"not preferred for {c.group}.{lost[0]}" + (
                    f" ({other.name} is)" if other is not None else ""
                )
                changed = True
        if changed:
            continue
        # contract tokens, to a fixed point
        chosen = tuple(dict.fromkeys(first.values()))
        provided = {t for c in chosen for t in c.provides} | {c.group for c in chosen}
        for c in chosen:
            unmet = [r for r in c.requires if r not in provided]
            if unmet:
                passed[c.key] = (
                    f"requires {unmet[0]}, which no unit provides at {rows} rows"
                )
                changed = True
        if not changed:
            return Composition(mode=mode, rows=rows, owners=first, passed=passed)


def _provided(value: Any) -> Any:
    """A provider's attribute as a class attribute of the composed group: a bound method
    stays bound to the provider; a plain function (a static method) stays unbound."""
    return staticmethod(value) if isinstance(value, types.FunctionType) else value


# group -> step -> the helper methods its body (and the group's shared paths) call: a
# composed group takes them from the provider that claims the step.
# AttentionStep runs index_k / index_q through _index_kw (and, when both are stock,
# its two-stream path).
STEP_HELPERS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "attention": {"index_k": ("_index_kw",), "index_q": ("_index_kw",)},
    "draft_attention": {"index_k": ("_index_kw",), "index_q": ("_index_kw",)},
}


def compose_group(
    stock: Any,
    steps: Sequence[str],
    owners: Mapping[str, Any],
    helpers_of: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> Any:
    """The step-group object serving ``owners`` (step -> provider object) and ``stock``
    elsewhere (step 3 of the module docstring).

    A mixed group is an instance of a subclass of the stock group's class whose
    claimed steps are the providers' bound methods (class attributes, so the stock
    group's own type checks, its calls through ``self`` and ``Providers.describe``
    see them); a claimed step the provider inherits stays the stock body. A step's helpers (``helpers_of``: step -> the helper methods its body
    calls, e.g. ``index_k`` -> ``_index_kw``) come from the provider that claims the
    step, so a claimed step keeps its provider's helpers in a mixed group; steps
    sharing a helper must not be claimed by providers that differ in it."""
    objects = list({id(o): o for o in owners.values()}.values())
    if not objects:
        return stock
    if helpers_of is None:
        # the group's own table (attention and draft_attention share their steps)
        group = next((g for g, gs in GROUPS.items() if gs == tuple(steps)), None)
        helpers_of = STEP_HELPERS.get(group, {})
    helpers: Dict[str, Any] = {}  # helper -> the provider whose helper serves it
    for s, provider in owners.items():
        for h in helpers_of.get(s, ()):
            other = helpers.setdefault(h, provider)
            if getattr(type(other), h, None) is not getattr(type(provider), h, None):
                raise ValueError(
                    f"{h} serves steps claimed by {type(other).__name__} and "
                    f"{type(provider).__name__}, which define it differently"
                )
    if len(objects) == 1:
        (only,) = objects
        unclaimed = [s for s in steps if s not in owners]
        unclaimed += [
            h for s in unclaimed for h in helpers_of.get(s, ()) if h not in helpers
        ]
        if all(
            getattr(type(only), s, None) is getattr(type(stock), s, None)
            for s in unclaimed
        ):
            return only
    base = type(stock)
    # a claimed step whose body the provider inherits stays the stock function (bound to
    # the composed group, so the group's type checks still see the stock step); its
    # helpers are the provider's
    attrs = {
        s: (
            getattr(base, s)
            if getattr(type(provider), s, None) is getattr(base, s, None)
            else _provided(getattr(provider, s))
        )
        for s, provider in owners.items()
    }
    attrs.update(
        {h: _provided(getattr(provider, h)) for h, provider in helpers.items()}
    )
    composed = type(f"Composed{base.__name__}", (base,), attrs)
    obj = composed.__new__(composed)
    obj.__dict__.update(vars(stock))
    return obj


@dataclass
class Build:
    """What ``build`` made for one model."""

    profile: Profile
    families: Dict[str, Any]  # built families by name, in registration order
    compositions: Dict[Tuple[str, int], Composition]
    # (mode, rows) -> group -> step-group object
    groups: Dict[Tuple[str, int], Dict[str, Any]]
    candidates: Dict[Tuple[str, str], Candidate]

    def stock_reasons(self, mode: str, rows: int) -> Dict[str, str]:
        return self.compositions[(mode, rows)].stock_reasons(self.candidates)


def collect(modules: Iterable[Any]) -> Tuple[Tuple[Candidate, ...], Dict[str, Any]]:
    """The ``CANDIDATES`` and ``FAMILIES`` the family modules declare."""
    candidates: List[Candidate] = []
    families: Dict[str, Any] = {}
    for module in modules:
        candidates += list(getattr(module, "CANDIDATES", ()))
        for name, factory in getattr(module, "FAMILIES", {}).items():
            if families.setdefault(name, factory) is not factory:
                raise ValueError(f"family {name!r} is declared twice")
    seen = set()
    for c in candidates:
        if c.key in seen:
            raise ValueError(f"candidate {c.name} of {c.group} is declared twice")
        seen.add(c.key)
        if c.family not in families:
            raise ValueError(f"candidate {c.name}: no family {c.family!r}")
    return tuple(candidates), families


def target_modules() -> Tuple[Any, ...]:
    """The modules whose candidates serve the target's verify step (the TP4 profiles;
    the TP8 envelope builds its families through plan.build_providers)."""
    from sgl_nvidia_mega.glm5_3 import ll

    return tuple(ll.modules().values())


def _single(obj: Any) -> List[Any]:
    return [obj]


def _describe(e: BaseException) -> str:
    text = " ".join(str(e).split())
    return f"{type(e).__name__}: {text[:400]}"


def build(
    model: Any,
    derived: Any,
    profile: Profile,
    *,
    steps: Iterable[Tuple[str, int]],
    groups: Mapping[str, Callable[[], Any]],
    candidates: Iterable[Candidate],
    families: Mapping[str, Callable[..., Any]],
    preference: Optional[Mapping[Tuple[str, str], Sequence[Pref]]] = None,
    allowed: Optional[Callable[[Candidate], bool]] = None,
    gather: Callable[[Any], List[Any]] = _single,
    rank: int = 0,
    what: str = "NVIDIA megakernel: GLM-5.3 / GLM-5.2",
) -> Build:
    """Compose every (mode, rows) of ``steps``, build and compile the families the
    compositions take, and make each width's step-group objects.

    ``groups``: group -> a factory of its stock step-group object; ``gather``: an
    all-gather of a host object over the tensor-parallel group (every rank calls it
    alike); ``rank``: this rank in that group (rank 0 logs)."""
    from sgl_nvidia_mega.glm5_3 import units
    from sgl_nvidia_mega.glm5_3.ll.jit import FORCE_ERROR_ENV, forced_error

    preference = units.PREFERENCE if preference is None else preference
    allowed = units.allowed if allowed is None else allowed
    steps = tuple(dict.fromkeys(steps))
    group_names = tuple(groups)
    cands = {c.key: c for c in candidates if c.group in groups}
    excluded: Dict[Tuple[str, int], Dict[Tuple[str, str], str]] = {s: {} for s in steps}

    def compose(key: Tuple[str, int]) -> Composition:
        return select(
            profile,
            key[1],
            key[0],
            candidates=cands.values(),
            preference=preference,
            groups=group_names,
            allowed=allowed,
            excluded=excluded[key],
        )

    comps = {key: compose(key) for key in steps}
    built: Dict[str, Any] = {}

    def build_needed() -> None:
        """Build and compile, in registration order, every family a composition takes
        that is not built yet; any rank's failure raises on every rank."""
        taken = {c.family for comp in comps.values() for c in comp.chosen}
        for name, factory in families.items():
            if name not in taken or name in built:
                continue
            mine = [c for c in cands.values() if c.family == name and allowed(c)]
            widths = tuple(
                sorted(
                    {
                        rows
                        for mode, rows in steps
                        if any(c.covers(profile, rows, mode) is None for c in mine)
                    }
                )
            )
            failure = None
            try:
                if forced_error(name):
                    raise RuntimeError(f"forced build error ({FORCE_ERROR_ENV})")
                family = factory(model, derived, profile)
                if family is None:
                    raise RuntimeError("the family build returned None")
                family.compile(widths)
            except Exception as e:  # noqa: BLE001 - gathered, then raised on every rank
                failure = _describe(e)
                logger.exception("%s: building the %s family failed", what, name)
            failures = gather(failure)
            if any(failures):
                units_on = sorted({c.name for c in mine})
                raise RuntimeError(
                    f"{what}: building the {name!r} kernel family failed (widths "
                    f"{list(widths)}): "
                    + "; ".join(f"rank {r}: {f}" for r, f in enumerate(failures) if f)
                    + ". To serve without them, leave these slots out of "
                    "SGLANG_NVIDIA_MEGAKERNEL_SLOTS: " + ", ".join(units_on)
                )
            built[name] = family

    objects: Dict[Tuple[str, int], Dict[Tuple[str, str], Any]] = {}
    pending = list(steps)
    while pending:
        build_needed()
        declines: Dict[Tuple[str, int], Dict[Tuple[str, str], str]] = {}
        failure = None
        for key in pending:
            mode, rows = key
            objects[key] = {}
            for c in comps[key].chosen:
                try:
                    obj = built[c.family].provider(c, rows, mode)
                    if obj is None:
                        raise RuntimeError("the provider is None")
                except Decline as d:
                    declines.setdefault(key, {})[c.key] = str(d) or "declined at build"
                    continue
                except Exception as e:  # noqa: BLE001
                    failure = (
                        failure or f"{c.name} at {rows} rows ({mode}): {_describe(e)}"
                    )
                    logger.exception("%s: %s at %d rows failed", what, c.name, rows)
                    continue
                objects[key][c.key] = obj
        gathered = gather(
            (failure, sorted((k, sorted(v.items())) for k, v in declines.items()))
        )
        failures = [f for f, _ in gathered]
        if any(failures):
            raise RuntimeError(
                f"{what}: a kernel unit failed to build: "
                + "; ".join(f"rank {r}: {f}" for r, f in enumerate(failures) if f)
            )
        agreed: Dict[Tuple[str, int], Dict[Tuple[str, str], str]] = {}
        for r, (_, rank_declines) in enumerate(gathered):
            for key, items in rank_declines:
                for ckey, why in items:
                    agreed.setdefault(tuple(key), {}).setdefault(
                        tuple(ckey), why if r == 0 else f"rank {r}: {why}"
                    )
        pending = []
        for key in steps:
            if key in agreed:
                excluded[key].update(agreed[key])
                comps[key] = compose(key)
                pending.append(key)

    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key in steps:
        owners = comps[key].owners
        out[key] = {
            g: compose_group(
                stock(),
                GROUPS[g],
                {s: objects[key][owners[(g, s)].key] for (gg, s) in owners if gg == g},
                STEP_HELPERS.get(g),
            )
            for g, stock in groups.items()
        }
    # step 4: wire each consumed contract token into its providers, by width
    for key in sorted(steps, key=lambda k: (k[1], k[0])):
        chosen = comps[key].chosen
        for consumer in chosen:
            for token in consumer.requires:
                for p in chosen:
                    if p is not consumer and token in p.provides:
                        wire = getattr(objects[key][p.key], "wire", None)
                        if wire is not None:
                            wire(token)
    result = Build(
        profile=profile,
        families=dict(built),
        compositions=comps,
        groups=out,
        candidates=cands,
    )
    if rank == 0:
        for mode, rows in sorted(steps, key=lambda k: (k[0], k[1])):
            why = result.stock_reasons(mode, rows)
            if why:
                logger.info(
                    "%s: stock at %d rows%s: %s",
                    what,
                    rows,
                    "" if mode == "target_verify" else f" ({mode})",
                    "; ".join(f"{n}: {r}" for n, r in why.items()),
                )
    return result
