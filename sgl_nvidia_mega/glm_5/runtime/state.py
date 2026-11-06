# SPDX-License-Identifier: Apache-2.0
"""Allocate and retain persistent resources using compiled Config metadata."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelState:
    metadata: Any
    workspace: Any
    epochs: Any
    peer_ptrs: tuple[int, ...]
    rank: int
    owners: tuple[Any, ...] = ()

    @classmethod
    def allocate(cls, metadata, device, *, group=None) -> "KernelState":
        """Allocate before capture. Keeping this object retains all peer mappings."""
        import torch
        import torch.distributed as dist

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GLM5 kernel state requires a CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot allocate GLM5 state inside CUDA-graph capture")
        scratch = torch.empty(
            metadata.workspace_bytes, dtype=torch.uint8, device=device
        )
        scratch.fill_(metadata.workspace_fill)
        epochs = torch.zeros(metadata.epoch_count, dtype=torch.int32, device=device)
        if not dist.is_initialized():
            peer = torch.zeros(metadata.peer_bytes, dtype=torch.uint8, device=device)
            return cls(metadata, scratch, epochs, (peer.data_ptr(),), 0, (peer,))
        group = group if group is not None else dist.group.WORLD
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if world != metadata.variant.tp_size:
            raise ValueError(f"{metadata.variant}: process group has {world} ranks")
        owners, pointers = _symmetric_peers(device, metadata.peer_bytes, group)
        return cls(metadata, scratch, epochs, pointers, rank, owners)


def _symmetric_peers(device, nbytes: int, group):
    import torch
    import torch.distributed as dist
    from torch._C._distributed_c10d import _SymmetricMemory

    world = dist.get_world_size(group)
    region = _SymmetricMemory.empty_strided_p2p(
        (nbytes,), [1], torch.uint8, device, group.group_name
    )
    symmetric = _SymmetricMemory.rendezvous(region)
    buffers = tuple(
        symmetric.get_buffer(rank, [nbytes], torch.uint8) for rank in range(world)
    )
    pointers = tuple(buffer.data_ptr() for buffer in buffers)
    if len(set(pointers)) != world or any(not ptr or ptr % 128 for ptr in pointers):
        raise RuntimeError("invalid GLM5 symmetric peer pointers")
    region.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return (region, symmetric, *buffers), pointers


def check_binding(state, metadata, device):
    """Fixed resource checks belong to binding, outside graph capture."""
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("bind GLM5 weights and state before CUDA-graph capture")
    if state.metadata != metadata or state.workspace.device != device:
        raise ValueError("GLM5 state does not match the compiled metadata/device")
