# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team): the source's
# python/sglang/srt/layers/attention/qsa/capture_fork.py. Changed: the driver library
# loads at the first call instead of at import, so the module imports without a GPU.
"""Fork a captured side stream from completed projection, without a main event."""

import ctypes as c
import functools


@functools.cache
def _driver():
    driver = c.CDLL("libcuda.so.1")
    info = driver.cuStreamGetCaptureInfo_v3
    info.argtypes = [
        c.c_void_p,
        c.POINTER(c.c_int),
        c.POINTER(c.c_uint64),
        c.POINTER(c.c_void_p),
        c.POINTER(c.POINTER(c.c_void_p)),
        c.POINTER(c.c_void_p),
        c.POINTER(c.c_size_t),
    ]
    info.restype = c.c_int
    update = driver.cuStreamUpdateCaptureDependencies_v2
    update.argtypes = [
        c.c_void_p,
        c.POINTER(c.c_void_p),
        c.c_void_p,
        c.c_size_t,
        c.c_uint,
    ]
    update.restype = c.c_int
    return info, update


def projection_frontier(stream):
    info, _ = _driver()
    status, capture_id, graph = c.c_int(), c.c_uint64(), c.c_void_p()
    nodes, edge_data, count = c.POINTER(c.c_void_p)(), c.c_void_p(), c.c_size_t()
    error = info(
        stream.cuda_stream,
        c.byref(status),
        c.byref(capture_id),
        c.byref(graph),
        c.byref(nodes),
        c.byref(edge_data),
        c.byref(count),
    )
    if error:
        raise RuntimeError(f"cuStreamGetCaptureInfo_v3: {error}")
    if status.value == 0:
        return None
    if status.value != 1 or not count.value:
        raise RuntimeError("Active projection capture has no dependency frontier")
    # The driver-owned array expires at the next stream API. Copy node handles
    # now; the nodes live until the graph is destroyed. Query edge data as well
    # to avoid a lossy query of the main stream's programmatic dependencies.
    return tuple(nodes[i] for i in range(count.value))


def fork_after_projection(stream, nodes):
    _, update = _driver()
    array = (c.c_void_p * len(nodes))(*nodes)
    # NULL edge data gives FULL completion dependencies: Qprep reads projected
    # qk only after it is written. Replace the temporary QKV event dependency.
    error = update(stream.cuda_stream, array, None, len(nodes), 1)
    if error:
        raise RuntimeError(f"cuStreamUpdateCaptureDependencies_v2: {error}")
