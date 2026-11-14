"""runtime/capture_fork.py under a real CUDA graph capture (the PDL-index branch's fork):
the side stream's first captured kernel depends on the projection's nodes only, not on
the main-stream work captured after them, and the replay reads the finished
projection. Outside a capture the frontier is None."""

import ctypes as c

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.runtime import capture_fork

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _dependencies(node):
    driver = c.CDLL("libcuda.so.1")
    count = c.c_size_t()
    try:  # cuGraphNodeGetDependencies_v2(node, deps, edge data, count)
        get = driver.cuGraphNodeGetDependencies_v2
        args = lambda deps: (c.c_void_p(node), deps, None, c.byref(count))
    except AttributeError:
        get = driver.cuGraphNodeGetDependencies
        args = lambda deps: (c.c_void_p(node), deps, c.byref(count))
    assert get(*args(None)) == 0
    deps = (c.c_void_p * count.value)()
    assert get(*args(deps)) == 0
    return {d for d in deps}


def test_side_stream_depends_on_the_projection_only():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(8, 2560, generator=g, device="cuda").bfloat16()
    w = torch.randn(640, 2560, generator=g, device="cuda").bfloat16()
    main, alt = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(main):
        assert capture_fork.projection_frontier(main) is None  # not capturing
        x @ w.T  # cuBLAS initializes its handle outside the capture
    main.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        current = torch.cuda.current_stream()
        qk = x @ w.T  # the index projection
        nodes = capture_fork.projection_frontier(current)
        later = x * 3  # the QKV after it
        later_nodes = capture_fork.projection_frontier(current)
        alt.wait_stream(current)
        capture_fork.fork_after_projection(alt, nodes)
        with torch.cuda.stream(alt):
            side = qk * 2
            (side_node,) = capture_fork.projection_frontier(alt)
            # while the captured graph exists (torch drops it once instantiated)
            deps = _dependencies(side_node)
        qk.record_stream(alt)
        current.wait_stream(alt)
    assert nodes and later_nodes and set(nodes).isdisjoint(later_nodes)
    assert deps == set(nodes)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(side, qk * 2)
    assert torch.equal(later, x * 3)
