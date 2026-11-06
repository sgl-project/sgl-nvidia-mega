# GLM5 SM10x operator backends

include/ contains common/, comm/, mla/ and moe/. Args, Config, pipelines and
device stages stay separate from host-only TVM-FFI headers. src/ contains the
two specialization entry units and exports Config metadata with each cubin.

Python load compiles a cubin with NVCC against external CUTLASS headers and
JIT-compiles the typed C++ binding. Install its optional dependency with
`pip install -e ".[ffi]"`; a C++ compiler and Apache TVM-FFI >= 0.1.11 are required.

`load -> bind -> call` keeps native weights, cache and state alive without a
weight pack. C++ binds typed Args and TMA descriptors once, then checks dynamic
tensors and launches on the framework's current CUDA stream through the driver.
PDL stays disabled. Config metadata supplies resource sizes and launch geometry;
Python allocates and retains state. The eight-row kernel covers 4x2 and 8x1.

Device synchronization uses native CUTLASS barrier waits and marker polling.
GLM5 adds no device timeout, timer checks or device logging. Serving failure
detection and termination rely on the upper-layer watchdog or process controls.
See docs/glm5_megakernel_integration_plan.md for current evidence and limitations.
