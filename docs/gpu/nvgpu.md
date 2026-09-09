# nvgpu

Notes from the [allbilly/nvgpu repository](https://github.com/allbilly/nvgpu),
which runs a small CUDA-style add kernel on an RTX 3080 eGPU using a handwritten
Python userspace stack.

## What is implemented

The live path reimplements the userspace pieces needed to boot the GPU System
Processor (GSP), create channels, build a GPFIFO launch, upload a hand-built
SM86 cubin, and execute the kernel. It does not import the tinygrad runtime on
the live path; tinygrad is retained only as a reference for constants and
health checks.

The cubin contains a small vector operation with four floating-point additions,
two loads and one store. On Linux, the hardware boundary is the open NVIDIA
kernel interface; on macOS, the project uses the TinyGPU helper for PCIe and
BAR access.

## Quick start

The offline self-test checks cubin construction and launch-word generation
without requiring an eGPU:

```bash
python3 examples/middle_nv.py --middle-selftest
```

With the RTX 3080 eGPU attached:

```bash
python3 examples/add.py
```

The expected result is:

```text
[11.0, 22.0, 33.0, 44.0]
```

The repository reports that device setup and GSP boot dominate the runtime;
the small add kernel itself takes only a few milliseconds.

## CUDA tools on macOS

macOS cannot run NVIDIA CUDA tools natively. The project uses Docker for tools
such as `nvdisasm`, without GPU passthrough:

```bash
python3 -c "import examples.middle_nv as a; open('add.cubin','wb').write(a.build_cubin())"
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 nvdisasm add.cubin
```

## References

- [nvgpu repository](https://github.com/allbilly/nvgpu)
- [tinygrad NVIDIA runtime reference](https://github.com/tinygrad/tinygrad)
