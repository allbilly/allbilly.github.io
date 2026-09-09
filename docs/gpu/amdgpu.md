# amdgpu

Notes from the [allbilly/amdgpu repository](https://github.com/allbilly/amdgpu),
which explores direct PM4 compute submission and trap-handler debugging on AMD
GPUs.

## What is implemented

The proof-of-concept opens a DRM render node, initializes `libdrm_amdgpu`,
allocates GPU buffer objects, maps them into the GPU virtual address space,
uploads a raw shader, emits PM4 compute packets, submits an indirect buffer and
waits for a fence.

The repository also includes register access through debugfs `regs2`, an
offscreen RADV triangle reference, raw GFX PM4 experiments and small GCN shader
examples.

## Build and run

On a Fedora-like system:

```bash
sudo dnf install libdrm-devel clang llvm vulkan-tools \
  mesa-vulkan-drivers vulkan-headers vulkan-loader-devel
make
build/amdgpu-poc --card /dev/dri/renderD128
```

Compile the shader examples for the tested Renoir target:

```bash
AMDGPU_MCPU=gfx90c make shaders
build/amdgpu-poc --shader build/add-output.bin --pass-output-va \
  --output-bytes 16
```

The expected first output bytes are `05 00 00 00`, representing `2 + 3 = 5`
in little-endian form. A multiply example writes `42` from `6 × 7`.

## Driver reference path

The project uses RADV through the normal Vulkan loader as a known-good triangle
reference before comparing the resulting command stream with direct GFX PM4
submission. The tested local GPU is AMD Renoir with LLVM target `gfx90c`.

## References

- [amdgpu repository](https://github.com/allbilly/amdgpu)
- [AMDGPU debugger notes](https://github.com/allbilly/amdgpu/blob/main/thegeeko.md)
- [tinygrad AMD runtime reference](https://github.com/tinygrad/tinygrad)
