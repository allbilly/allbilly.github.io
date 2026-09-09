# Apple ANE

Notes from the [allbilly/ane repository](https://github.com/allbilly/ane), which
runs Apple Neural Engine operations from pure Python on M1 Asahi Linux. The
project programs the NPU directly and does not depend on Espresso, Core ML,
Metal, `.mlmodel` files, or private Apple APIs.

## Status and platform

The project is tested on Apple Silicon running Asahi Linux. The documented
bring-up path requires an ANE-enabled Asahi kernel and kernel module (KMD), so
these examples are intended for an Apple Silicon Linux installation rather
than a normal macOS environment.

## Supported operations

The example programs cover:

```text
ADD  MUL  MIN  MAX  SUMSQ  CONV  CONCAT  GEMM  RELU  SIGMOID
```

With the kernel support installed, basic examples can be run with Python:

```bash
pip install numpy
python examples/elementwise.py add
python examples/elementwise.py mul
python examples/conv.py
python examples/concat.py
python examples/gemm.py
python examples/relu.py
python examples/sigmoid.py
```

NumPy is used by many examples but is optional in the core approach.

## Execution paths

There are two complementary ways to run an operation:

1. Compile a model with `anecc` into an `.ane` file, then run it through the
   Python binding.
2. Extract the command buffer from an HWX file and replay it directly from
   Python, without an `.ane` file or `anecc`.

The repository’s `experimental/hwx2.py` tool generates a readable Python
program from an HWX file:

```bash
python experimental/hwx2.py hwx/sum.hwx -o sum_from_hwx.py
python sum_from_hwx.py
```

This makes the register programming and command-buffer layout inspectable.

## Example: convolution

The repository demonstrates convolution both through a compiled `.ane` model
and through direct register programming. The direct examples use the same
firmware and kernel data, and are intended to produce identical results.

The input layout is significant: the examples use a stride of `32` between
channels, matching the firmware’s expected layout. Kernel data must also be
byte-exact; an incorrectly positioned weight blob can make output channels
appear to be zero even when the register setup is otherwise correct.

## Kernel setup

The repository documents building an Asahi kernel with ANE support and
installing the KMD. The high-level flow is:

```bash
git clone https://github.com/AsahiLinux/linux.git --branch fairydust --single-branch
make -j$(nproc)
make dtbs -j$(nproc)
sudo make modules_install
sudo make dtbs_install
sudo make install
sudo reboot
```

After booting the new kernel, the ANE module and device should be available
before running the examples. See the repository README for the exact kernel
patches, device-tree details and recovery steps for kernel updates.

## References

- [Apple ANE repository](https://github.com/allbilly/ane)
- [Register Programming Guide](https://github.com/allbilly/ane/blob/main/REGISTER_PROGRMMING.md)
- [Asahi Linux](https://asahilinux.org/)
- [tinygrad ANE experiments](https://github.com/tinygrad/tinygrad/tree/v0.10.3/extra/accel/ane/ops)
