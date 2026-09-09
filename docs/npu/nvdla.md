# NVDLA

Notes from the [allbilly/nvdla repository](https://github.com/allbilly/nvdla),
which explores pure register programming on the NVDLA neural processing unit.

## Quick start

The repository uses the NVDLA virtual platform to test loadable programs and
compare direct register programming with the reference runtime:

```bash
docker run -it --rm -p 6667:6667 \
  -v ./examples/vp:/usr/local/nvdla/vp:z \
  -v ./examples:/usr/local/nvdla/examples:z \
  -w /usr/local/nvdla/ \
  onnc/vp aarch64_toplevel -c aarch64_nvdla.lua

ssh -p 6667 root@127.0.0.1
mount -t 9p -o trans=virtio r /mnt
cd /mnt && insmod drm.ko && insmod opendla.ko
./nvdla_runtime --loadable test_Add.nvdla --image input1x5x7.pgm --rawdump
```

The Python examples include simple add, convolution and `1×1` convolution
experiments:

```bash
python3 examples/simple_add.py
python3 examples/conv.py
python3 examples/conv_1x1x8.py
```

## Loadable parsing

An NVDLA loadable can be inspected with:

```bash
python3 parse.py examples/vp/test_Add.nvdla
```

The repository uses this to connect the serialized loadable format with the
register programming needed by the virtual platform.

## Capture caveat

The logging kernel module is useful for capturing registers, but its submit
path is not the validation path. First confirm that the operation works with
`opendla.ko`; then use the logging module for investigation. The README also
notes that verbose buffer dumping can be slow and may leak temporary GEM
handles during long capture runs.

## References

- [NVDLA repository](https://github.com/allbilly/nvdla)
- [NVDLA virtual platform](https://github.com/nvdla/vp)
- [NVDLA environment setup](https://nvdla.org/hw/v2/environment_setup_guide.html)
