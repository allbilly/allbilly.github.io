# RK3588 Mali GPU

Notes from the [allbilly/rk3588gpu repository](https://github.com/allbilly/rk3588gpu),
which brings up the RK3588 Mali GPU from Python by opening the device, issuing
ioctls and submitting work without libmali or Mesa in the direct path.

## Two driver paths

The repository documents both GPU stacks found on RK3588 systems:

| Stack | Device | Submission path |
| --- | --- | --- |
| Panthor mainline | `/dev/dri/renderD128` | DRM Panthor ioctls and a CSF stream |
| Proprietary kbase | `/dev/mali0` | Mali kbase CSF ioctls |

On many vendor images, `/dev/mali0` is the Mali GPU while `renderD128` belongs
to the display subsystem. The correct device node therefore depends on the
kernel and driver stack installed on the board.

## Submission flow

The kbase path follows this sequence:

```text
Python → VERSION_CHECK / SET_FLAGS → global interface query
       → queue group creation → memory allocation → command-stream fill
       → queue kick → Mali G610 → completion event
```

Captured `.mcap` traces can be decoded and replayed, while the command-stream
disassembler exposes the CSF words used by the examples.

## Quick start

For the mainline Panthor path:

```bash
python3 examples/add.py
python3 examples/add.py --dry-run -v
```

The standalone vector-add example is expected to produce `[11, 22, 33, 44]`.
For a vendor kbase device, the repository provides capture and replay tools:

```bash
python3 experiemental/tools/mcap.py gen-sample -o /tmp/test.mcap
python3 experiemental/replay.py /tmp/test.mcap --dry-run
```

Live replay requires `/dev/mali0` and suitable `video` or `render` group
permissions. A GPU reset or reboot may be required after a fatal command-stream
error.

## References

- [RK3588 GPU repository](https://github.com/allbilly/rk3588gpu)
- [Mali Panfrost documentation](https://docs.mesa3d.org/drivers/panfrost.html)
- [Linux Panfrost UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/drm/panfrost_drm.h)
