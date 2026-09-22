# How to add a new backend to tinygrad: A Rockchip NPU example  
Last update: Sep 20 2026

TLDR: This blog will mainly use DPU EW op for the 28 GroupOps.ALU, treat CMAC like tensor core for GEMM/CONV if the shape matches, otherwise will just use EW add/mul for GEMM. 

Tinygrad is a zero-dependency minmial codebase (25407 core lines @20260920) to do ML in python, those lines already included a PyTorch like frontend and kernel space GPU driver down to MMIO written in user space, so makes it the perfect place to support USB3 eGPU thats can drives a car (https://www.youtube.com/watch?v=nmTepfv3Itg) and add new accelorator support. 

"Your accelerator of choice only needs to support a total of ~25 low level ops."
-- from tinygrad README (https://github.com/tinygrad/tinygrad/blob/ebf163682acfa4a6be2775c5efa02a503c86a220/README.md?plain=1#L112)

We will 
- Part 1: Modify ops_python.py as starting point to understand the NPU and implements the 25 Ops
- Part 2: Run test_ops.py to see how many passes we can achieve with it  
- Part 3: Write proper ops_rockchip.py from scratch before fixig failed cases
- Part 4: Fail case fix by Pattern Matcher
- Part 5: Add tests cases and Emulator for CI
- Part 5: Issues to be solved before a PR
- Part 6: Repeat 1 - 5 on Apple ANE

U can follow among if u own an OrangePi 5 running the Orange pi Ubuntu 22.02 image.

## Part 1: modify ops_python.py as starting point to understand the NPU and implements the 25 Ops

This approach was inspired by liej6799 (https://github.com/liej6799/tinygrad/blob/3588-new/tinygrad/runtime/ops_rockchip.py) who made the simplest ADD works on tinygrad while I was struggling how to port the registers and Ops I reversed (https://github.com/allbilly/npu/blob/master/include/rknnops.h) to tinygrad.

Lets begin with
```
git clone https://github.com/tinygrad/tinygrad 
cd tinygrad && git checkout ebf1636
uv venv && source .venv/bin/activate
cp tinygrad/runtime/ops_python.py tinygrad/runtime/ops_rockchip.py
```

In tinygrad, u can choose a runtime with env like DEV=ROCKCHIP. If u want to run tests, some dependenceis are still needed like pytest and numpy/torch to generte reference output. 
```
uv pip install -e . numpy torch --torch-backend=cpu
DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add
```

```
(tinygrad) orangepi@orangepi5:~/tinygrad$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add
  Traceback (most recent call last):
    File "/home/orangepi/tinygrad/test/backend/test_ops.py", line 92, in <module>
      class TestOps(unittest.TestCase):
    File "/home/orangepi/tinygrad/test/backend/test_ops.py", line 727, in TestOps
      @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, NIRRenderer), "TODO: broken in LVP")
                                  ~~~~~~^^^^^^^^^^^^^^^^
    File "/home/orangepi/tinygrad/tinygrad/device.py", line 32, in __getitem__
      return self.__get_canonicalized_item(ix)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/orangepi/tinygrad/tinygrad/device.py", line 40, in __get_canonicalized_item
      ret = self.get_class(ix)(ix)
            ^^^^^^^^^^^^^^^^^^
    File "/home/orangepi/tinygrad/tinygrad/device.py", line 37, in get_class
      return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.runtime.ops_{x}')) if
  (cname.lower() == x + "device")][0]

  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~~~~~~~~~~~~^^^
  IndexError: list index out of range
```

This is because RockchipDevice was not found in our new ops_rockchip.py. Lets replace all "Python" with "Rockchip"

```diff
+ops_map = {Ops.ADD: 2}
+
-class PythonProgram(Program['PythonDevice']):
-  def __init__(self, dev:'PythonDevice', obj:TinyELF):
+class RockchipProgram(Program['RockchipDevice']):
+  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
     self.uops: list[UOp] = pickle.loads(obj.lib)
-    self.tensor_cores = PythonRenderer(obj.target).tensor_cores
+    self.tensor_cores = RockchipRenderer(obj.target).tensor_cores
     self.uop_to_index: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
     self.loop_ends: dict[UOp, int] = {u.src[1]:i for i, u in enumerate(self.uops) if u.op == Ops.END}
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@ -156,12 +156,12 @@ class PythonProgram(Program['PythonDevice']):
         i += 1
     return time.perf_counter() - st

-class PythonCompiler(Compiler):
+class RockchipCompiler(Compiler):
   def compile(self, src:str) -> bytes: return base64.b64decode(src)

-class PythonRenderer(Renderer):
+class RockchipRenderer(Renderer):
-  code_for_op = python_alu
+  code_for_op = {op: python_alu.get(op, lambda: None) for op in ops_map}
-  compiler = PythonCompiler()
+  compiler = RockchipCompiler()

   def __init__(self, target:Target):
     assert (emu:=getenv("EMULATE", "")) == "", ("EMULATE is deprecated, use DEV=PYTHON::" +
@@ -181,6 +181,6 @@ class PythonRenderer(Renderer):

   def supported_dtypes(self): return {d for d in super().supported_dtypes() if d != dtypes.half or sys.version_info >= (3, 12)}

-class PythonDevice(Compiled):
+class RockchipDevice(Compiled):
   def __init__(self, device:str):
-    super().__init__(device, HostAllocator(self), [PythonRenderer], PythonProgram)
+    super().__init__(device, HostAllocator(self), [RockchipRenderer], RockchipProgram)
```

```
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add

test_add (__main__.TestOps.test_add) ...
testing                     [(45, 68), (45, 68)]   torch/tinygrad fp: 0.26 / 263.81 ms  bp: 1.07 / 2.80 ms
testing                                       []   torch/tinygrad fp: 0.27 / 0.25 ms  bp: nan / nan ms
testing                     [(45, 68), (45, 68)]   torch/tinygrad fp: 0.06 / 107.37 ms  bp: 0.49 / 0.73 ms
testing                                 [(), ()]   torch/tinygrad fp: 0.05 / 0.09 ms  bp: 0.44 / 0.39 ms ok

----------------------------------------------------------------------
Ran 1 test in 0.494s

OK
```

Great, but it was running on the CPU not the NPU. We need to understand a bit more on the NPU first. 

Rockchip NPU was derived from the open NVDLA (doc link) NPU from nvidia as described in mtx512 blog(https://jas-hacks.blogspot.com/2024/02/rk3588-reverse-engineering-rknn.html). Rockchip neither release the verilog nor RKNN (link) compiler and runtime, but they released the linux KMD rknpu_driver(https://github.com/allbilly/rknpu_driver) because of linux GPLv2 license. Thanks to prior work from phhusson (https://github.com/phhusson/rknpu-reverse-engineering) that introduced me the concept of IOCTL and GEM. Reading the rknpu_driver code we can easily find DRM_IOCTL_RKNPU_SUBMIT and the submit struct it needed. Rockchip proprietary runtime being closed source is not really a problem for us because it still needed some way to talk to linux KMD, so we can capture the RKNN bytes, replays it, and rewrite into readable code. With GDB and enough pantient, i captured the submit struct and bytes in each allocated BO/GEM and rewrote in C(https://github.com/allbilly/npu/blob/master/include/rknnops.h) according to the leaked TRM(https://github.com/liej6799/rk3588/blob/main/trm.pdf) and prior work from mtx512(https://github.com/mtx512/rk3588-npu) who did the GEMM reverse. Its a year long frustrating but rewarding process as SOTA GPT-5.1 back then was bad in register programming, it helps with decoding a bit but dont even think about auto debug why the replay/rewrite is not working, which forces me to read/program/debug everything myself. Most effort was in decoding CONV input/weight bytes packing for every shapes(https://github.com/allbilly/npu/blob/master/ops_reg/main.c) and decoding/programming the LUT silu (https://github.com/allbilly/npu/blob/master/ops_rknn/act/silu.py) and sigmoid(https://github.com/allbilly/npu/blob/master/ops_rknn/act/sigmoid.py). Details are not included in this blog, if u are still interested in my frustration, just point ur agent to commit history in allbilly/npu and allbilly/rk3588.

Rockchip and NVDLA
![alt text](image-1.png)

The NPU is a 3 core fixed stage pipeline processor, input and weight go through CORE (MAC array) -> DPU (elementwise op) -> PPU (pooling/reduction op). In NVDLA terms, CORE is CSC/CMAC/CACC(https://github.com/nvdla/hw/tree/nvdlav1/vmod/nvdla), DPU is SDP(https://github.com/nvdla/hw/tree/nvdlav1/vmod/nvdla/sdp), PPU is PDP(https://github.com/nvdla/hw/tree/nvdlav1/vmod/nvdla/pdp). This blog will mainly use DPU EW op for the 25 Ops, treat CMAC like tensor core for GEMM/CONV if the shape matches, otherwise will just use EW add/mul for GEMM. 

Now to do a simple fp16 ADD example, this is also my NPU health check script.
```
cd ~ && git clone https://github.com/allbilly/rk3588
python rk3588/examples/simple_add.py

ret=0, handle=1, obj_addr=0xffffff8137dadc00, dma_addr=0xfffff000
Memory mapped at offset=0x100000000
ret=0, handle=2, obj_addr=0xffffff8137da8000, dma_addr=0xffffe000
Memory mapped at offset=0x100001000
ret=0, handle=3, obj_addr=0xffffff8137daec00, dma_addr=0xff800000
Memory mapped at offset=0x100002000
ret=0, handle=4, obj_addr=0xffffff8137dac400, dma_addr=0xff400000
Memory mapped at offset=0x100402000
ret=0, handle=5, obj_addr=0xffffff8137dab000, dma_addr=0xff000000
Memory mapped at offset=0x100802000
reset_npu ret=0
SUBMIT ret=0
ADD  NPU=[8 8 8 8 8 8 8 8] expected=[8 8 8 8 8 8 8 8] PASS
```

what the script does is first open the NPU device to obtain fd, then use ctypes to create C struct, ioctl DRM_IOCTL_RKNPU_MEM_CREATE to allocate BO for task/regcmd/input/weight/output, then update dma_addr of input/weight/output in the register sequence, copy the registers to C array regcmd, set regcmd address in a task, and DRM_IOCTL_RKNPU_SUBMIT to submit the task. npu_regs was captured from RKNN simple add and removed all unesscary lines.

```python
npu_regs = [
        0x1001000001e5400c,
        0x1001480000024010,
        0x100100070007403c,
        0x1001000000004030,
        0x1001108202c04070,
        0x200100000000500c,
        0x2001000000005010,
        0x2001000700075014,
        0x2001400000085034,
        (reg.TARGET_DPU  << 48) | ((output_mem_create.dma_addr & 0xFFFFFFFF) << 16) | reg.DST_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((input_mem_create.dma_addr & 0xFFFFFFFF) << 16) | reg.RDMA_SRC_BASE_ADDR,
        (reg.TARGET_RDMA << 48) | ((weight_mem_create.dma_addr & 0xFFFFFFFF) << 16) | reg.RDMA_EW_BASE_ADDR,
        0x2001000178495044,
        0x0081000000180008,
    ]
```

Dont worry, we will decode it later. The NPU programming model is mulitple NPU submits by host -> mulitple tasks in one submit -> fused mulitple ops in one task, e.g. CONV/MAC+ELEMENTWISE+BN+BS+RELU+POOL in one task.

Great, now we can port this to ops_rockchip.py and open the NPU device. In simple_add.py, we used os.open(f"/dev/dri/card1", os.O_RDWR), but in tinygrad we will use FileIOInterface like other runtime do. 

AGENTS.md
```diff
+- DO NOT run -n12 for RK NPU use -n0 instead, otherwise will race condition crashes the NPU. 
```

```diff
-import pickle, base64, itertools, time, sys, ctypes
+import pickle, base64, itertools, time, sys, ctypes, os
+from tinygrad.runtime.support.hcq import FileIOInterface
  ...
  class RockchipDevice(Compiled):
    def __init__(self, device:str):
+    self.fd_ctl = FileIOInterface("/dev/dri/card1", os.O_RDWR)
```

we also need the C struct and registers offset from rknpu_driver/inlcude/*.h (https://github.com/allbilly/rknpu_driver/tree/main/include) to be imported in python, here i will just reuse the autogen from liej6799(https://github.com/liej6799/tinygrad/blob/3588-new/tinygrad/runtime/autogen/rockchip.py)

```
cd ~/tinygrad/tinygrad/runtime/autogen
wget https://raw.githubusercontent.com/liej6799/tinygrad/refs/heads/3588-new/tinygrad/runtime/autogen/rockchip.py
```

And allocate task/regcmd/input/weight/output Buffer Object
```diff
-import pickle, base64, itertools, time, sys, ctypes, os
+import pickle, base64, itertools, time, sys, ctypes, os, mmap
+from tinygrad.runtime.autogen import rockchip as rk

class RockchipDevice(Compiled):
     def __init__(self, device:str):
       self.fd_ctl = FileIOInterface("/dev/dri/card1", os.O_RDWR)
       super().__init__(device, HostAllocator(self), [RockchipRenderer], RockchipProgram)
+      self.task_buf, self.task_mem = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING)
+      self.regcmd_buf, self.regcmd_mem = self._gpu_alloc(8192)
+      self.input_buf, self.input_mem = self._gpu_alloc(4194304)
+      self.weight_buf, self.weight_mem = self._gpu_alloc(4194304)
+      self.output_buf, self.output_mem = self._gpu_alloc(4194304)
+  
+    def _gpu_alloc(self, size:int, flags:int=0) -> tuple[int, rk.struct_rknpu_mem_create]:
+      mem = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
+      try:
+        mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem.handle)
+        addr = self.fd_ctl.mmap(0, mem.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset)
+      except Exception:
+        rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=mem.handle, obj_addr=mem.obj_addr)
+        raise
+      return addr, mem
+  
+    def _gpu_free(self, addr:int, mem:rk.struct_rknpu_mem_create) -> None:
+      FileIOInterface.munmap(addr, mem.size)
+      rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=mem.handle, obj_addr=mem.obj_addr)
```

Next, update dma_addr of input/weight/output in the register sequence, dma_addr are owned by RockchipDevice so we can get it with self.dev.output_mem.dma_addr

```diff
 class RockchipProgram(Program['RockchipDevice']):
   def __init__(self, dev:'RockchipDevice', obj:TinyELF):
+    self.dev = dev
     self.uops: list[UOp] = pickle.loads(obj.lib)
     self.tensor_cores = RockchipRenderer(obj.target).tensor_cores
     self.uop_to_index: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
     self.loop_ends: dict[UOp, int] = {u.src[1]:i for i, u in enumerate(self.uops) if u.op == Ops.END}
+    self.npu_regs = [
+      0x1001000001e5400c,
+      0x1001480000024010,
+      0x100100070007403c,
+      0x1001000000004030,
+      0x1001108202c04070,
+      0x200100000000500c,
+      0x2001000000005010,
+      0x2001000700075014,
+      0x2001400000085034,
+      ((rk.DPU + 1) << 48) | ((self.dev.output_mem.dma_addr & 0xFFFFFFFF) << 16) | rk.REG_DPU_DST_BASE_ADDR,
+      ((rk.DPU_RDMA + 1) << 48) | ((self.dev.input_mem.dma_addr & 0xFFFFFFFF) << 16) | rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
+      ((rk.DPU_RDMA + 1) << 48) | ((self.dev.weight_mem.dma_addr & 0xFFFFFFFF) << 16) | rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
+      0x2001000178495044,
+      0x0081000000180008,
+    ]
```

Next, copy the registers to the C array regcmd, set the regcmd DMA address in a task, prepare the submut struct and submit with DRM_IOCTL_RKNPU_SUBMIT 
we will add a function submit() in RockchipProgram

```diff
+  def submit(self) -> None:
+    regs = self.npu_regs
+    # copy the registers to the C array regcmd
+    ctypes.memset(self.dev.regcmd_buf, 0, self.dev.regcmd_mem.size)
+    regcmd = (ctypes.c_uint64 * len(regs)).from_address(self.dev.regcmd_buf)
+    regcmd[:] = list(regs)
+
+    # set the regcmd DMA address in a task
+    tasks = ctypes.cast(self.dev.task_buf, ctypes.POINTER(rk.struct_rknpu_task))
+    tasks[0] = rk.struct_rknpu_task(
+      flags=0,
+      op_idx=4,
+      enable_mask=0x18,
+      int_mask=0x300,
+      int_clear=0x1ffff,
+      int_status=0,
+      regcfg_amount=len(regs),
+      regcfg_offset=0,
+      regcmd_addr=self.dev.regcmd_mem.dma_addr
+    )
+
+    # prepare the submut struct
+    submit = rk.struct_rknpu_submit(
+      flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
+      timeout=6000,
+      task_start=0,
+      task_number=1,
+      task_counter=0,
+      priority=0,
+      task_obj_addr=self.dev.task_mem.obj_addr,
+      core_mask=1,
+      fence_fd=-1
+    )
+    submit.subcore_task[0] = rk.struct_rknpu_subcore_task(task_start=0, task_number=1) # assigned to core 0
+    submit.subcore_task[1] = rk.struct_rknpu_subcore_task(task_start=1, task_number=0)
+    submit.subcore_task[2] = rk.struct_rknpu_subcore_task(task_start=2, task_number=0)
+
+    # submit with DRM_IOCTL_RKNPU_SUBMIT
+    rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl, __payload=submit)
```

and then pack our inputs


```diff
-import pickle, base64, itertools, time, sys, ctypes, os, mmap
+import pickle, base64, itertools, time, sys, ctypes, os, mmap, struct

 class RockchipProgram(Program['RockchipDevice']):
   def __init__(self, dev:'RockchipDevice', obj:TinyELF):
     self.dev = dev
+    self.ops_map = ops_map
```

`self.ops_map` uses the shared module-level map introduced above. It currently contains only ADD, whose EW algorithm code is 2. At this step the runtime uses map membership to select the NPU execution path, and the renderer uses the same keys to select code-generation rewrites. The captured register sequence remains hardcoded for ADD.

```diff
+  def add(self, a:list[float], b:list[float]) -> list[float]:
+    assert b is not None and len(a) == len(b)
+    result:list[float] = []
+    # The captured register sequence processes eight FP16 elements per submission.
+    for start in range(0, len(a), 8):
+      lhs, rhs = a[start:start+8], b[start:start+8]
+      to_mv(self.dev.input_buf, 16)[:] = struct.pack("<8e", *(lhs + [0.0] * (8 - len(lhs))))
+      to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8e", *(rhs + [0.0] * (8 - len(rhs))))
+      self.submit()
+      result.extend(struct.unpack("<8e", to_mv(self.dev.output_buf, 16))[:len(lhs)])
+    return result
```

but exec_alu() is still doing the calculation on CPU and stores result to values[u], we want it runs on NPU instead.

```diff
-          values[u] = [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)]
+          if dtypes.is_float(u.dtype):
+            if u.op not in self.ops_map or u.dtype != dtypes.half:
+              raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
+            values[u] = self.add(*src_values)
+          else:
+            values[u] = [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)]
```

Floating-point ALU operations now either run on the NPU or raise a clear error. Our map only contains ADD and the captured registers use FP16, so FP32 ADD and unsupported floating-point operations are rejected. Non-floating-point ALU operations, including integer address calculations, still use the CPU interpreter. This avoids a successful ROCKCHI-labelled floating-point kernel silently calculating on the CPU.

and for simplicity, comment the other cases and leave the first one  
```python
def test_add(self):
  helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
  # helper_test_op([], lambda: torch.tensor(1)+0.5, lambda: Tensor(1)+0.5, forward_only=True)
  # helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
  # helper_test_op([(), ()], lambda x,y: x+y)
```

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add

NotImplementedError: ROCKCHIP NPU does not support Ops.ADD with dtypes.float

----------------------------------------------------------------------
Ran 1 test in 0.191s

FAILED (errors=1)
```

Its raise NotImplementedError as DEFAULT_FLOAT is fp32, lets set as HALF
```
$ DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add

test_add (__main__.TestOps.test_add) ...
testing                     [(45, 68), (45, 68)]   torch/tinygrad fp: 0.64 / 390.55 ms  bp: 0.98 / 3.25 ms ok

----------------------------------------------------------------------
Ran 1 test in 0.496s

OK
```

Great its run, lets check whats happening with DEBUG=5
```bash
$ DEBUG=5 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add | grep "\*\*\*"

compiling E_255_4_3                               : 100%|█| 1/1 [00:00<00:00, 12.38it/s]
compiling E_1020_3                                : 100%|█| 1/1 [00:00<00:00, 36.65it/s]
ok

----------------------------------------------------------------------
Ran 1 test in 0.615s

OK
*** ROCKCHI    1 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   4985.97us/     4.99ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    2 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   2234.13us/     7.22ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    3 E_255_4_3                                      arg  3 mem   0.00 GB tm    323.77ms/   330.99ms (      0 GFLOPS    0|0      GB/s)
*** CPU        4 E_1020_3                                       arg  1 mem   0.00 GB tm     28.58us/   331.02ms (      0 GFLOPS    0|0      GB/s)
*** CPU        5 E_1020_3                                       arg  1 mem   0.00 GB tm     23.92us/   331.04ms (      0 GFLOPS    0|0      GB/s)
```

Why the last 2 kernels running on CPU not ROCKCHIP? Turns out those are the backward kernel that we shd just skips at this stage with FORWARD_ONLY=1
```
$ DEBUG=5 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add | grep "\*\*\*"

compiling E_255_4_3                               : 100%|█| 1/1 [00:00<00:00, 12.31it/s]
ok

----------------------------------------------------------------------
Ran 1 test in 0.411s

OK
*** ROCKCHI    1 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   5080.76us/     5.08ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    2 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   2346.71us/     7.43ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    3 E_255_4_3                                      arg  3 mem   0.00 GB tm    220.15ms/   227.58ms (      0 GFLOPS    0|0      GB/s)
```

Great now all kernels are on ROCKCHIP and we shd uncomment the remaining cases in test_add one by one
2. torch.tensor(1)+0.5 : showed a CPU kernel even with NOOPT=1 because constant folding in symbolic.py still optimized it
3. [(45,68), (45,68)]  : ROCKCHIP only kernels
4. [(), ()]            : showed a CPU kernel just like (2)

We passed test_add on NPU, next for MUL. To add NPU MUL support, we shdnt rely on the hardcoded hex blob any more.
I wrote a decode script(https://github.com/allbilly/npu/blob/master/ops_reg/dump.py) to decode the RKNN weight BO, why weight u might ask, because RKNN put weight and regcmd in the same BO.
You can use it with RKNN gdb here(https://github.com/allbilly/npu/blob/master/ops_reg/run.sh) and here(https://github.com/allbilly/npu/blob/master/ops_reg/test.gdb)

The decoded registers are here (https://github.com/allbilly/rk3588/blob/c6944a6/examples/elementwise.py#L295-L352). This version also has `CAST_BOOL_HALF`, which runs an INT16 MUL on 0/1 lanes to form the FP16 bit patterns. We will expose that CAST to tinygrad later; for now we can carry its register mode into the builder.
Most important one for us is EW_CFG, and TRM shows

| Bit   | Attr | Reset | Description |
|------:|:----:|:-----:|-------------|
|    31 |  RW  | `0x0` | **ew_cvt_type** — Convert type of EW input conversion when input is 0.5.<br>`1'd0`: Mul first<br>`1'd1`: Add first |
|    30 |  RW  | `0x0` | **ew_cvt_round** — Rounding type of EW input conversion when input is 0.5.<br>`1'd0`: If the integer is odd, carry 1<br>`1'd1`: Carry 1 no matter what the integer is |
| 29:28 |  RW  | `0x0` | **ew_data_mode** — Data mode of the data from ERDMA. |
| 27:24 |  RO  | `0x0` | Reserved. |
| 23:22 |  RW  | `0x0` | **edata_size** — Data size of the cube from ERDMA.<br>`2'd0`: 4-bit<br>`2'd1`: 8-bit<br>`2'd2`: 16-bit<br>`2'd3`: 32-bit |
|    21 |  RW  | `0x0` | **ew_equal_en** — Min/max equal enable.<br>`1'd0`: Disable<br>`1'd1`: Enable |
|    20 |  RW  | `0x0` | **ew_binary_en** — Min/max binary enable.<br>`1'd0`: Disable<br>`1'd1`: Enable |
| 19:16 |  RW  | `0x0` | **ew_alu_algo** — EW core ALU operation type.<br>`4'd0`: Max<br>`4'd1`: Min<br>`4'd2`: Add<br>`4'd3`: Div<br>`4'd4`: Minus<br>`4'd5`: Abs<br>`4'd6`: Neg<br>`4'd7`: Floor<br>`4'd8`: Ceil |
| 15:11 |  RO  | `0x0` | Reserved. |
|    10 |  RW  | `0x0` | **ew_relux_en** — Enable RELUX.<br>`1'd0`: Disable<br>`1'd1`: Enable |
|     9 |  RW  | `0x0` | **ew_relu_bypass** — Bypass EW core RELU operation.<br>`1'd0`: Do not bypass<br>`1'd1`: Bypass |
|     8 |  RW  | `0x0` | **ew_op_cvt_bypass** — Bypass EW input converter.<br>`1'd0`: Do not bypass<br>`1'd1`: Bypass |
|     7 |  RW  | `0x0` | **ew_lut_bypass** — Bypass LUT.<br>`1'd0`: Do not bypass<br>`1'd1`: Bypass |
|     6 |  RW  | `0x0` | **ew_op_src** — Operand source.<br>`1'd0`: From configure register<br>`1'd1`: From outside |
|     5 |  RW  | `0x0` | **ew_mul_prelu** — Enable MUL PRELU.<br>`1'd0`: Disable<br>`1'd1`: Enable |
|   4:3 |  RO  | `0x0` | Reserved. |
|     2 |  RW  | `0x0` | **ew_op_type** — Operator type.<br>`1'd0`: ALU<br>`1'd1`: MUL |
|     1 |  RW  | `0x0` | **ew_op_bypass** — Bypass EW core ALU and MUL operation.<br>`1'd0`: Do not bypass<br>`1'd1`: Bypass |
|     0 |  RW  | `0x0` | **ew_bypass** — Bypass EW core.<br>`1'd0`: Do not bypass EW core<br>`1'd1`: Bypass EW core |

At first glance, ew_alu_algo looks interesting, but it doesnt contains MUL we want

ew_alu_algo
| Value  | Operation |
|:------:|-----------|
| `4'd0` | Max       |
| `4'd1` | Min       |
| `4'd2` | Add       |
| `4'd3` | Div       |
| `4'd4` | Minus     |
| `4'd5` | Abs       |
| `4'd6` | Neg       |
| `4'd7` | Floor     |
| `4'd8` | Ceil      |

MUL is is in ew_op_type
| Value  | Operator Type |
|:------:|---------------|
| `1'd0` | ALU           |
| `1'd1` | MUL           |

And from the RKNN capture and playing around with different val, "MUL" would need to set not only DPU_EW_CFG_EW_OP_TYPE but also DPU_EW_CFG_EW_OP_CVT_BYPASS

The following is extracted from elementwise.py in allbilly/rk3588
```python
int16_mode = op == "CAST_BOOL_HALF"
out_precision = 1 if int16_mode else (2 if hw_out_fp16 else 5)
precision = 1 if int16_mode else 2
out_cvt_scale = (1 if fdiv_op or int16_mode else ((1 << 16) | 1)) if hw_out_fp16 else 0
scale_reg = reg.EW_CVT_SCALE_VALUE if int16_mode else reg.OUT_CVT_SCALE

task_regs.append([
    E(reg.DPU,  reg.S_POINTER, 0x0000000E),
    E(reg.DPU,  reg.FEATURE_MODE_CFG,
        ((15 << 5) |                          # DPU_FEATURE_MODE_CFG_BYPASS
          (2 << 1)  |                          # DPU_FEATURE_MODE_CFG_MODE
          1)                                    # DPU_FEATURE_MODE_CFG_FLYING_MODE
    ),
    E(reg.DPU,  reg.DATA_FORMAT,
        ((out_precision << 29) |              # DPU_DATA_FORMAT_OUT_PRECISION
          (precision << 26) |                  # DPU_DATA_FORMAT_IN_PRECISION
          precision)                            # DPU_DATA_FORMAT_PROC_PRECISION
    ),
    E(reg.DPU,  reg.DATA_CUBE_WIDTH, dataout_width),
    E(reg.DPU,  reg.DATA_CUBE_HEIGHT, 0),
    E(reg.DPU,  reg.DATA_CUBE_NOTCH, 0),
    E(reg.DPU,  reg.DATA_CUBE_CHANNEL,
        ((7 << 16) |                          # DPU_DATA_CUBE_CHANNEL_CUBE
          7)                                    # DPU_DATA_CUBE_CHANNEL_ATOMICS
    ),
    E(reg.DPU,  reg.EW_CFG,
        # selects MAX=0, ADD=2, FDIV=3, SUB=4 when EW_OP_TYPE[2]=0.
        # MUL/NEG/CAST_BOOL_HALF use EW_OP_TYPE[2]=1 instead of an ALU selector.
        # Base config: data_mode=1, data_size=2, relu_bypass=1, lut_bypass=1, op_src=1
        ((1 << 28) |                         # DPU_EW_CFG_EW_DATA_MODE
          (2 << 22) |                         # DPU_EW_CFG_EDATA_SIZE
          ({"MAX": 0, "ADD": 2, "FDIV": 3, "SUB": 4}.get(op, 0) << 16) |  # DPU_EW_CFG_EW_ALU_ALGO
          (1 << 9)  |                         # DPU_EW_CFG_EW_RELU_BYPASS
          ((op in ("MUL", "NEG", "FDIV") and not int16_mode) << 8) |  # DPU_EW_CFG_EW_OP_CVT_BYPASS
          (1 << 7)  |                         # DPU_EW_CFG_EW_LUT_BYPASS
          (1 << 6)  |                         # DPU_EW_CFG_EW_OP_SRC
          ((op in ("MUL", "NEG", "CAST_BOOL_HALF")) << 2))  # DPU_EW_CFG_EW_OP_TYPE: MUL (0 selects ALU)
    ),
    E(reg.DPU,  scale_reg, out_cvt_scale),
    E(reg.RDMA, reg.RDMA_S_POINTER, 0x0000000E),
    E(reg.RDMA, reg.RDMA_DATA_CUBE_WIDTH, dataout_width),
    E(reg.RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
    E(reg.RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 7),
    E(reg.RDMA, reg.RDMA_ERDMA_CFG,
        ((1 << 30) |                          # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE
          (2 << 2))                            # DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE
    ),
    E(reg.DPU,  reg.DST_BASE_ADDR, output_addr),
    E(reg.RDMA, reg.RDMA_SRC_BASE_ADDR, input_addr),
    E(reg.RDMA, reg.RDMA_EW_BASE_ADDR, weight_addr),
    E(reg.RDMA, reg.RDMA_FEATURE_MODE_CFG,
        ((precision << 15) |                  # DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION
          (15 << 11) |                         # DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN
          (precision << 5) |                   # DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION
          ((not fdiv_op and not int16_mode) << 3) |  # DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN
          1)                                    # DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE
    ),
])
```

Now replace the hardcoded hex blob in npu_regs in RockchipProgram. 
Note that `elementwise.py` is desiged to be single python file so it uses hardcoded numeric shifts
Here we use the `rk` shifts CONSTANT from autogen 

```diff
-ops_map = {Ops.ADD: 2}
+ops_map = {Ops.ADD: 2, Ops.MUL: 0}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
+  @staticmethod
+  def EMIT(target:int, reg:int, value:int) -> int: return ((target + 1) << 48) | ((value & 0xFFFFFFFF) << 16) | reg
+
@@
+    self.npu_regs:list[int] = []
+
+  def build_registers(self, op:Ops, int16_mode:bool=False, custom:str|None=None) -> None:
+    E = self.EMIT
+    zero_tag, mask_bool = custom == "rk_zero_tag", custom == "rk_mask_to_bool"
+    assert custom is None or (op is Ops.CUSTOM and custom in ("rk_zero_tag", "rk_mask_to_bool") and not int16_mode)
+    def fp16(value:float) -> int: return int.from_bytes(struct.pack("<e", value), "little")
+    pc_enable = 0x80 # E adds 1: operation-enable target 0x0081, distinct from rk.PC (PC register writes).
+    precision = 1 if int16_mode else 2
     self.npu_regs = [
-      0x1001000001e5400c,
-      0x1001480000024010,
-      0x100100070007403c,
-      0x1001000000004030,
-      0x1001108202c04070,
-      0x200100000000500c,
-      0x2001000000005010,
-      0x2001000700075014,
-      0x2001400000085034,
-      ((rk.DPU + 1) << 48) | ((self.dev.output_mem.dma_addr & 0xFFFFFFFF) << 16) | rk.REG_DPU_DST_BASE_ADDR,
-      ((rk.DPU_RDMA + 1) << 48) | ((self.dev.input_mem.dma_addr & 0xFFFFFFFF) << 16) | rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
-      ((rk.DPU_RDMA + 1) << 48) | ((self.dev.weight_mem.dma_addr & 0xFFFFFFFF) << 16) | rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
-      0x2001000178495044,
-      0x0081000000180008,
+      E(rk.DPU, rk.REG_DPU_S_POINTER, 0xE),
+      E(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
+        (15 << rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
+        (2 << rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT) |
+        (1 << rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_DATA_FORMAT,
+        ((1 if mask_bool else precision) << rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT) |
+        (precision << rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT) |
+        (precision << rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0),
+      E(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0),
+      E(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
+      E(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
+        (7 << rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT) |
+        (7 << rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_BS_CFG,
+        (2 << rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT) |
+        (1 << rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT) |
+        (1 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT) if zero_tag else
+        (1 << rk.DPU_BS_CFG_BS_BYPASS__SHIFT) | (1 << rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT) |
+        (1 << rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_BN_CFG,
+        (1 << rk.DPU_BN_CFG_BN_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT) |
+        (1 << rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_BS_ALU_CFG, 0),
+      E(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
+        (fp16(float("inf")) << rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT) if zero_tag else 0),
+      E(rk.DPU, rk.REG_DPU_BS_OW_CFG, 1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT),
+      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_0, 7),
+      E(rk.DPU, rk.REG_DPU_WDMA_SIZE_1, 0),
+      E(rk.DPU, rk.REG_DPU_BN_MUL_CFG, 0),
+      E(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0),
+      E(rk.DPU, rk.REG_DPU_EW_CFG,
+        (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT) |
+        (1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
+        (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) if zero_tag else
+        (1 << rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT) |
+        (2 << rk.DPU_EW_CFG_EDATA_SIZE__SHIFT) |
+        ((0 if mask_bool else self.ops_map[op]) << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) |
+        (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) |
+        ((not int16_mode and (mask_bool or op in (Ops.MUL, Ops.NEG, Ops.FDIV))) << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) |
+        (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
+        ((op is not Ops.NEG) << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
+        ((mask_bool or op in (Ops.MUL, Ops.NEG)) << rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, (16 if zero_tag else 0) << rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT),
+      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, (2 if int16_mode or zero_tag or mask_bool else 4) << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
+        ((not int16_mode and not mask_bool and op is not Ops.FDIV) << rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT) |
+        (1 << rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT)),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xE),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, 0),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
+        (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT) |
+        (2 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT) |
+        ((zero_tag or op is Ops.NEG) << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, self.dev.output_mem.dma_addr),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, self.dev.input_mem.dma_addr),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
+        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT) |
+        (15 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
+        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT) |
+        ((not int16_mode and op is not Ops.FDIV) << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT) |
+        (1 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT)),
+    ]
+    if zero_tag or mask_bool:
+      self.npu_regs += [
+        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BRDMA_CFG, 1),
+        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_NRDMA_CFG, 1),
+        E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, 2 << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
+        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 2 << rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT),
+      ]
+    if op is Ops.NEG:
+      self.npu_regs.append(E(rk.DPU, rk.REG_DPU_EW_OP_VALUE_0, fp16(-1.0)))
+    else:
+      self.npu_regs.append(E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, self.dev.weight_mem.dma_addr))
+    self.npu_regs.append(E(pc_enable, rk.REG_PC_OPERATION_ENABLE,
+      rk.GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK | rk.GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK))
@@
-  def add(self, a:list[float], b:list[float]) -> list[float]:
+  def alu(self, op:Ops, a:list[float], b:list[float]) -> list[float]:
     assert b is not None and len(a) == len(b)
+    self.build_registers(op)
     result:list[float] = []
@@
           if dtypes.is_float(u.dtype):
             if u.op not in self.ops_map or u.dtype != dtypes.half:
               raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
-            values[u] = self.add(*src_values)
+            values[u] = self.alu(u.op, *src_values)
           else:
             values[u] = [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)]
```

The decoded builder includes both operand routes. Binary ADD and MUL use `EW_OP_SRC=1` and read the second tensor through ERDMA. NEG uses the MUL datapath with `EW_OP_SRC=0`, reads FP16 `-1.0` from `EW_OP_VALUE_0`, and disables ERDMA. The FDIV converter settings are here too. Neither NEG nor FDIV is exposed in `ops_map` yet; this keeps the register setup together while the next step still tests only ADD and MUL.

The builder also carries the example's `CAST_BOOL_HALF` INT16 mode now, but does not expose a general INT16 ALU op to tinygrad. `elementwise.py` resets the NPU before each submit; `RockchipProgram` does not. For the INT16 task, we therefore also write the BS/BN bypass, output scale/shift, and surface registers so it initializes that path after an FP16 task. Those extra writes are a runtime adaptation, not lines copied from the example. The later CAST step only needs to pack its bool lanes and select `int16_mode=True`.

now lets run test_add, test_tiny_mul and test_mul

```bash
$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tiny_mul

*** ROCKCHI    3 E_64                                           arg  3 mem   0.00 GB tm     13.12ms/    24.23ms (      0 GFLOPS    0|0      GB/s)
Ran 1 test in 0.150s

OK

$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add

*** ROCKCHI    1 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   5097.38us/     5.10ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    2 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   2259.80us/     7.36ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    3 E_3060                                         arg  3 mem   0.00 GB tm    752.22ms/   759.58ms (      0 GFLOPS    0|0      GB/s)
*** CPU        4 E                                              arg  1 mem   0.00 GB tm     21.87us/   759.60ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    5 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   1292.36us/   760.89ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    6 copy    5.98 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   1045.32us/   761.94ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    7 E_3060                                         arg  3 mem   0.00 GB tm    568.81ms/  1330.75ms (      0 GFLOPS    0|0      GB/s)
*** CPU        8 E                                              arg  1 mem   0.00 GB tm     11.37us/  1330.76ms (      0 GFLOPS    0|0      GB/s)
Ran 1 test in 1.581s

OK

$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mul

*** ROCKCHI    1 copy    8.00 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   5286.96us/     5.29ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    2 copy    8.00 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   2315.21us/     7.60ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    3 E_4096                                         arg  3 mem   0.00 GB tm    891.25ms/   898.85ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    4 copy    8.00 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   1318.31us/   900.17ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    5 copy    8.00 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   1114.44us/   901.28ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    6 E_4096                                         arg  3 mem   0.00 GB tm    907.43ms/  1808.71ms (      0 GFLOPS    0|0      GB/s)
*** CPU        7 E                                              arg  1 mem   0.00 GB tm     23.62us/  1808.73ms (      0 GFLOPS    0|0      GB/s)
Ran 1 test in 2.141s

OK
```

CPU kernel was observed when running test_mul [(), ()], with same reason mentioend in test_add

next we do test_sub and test_neg, 
test_sub is simply add Ops.SUB: 4 to ops_map, while Ops.NEG is an unary Ops, so we set Ops.NEG as default 0 and need some fix to expect single input here


```diff
-ops_map = {Ops.ADD: 2, Ops.MUL: 0}
+ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0}
@@
-  def alu(self, op:Ops, a:list[float], b:list[float]) -> list[float]:
+  def alu(self, op:Ops, a:list[float], b:list[float]|None=None) -> list[float]:
-    assert b is not None and len(a) == len(b)
+    if op is Ops.NEG: assert b is None
+    else: assert b is not None and len(a) == len(b)
     self.build_registers(op)
@@
-      lhs, rhs = a[start:start+8], b[start:start+8]
+      lhs = a[start:start+8]
       to_mv(self.dev.input_buf, 16)[:] = struct.pack("<8e", *(lhs + [0.0] * (8 - len(lhs))))
-      to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8e", *(rhs + [0.0] * (8 - len(rhs))))
+      if b is not None:
+        rhs = b[start:start+8]
+        to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8e", *(rhs + [0.0] * (8 - len(rhs))))
```

```bash
$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest -n0 -q -s \
    test/backend/test_ops.py::TestOps::test_sub \
    test/backend/test_ops.py::TestOps::test_neg \
    test/backend/test_ops.py::TestOps::test_add \
    test/backend/test_ops.py::TestOps::test_tiny_mul \
    test/backend/test_ops.py::TestOps::test_mul

*** ROCKCHI    3 E_2925                                         arg  3 mem   0.00 GB tm    565.83ms/   573.58ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    6 E_2925                                         arg  3 mem   0.00 GB tm    567.44ms/  1143.54ms (      0 GFLOPS    0|0      GB/s)
*** CPU        7 E                                              arg  1 mem   0.00 GB tm     21.00us/  1143.56ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    9 E_2925                                         arg  2 mem   0.00 GB tm    503.06ms/  1648.11ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI   11 E_2925                                         arg  2 mem   0.00 GB tm    495.21ms/  2144.67ms (      0 GFLOPS    0|0      GB/s)
*** CPU       12 E                                              arg  1 mem   0.00 GB tm     10.50us/  2144.68ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI   15 E_3060                                         arg  3 mem   0.00 GB tm    667.54ms/  2814.86ms (      0 GFLOPS    0|0      GB/s)
*** CPU       16 E                                              arg  1 mem   0.00 GB tm      9.33us/  2814.87ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI   19 E_3060                                         arg  3 mem   0.00 GB tm    582.86ms/  3400.13ms (      0 GFLOPS    0|0      GB/s)
*** CPU       20 E                                              arg  1 mem   0.00 GB tm      8.75us/  3400.14ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI   23 E_64                                           arg  3 mem   0.00 GB tm     12.35ms/  3415.11ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI   26 E_4096                                         arg  3 mem   0.00 GB tm    796.47ms/  4214.25ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI   29 E_4096                                         arg  3 mem   0.00 GB tm    811.10ms/  5027.76ms (      0 GFLOPS    0|0      GB/s)
*** CPU       30 E                                              arg  1 mem   0.00 GB tm      9.04us/  5027.77ms (      0 GFLOPS    0|0      GB/s)
5 passed in 8.14s
```

Great it works without the `emulating in python` warnings, also some cpu kernels are expected for costand folding case.

now we have ADD/MUL/SUB/NEG working, what are the remaining ones?
The deifintion of the ~25 Ops is in GroupOp.ALU of tinygrad/uop/__init__.py, while the living definition is in ops_python.py

```python
class GroupOp:
  Unary = {Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.CDIV, Ops.MAX, Ops.CMOD, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ,
            Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV, Ops.POW, Ops.FLOORDIV, Ops.FLOORMOD}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)
  Broadcastable = set.union(Binary, Ternary)

  # TODO: is BITCAST always Elementwise if it's shape changing?
  Elementwise = set.union(ALU, {Ops.CAST, Ops.BITCAST})
```

so GroupOp.ALU counts 28 Ops, and GroupOp.Elementwise counts 30 Ops with Ops.CAST and Ops.BITCASE
and we already got NEG/ADD/MUL/SUB running,

| Group                | Working now         | Remaining                           |
|----------------------|---------------------|-------------------------------------|
| `GroupOp.Unary`      | `NEG`               | `EXP2`, `LOG2`, `RECIPROCAL`        |
|                      |                     | `SIN`, `SQRT`, `TRUNC`              |
| `GroupOp.Binary`     | `ADD`, `MUL`, `SUB` | `AND`, `CDIV`, `CMOD`               |
|                      |                     | `CMPEQ`, `CMPLT`, `CMPNE`           |
|                      |                     | `FDIV`, `FLOORDIV`, `FLOORMOD`      |
|                      |                     | `MAX`, `OR`, `POW`                  |
|                      |                     | `SHL`, `SHR`, `THREEFRY`, `XOR`     |
| `GroupOp.Ternary`    | —                   | `MULACC`, `WHERE`                    |
| `Elementwise` extras | —                   | `CAST`, `BITCAST`                    |
| **Total**            | **4 / 30**          | **26 / 30**                          |                                                                                                                                       |                                                                                                                            |
Just like what we did on Ops.SUB and Ops.NEG, we will expand the coverage to ops_map to see what all those DPU_EW_ALU_ALGO bring us

```diff
-ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0}
+ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX:0 }
```

```bash
$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div

*** ROCKCHI    1 copy    5.71 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   5282.29us/     5.28ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    2 copy    5.71 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   2330.96us/     7.61ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    3 E_2925                                         arg  3 mem   0.00 GB tm    539.16ms/   546.78ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    4 copy    5.71 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   1314.81us/   548.09ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    5 copy    5.71 KB, ROCKCHI <- NPY                arg  2 mem   0.00 GB tm   1138.94us/   549.23ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    6 E_2925                                         arg  3 mem   0.00 GB tm    508.98ms/  1058.21ms (      0 GFLOPS    0|0      GB/s)
*** CPU        7 E                                              arg  1 mem   0.00 GB tm     42.29us/  1058.25ms (      0 GFLOPS    0|0      GB/s)
1 passed in 3.95s
```

Ops.FDIV is an easy win and passed test_div, note that the NPU edge case handling is non-standard, 
e.g.`+0 / -2` returns `+0` and `-0 / -2` returns `-0`, which is opposite of IEEE division, comparison still pass but keep this in mind, we might need to handle them with pattern matcher later. 

How about RECIPROCAL? We can do `RECIP = 1 / x` with FDIV

```diff
-ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX:0 }
+ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3}
@@
-import pickle, base64, itertools, time, sys, ctypes, os, mmap, struct
+import pickle, base64, itertools, time, sys, ctypes, os, mmap, struct, math
+
@@
-  def alu(self, op:Ops, a:list[float], b:list[float]|None=None) -> list[float]:
+  def run_npu(self, op:Ops, a:list, b:list|None=None) -> list:
+    if op is Ops.RECIPROCAL:
+      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in a):
+        raise NotImplementedError("ROCKCHIP NPU RECIPROCAL does not preserve the sign of negative zero or negative infinity")
+      return self.run_npu(Ops.FDIV, [1.0] * len(a), a)
@@
-          if dtypes.is_float(u.dtype):
-            if u.op not in self.ops_map or u.dtype != dtypes.half:
-              raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
-            values[u] = self.alu(u.op, *src_values)
-          else:
-            values[u] = [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)]
+          if u.op not in self.ops_map or u.dtype != dtypes.half:
+            raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
+          values[u] = self.run_npu(u.op, *src_values)
```

Python only prepares the constant ones; FDIV runs on the NPU. No new register mode is needed. The direct RECIPROCAL check passed 14 values, including positive and negative finite inputs, +0, +inf and NaN. But FDIV gave +inf for `1 / -0` and +0 for `1 / -inf`, so this RECIPROCAL path rejects those two inputs for now. Expressions already lowered to FDIV still have the FDIV limitations above.

Rerun the division test:

```bash
$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div
*** ROCKCHI    3 E_2925                                         arg  3 mem   0.00 GB tm    541.86ms/   549.35ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    6 E_2925                                         arg  3 mem   0.00 GB tm    527.58ms/  1079.32ms (      0 GFLOPS    0|0      GB/s)
Ran 1 test in 1.322s
OK
```

Ops.MAX is different though, we didnt pass test_maximum (not using test_max here, its for reduction)

```bash
$ DEBUG=5 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum

NotImplementedError: ROCKCHIP NPU does not support Ops.OR with dtypes.bool

----------------------------------------------------------------------
Ran 1 test in 0.906s

FAILED (errors=1)
```

why test_maximum is calling Ops.OR? 
We can check each test cases inside test_maximum and find out its caused by this boolean case
```python
helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], [True, True, False]], forward_only=True)
```

lets comment out other case and inspect its UOps list with TRACE=1

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum

test_maximum (__main__.TestOps.test_maximum) ... 0 Ops.PARAM dtypes.bool ParamArg(0, dtypes.bool, 3, device='ROCKCHIP') [] []
1 Ops.PARAM dtypes.bool ParamArg(1, dtypes.bool, 3, device='ROCKCHIP') [] []
2 Ops.PARAM dtypes.bool ParamArg(2, dtypes.bool, 3, device='ROCKCHIP') [] []
3 Ops.CONST dtypes.weakint 3 [] []
4 Ops.CAST dtypes.int dtypes.int [[3]] [dtypes.weakint]
5 Ops.SPECIAL dtypes.int gidx0 [[3]] [dtypes.int]
6 Ops.INDEX dtypes.bool None [[<memory at 0x7f6500fdc0>], [0]] [dtypes.bool, dtypes.int]
7 Ops.LOAD dtypes.bool None [[(<memory at 0x7f6500fdc0>, 0)]] [dtypes.bool]
8 Ops.INDEX dtypes.bool None [[<memory at 0x7f6500ff40>], [0]] [dtypes.bool, dtypes.int]
9 Ops.LOAD dtypes.bool None [[(<memory at 0x7f6500ff40>, 0)]] [dtypes.bool]
10 Ops.INDEX dtypes.bool None [[<memory at 0x7f6500fa00>], [0]] [dtypes.bool, dtypes.int]
11 Ops.OR dtypes.bool None [[True], [True]] [dtypes.bool, dtypes.bool]
```

OR using VIZ=1
```bash
$ VIZ=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum
```
![alt text](image-1.png)

We found tinygrad rewrite Ops.MAX(dtypes.bool) into Ops.OR(dtypes.bool) for bool input, which we were only allowing dtypes.float16 before.
The solution is Pattern Matcher, its rewrite the UOps tree according to a predefined rule.
We can add one to rewrite Ops.OR(dtypes.bool) into Ops.MAX(dtypes.half) in `RockchipRenderer.extra_matcher` and `RockchipProgram` would recieved the rewritten Uops tree

TODO: standardize use dtypes.half or dtypes.float16


```diff
-from tinygrad.uop.ops import python_alu, Ops, UOp, GroupOp
+from tinygrad.uop.ops import python_alu, Ops, UOp, GroupOp, PatternMatcher, UPat

 class RockchipRenderer(Renderer):
   code_for_op = {op: python_alu.get(op, lambda: None) for op in ops_map}
+  extra_matcher = PatternMatcher([
+    (UPat(Ops.OR, dtypes.bool, name="u"),
+     lambda u: u.src[0].cast(dtypes.half).maximum(u.src[1].cast(dtypes.half)).cast(dtypes.bool)),
+  ])
```

As we are using cast, we need another NPU gate around `elif u.op is Ops.CAST`

```diff
 elif u.op is Ops.CAST:
+  if (src_dtypes[0], u.dtype) == (dtypes.bool, dtypes.half):
+    raise NotImplementedError(f"ROCKCHIP NPU CAST from {src_dtypes[0]} to {u.dtype} is not implemented")
   values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
```

```bash
$ TRACE=1 DEV=ROCKCHIP NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python test/backend/test_ops.py TestOps.test_maximum

test_maximum (__main__.TestOps.test_maximum) ... 0 Ops.PARAM dtypes.bool ParamArg(0, dtypes.bool, 3, device='ROCKCHIP') [] []
1 Ops.PARAM dtypes.bool ParamArg(1, dtypes.bool, 3, device='ROCKCHIP') [] []
2 Ops.PARAM dtypes.bool ParamArg(2, dtypes.bool, 3, device='ROCKCHIP') [] []
3 Ops.CONST dtypes.weakint 3 [] []
4 Ops.CAST dtypes.int dtypes.int [[3]] [dtypes.weakint]
5 Ops.SPECIAL dtypes.int gidx0 [[3]] [dtypes.int]
6 Ops.INDEX dtypes.bool None [[<memory at 0x7f5e7bfb80>], [0]] [dtypes.bool, dtypes.int]
7 Ops.LOAD dtypes.bool None [[(<memory at 0x7f5e7bfb80>, 0)]] [dtypes.bool]
8 Ops.INDEX dtypes.bool None [[<memory at 0x7f5e7bfd00>], [0]] [dtypes.bool, dtypes.int]
9 Ops.LOAD dtypes.bool None [[(<memory at 0x7f5e7bfd00>, 0)]] [dtypes.bool]
10 Ops.INDEX dtypes.bool None [[<memory at 0x7f5e7bf7c0>], [0]] [dtypes.bool, dtypes.int]
11 Ops.CAST dtypes.half dtypes.half [[True]] [dtypes.bool]
ERROR

NotImplementedError: ROCKCHIP NPU bool-to-half CAST is not implemented
```

As expected, NPU Ops.CAST not implemeted error raised
Lets implement the bool-to-half CAST on NPU with bool_mask * 0x3c00 (1.0 in fp16)

```python
def cast_bool_half(self, values:list[bool]) -> list[float]:
  self.build_registers(Ops.MUL, int16_mode=True)
  # Note: result = bool_mask * 0x3c00 (1.0 in fp16)
  weights = struct.pack("<H", 0x3c00) * 8
  result:list[float] = []
  for start in range(0, len(values), 8):
    lanes = values[start:start+8]
    to_mv(self.dev.input_buf, 16)[:] = struct.pack("<8h", *(lanes + [False] * (8-len(lanes))))
    to_mv(self.dev.weight_buf, 16)[:] = weights
    self.submit()
    result.extend(struct.unpack("<8e", to_mv(self.dev.output_buf, 16))[:len(lanes)])
  return result
```

```diff
 elif u.op is Ops.CAST:
   if (src_dtypes[0], u.dtype) == (dtypes.bool, dtypes.half):
-    raise NotImplementedError(f"ROCKCHIP NPU CAST from {src_dtypes[0]} to {u.dtype} is not implemented")
+    values[u] = self.cast_bool_half(src_values[0])
-  values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
+  else: values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
```


```diff
-  def cast_bool_half(self, values:list[bool]) -> list[float]:
-    self.build_registers(Ops.MUL, int16_mode=True)
-    # Note: result = bool_mask * 0x3c00 (1.0 in fp16)
-    weights = struct.pack("<H", 0x3c00) * 8
-    result:list[float] = []
-    for start in range(0, len(values), 8):
-      lanes = values[start:start+8]
-      to_mv(self.dev.input_buf, 16)[:] = struct.pack("<8h", *(lanes + [False] * (8-len(lanes))))
-      to_mv(self.dev.weight_buf, 16)[:] = weights
-      self.submit()
-      result.extend(struct.unpack("<8e", to_mv(self.dev.output_buf, 16))[:len(lanes)])
-    return result
@@
   def run_npu(self, op:Ops, a:list, b:list|None=None) -> list:
@@
-    if op is Ops.NEG: assert b is None
-    else: assert b is not None and len(a) == len(b)
-    self.build_registers(op)
-    result:list[float] = []
+    assert (b is None and op in (Ops.NEG, Ops.CAST)) or (b is not None and len(a) == len(b))
+    if op is Ops.CAST:
+      self.build_registers(Ops.MUL, int16_mode=True)
+      to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<H", 0x3c00) * 8
+    else: self.build_registers(op)
+    result:list = []
@@
-      lhs = a[start:start+8]
-      to_mv(self.dev.input_buf, 16)[:] = struct.pack("<8e", *(lhs + [0.0] * (8 - len(lhs))))
+      lanes = a[start:start+8]
+      packed = struct.pack("<8h" if op is Ops.CAST else "<8e", *(lanes + [0] * (8-len(lanes))))
+      to_mv(self.dev.input_buf, 16)[:] = packed
@@
-      result.extend(struct.unpack("<8e", to_mv(self.dev.output_buf, 16))[:len(lhs)])
+      result.extend(struct.unpack("<8e", to_mv(self.dev.output_buf, 16))[:len(lanes)])
@@
-    values[u] = self.cast_bool_half(src_values[0])
+    values[u] = self.run_npu(Ops.CAST, src_values[0])
```

```bash
$ TRACE=1 DEV=ROCKCHIP NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python test/backend/test_ops.py TestOps.test_maximum

11 Ops.CAST dtypes.half dtypes.half [[True]] [dtypes.bool]
12 Ops.CAST dtypes.half dtypes.half [[True]] [dtypes.bool]
15 Ops.MAX dtypes.half None [[1.0], [1.0]] [dtypes.half, dtypes.half]
16 Ops.CMPNE dtypes.bool None [[1.0], [0.0]] [dtypes.half, dtypes.half]

NotImplementedError: ROCKCHIP NPU does not support Ops.CMPNE with dtypes.bool
Ran 1 test in 0.223s
FAILED (errors=1)
```

Great we have Ops.CAST working on NPU already, but we have Ops.CMPNE NotImplementedError
It is just the last step in our pattern matcher cast fp16 back to bool usig x != 0.0,

CMPEQ and CMPNE is very interesting here, as the NPU does not expose COMPARE/EQUAL/IF implementation, all we got are just MUL and those in EW_ALU_ALGO and RELU if u already found it in the registers name.
You cannot find any CMPEQ/CMPNQ alternative in RKNN compiled capture either, it just offload it to CPU.

I have stucked for a week or two, and comes up with this idea by myself during a shower, GPT back then cant even port my NPU C code to Python. 
```
CMPEQ_FP16(A, B) = ReLU((exponent_shift_minus_16((B-A)*inf)-1)*1024)
```

Looks diffcult but not really that hard if u put it into a spreadsheet.

| Stage             | Operation / Constant |      delta |      delta |          delta |      delta |      delta |
|-------------------|----------------------|-----------:|-----------:|---------------:|-----------:|-----------:|
| task1 EW SUB      | `B - A`              |         -4 |         -2 |              0 |          2 |          4 |
| task2 BS MUL      | `* inf`              |     `-inf` |     `-inf` | `NaN (0x7c01)` |      `inf` |      `inf` |
|   EXPON SHF MINUS | `exp shift 16`       |         -1 |         -1 |   1.0009765625 |          1 |          1 |
| task3 BS SUB      | `- 1`                |         -2 |         -2 |   0.0009765625 |          0 |          0 |
|       BS MUL      | `* 1024`             |      -2048 |      -2048 |              1 |          0 |          0 |
|       BS ReLU     | `max(x, 0)`          |          0 |          0 |              1 |          0 |          0 |

1. First we find the delta between A and B, B-A or A-B doesnt matter, and we need delta == 0, but how can we do CMPEQ(delta, 0) while we are currently implementing CMPEQ itself? Its totally possible with the following bits trick.

2. Next, we dont need another CMPEQ, my shower thought was to trigger out fp16 edge cases could help us here. We want to make value with 0 different to other value, and other value got the same value, such that {"0": "A", "others":"B"} just like an IF. The floating edge cases mostly play with 0 and INF and overflow. So, 
```
anything * INF = INF, INF * INF = INF, 0 * INF = NaN. 
```
Viola! Thats exactly what we needed. In the table, we got -inf for negative delta and NaN for 0 and inf for positive inf, but we still need to normalize -inf and inf into same value later.

3. Next, we need to turn the inf and NaN back to numbers. I found an interesting register named DPU_OUT_CVT_SHIFT_MINUS_EXP which minus exponent value from a fp16 value. So, with a floating point playground(https://evanw.github.io/float-toy/) we can see fp16
```
-inf = 0xFC00 = -1 * 2^16 * 1
NaN  = 0x7C01 (observed on this NPU; fraction bits = 1/1024)
inf  = 0x7C00 =  1 * 2^16 * 1

set f = exponent_shift_minus_16
f(-inf) = -1 * 2^(16-16) * 1   = -1 * 2^(0) * 1   = -1
f(NaN)  =  1 * 2^(16-16) * (1 + 1/1024) = 1.0009765625 (0x3C01)
f(inf)  =  1 * 2^(16-16) * 1   =  1 * 2^(0) * 1   =  1
```

4. Next, subtract 1 so -1 and 1 become -2 and 0. Simple RELU can bring both to 0. So its explained the SUB and RELU. But what about MUL 1024? 
If we apply RELU(X-1) to last step result, we got
```
Step3 result -1, -1, 1.0009765625, 1, 1
SUB 1        -2, -2, 0.0009765625, 0, 0
RELU          0,  0, 0.0009765625, 0, 0
```
A simple MUL 1024 gets the CMPEQ result we need: 0.0009765625 * 1024 is exactly 1.0 (fp16 0x3c00), verified on the NPU with ReLUX disabled. Ordinary RELU removes the negative values, so no upper clamp is needed.

With MUL 1024 and ordinary RELU, 4112 mixed comparison lanes gave exact fp16 0/1, including subnormals, signed zero and partial atoms. The seven tests `test_maximum`, `test_add`, `test_sub`, `test_neg`, `test_mul`, `test_tiny_mul` and `test_div` gave `7 passed in 9.51s`.

And thats our trick, now looking at the formula again and its not that diffcult to understand. 
And u might ask, what are those BS MUL in the table? Why not use EW MUL we have been using?

> what are those BS MUL in the table?
If u remember we have mentioned the NPU programming model is mulitple NPU submits by host -> mulitple tasks in one submit -> fused mulitple ops in one task, e.g. CONV/MAC+ELEMENTWISE+BN+BS+RELU+POOL in one task. Its because NVDLA was released in 2017, and it was designed for pipeline
```
CONV(MAC) → BS(Norm/ReLUX) → BN(Another Norm/ReLUX) → EW(ALU_ALGO/ReLUX) → POOL 
```
matching those in popular Convulution model in 2017 like YOLO. On BS/BN, the input floating point precision requirement is different, and TRM also mentioned that they can perform RELUX as well.

```
BS: ReLU-X((x + FP32 bias) × FP16 scale)
BN: ReLU-X((x × FP16 scale) + FP32 bias)
```

ReLU-X is optional in each stage; without it, the formulas are just the affine operations inside the parentheses.

> Why not use EW MUL we have been using?
Because we want to fuse multiple Ops into one task instead of issusing many task per Ops, like GPU kerenl fusion. But the pipeline runs in squential order, we cant put everything into one task. I have made CMPEQ into 3 task, task1 EW_SUB, task2 BS_MUL+EXPON_SHF_MINUS, task3 BS_SUB+BS_MUL+BS_RELU. Other combination might still works and might be faster as well.

Here we used BS RELU, enabled by setting BS_RELU_BYPASS to 0. BS_RELUX_EN stays 0, so no upper-bound register is needed.

Like this,

```python
E(rk.DPU, rk.REG_DPU_BS_CFG,
  (0 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT) |
  (0 << rk.DPU_BS_CFG_BS_RELUX_EN__SHIFT) |
  ...)
```

for EW RELU, we just set EW_RELU_BYPASS as 0
```python
E(rk.DPU, rk.REG_DPU_EW_CFG,
  (0 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) |
  ...
  ) 
```

and for EW RELUX we need EW_RELU_BYPASS as 0, EW_RELUX_EN as 1 and set fp32 X in EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT

```python
E(rk.DPU, rk.REG_DPU_EW_CFG,
  (0 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT) |
  (1 << rk.DPU_EW_CFG_EW_RELUX_EN__SHIFT) |
  ...)
E(rk.DPU, rk.REG_DPU_EW_RELUX_CMP_VALUE,
  fbits(1.0, "<f") << rk.DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT__SHIFT)
```

One more thing before implementing Ops.CMPEQ, we cannot just put registers belongs to mutiple task into one NPU submit, we need to advance the Program Counter PC so NPU know what next.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def pc_tail(self, next_addr:int|None, next_amount:int=0) -> list[int]:
+    E = self.EMIT # EMIT adds 1 to the target before packing it.
+    return [
+      E(rk.PC, rk.REG_PC_BASE_ADDRESS,
+        0 if next_addr is None else next_addr & rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK),
+      E(rk.PC, rk.REG_PC_REGISTER_AMOUNTS, 0 if next_addr is None else next_amount),
+      E(0x80, rk.REG_PC_OPERATION_ENABLE,
+        rk.GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK | rk.GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK),
+    ]

   def build_registers(self, op:Ops, int16_mode:bool=False, custom:str|None=None) -> None:
@@
-    pc_enable = 0x80 # E adds 1: operation-enable target 0x0081, distinct from rk.PC (PC register writes).
@@
-    self.npu_regs.append(E(pc_enable, rk.REG_PC_OPERATION_ENABLE,
-      rk.GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK | rk.GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK))
@@
   def submit(self) -> None:
-    regs = self.npu_regs
+    guard_offset = (len(self.npu_regs) + 3 + 1) // 2 * 16
+    # The final PC fetch points to a mapped zero-filled page, not address 0.
+    assert guard_offset + mmap.PAGESIZE <= self.dev.regcmd_mem.size
+    regs = self.npu_regs + self.pc_tail(self.dev.regcmd_mem.dma_addr + guard_offset, 0)
     # copy the registers to the C array regcmd
     ctypes.memset(self.dev.regcmd_buf, 0, self.dev.regcmd_mem.size)
```

From the RKNN captured register pattern in weight/regcmd GEM, we observed a PC tail with REG_PC_BASE_ADDRESS, REG_PC_REGISTER_AMOUNTS and REG_PC_OPERATION_ENABLE exists after the task registers. We put next task dma address into REG_PC_BASE_ADDRESS and the task register count (excluding its PC tail) into REG_PC_REGISTER_AMOUNTS.

And the last task shd sets `PC_REGISTER_AMOUNTS=0` and shd be pointed to a mapped zero-filled page like `pc_tail(guard_addr, 0)`
so lets implement Ops.CMPEQ in ops_rockchip.py and do Ops.CMPNE with `1 - CMPEQ(A, B)`

```diff
+CMP = 9  # Internal multi-stage comparison marker, not an EW algorithm.
-ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3}
+ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.CMPEQ: CMP, Ops.CMPNE: CMP}
@@
     assert custom is None or (op is Ops.CUSTOM and custom in ("rk_zero_tag", "rk_mask_to_bool") and not int16_mode)
+    assert op is Ops.CUSTOM or self.ops_map[op] != CMP, "comparisons must be lowered by the renderer"
```



The ref in `~/rk3588/experimental/ops_rockchip.py` uses named CUSTOM stages. We can reuse SUB, MUL and MAX for the arithmetic, and keep CUSTOM only for the hardware-specific parts.

```text
a, b → SUB → rk_zero_tag → SUB 1 → MUL 1024 → MAX 0 → rk_mask_to_bool
                                                    ↳ SUB from 1 for CMPNE, then rk_mask_to_bool
```

`rk_zero_tag` does MUL inf and exponent shift 16. This is not ordinary IEEE arithmetic, so a normal MUL cannot describe the whole operation. It also takes the original operands to reject NaN/inf inputs without rejecting a valid subtraction that overflows.

`rk_mask_to_bool` only accepts an FP16 0/1 mask. EW MUL by 1 with INT16 output and `FP32TOFP16_EN=0` writes integer 0/1 on the NPU. We unpack those output words; Python does not calculate the comparison. This avoids casting back to bool through CMPNE again.

The map still advertises CMPEQ/CMPNE, but the renderer must lower them before execution.

Use names for what each CUSTOM does, not just stage 1/2/3. In this tinygrad version, `arg=(name, dtype)` supplies the CUSTOM result dtype.

```diff
@@
 class RockchipRenderer(Renderer):
+  @staticmethod
+  def lower_compare(u:UOp) -> UOp:
+    a, b = (x.cast(dtypes.half) for x in u.src)
+    delta = a.alu(Ops.SUB, b)
+    # MUL inf + exponent-field shift is hardware-specific, not IEEE arithmetic.
+    tag = UOp(Ops.CUSTOM, src=(delta, a, b), arg=("rk_zero_tag", dtypes.half))
+    mask = tag.alu(Ops.SUB, tag.const_like(1)).alu(Ops.MUL, tag.const_like(1024)).maximum(tag.const_like(0))
+    if u.op is Ops.CMPNE: mask = mask.const_like(1).alu(Ops.SUB, mask)
+    # This converts an already-normalized mask; it is not a general x != 0.
+    return UOp(Ops.CUSTOM, src=(mask,), arg=("rk_mask_to_bool", dtypes.bool))
+
   code_for_op = {op: python_alu.get(op, lambda: None) for op in ops_map}
   extra_matcher = PatternMatcher([
     (UPat(Ops.OR, dtypes.bool, name="u"),
      lambda u: u.src[0].cast(dtypes.half).maximum(u.src[1].cast(dtypes.half)).cast(dtypes.bool)),
+    (UPat.var("a", dtypes.half).ne(UPat.var("b", dtypes.half)).ne(UPat.const(True, dtypes.bool)),
+     lambda a,b: a.alu(Ops.CMPEQ, b)),
+    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=(dtypes.half, dtypes.weakfloat)),
+                                     UPat(dtype=(dtypes.half, dtypes.weakfloat))), name="u"),
+     lambda u: RockchipRenderer.lower_compare(u)),
+    (UPat(Ops.CMPNE, src=(UPat.var("x", dtypes.bool), UPat.const(True, dtypes.bool))),
+     lambda x: UOp(Ops.CUSTOM, src=(UOp.const(1, dtypes.half).alu(Ops.SUB, x.cast(dtypes.half)),),
+                  arg=("rk_mask_to_bool", dtypes.bool))),
   ])
```

The zero constant can still have `dtypes.weakfloat` when the matcher runs, so we accept it and cast it to half. The boolean inversion rule handles `CMPNE(bool, True)` if the inner comparison was already lowered.

Each UOp now runs one task. We no longer build comparison stages or their intermediate DMA offsets inside `run_npu`.

```diff
@@
+  def run_npu(self, op:Ops, a:list, b:list|None=None, custom:str|None=None) -> list:
-  def run_npu(self, op:Ops, a:list, b:list|None=None) -> list:
     if op is Ops.RECIPROCAL:
       if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in a):
         raise NotImplementedError("ROCKCHIP NPU RECIPROCAL does not preserve the sign of negative zero or negative infinity")
       return self.run_npu(Ops.FDIV, [1.0] * len(a), a)
+    assert (b is None and op in (Ops.NEG, Ops.CAST, Ops.CUSTOM)) or (b is not None and len(a) == len(b))
-    assert (b is None and op in (Ops.NEG, Ops.CAST)) or (b is not None and len(a) == len(b))
     if op is Ops.CAST:
       self.build_registers(Ops.MUL, int16_mode=True)
       to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<H", 0x3c00) * 8
+    else:
+      self.build_registers(op, custom=custom)
+      if custom == "rk_mask_to_bool":
+        if any(x not in (0.0, 1.0) for x in a): raise NotImplementedError("rk_mask_to_bool requires an FP16 0/1 mask")
+        to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8e", *([1.0] * 8))
-    else: self.build_registers(op)
     result:list = []
     for start in range(0, len(a), 8):
@@
         to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8e", *(rhs + [0.0] * (8-len(rhs))))
       self.submit()
+      result.extend(struct.unpack("<8H" if custom == "rk_mask_to_bool" else "<8e", to_mv(self.dev.output_buf, 16))[:len(lanes)])
-      result.extend(struct.unpack("<8e", to_mv(self.dev.output_buf, 16))[:len(lanes)])
     return result
@@
+        elif u.op is Ops.CUSTOM:
+          if u.arg == ("rk_zero_tag", dtypes.half) and u.dtype == dtypes.half and src_dtypes == [dtypes.half] * 3:
+            # The original operands retain the finite-input check even if SUB overflows.
+            if any(not math.isfinite(x) for xs in src_values[1:] for x in xs):
+              raise NotImplementedError("ROCKCHIP NPU FP16 comparisons do not support NaN or infinity")
+          elif not (u.arg == ("rk_mask_to_bool", dtypes.bool) and u.dtype == dtypes.bool and src_dtypes == [dtypes.half]):
+            raise NotImplementedError(f"ROCKCHIP NPU does not support CUSTOM {u.arg}")
+          values[u] = self.run_npu(Ops.CUSTOM, src_values[0], custom=u.arg[0])
         elif u.op in GroupOp.ALU:
@@
-          if u.op not in self.ops_map or u.dtype != dtypes.half:
+          if u.op not in self.ops_map or self.ops_map[u.op] == CMP or u.dtype != dtypes.half:
             raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
           values[u] = self.run_npu(u.op, *src_values)
```

The PC tail stays in `submit`. These are separate blocking submissions, not a PC chain.

The arithmetic is now visible in TRACE/VIZ. The tradeoff is six tasks for CMPEQ and seven for CMPNE instead of the fused three/four; intermediate values also go through the existing Python interpreter's packing/readback. The arithmetic and mask conversion still run on the NPU.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum

20 Ops.SUB dtypes.half
21 Ops.CUSTOM dtypes.half ('rk_zero_tag', dtypes.half)
22 Ops.SUB dtypes.half
23 Ops.MUL dtypes.half
24 Ops.MAX dtypes.half
25 Ops.SUB dtypes.half
26 Ops.CUSTOM dtypes.bool ('rk_mask_to_bool', dtypes.bool)

Ran 1 test in 0.132s
OK
```

Great test_maximum passed. But having rk_zero_tag and rk_mask_to_bool is not very readable in the UOps tree.
We will try to replace rk_zero_tag with
```
MUL inf
↓
BITCAST half → int16/uint16
↓
integer manipulation of exponent bits
↓
BITCAST int16/uint16 → half
```

For that, we need a Ops.BITCAST implementation.

==== keep everything above uncahnge
==== only fix below

BITCAST keeps the bytes and changes their interpretation. For example, `0x3c00` is fp16 `1.0` or int16 `15360`. No NPU arithmetic is needed.

we keep a storage reference and its dtype. BITCAST creates a new typed reference to the same bytes, without unpacking, repacking or submitting an NPU task. This also preserves the NaN payload `0x7c01`.

```diff
-from dataclasses import replace
+from dataclasses import dataclass, replace
@@
+# BITCAST changes only the dtype of a storage reference, like the 1500 branch's carrier.
+@dataclass(frozen=True)
+class RockchipValue:
+  raw:bytes
+  dtype:DType
+
+  def bitcast(self, dtype:DType) -> 'RockchipValue': return RockchipValue(self.raw, dtype)
+
+def scalar16(value):
+  return struct.unpack("<e" if value.dtype == dtypes.half else "<h", value.raw)[0] if isinstance(value, RockchipValue) else value
+
+def raw16(value, dtype:DType) -> bytes:
+  if isinstance(value, RockchipValue): return value.raw
+  return struct.pack("<e" if dtype == dtypes.half else "<h", value)
+
 def _load(m, i, dtype: DType):
   if i is None: return 0.0
   if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
+  if m.itemsize == 2 and dtype in (dtypes.half, dtypes.int16):
+    return RockchipValue(bytes(m.cast("B")[i*2:i*2+2]), dtype)
@@
 def _store(m, i, v, dtype: DType):
   if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
+  if m.itemsize == 2 and dtype in (dtypes.half, dtypes.int16):
+    m.cast("B")[i*2:i*2+2] = raw16(v, dtype)
+    return
```

Now half ↔ int16 BITCAST shares those same bytes instead of packing the Python float again:

```diff
-        elif u.op is Ops.BITCAST: values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
+        elif u.op is Ops.BITCAST:
+          if {src_dtypes[0], u.dtype} == {dtypes.half, dtypes.int16}:
+            values[u] = [(x if isinstance(x, RockchipValue) else RockchipValue(raw16(x, src_dtypes[0]), src_dtypes[0])).bitcast(u.dtype)
+                         for x in src_values[0]]
+          else: values[u] = [bitcast(scalar16(x), src_dtypes[0], u.dtype) for x in src_values[0]]
```

The NPU packing and readback must preserve the bytes too. Copy the output before the next submit reuses its buffer:

```diff
     for start in range(0, len(a), 8):
       lanes = a[start:start+8]
-      packed = struct.pack("<8h" if op is Ops.CAST else "<8e", *(lanes + [0] * (8-len(lanes))))
+      packed = b"".join(raw16(x, dtypes.int16 if op is Ops.CAST else dtypes.half) for x in lanes) + bytes(2*(8-len(lanes)))
       to_mv(self.dev.input_buf, 16)[:] = packed
       if b is not None:
         rhs = b[start:start+8]
-        to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8e", *(rhs + [0.0] * (8-len(rhs))))
+        to_mv(self.dev.weight_buf, 16)[:] = b"".join(raw16(x, dtypes.half) for x in rhs) + bytes(2*(8-len(rhs)))
       self.submit()
-      result.extend(struct.unpack("<8H" if custom == "rk_mask_to_bool" else "<8e", to_mv(self.dev.output_buf, 16))[:len(lanes)])
+      # Snapshot the output before the next submission reuses the DMA buffer.
+      out = bytes(to_mv(self.dev.output_buf, 16))
+      if custom == "rk_mask_to_bool": result.extend(struct.unpack("<8H", out)[:len(lanes)])
+      else: result.extend(RockchipValue(out[i:i+2], dtypes.half) for i in range(0, len(lanes)*2, 2))
```

The existing checks and Python CAST fallback still need numeric values, so decode only there, not in BITCAST:

```diff
@@
-      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in a):
+      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in map(scalar16, a)):
@@
-        if any(x not in (0.0, 1.0) for x in a): raise NotImplementedError("rk_mask_to_bool requires an FP16 0/1 mask")
+        if any(scalar16(x) not in (0.0, 1.0) for x in a): raise NotImplementedError("rk_mask_to_bool requires an FP16 0/1 mask")
@@
-          else: values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
+          else: values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(scalar16(x))) for x in src_values[0]]
@@
-            if any(not math.isfinite(x) for xs in src_values[1:] for x in xs):
+            if any(not math.isfinite(scalar16(x)) for xs in src_values[1:] for x in xs):
```

BITCAST itself aliases storage; the existing LOAD/STORE and NPU packing/readback still copy bytes. This is not yet the 1500 branch's device-buffer pipeline. Python constants are packed once when making their storage reference.

Both directions and round trips preserved all 65536 bit patterns. The typed references shared the same byte object with pack/unpack disabled during BITCAST.

The Tensor round trip also preserved all 65536 patterns with NPU submit disabled. EW MUL `0 * inf` followed by BITCAST preserved `0x7c01`. The same seven regression tests passed in 8.78s.

This step adds raw-bit-preserving half ↔ int16 BITCAST only. Other BITCAST pairs still use the inherited Python implementation. INT16 SUB is not supported yet, so we have not replaced `rk_zero_tag` here.

The seven tests `test_maximum`, `test_add`, `test_sub`, `test_neg`, `test_mul`, `test_tiny_mul` and `test_div` gave `7 passed in 8.83s`. Another 390 Tensor equality/inequality lanes, all four boolean maximum pairs and NaN/inf rejection passed.

The generated comparison graph also passed 4112 lanes across full and partial atoms. `rk_mask_to_bool` rejects values other than 0/1.

Progress so far

| Group                | Working now                           | Remaining                                      |
|----------------------|---------------------------------------|------------------------------------------------|
| `GroupOp.Unary`      | `NEG`, `RECIPROCAL`                   | `EXP2`, `LOG2`                                  |
|                      |                                       | `SIN`, `SQRT`, `TRUNC`                         |
| `GroupOp.Binary`     | `ADD`, `MUL`, `SUB`                   | `AND`, `CDIV`, `CMOD`                          |
|                      | `FDIV`, `MAX`                         | `CMPLT`, `FLOORDIV`, `FLOORMOD`                |
|                      | `CMPEQ`, `CMPNE`                      | `POW`, `SHL`, `SHR`                            |
|                      | `OR` (bool)                           | `THREEFRY`, `XOR`                              |
| `GroupOp.Ternary`    | —                                     | `MULACC`, `WHERE`                              |
| `Elementwise` extras | `CAST` (bool → FP16)                  | —                                              |
|                      | `BITCAST` (FP16 ↔ INT16)              |                                                |
| **Total**            | **12 / 30**                           | **18 / 30**                                    |

This counts the paths implemented so far, not every dtype or test case. Arithmetic is FP16; CMPEQ/CMPNE accept finite FP16 inputs only. Bool OR uses the CAST → MAX → CMPNE matcher. General CAST back to bool lowers to CMPNE(x, 0); the matcher ends its normalized mask with rk_mask_to_bool.
