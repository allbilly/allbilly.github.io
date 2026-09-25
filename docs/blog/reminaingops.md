# Remaining ops

## Ops.AND

Next we will do Ops.AND. And lets try to run it directly

TOREVIEW1: With the later integer AND dispatch and bool AND matcher disabled in memory, the existing test stops here:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.int
Ran 1 test in 0.316s
FAILED (errors=1)
```

This is a disabled-feature probe of the current runtime, not a full replay of every earlier diff. No runtime or test files were changed.

| a     | b     | FP16 a * b | AND   |
| ----- | ----- | ---------: | ----- |
| False | False |          0 | False |
| False | True  |          0 | False |
| True  | False |          0 | False |
| True  | True  |          1 | True  |

For bool inputs, we can just cast both input into FP16 and MUL them.

```text
a, b → CAST(half) → MUL → CAST(bool)
```

Lets add a pattern matcher and see if this approach works out

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Bool AND multiplies 0/1 masks; keep the final CAST after general rewrites.
+    (UPat(Ops.AND, dtypes.bool, name="u"),
+     lambda u: u.src[0].cast(dtypes.half).alu(Ops.MUL, u.src[1].cast(dtypes.half)).cast(dtypes.bool)),
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.int

Ran 1 test in 0.058s

FAILED (errors=1)
```

Lets us comment each test case to see which caused the error.

```text
ten & ten: PASS
ten & 0x1337: ROCKCHIP NPU does not support Ops.AND with dtypes.int
```
INT16 can hold every input in this case, including `0x1337 = 4919`, but CAST does not implement the missing bitwise AND. We would still need an INT16 AND algorithm. FP16 is lossy here: 4919 rounds to 4920, changing its low bits (`1 & 4919 = 1`, but `1 & 4920 = 0`). MUL only implements AND for 0/1 bool inputs, not these integers.

TOREVIEW1: Lets try the INT16 CAST before rejecting it:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Try narrower integer operands; this changes their dtype, not the AND operation.
+    (UPat(Ops.AND, dtypes.int32, name="u"),
+     lambda u: u.src[0].cast(dtypes.int16).alu(Ops.AND, u.src[1].cast(dtypes.int16)).cast(dtypes.int32)),
```

With the later integer AND helper disabled in memory, the existing test gives:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.short
Ran 1 test in 0.166s
FAILED (errors=1)
```

Now try FP16 instead:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
@@
-     lambda u: u.src[0].cast(dtypes.int16).alu(Ops.AND, u.src[1].cast(dtypes.int16)).cast(dtypes.int32)),
+     lambda u: u.src[0].cast(dtypes.half).alu(Ops.AND, u.src[1].cast(dtypes.half)).cast(dtypes.int32)),
```

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.half
Ran 1 test in 0.175s
FAILED (errors=1)
```

These ran `TestOps.test_and` with NOOPT=1, FORWARD_ONLY=1 and DEFAULT_FLOAT=HALF, using the current runtime's CAST support. They test the proposed matcher, not a full reconstruction of this tutorial stage. Neither supplies a bitwise AND implementation.

Remove the unsuccessful rule before continuing:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
-    # Try narrower integer operands; this changes their dtype, not the AND operation.
-    (UPat(Ops.AND, dtypes.int32, name="u"),
-     lambda u: u.src[0].cast(dtypes.half).alu(Ops.AND, u.src[1].cast(dtypes.half)).cast(dtypes.int32)),
```

TOREVIEW1: We are back at the original INT32 AND gate. The new errors above follow explicit matcher diffs; no gate was released. An assertion from the later runtime's input packing is not a result of these tutorial steps.

```python
data = [[1,-8,1],[32,1,6]]
tor = torch.tensor(data, dtype=torch.int)
ten = Tensor(data, dtype=dtypes.int32)
helper_test_op([], lambda: tor&tor, lambda: ten&ten, forward_only=True)
helper_test_op([], lambda: tor&0x1337, lambda: ten&0x1337, forward_only=True)
```

MUL is not bitwise AND for whole integers: `2 * 3 = 6`, but `2 & 3 = 2`. We need another decomposition.

How small should each piece be? One byte has 256 values, so a pair would need 65,536 table entries. A two-bit digit has only four values, giving 16 pairs. Lets split each byte into four base-4 digits and reuse SHR's exact floor conversion to extract them. No 32-bit-plane array is needed.

For a signed byte s:

```text
q0 = s
q1 = floor(s/4)
q2 = floor(s/16)
q3 = floor(s/64)

d0 = q0 - 4*q1
d1 = q1 - 4*q2
d2 = q2 - 4*q3
d3 = q3

s = d0 + 4*d1 + 16*d2 + 64*d3
```

The low three digits are 0..3. The top digit is -2..1 because we read the byte as INT8. For example, `0xCD` is -51:

| Stage | Formula        | q0 / d0 | q1 / d1 | q2 / d2 | q3 / d3 |
| ----- | -------------- | ------: | ------: | ------: | ------: |
| Input | quotients      |     -51 |     -13 |      -4 |      -1 |
| Split | base-4 digits  |       1 |       3 |       0 |      -1 |
| Join  | weighted digit |       1 |      12 |       0 |     -64 |

`1 + 12 + 0 - 64 = -51`, so the same byte is preserved. The top digit -1 represents the two-bit pattern `11`.

Each digit pair only needs this small AND table:

| a   |  b=0 |  b=1 |  b=2 |  b=3 |
| --- | ---: | ---: | ---: | ---: |
| 0   |    0 |    0 |    0 |    0 |
| 1   |    0 |    1 |    0 |    1 |
| 2   |    0 |    0 |    2 |    2 |
| 3   |    0 |    1 |    2 |    3 |

But how can convolution look up a table? Turn the pair into `index = 4*a + b`, then generate integer step values with ReLU-X capped at 1:

```text
step[t] = RELUX1(index - t + 1)

result = table[0]
       + (table[1] - table[0])*step[1]
       + (table[2] - table[1])*step[2]
       + ...
       + (table[15] - table[14])*step[15]
```

For `a=2, b=3`, index is 11. Steps 1..11 are 1 and steps 12..15 are 0. The differences cancel each other, leaving `table[11] = 2`. Both the steps and the weighted sum run on the NPU; Python only prepares the constant table weights.

For the signed top digits, use `index = 4*(a+2) + (b+2) = 4*a+b+10`. Reorder the constant table for input order -2, -1, 0, 1, and store output digits 2/3 as -2/-1. This keeps the reconstructed byte within INT8 instead of saturating values above 127.

The tasks are:

1. Upload the two words as eight original bytes. Three CONV/CVT tasks compute q1, q2 and q3. For digit k, weight 2, offset `-(4**k-1)` and shift `2*k+1` give `floor(s/4**k)`.
2. One convolution subtracts adjacent quotients to make all 32 digits, 16 per input word.
3. Eight convolution tasks generate the steps, two digit pairs per task. Integer RELUX1 writes each step as 0 or 1.
4. One convolution applies the table's constant difference weights to those steps, producing 16 result digits.
5. One convolution joins each group of four digits with weights 1, 4, 16 and 64. The four output bytes are already the result word.

That plan needs 14 tasks per pair of words. The quotient, threshold and reconstruction values stay in NPU storage; Python prepares the constant weights. The test below will check whether the register setup implements this table.

First extend our shared convolution helper. SHR already added a selectable channel count. These tasks also need a selectable input DMA address. Enable integer ReLU-X only for the step tasks:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def conv_shl_subtask(self, weights:list[list[int]], out_addr:int, offset:int=0, shift:int=0,
-                       unsigned:bool=False, scratch_input:bool=False, channels:int=64) -> None:
+  def conv_shl_subtask(self, weights:list[list[int]], out_addr:int, offset:int=0, shift:int=0,
+                    unsigned:bool=False, scratch_input:bool=False, channels:int=64, input_addr:int|None=None, relux:bool=False) -> None:
@@
+    if input_addr is not None: self.npu_regs.append(E(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR, input_addr))
+    # Tasks 5–12: turn each digit-pair threshold index-t+1 into the 0/1 step used by Task 13's lookup sum.
+    if relux:
+      self.npu_regs += [
+        E(rk.DPU, rk.REG_DPU_BN_CFG, (1 << rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT) |
+          (1 << rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT)),
+        E(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 1), # Integer mode: cap at integer 1, not FP32 bits.
+      ]
     self.submit(cna=True)
```
TOREVIEW1: This is plan step 3, the eight threshold convolutions (Tasks 5–12). Only these calls enable relux; Task 13 combines their step masks with the table-difference weights.

Keep the scratch layout explicit before writing the tasks:

| Offset | Contents                        |
| -----: | ------------------------------- |
|      0 | Two original words, eight bytes |
|     32 | q1 for all eight bytes          |
|     64 | q2                              |
|     96 | q3                              |
|    128 | 32 base-4 digits                |
|    160 | Constant 1 for the step offsets |
|    256 | 16 groups of 16 step lanes      |
|    512 | Constant 1 for table[0]         |
|    544 | 16 result digits                |

Each step group uses 15 lanes; the last lane stays zero. The two constant 1 bytes provide biases through convolution weights.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def conv_and_word(self, raw:bytes) -> bytes:
+    assert len(raw) == 8
+    scratch = bytearray(640)
+    scratch[:8], scratch[160], scratch[512] = raw, 1, 1
+    to_mv(self.dev.input_buf, len(scratch))[:] = scratch
+    base = self.dev.input_mem.dma_addr
+    # Tasks 1–3: signed quotients for all eight original bytes.
+    for digit in range(1, 4):
+      self.conv_shl_subtask([[2*int(i==j) for j in range(32)] for i in range(8)], base+32*digit,
+        offset=-(4**digit-1), shift=2*digit+1, scratch_input=True, channels=32)
+    # Task 4: adjacent quotients give four base-4 digits per byte.
+    weights = [[0]*128 for _ in range(32)]
+    for byte in range(8):
+      for digit in range(4):
+        weights[byte*4+digit][digit*32+byte] = 1
+        if digit < 3: weights[byte*4+digit][(digit+1)*32+byte] = -4
+    self.conv_shl_subtask(weights, base+128, scratch_input=True, channels=128)
+    # Tasks 5–12: two digit pairs per task, 15 threshold steps per pair.
+    for pair in range(8):
+      weights = [[0]*64 for _ in range(32)]
+      for part in range(2):
+        digit = pair*2+part
+        for t in range(1, 16):
+          row = weights[part*16+t-1]
+          row[digit], row[16+digit], row[32] = 4, 1, 1-t+(10 if digit%4 == 3 else 0)
+      self.conv_shl_subtask(weights, base+256+pair*32, scratch_input=True, channels=64, input_addr=base+128, relux=True)
+    # Task 13: sum table differences selected by the threshold steps.
+    lookup = (0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3)
+    weights = [[0]*288 for _ in range(16)]
+    for digit in range(16):
+      table = list(lookup)
+      if digit%4 == 3:
+        table = [lookup[4*((i//4+2)%4)+(i%4+2)%4] for i in range(16)]
+        table = [x if x < 2 else x-4 for x in table]
+      weights[digit][256] = table[0]
+      for t in range(1, 16): weights[digit][digit*16+t-1] = table[t]-table[t-1]
+    self.conv_shl_subtask(weights, base+544, scratch_input=True, channels=288, input_addr=base+256)
+    # Task 14: reconstruct four signed bytes directly into the output buffer.
+    weights = [[0]*32 for _ in range(4)]
+    for byte in range(4):
+      for digit in range(4): weights[byte][byte*4+digit] = 4**digit
+    self.conv_shl_subtask(weights, self.dev.output_mem.dma_addr, scratch_input=True, channels=32, input_addr=base+544)
+    return bytes(to_mv(self.dev.output_buf, 4))
```

The wrapper only packs the original words and reads the result. Unlike signed SHR, INT32 AND needs no sign-fill task: signed and unsigned words use the same 32 bits.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_u32_and(self, a:list, b:list, dtype:DType) -> list:
+    assert len(a) == len(b)
+    fmt = "<I" if dtype == dtypes.uint else "<i"
+    result:list = []
+    for x,y in zip(a,b):
+      raw = self.conv_and_word(struct.pack(fmt, x) + struct.pack(fmt, y))
+      result.append(struct.unpack(fmt, raw)[0])
+    return result
```

Advertise AND as a composite operation, not an EW algorithm number. Bool inputs still use the matcher; INT32/UINT32 use the convolution helper:

Composite helpers now dispatch before the ordinary EW gate; otherwise `CMP` would reject AND before its helper runs.

```diff
-CMP = 9  # Internal multi-stage comparison marker, not an EW algorithm.
+CMP = 9  # Internal lowering/dispatch marker, not an EW algorithm.
@@
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
-           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0}
+           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          if u.op not in self.ops_map or self.ops_map[u.op] == CMP or \
-             u.dtype not in ((dtypes.int16, dtypes.int, dtypes.uint) if u.op is Ops.SHL else
-                             (dtypes.int, dtypes.uint) if u.op is Ops.SHR else (dtypes.half,)):
-            raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
-          elif u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
+          if u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_shift(Ops.SHL, src_values[0], src_values[1], u.dtype)
@@
           elif u.op is Ops.SHR and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_shift(Ops.SHR, src_values[0], src_values[1], u.dtype)
+          elif u.op is Ops.AND and u.dtype in (dtypes.int, dtypes.uint):
+            values[u] = self.run_u32_and(src_values[0], src_values[1], u.dtype)
+          elif u.op not in self.ops_map or self.ops_map[u.op] == CMP or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
+            raise NotImplementedError(f"ROCKCHIP NPU does not support {u.op} with {u.dtype}")
           else: values[u] = self.run_npu(u.op, *src_values, dtype=u.dtype)
```

Run test_and again:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

test_and (__main__.TestOps.test_and) ... ok

Ran 1 test in 22.234s

OK
```

The two AND helpers extracted from these diffs also passed the same test in 20.790s. The test covers INT32 inputs; the four bool pairs were checked separately. XOR is next.

## Ops.XOR

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_xor
```
TOREVIEW1: The disabled-feature probe removed the later integer XOR dispatch and bool XOR matcher in memory, keeping other current runtime support. It ran the existing test with fail-fast and no TRACE; this is not a full tutorial reconstruction:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.XOR with dtypes.int
Ran 1 test in 0.160s
FAILED (errors=1)
```


The probe establishes this missing path, not an accuracy failure in a built-in decomposition.

Next Ops.XOR. For bools, XOR means the two inputs are different, so reuse CMPNE:

```text
a, b → CAST(half) → CMPNE → existing comparison formula → CAST(bool)
```

| a     | b     | XOR   |
| ----- | ----- | ----- |
| False | False | False |
| False | True  | True  |
| True  | False | True  |
| True  | True  | False |

Add only the bool rule first:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Bool XOR is inequality of the FP16 0/1 inputs.
+    (UPat(Ops.XOR, dtypes.bool, name="u"),
+     lambda u: u.src[0].cast(dtypes.half).ne(u.src[1].cast(dtypes.half))),
```

This also handles the bool XOR in test_minimum. It does not handle the INT32 inputs in test_xor.

For integer XOR, reuse the AND decomposition. Only the two-bit table changes:

| a   |  b=0 |  b=1 |  b=2 |  b=3 |
| --- | ---: | ---: | ---: | ---: |
| 0   |    0 |    1 |    2 |    3 |
| 1   |    1 |    0 |    3 |    2 |
| 2   |    2 |    3 |    0 |    1 |
| 3   |    3 |    2 |    1 |    0 |

1. Keep the same base-4 digits and `index = 4*a+b`.
2. Keep the same RELUX1 threshold lanes.
3. Replace the table-difference weights. For a=2 and b=3, index=11 now selects XOR=1 instead of AND=2.
4. Keep the same signed top-digit adjustment and byte reconstruction.

Rename the helper now that it handles two operations:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def conv_and_word(self, raw:bytes) -> bytes:
-    assert len(raw) == 8
+  def conv_bitwise_word(self, op:Ops, raw:bytes) -> bytes:
+    assert op in (Ops.AND, Ops.XOR) and len(raw) == 8
@@
-    lookup = (0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3)
+    lookup = ((0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3) if op is Ops.AND else
+              (0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0))
```

Pass the operation through the wrapper. Input/output packing stays unchanged:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_u32_and(self, a:list, b:list, dtype:DType) -> list:
+  def run_u32_bitwise(self, op:Ops, a:list, b:list, dtype:DType) -> list:
@@
-      raw = self.conv_and_word(struct.pack(fmt, x) + struct.pack(fmt, y))
+      raw = self.conv_bitwise_word(op, struct.pack(fmt, x) + struct.pack(fmt, y))
```

Then advertise XOR and extend dispatch:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
-           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: CMP}
+           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: CMP, Ops.XOR: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op is Ops.AND and u.dtype in (dtypes.int, dtypes.uint):
-            values[u] = self.run_u32_and(src_values[0], src_values[1], u.dtype)
+          elif u.op in (Ops.AND, Ops.XOR) and u.dtype in (dtypes.int, dtypes.uint):
+            values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_xor

test_xor (__main__.TestOps.test_xor) ... ok

Ran 1 test in 0.253s

OK
```

Both operations use 14 convolution tasks per word pair. Python builds constant weights and copies original/final storage; the NPU calculates the digits, table selection and result bytes.

The helper diffs also passed 196 INT32/UINT32 AND/XOR pairs, including negative values and word boundaries. All four bool XOR pairs passed too.

Progress: **17 / 30** paths covered. AND/XOR cover bool and INT32/UINT32 here, not every integer width.

## Ops.TRUNC

TOREVIEW1: First run this at the tutorial state, before the new diffs below. Check whether TRUNC remains in the UOps before selecting native FLOOR/CEIL.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc
```
TOREVIEW1: The disabled-feature probe removed the later TRUNC matcher in memory, keeping other current runtime support. It ran the existing test with fail-fast and no TRACE; this is not a full tutorial reconstruction:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.TRUNC with dtypes.half
Ran 1 test in 0.229s
FAILED (errors=1)
```


The probe establishes this missing path, not an accuracy failure in a built-in decomposition.

Next Ops.TRUNC. Does the CVT right shift help here? Not directly: `2.9 >> 1` would scale the number, while TRUNC needs 2. We need to remove the fraction towards zero.

The TRM's FLOOR and CEIL selectors are candidates: FLOOR works for positive inputs, CEIL for negative inputs. The recorded selector probe confirmed 7 = FLOOR and 8 = CEIL on FP16. Lets combine those results without a Python sign branch:

```text
TRUNC(x) = MAX(FLOOR(x), MIN(CEIL(x), 0))
MIN(y, 0) = NEG(MAX(NEG(y), 0))
```

| Stage                   | -2.9 | -0.9 |  0.9 |  2.9 |
| ----------------------- | ---: | ---: | ---: | ---: |
| FLOOR(x)                |   -3 |   -1 |    0 |    2 |
| CEIL(x)                 |   -2 |   -0 |    1 |    3 |
| MIN(CEIL(x), 0)         |   -2 |   -0 |    0 |    0 |
| MAX(FLOOR(x), previous) |   -2 |   -0 |    0 |    2 |

1. For positive x, the MIN gives zero, so the final MAX keeps FLOOR(x).
2. For negative x, CEIL(x) is closer to zero than FLOOR(x), so the final MAX keeps CEIL(x).
3. Use CUSTOM for native FLOOR/CEIL, and ordinary NEG/MAX UOps for the rest. No CPU sign check, lossy integer CAST or LUT is needed.

First advertise TRUNC as another operation that must be lowered:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
-           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: CMP, Ops.XOR: CMP}
+           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.TRUNC: CMP}
```

Select the native EW algorithms in build_registers. FLOOR/CEIL are unary: disable operand DMA and select the immediate operand, just like NEG. Their unused immediate is zero instead of NEG's -1 multiplier.

```diff
   def build_registers(self, op:Ops, int16_mode:bool=False, custom:str|None=None, byte_output:bool=False,
                       input_addr:int|None=None, weight_addr:int|None=None, output_addr:int|None=None) -> None:
@@
     exp_shift = custom == "fp16_exponent_shift_minus(16)"
-    assert custom is None or (op is Ops.CUSTOM and custom == "fp16_exponent_shift_minus(16)" and not int16_mode)
+    rounding = {"FLOOR": 7, "CEIL": 8}.get(custom or "")
+    unary = op is Ops.NEG or rounding is not None
+    assert custom is None or (op is Ops.CUSTOM and (exp_shift or rounding is not None) and not int16_mode)
```

FLOOR selects algorithm 7, CEIL selects 8. Other operations keep their ops_map value:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def build_registers(self, op:Ops, int16_mode:bool=False, custom:str|None=None, byte_output:bool=False,
                       input_addr:int|None=None, weight_addr:int|None=None, output_addr:int|None=None) -> None:
@@
-        (self.ops_map[op] << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) |
+        ((rounding if rounding is not None else self.ops_map[op]) << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) |
```

These are unary operations. Disable the unused operand DMA and use an immediate zero; NEG keeps its existing -1 multiplier:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def build_registers(self, op:Ops, int16_mode:bool=False, custom:str|None=None, byte_output:bool=False,
                       input_addr:int|None=None, weight_addr:int|None=None, output_addr:int|None=None) -> None:
@@
-        ((op is not Ops.NEG) << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
+        ((not unary) << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
@@
-        ((exp_shift or op is Ops.NEG) << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT)),
+        ((exp_shift or unary) << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT)),
@@
-    if op is Ops.NEG:
-      self.npu_regs.append(E(rk.DPU, rk.REG_DPU_EW_OP_VALUE_0, fp16(-1.0)))
+    if unary:
+      self.npu_regs.append(E(rk.DPU, rk.REG_DPU_EW_OP_VALUE_0, fp16(-1.0) if op is Ops.NEG else 0))
```

Allow the two unary CUSTOM modes in the interpreter:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.arg == ("RELUX", dtypes.half) and u.dtype == dtypes.half and src_dtypes == [dtypes.half]: pass
+          elif u.arg in (("RELUX", dtypes.half), ("FLOOR", dtypes.half), ("CEIL", dtypes.half)) and \
+              u.dtype == dtypes.half and src_dtypes == [dtypes.half]: pass
```

Now lower the formula, reusing NEG/MAX:

```diff
 class RockchipRenderer(Renderer):
+  @staticmethod
+  def _pm_lower_trunc(x:UOp) -> UOp:
+    floor, ceil = (UOp(Ops.CUSTOM, src=(x,), arg=(mode, dtypes.half)) for mode in ("FLOOR", "CEIL"))
+    # trunc(x) = max(floor(x), min(ceil(x), 0)); min uses existing NEG/MAX.
+    return floor.maximum(ceil.alu(Ops.NEG).maximum(x.const_like(0)).alu(Ops.NEG))
+
@@
   comparison_matcher = PatternMatcher([
+    # Round toward zero using native FP16 FLOOR/CEIL, without a lossy integer CAST.
+    (UPat(Ops.TRUNC, dtypes.half, src=(UPat.var("x", dtypes.half),)),
+     lambda x: RockchipRenderer._pm_lower_trunc(x)),
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc

test_trunc (__main__.TestOps.test_trunc) ... ok

Ran 1 test in 1.336s

OK
```

All 65,536 FP16 encodings passed the direct NPU check against numpy.trunc. Non-NaN outputs matched bit for bit, including negative zero, subnormals and both infinities. NaN inputs remained NaN; their payload bits are not preserved. This is FP16 TRUNC only, with six NPU stages per atom, not a fused single-task implementation.

Progress: **18 / 30** paths covered, **12 / 30** remaining. TRUNC joins NEG and RECIPROCAL; AND/XOR now cover bool and 32-bit integers.

The earlier combined regression run gave `Ran 12 tests in 25.069s`, `OK`, with the same NOOPT/FP16/forward-only settings. The standalone result above checks this TRUNC step.

## Ops.OR

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or
```
TOREVIEW1: The disabled-feature probe removed the later integer OR dispatch (the earlier bool OR matcher stays enabled) in memory, keeping other current runtime support. It ran the existing test with fail-fast and no TRACE; this is not a full tutorial reconstruction:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.OR with dtypes.int
Ran 1 test in 0.163s
FAILED (errors=1)
```


The probe establishes this missing path, not an accuracy failure in a built-in decomposition.

Next extend Ops.OR to INT32/UINT32. We already split and rejoin the digits for AND/XOR. OR changes which of the two bits are kept, not their positions, so only the two-bit table should change:

| a   |  b=0 |  b=1 |  b=2 |  b=3 |
| --- | ---: | ---: | ---: | ---: |
| 0   |    0 |    1 |    2 |    3 |
| 1   |    1 |    1 |    3 |    3 |
| 2   |    2 |    3 |    2 |    3 |
| 3   |    3 |    3 |    3 |    3 |

The same `index = 4*a+b`, RELUX1 thresholds and table-difference weights now give a OR b. Signed top-digit handling and byte assembly stay the same. Bool OR still uses its existing CAST → MAX path.

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
            Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.TRUNC: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP}
@@
   def conv_bitwise_word(self, op:Ops, raw:bytes) -> bytes:
-    assert op in (Ops.AND, Ops.XOR) and len(raw) == 8
+    assert op in (Ops.AND, Ops.XOR, Ops.OR) and len(raw) == 8
@@
-    lookup = ((0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3) if op is Ops.AND else
-              (0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0))
+    lookup = {Ops.AND: (0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3),
+              Ops.XOR: (0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0),
+              Ops.OR:  (0,1,2,3,1,1,3,3,2,3,2,3,3,3,3,3)}[op]
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op in (Ops.AND, Ops.XOR) and u.dtype in (dtypes.int, dtypes.uint):
+          elif u.op in (Ops.AND, Ops.XOR, Ops.OR) and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or

test_or (__main__.TestOps.test_or) ... ok

Ran 1 test in 0.292s

OK
```

OR was already counted for bool, so progress stays **18 / 30**. Its coverage now includes INT32/UINT32, with the same 14 convolution tasks per word pair and no host arithmetic on intermediate values.

Four additional INT32/UINT32 Tensor checks passed (144 lanes), including broadcasting, random full-width values, sign-bit boundaries and alternating-bit patterns. Every check submitted NPU work.

## Ops.MULACC

For Ops.MULACC, which is `a*b+c`. We can try normal MUL then ADD? 

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q -rs \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc
```
TOREVIEW1: The disabled-feature probe removed the later MULACC dispatch in memory, keeping other current runtime support. It ran the existing test with fail-fast and no TRACE; this is not a full tutorial reconstruction:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.MULACC with dtypes.half
Ran 1 test in 0.057s
FAILED (errors=1)
```



The probe uses the current test's ROCKCHIP allowance, which is discussed later in this section; an earlier test version may skip instead. A skip is not a pass. The separate MUL→ADD idea below is a candidate, not evidence that tinygrad lowers a direct MULACC that way.

MULACC should round the combined result once; our two FP16 tasks would round the product first.

BS MUL followed by EW ADD looks promising. BRDMA can supply b and ERDMA can supply c, keeping the product FP32 until the output converter. Lets check ordinary values, overflow cancellation, halfway rounding and signed zero. The recorded native-task probe gave:

| Inputs                                      | Separate FP16 MUL then ADD | One BS → EW task | Correctly rounded FP16 MULACC |
| ------------------------------------------- | -------------------------: | ---------------: | ----------------------------: |
| `1.0009765625 * 1.0009765625 - 1.001953125` |                          0 |          `2^-20` |                       `2^-20` |
| `65504 * 2 - 65504`                         |                        inf |            65504 |                         65504 |
| `11.8125 * -2688 - 0.00023484230041503906`  |                     -31744 |           -31744 |                        -31760 |
| `(-0) * 1 + (-0)`                           |                         -0 |               +0 |                            -0 |

The first two cases show why keeping the product in FP32 helps. But the third exposes another rounding step: the exact product is -31752, halfway between two FP16 values. The small negative addend should move it towards -31760. FP32 addition loses that small amount, so the final FP16 conversion rounds the halfway value to -31744 instead.

The converter controls we tried did not fix this. Bypassing EW operand conversion misreads FP16 addends; changing its offset to negative zero or changing CVT_TYPE also did not preserve the negative-zero case.

The probe is saved in `~/npu/ops_reg/probe_fp16_mulacc.py`. A 4,096-triple random check had no numeric mismatches, but targeted halfway cases did. Passing ordinary random tests is not enough to claim correctly rounded MULACC.

Dont add MULACC to ops_map yet. The native task loses the rounding information in the third case. First correct that on the NPU; progress stays **18 / 30**.

### Recover the lost rounding information

The failing case tells us what is missing: the small addend disappears when the FP32 sum rounds. Keeping the sum in FP32 alone cannot fix that. Can we retain the rounding error separately?

Two FP16 significands have at most 11 bits each, so their product needs at most 22 bits and fits FP32's 24-bit significand. Start with that exact product, then use TwoSum for the addition. The private precision-5 mode keeps these intermediate results in FP32 storage.

TwoSum recovers that missing part. Each line below uses FP32 NPU ADD/SUB:

```text
p = a * b
s = p + c
v = s - p
e = (p - (s - v)) + (c - v)
```

For finite inputs, `s + e` represents the exact sum. In our failing example, s is -31752 and e is the small negative addend that disappeared before.

Next use round-to-odd before the final FP16 conversion. Read the same FP32 storage as INT32 bits, without numeric conversion. If e is finite and nonzero and s has an even low bit, move s by one FP32 bit towards e. If its low bit is already odd, leave it alone. The NPU does these comparisons and integer operations too. This keeps the information needed to round the final FP16 result, without adding e back and losing it again.

There is one separate zero case. FP32 ADD changes `-0 + -0` to +0. The product and c still retain their sign bits, so detect both negative-zero operands on the NPU and restore the result's sign bit before conversion.

| Case                                        | Native one-task result | Corrected result |
| ------------------------------------------- | ---------------------: | ---------------: |
| `11.8125 * -2688 - 0.00023484230041503906`  |                 -31744 |           -31760 |
| `1.0009765625 * 1.0009765625 - 1.001953125` |                `2^-20` |          `2^-20` |
| `65504 * 2 - 65504`                         |                  65504 |            65504 |
| `(-0) * 1 + (-0)`                           |                     +0 |               -0 |

The correction probe is `~/npu/ops_reg/probe_fp16_mulacc_correct.py`. Intermediate values stay in DMA storage; Python only uploads inputs/constants, submits tasks and reads the final result. It uses 32 NPU tasks per eight lanes, so this fixes the tested numerical limits but is much more expensive than the native task. It is still a probe, not advertised MULACC support in ops_rockchip.py.

```bash
$ PYTHONPATH=$PWD .venv/bin/python ~/npu/ops_reg/probe_fp16_mulacc_correct.py

4096 triples PASS
8192 triples PASS
12288 triples PASS
16384 triples PASS
17723 triples PASS; non-NaN results bit-exact, including signed zero
```

Run from the tinygrad repo root. This covers eight targeted cases, 16,384 random triples and 1,331 special-value combinations against a float64 multiply-add rounded to FP16. NaNs are checked as NaNs, not by payload. This is not an exhaustive test of all FP16 triples; progress stays **18 / 30** until the corrected path is integrated.

Now port the correction to ops_rockchip.py. The input and result are still FP16; FP32 and INT32 are private intermediate modes, not general dtype support.

First add a task helper. It starts from build_registers, then selects precision 5 for FP32 or 4 for INT32. Algorithm -1 bypasses EW for the two input conversions and the final output conversion. BS MUL reads b through BRDMA only for the product task; every other task disables BRDMA again.

The stride fields start at bit 4: `1 << ...__SHIFT` writes a 16-byte stride. Shifting 16 instead wrote 256 bytes and left the last four output lanes missing in the first port. The final FP32 → FP16 task writes four half lanes at offset 0 and four at offset 16.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def mulacc_stage(self, algo:int, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
+                   shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
+    # Private FP32/INT32 stages: precision 2 = FP16, 4 = INT32, 5 = FP32; algo -1 bypasses EW.
+    self.build_registers(Ops.ADD, input_addr=lhs, weight_addr=rhs, output_addr=out)
+    E = self.EMIT
+    ew = ((1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
+          (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT))
+    if algo < 0:
+      ew |= (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT)
+    else:
+      ew |= ((1 << rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT) | (3 << rk.DPU_EW_CFG_EDATA_SIZE__SHIFT) |
+             (algo << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
+             (binary << rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT) | (mul << rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT))
```

Select input/output precision. Keep the converter scale at 1; only the final FP16 output enables FP32TOFP16.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
                    shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
@@
              (algo << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
              (binary << rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT) | (mul << rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT))
+    self.npu_regs += [
+      E(rk.DPU, rk.REG_DPU_DATA_FORMAT,
+        (output << rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT) | (precision << rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT) |
+        (precision << rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
+        ((output == 2) << rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT) | (1 << rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
+        (1 << rk.DPU_OUT_CVT_SHIFT_CVT_TYPE__SHIFT) | ((shift != 0) << rk.DPU_OUT_CVT_SHIFT_CVT_ROUND__SHIFT) |
+        (shift << rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT)),
+    ]
```

Now set the output layout. One stride unit is 16 bytes. FP32 has four lanes per surface; FP16 output still keeps that surface stride.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
                    shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
@@
         (shift << rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT)),
     ]
+    self.npu_regs += [
+      E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, 1 << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
+      E(rk.DPU, rk.REG_DPU_BS_OW_CFG,
+        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
+        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) |
+        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT) | (1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, 1 << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
+      E(rk.DPU, rk.REG_DPU_EW_CFG, ew),
+    ]
```

Configure the input streams for the same precision. Disable BRDMA/NRDMA by default, and disable ERDMA when EW is bypassed.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
                    shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
@@
       E(rk.DPU, rk.REG_DPU_EW_CFG, ew),
     ]
+    self.npu_regs += [
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 1 << rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
+        (1 << rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT) | (1 << rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT) |
+        (1 << rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT) | (1 << rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT)),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
+        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT) |
+        (15 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT) |
+        (precision << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT) |
+        ((precision == 2) << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT) |
+        (1 << rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT)),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BRDMA_CFG, 1),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_NRDMA_CFG, 1),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
+        (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT) if algo < 0 else
+        (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT) | (3 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT)),
+    ]
```

The product task alone enables BS operand DMA. Point BRDMA at b, keep BS ALU/ReLU bypassed, then submit. The next task starts from the shared initialization again.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
                    shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
@@
         (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT) | (3 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT)),
     ]
+    if bs_mul is not None:
+      self.npu_regs += [
+        E(rk.DPU, rk.REG_DPU_BS_CFG,
+          (1 << rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT) | (1 << rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT)),
+        E(rk.DPU, rk.REG_DPU_BS_MUL_CFG, 1 << rk.DPU_BS_MUL_CFG_BS_MUL_SRC__SHIFT),
+        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BRDMA_CFG, 4 << rk.DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE__SHIFT),
+        E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BS_BASE_ADDR, bs_mul),
+      ]
+    self.submit()
```

Next allocate a separate scratch slot for every stage. The INT32 operations below operate on the stored FP32 bits, not on numerically converted integers:

| Step       | NPU operation                                                       |
| ---------- | ------------------------------------------------------------------- |
| Product    | FP16 BS MUL → exact FP32 p                                          |
| Sum        | FP32 ADD(p, c) → s                                                  |
| Error      | FP32 TwoSum → e                                                     |
| Low bit    | INT32 `q = floor(s_bits/2)`, `low = s_bits - 2*q`                   |
| Correction | If low is 0 and e is finite/nonzero, move s_bits one unit towards e |
| Zero sign  | Restore -0 when both p and c are -0                                 |
| Output     | FP32 → FP16 conversion                                              |

For the low bit, `round_away((x - int(x > 0))/2)` gives `floor(x/2)`. This avoids overflowing `2*x`. For negative FP32 values, moving the bits up makes the value more negative, so the sign comparison selects +1 or -1.

The finite/nonzero error check disables the correction for infinities and NaNs. We return the hardware NaN, without promising its payload bits. The zero-sign check uses `bits < INT_MIN+1`, which matches only INT_MIN, the stored FP32 negative zero.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_mulacc(self, a:list, b:list, c:list) -> list:
+    assert len(a) == len(b) == len(c)
+    base = self.dev.input_mem.dma_addr
+    # Each stage owns 64 scratch bytes. All intermediate arithmetic stays in this BO.
+    assert self.dev.input_mem.size >= 64*42
+    constants = (0, 1, 2, 0x7f800000, -2147483647, -2147483648)
+    zero, one, two, infinity, min_plus_one, min_int = (base + i*64 for i in range(3, 9))
+    for i,value in enumerate(constants, 3): to_mv(self.dev.input_buf+i*64, 32)[:] = struct.pack("<i", value)*8
+    result:list = []
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      for i,values in enumerate((a, b, c)):
+        to_mv(self.dev.input_buf+i*64, 16)[:] = struct.pack("<8e", *(values[start:start+8] + [0.0]*(8-count)))
+      slot = 10
+      def calc(algo:int, x:int, y:int=zero, precision:int=5, output:int=5, **kw) -> int:
+        nonlocal slot
+        out = base + 64*slot
+        slot += 1
+        self.mulacc_stage(algo, x, y, out, precision, output, **kw)
+        return out
+      def iadd(x:int, y:int) -> int: return calc(2, x, y, 4, 4)
+      def isub(x:int, y:int) -> int: return calc(4, x, y, 4, 4)
+      def imul(x:int, y:int) -> int: return calc(0, x, y, 4, 4, mul=True)
+      def ilt(x:int, y:int) -> int: return calc(1, x, y, 4, 4, binary=True)
```

Now calculate p and c in FP32, then TwoSum. Each calc call writes a new scratch slot, so later stages cannot overwrite a value still needed by the formula.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_mulacc(self, a:list, b:list, c:list) -> list:
@@
       def imul(x:int, y:int) -> int: return calc(0, x, y, 4, 4, mul=True)
       def ilt(x:int, y:int) -> int: return calc(1, x, y, 4, 4, binary=True)
+      product = calc(-1, base, precision=2, bs_mul=base+64)
+      addend = calc(-1, base+128, precision=2)
+      total = calc(2, product, addend)
+      # TwoSum: exact a*b+c = total + error for finite FP16 inputs.
+      v = calc(4, total, product)
+      error = calc(2, calc(4, product, calc(4, total, v)), calc(4, addend, v))
```

Find whether total's FP32 bit pattern is even. The same address is read in INT32 mode; this is a storage reinterpretation, not a numeric FP32 → INT32 CAST.

```text
half = floor(total_bits/2)
low_bit = total_bits - 2*half
even = 1 - low_bit
```

For signed INT32, the converter uses `round_away((x-int(x>0))/2)` to obtain floor(x/2) without overflowing 2*x.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_mulacc(self, a:list, b:list, c:list) -> list:
@@
       v = calc(4, total, product)
       error = calc(2, calc(4, product, calc(4, total, v)), calc(4, addend, v))
+      # Reinterpret FP32 storage as INT32. Get parity without overflowing 2*total_bits.
+      half = calc(4, total, ilt(zero, total), 4, 4, shift=1)
+      even = isub(one, isub(total, imul(half, two)))
```

Only a finite, nonzero error needs correction. ABS(error) must be between zero and infinity. Its positive FP32 encoding can be compared as INT32.

If error and total have opposite signs, decrement the encoding; otherwise increment it. Multiply this direction by even and valid, so odd encodings and special values stay unchanged.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_mulacc(self, a:list, b:list, c:list) -> list:
@@
       half = calc(4, total, ilt(zero, total), 4, 4, shift=1)
       even = isub(one, isub(total, imul(half, two)))
+      abs_error = calc(5, error)  # EW algorithm 5 = ABS.
+      valid = imul(ilt(zero, abs_error), ilt(abs_error, infinity))
+      opposite = calc(5, isub(ilt(error, zero), ilt(total, zero)), precision=4, output=4)
+      direction = isub(one, imul(opposite, two))
+      odd = iadd(total, imul(imul(even, valid), direction))
```

Finally restore negative zero when both product and addend were -0. Convert the corrected FP32 result to FP16 and read the two output surfaces.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_mulacc(self, a:list, b:list, c:list) -> list:
@@
       direction = isub(one, imul(opposite, two))
       odd = iadd(total, imul(imul(even, valid), direction))
+      # FP32 ADD turns -0 + -0 into +0; restore that sign from the original operands.
+      both_neg_zero = imul(ilt(product, min_plus_one), ilt(addend, min_plus_one))
+      signed = iadd(odd, imul(min_int, both_neg_zero))
+      out = calc(-1, signed, output=2) - base
+      # Four FP32 lanes per surface: half output retains the 16-byte surface stride.
+      raw = bytes(to_mv(self.dev.input_buf+out, 8)) + bytes(to_mv(self.dev.input_buf+out+16, 8))
+      result.extend(struct.unpack("<8e", raw)[:count])
+    return result
```

The helpers reconstructed from these smaller diffs passed 1,030 triples on the NPU: 1,024 seeded random triples plus six rounding, signed-zero and special-value cases. Non-NaN output bits matched the FP16 reference; NaNs were checked as NaNs. This checks these tutorial helpers, not every possible triple.

Now advertise MULACC and dispatch its three inputs. tinygrad also fuses integer MUL+ADD when MULACC is advertised, so decompose those non-FP16 cases again in the late matcher. This does not add general INT32 or FP32 MULACC support.

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
            Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op in (Ops.AND, Ops.XOR, Ops.OR) and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
+          elif u.op is Ops.MULACC and u.dtype == dtypes.half:
+            values[u] = self.run_mulacc(*src_values)
@@
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Global MUL+ADD fusion also creates integer MULACC; keep the existing non-FP16 paths.
+    (UPat(Ops.MULACC, name="u"),
+     lambda u: u.src[0].alu(Ops.MUL, u.src[1]).alu(Ops.ADD, u.src[2]) if u.dtype != dtypes.half else None),
```

Test the integrated backend, not just the standalone register probe. The first sweep includes all FP16 encodings in identity and negation; the second adds halfway products, overflow boundaries and all encodings multiplied by 0.5. Final summary lines:

```bash
$ PYTHONPATH=$PWD .venv/bin/python ~/npu/ops_reg/probe_fp16_mulacc_correct.py --backend --exhaustive-unary
148795 triples PASS; non-NaN results bit-exact, including signed zero

$ PYTHONPATH=$PWD .venv/bin/python ~/npu/ops_reg/probe_fp16_mulacc_correct.py --backend --rounding-boundaries
125665 triples PASS; non-NaN results bit-exact, including signed zero
```

Now check actual UOp dispatch and Tensor fusion. This also checks that a CNA task before MULACC, and ordinary EW ADD after it, do not leave stale register state. Scratch reuse must not overwrite earlier results.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP PYTHONPATH=$PWD \
    .venv/bin/python ~/npu/ops_reg/probe_fp16_mulacc_correct.py --integration-only

CNA → MULACC → EW, scratch lifetime and 0/1/7/8/9/17/31 lanes PASS
explicit MULACC dispatch: 3 calls
Tensor fused, tail and following ADD: 1 PASS
Tensor fused, tail and following ADD: 7 PASS
Tensor fused, tail and following ADD: 8 PASS
Tensor fused, tail and following ADD: 9 PASS
Tensor fused, tail and following ADD: 17 PASS
Tensor fused, tail and following ADD: 31 PASS
broadcast and transposed Tensor PASS
PASS 100 actual MULACC dispatches
```

The Tensor checks also passed with NOOPT=0, with 67 MULACC dispatches. These checks count calls to run_mulacc, so a constant-folded expression cannot silently pass as an NPU MULACC test.

The existing test_mulacc_with_zero_strides still fails under DEFAULT_FLOAT=HALF:

```text
dtype mismatch: tinygrad=float32 | torch=float16
```

It fails the same way with MULACC fusion disabled, before testing this path. We have not changed its dtype assertions or counted it as passing.

Rerun the existing arithmetic, comparison, selection and integer paths:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/backend/test_ops.py::TestOps::test_add test/backend/test_ops.py::TestOps::test_sub \
    test/backend/test_ops.py::TestOps::test_neg test/backend/test_ops.py::TestOps::test_mul \
    test/backend/test_ops.py::TestOps::test_div test/backend/test_ops.py::TestOps::test_maximum \
    test/backend/test_ops.py::TestOps::test_minimum test/backend/test_ops.py::TestOps::test_cmp_eq \
    test/backend/test_ops.py::TestOps::test_cmp_lt test/backend/test_ops.py::TestOps::test_where \
    test/backend/test_ops.py::TestOps::test_trunc test/backend/test_ops.py::TestOps::test_and \
    test/backend/test_ops.py::TestOps::test_or test/backend/test_ops.py::TestOps::test_xor \
    test/backend/test_ops.py::TestOps::test_lshift test/backend/test_ops.py::TestOps::test_rshift \
    test/backend/test_ops.py::TestOps::test_lshift_signed test/backend/test_ops.py::TestOps::test_rshift_signed

..................                                                       [100%]
18 passed in 32.66s
```

Progress: **19 / 30** paths covered. MULACC now has an NPU-only FP16 implementation, including the tested rounding, overflow, underflow, infinity and signed-zero cases. NaNs remain NaNs; payload bits are not guaranteed. It costs 32 submissions per atom of up to eight lanes, and the NOOPT interpreter can submit only one lane at a time. This is not general FP32 MULACC support or an exhaustive check of every possible input triple.

Why does test_uops.test_mulacc say "only python supports MULACC"? Git blame points to two commits:

- [c13da83f1](https://github.com/tinygrad/tinygrad/commit/c13da83f128fba7afce2c00a23a086fee45d021b), July 8, 2024, "tests from lowerer branch (#5339)", added the test with `skipUnless(getenv("PYTHON"), ...)`.
- [bc180a963c](https://github.com/tinygrad/tinygrad/commit/bc180a963c264aa426ce2a0aa6bee097c3e5e597), March 26, 2026, "deprecate <dev>=1 in favor of DEV=<dev> (#15467)", only changed that condition to `Device.DEFAULT == "PYTHON"`.

The weird part: the PTX renderer already emitted fma.rn for MULACC in the 2024 commit. So MULACC was not designed for Python only. The commit message does not explain why this particular test was restricted; the skip text is not a hardware support list.

Lets allow ROCKCHIP in the existing test. It explicitly used dtypes.float, so DEFAULT_FLOAT=HALF alone would not change its inputs. Select FP16 for ROCKCHIP and keep Python's FP32 case unchanged. The existing input cases and assertions stay the same; these small integer-valued cases are exactly representable in FP16.

```diff
 class TestFloatUOps(TestUOps):
@@
-  @unittest.skipUnless(Device.DEFAULT == "PYTHON", "only python supports MULACC")
+  @unittest.skipUnless(Device.DEFAULT in ("PYTHON", "ROCKCHIP"), "MULACC test enabled for PYTHON and ROCKCHIP")
   def test_mulacc(self):
-    self._test_top_fxn(Ops.MULACC, lambda a,b,c: a*b+c, (dtypes.float, dtypes.float, dtypes.float))
+    dtype = dtypes.half if Device.DEFAULT == "ROCKCHIP" else dtypes.float
+    self._test_top_fxn(Ops.MULACC, lambda a,b,c: a*b+c, (dtype,)*3)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc

.                                                                        [100%]
1 passed in 0.68s
```

This test now passes on ROCKCHIP, not skips. It checks both buffer inputs and constants; the larger correction probes above still cover rounding and special values. This is separate from test_ops.test_mulacc_with_zero_strides, whose dtype mismatch remains.


## Ops.CDIV
TOREVIEW1: Start with division; CMOD has its own section below.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int
```
TOREVIEW1: The disabled-feature probe removed the later integer division/remainder dispatch in memory, keeping other current runtime support. It ran the existing test with fail-fast and no TRACE; this is not a full tutorial reconstruction:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.CMOD with dtypes.int
Ran 1 test in 0.178s
FAILED (errors=1)
```



`test_div_int` includes floor division too. Its first unsupported UOp was CMOD, so it does not isolate CDIV. The direct `TestNonFloatUOps.test_div_int32` check below does.

The probe establishes this missing path, not an accuracy failure in a built-in decomposition.

For CDIV, we already have TRUNC, so first check the obvious candidate, FDIV → TRUNC:

```text
4094 / 3 = 1364.666...
FP16 FDIV rounds it to 1365
TRUNC(1365) = 1365, but CDIV should be 1364
```

Both inputs fit FP16 exactly. This is quotient rounding, not just an input CAST problem. The direct NPU probe returned 1365 too.

The quotient must stay exact, so narrowing these integers to FP16 is not enough. Before implementing integer arithmetic, we need to keep their original bytes through LOAD, BITCAST and STORE.

## Ops.BITCAST

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitcast
```
TOREVIEW1: The current-runtime run, without TRACE, failed in the Torch reference before reaching Rockchip:

```text
RuntimeError: self.size(-1) must be divisible by 2 to view Half as Int (different element sizes), but got 3
Ran 1 test in 0.016s
FAILED (errors=1)
```


The first case views a (3,3) HALF tensor as INT32. Two half elements make one INT32, but its last dimension is odd. This result says nothing about NPU BITCAST. The payload-preservation checks below address a different question; do not describe this reference failure as a backend limitation.

BITCAST keeps the same bytes and changes their dtype. In this interpreter the host copies storage; it does not calculate a converted value or submit a fake identity operation.

For example, FP16 1.0 and UINT16 15360 have the same two bytes, `00 3c`. BITCAST changes which dtype reads them; CAST would calculate a new numeric representation.

Keep those bytes in a memoryview. No new wrapper class is needed. These helpers use the storage-format functions already imported by our starting Python interpreter:

```diff
@@
+def typed_view(raw:bytes|memoryview, dtype:DType) -> memoryview:
+  return memoryview(raw).cast("B").cast(storage_fmt_for_dtype(dtype))
+
+def raw16(value, dtype:DType) -> bytes|memoryview:
+  if isinstance(value, memoryview): return value.cast("B")
+  return struct.pack("<" + storage_fmt_for_dtype(dtype), to_storage_scalar(value, dtype))
+
 def _load(m, i, dtype: DType):
```

raw16 copies a view's bytes unchanged. Only constants that are still Python values need packing. Despite the old name, it handles the dtype's full size, not only 16 bits.

Some interpreter checks still need a scalar, such as an index or loop condition. Decode only at those boundaries:

```diff
@@
+def scalar16(value, dtype:DType|None=None):
+  if not isinstance(value, memoryview): return value
+  return from_storage_scalar(value[0], dtype) if dtype is not None else value[0]
+
 def _load(m, i, dtype: DType):
```

LOAD must keep the original bits, including NaN payloads. STORE copies them back without a numeric conversion:

```diff
 def _load(m, i, dtype: DType):
@@
   if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
+  if m.itemsize == dtype.itemsize:
+    return typed_view(bytes(m.cast("B")[i*m.itemsize:(i+1)*m.itemsize]), dtype)
```

```diff
 def _store(m, i, v, dtype: DType):
   if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
+  if isinstance(v, memoryview):
+    assert v.nbytes == dtype.itemsize
+    m.cast("B")[i*m.itemsize:i*m.itemsize+dtype.itemsize] = v.cast("B")
+    return
```

Now BITCAST is just a different view of the same bytes:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
         elif u.op is Ops.BITCAST:
-          values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
+          assert src_dtypes[0].itemsize == u.dtype.itemsize, "bitcast itemsize mismatch"
+          values[u] = [typed_view(raw16(x, src_dtypes[0]), u.dtype) for x in src_values[0]]
```

Remove the old numeric BITCAST helper from the import. Keep the storage-format helpers:

```diff
-from tinygrad.dtype import bitcast, DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
+from tinygrad.dtype import DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
```

Pack arithmetic inputs with raw16 rather than converting them back to Python floats. Bool-to-half still packs INT16 0/1; the byte-output CAST still takes FP16:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_npu(self, op:Ops, a:list, b:list|None=None, custom:str|None=None, dtype:DType=dtypes.half) -> list:
@@
-      packed = struct.pack("<8h" if op is Ops.SHL or (op is Ops.CAST and not byte_output) else "<8e", *(lanes + [0] * (8-len(lanes))))
+      input_dtype = dtypes.int16 if op is Ops.SHL or (op is Ops.CAST and not byte_output) else dtypes.half
+      packed = b"".join(raw16(x, input_dtype) for x in lanes) + bytes(2*(8-len(lanes)))
@@
-        to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8h" if op is Ops.SHL else "<8e", *(rhs + [0] * (8-len(rhs))))
+        to_mv(self.dev.weight_buf, 16)[:] = b"".join(raw16(x, input_dtype) for x in rhs) + bytes(2*(8-len(rhs)))
```

Copy each completed output before reusing the output buffer. This keeps a result alive without adding a DMA allocator or a storage wrapper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_npu(self, op:Ops, a:list, b:list|None=None, custom:str|None=None, dtype:DType=dtypes.half) -> list:
@@
-      fmt = "16?" if dtype == dtypes.bool else "16b" if dtype == dtypes.int8 else "8h" if dtype == dtypes.int16 else "8e"
-      result.extend(struct.unpack("<" + fmt, to_mv(self.dev.output_buf, 16))[:len(lanes)])
+      out = bytes(to_mv(self.dev.output_buf, 16))
+      result.extend(typed_view(out[i:i+dtype.itemsize], dtype) for i in range(0, len(lanes)*dtype.itemsize, dtype.itemsize))
```

The arithmetic still runs on the NPU. Python copies the input/output storage between tasks; this step does not make the interpreter fully device-resident.

The existing guards inspect values rather than their bytes. Decode there, without changing what each guard allows:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_npu(self, op:Ops, a:list, b:list|None=None, custom:str|None=None, dtype:DType=dtypes.half) -> list:
@@
-      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in a):
+      # Decode each typed view for the numeric guard; keep the original input bytes for the NPU.
+      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in map(scalar16, a)):
@@
     if op is Ops.SHL:
       assert b is not None
+      # Decode shift counts once into an indexable list for the uniform-count check and multiplier lookup.
+      b = list(map(scalar16, b))
@@
-      if byte_output and any(x not in (0.0, 1.0) for x in a):
+      if byte_output and any(scalar16(x) not in (0.0, 1.0) for x in a):
```
TOREVIEW1: SHL already works; BITCAST changes what its inputs look like. LOAD and the output loop above now return typed memoryviews instead of Python numbers. The old SHL count check and `powers[shift]` lookup need integers, so decode b before using them. RECIPROCAL's sign guard and CAST's 0/1 guard need numbers for the same reason. The SHL formula and NPU registers are unchanged.

Bool-to-half is a special case: a bool occupies one byte, but our multiplier takes INT16 0/1 lanes. Read the bool value before packing that input. The remaining Python CAST fallback also needs a scalar:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-            values[u] = self.run_npu(Ops.CAST, src_values[0], dtype=u.dtype)
+            values[u] = self.run_npu(Ops.CAST, [scalar16(x) for x in src_values[0]] if src_dtypes[0] == dtypes.bool else src_values[0],
+                                     dtype=u.dtype)
@@
-            values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
+            values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(scalar16(x, src_dtypes[0]))) for x in src_values[0]]
@@
-            if any(math.isnan(x) for xs in src_values[1:] for x in xs):
+            if any(math.isnan(scalar16(x)) for xs in src_values[1:] for x in xs):
```

A memoryview containing zero is still a non-empty Python object. Control-flow and LOAD gates must test its scalar value, not the object's truthiness:

```diff
 def load(inp, j, dtype: DType):
-  if len(inp) >= 3: return [_load(m, x+j*_step(m, dtype) if x is not None else None, dtype) if gate else alt for (m,x),alt,gate in zip(*inp[:3])]
+  if len(inp) >= 3:
+    return [_load(m, x+j*_step(m, dtype) if x is not None else None, dtype) if scalar16(gate) else alt for (m,x),alt,gate in zip(*inp[:3])]
```

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-            if values[u.src[2]][0]: i = self.uop_to_index[u.src[1]]
+            if scalar16(values[u.src[2]][0]): i = self.uop_to_index[u.src[1]]
@@
-          exec_masks.append([x and y for x,y in zip(exec_masks[-1], src_values[0])])
+          exec_masks.append([x and scalar16(y) for x,y in zip(exec_masks[-1], src_values[0])])
@@
-          if values[u][0] == src_values[0][0]:
+          if values[u][0] == scalar16(src_values[0][0]):
```

Likewise an INDEX needs an integer offset:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-            for m,o in zip(src_values[0], src_values[1]): ret.append((m[0], m[1]+o*scale) if isinstance(m, tuple) else (m, o*scale))
+            for m,o in zip(src_values[0], src_values[1]):
+              index = scalar16(o)
+              ret.append((m[0], m[1]+index*scale) if isinstance(m, tuple) else (m, index*scale))
```

The convolution wrappers only copy original/final storage. Change their packing boundary too; the register sequences stay the same:

Apply the storage change once to the shared shift wrapper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_u32_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
     assert op in (Ops.SHL, Ops.SHR)
+    b = list(map(scalar16, b))
@@
-    fmt = "<I" if dtype == dtypes.uint else "<i"
-    return [struct.unpack(fmt, self.conv_shift(op, struct.pack(fmt, x), int(b[0]), signed=dtype == dtypes.int))[0] for x in a]
+    return [typed_view(self.conv_shift(op, bytes(raw16(x, dtype)), int(b[0]), signed=dtype == dtypes.int), dtype) for x in a]
```

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_u32_bitwise(self, op:Ops, a:list, b:list, dtype:DType) -> list:
@@
-    fmt = "<I" if dtype == dtypes.uint else "<i"
     result:list = []
     for x,y in zip(a,b):
-      raw = self.conv_bitwise_word(op, struct.pack(fmt, x) + struct.pack(fmt, y))
-      result.append(struct.unpack(fmt, raw)[0])
+      raw = self.conv_bitwise_word(op, bytes(raw16(x, dtype)) + bytes(raw16(y, dtype)))
+      result.append(typed_view(raw, dtype))
```

MULACC keeps its arithmetic unchanged as well. Copy the original FP16 bytes in and return the final bytes as views:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_mulacc(self, a:list, b:list, c:list) -> list:
@@
-        to_mv(self.dev.input_buf+i*64, 16)[:] = struct.pack("<8e", *(values[start:start+8] + [0.0]*(8-count)))
+        to_mv(self.dev.input_buf+i*64, 16)[:] = b"".join(raw16(x, dtypes.half) for x in values[start:start+8]) + bytes(2*(8-count))
@@
-      result.extend(struct.unpack("<8e", raw)[:count])
+      result.extend(typed_view(raw[i*2:i*2+2], dtypes.half) for i in range(count))
```

The bit-pattern checks passed all 65,536 FP16 encodings, all 65,536 BF16 encodings, and selected FP32/FP64 zeros, infinities and NaN payloads. The existing test_bitcast passed with its default FP32 inputs.

## Ops.CDIV: exact integer arithmetic

TOREVIEW1: There is no `TestOps.test_cdiv`; the existing direct test is `TestNonFloatUOps.test_div_int32`. With the later integer division dispatch disabled in memory, it stops on CDIV itself:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestNonFloatUOps.test_div_int32

NotImplementedError: ROCKCHIP NPU does not support Ops.CDIV with dtypes.int
Ran 1 test in 0.061s
FAILED (errors=1)
```

This reuses current storage and other primitive support; it is a disabled-feature probe, not a full earlier-state replay. Implement the quotient path first, then enable CMOD in its separate section.

Now the original bytes survive the interpreter. How can we divide without FP16 rounding? Reuse the base-256 representation from shifts: each byte is a limb in 0..255. A padded INT32 lane can hold a byte product and carry exactly.

Before division, we need subtraction with borrow, comparison and selection. The private INT32 stages from MULACC provide those operations; this is why we add the following helpers here:

| Stage          | a        | b         | c         | d         |
| -------------- | -------- | --------- | --------- | --------- |
| UINT32 storage | byte 0   | byte 1    | byte 2    | byte 3    |
| Scratch INT32  | 0..255   | 0..255    | 0..255    | 0..255    |
| Arithmetic     | low limb | next limb | next limb | high limb |
| Result storage | low byte | low byte  | low byte  | low byte  |

1. Copy the original bytes into the scratch lanes. Python copies bytes and inserts zero padding; it does not split values with arithmetic.
2. For SUB, compute `t = a - b - borrow`. If t is negative, add 256 and pass borrow=1 to the next limb.
3. ADD can reuse SUB with the two's-complement negative of b.
4. MUL uses byte products. Each product is at most 65025; even an eight-byte word's accumulated products fit INT32.
5. Extract carry with our converter formula: `floor(t/256) = round((2*t - 255)/512)`. Keep `t - 256*carry` as the result byte.

These are private INT32 tasks through mulacc_stage, not FP16 casts. First add the scratch and limb helpers:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
+    # Base-256 limbs keep every intermediate exact in the INT32 datapath, including UINT64 division.
+    width = dtype.itemsize
+    assert dtype in dtypes.ints and all(len(x) == len(inputs[0]) for x in inputs)
+    base = self.dev.input_mem.dma_addr
+    result:list = []
+    for start in range(0, len(inputs[0]), 8):
+      count = min(8, len(inputs[0])-start)
+      slot = 0
+      def allocate() -> int:
+        nonlocal slot
+        addr = base + slot*64
+        slot += 1
+        if slot*64 > self.dev.input_mem.size: raise RuntimeError("ROCKCHIP integer scratch exhausted")
+        return addr
+      def put(raw:bytes) -> int:
+        addr = allocate()
+        to_mv(self.dev.input_buf+addr-base, 32)[:] = raw + bytes(32-len(raw))
+        return addr
```

Cache the constant lanes once per atom. calc submits an INT32 task and returns its scratch address; add/sub/mul/lt below are NPU operations, not Python arithmetic on input values.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
         to_mv(self.dev.input_buf+addr-base, 32)[:] = raw + bytes(32-len(raw))
         return addr
+      constants:dict[int, int] = {}
+      def const(value:int) -> int:
+        if value not in constants: constants[value] = put(struct.pack("<i", value)*8)
+        return constants[value]
+      zero, one, radix = const(0), const(1), const(256)
+      def calc(algo:int, x:int, y:int=zero, **kw) -> int:
+        out = allocate()
+        self.mulacc_stage(algo, x, y, out, precision=4, output=4, **kw)
+        return out
+      def add(x:int, y:int) -> int: return calc(2, x, y)
+      def sub(x:int, y:int) -> int: return calc(4, x, y)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
```

For a 0/1 mask, selection is `no + (yes-no)*mask`. These are small integer limbs, so this arithmetic selection is exact.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
       def mul(x:int, y:int) -> int: return calc(0, x, y, mul=True)
       def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
```

Subtract from the lowest byte upward. A negative result borrows 1 from the next byte and adds 256 to the current byte.

For `0x0100 - 0x0001`: low byte `0-1=-1` becomes 255 with borrow 1; high byte `1-0-1=0`. The result bytes are `ff 00`.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
       def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def subtract(a:list[int], b:list[int]) -> tuple[list[int], int]:
+        out, borrow = [], zero
+        for x,y in zip(a, b):
+          value = sub(sub(x, y), borrow)
+          borrow = lt(value, zero)
+          out.append(add(value, mul(radix, borrow)))
+        return out, borrow
```

Apply selection to each byte. Two's-complement negation is the same subtract helper with an all-zero left operand.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
           out.append(add(value, mul(radix, borrow)))
         return out, borrow
+      def choose(mask:int, a:list[int], b:list[int]) -> list[int]: return [select(mask, x, y) for x,y in zip(a, b)]
+      def negate(a:list[int]) -> list[int]: return subtract([zero]*len(a), a)[0]
```

Division will feed one bit at a time into a whole word. double computes `2*x+carry` per byte, keeps the low byte, and passes its carry upward.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
       def choose(mask:int, a:list[int], b:list[int]) -> list[int]: return [select(mask, x, y) for x,y in zip(a, b)]
       def negate(a:list[int]) -> list[int]: return subtract([zero]*len(a), a)[0]
+      def double(a:list[int], carry:int) -> list[int]:
+        out = []
+        for x in a:
+          value = add(add(x, x), carry)
+          carry = sub(one, lt(value, radix))
+          out.append(sub(value, mul(radix, carry)))
+        return out
```

All limb values are non-negative. Their sum is zero only when every limb is zero, so one comparison produces a nonzero mask.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
           out.append(sub(value, mul(radix, carry)))
         return out
+      def nonzero(a:list[int]) -> int:
+        total = zero
+        for x in a: total = add(total, x)
+        return lt(zero, total)
```

Next load the limbs and implement exact wrapping ADD/SUB/MUL, integer comparisons and WHERE. Integer WHERE selects raw bytes, so no integer is converted to FP16:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
+      # Only copy raw bytes into zero-padded INT32 lanes; all carries, signs and arithmetic are NPU operations.
+      operands = []
+      for index,values in enumerate(inputs):
+        dt = dtypes.bool if op is Ops.WHERE and index == 0 else dtype
+        raw = [bytes(raw16(x, dt)) for x in values[start:start+8]]
+        operands.append([put(b"".join(x[i:i+1]+bytes(3) for x in raw)) for i in range(dt.itemsize)])
+      a = operands[0]
+      b = operands[1] if len(operands) > 1 else [zero]*width
```

WHERE uses the bool mask to select corresponding limbs. NEG, ADD and SUB reuse the helpers above; discarding the final borrow gives the dtype's wrapping result.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
       a = operands[0]
       b = operands[1] if len(operands) > 1 else [zero]*width
+      if op is Ops.WHERE: out = choose(a[0], operands[1], operands[2])
+      elif op is Ops.NEG: out = negate(a)
+      elif op is Ops.ADD: out = subtract(a, negate(b))[0]
+      elif op is Ops.SUB: out = subtract(a, b)[0]
```

For multiplication, output byte i collects the byte products whose positions add to i:

```text
total[i] = carry + sum(a[j]*b[i-j], j=0..i)
carry = floor(total[i]/256)
out[i] = total[i] - 256*carry
```

The biased CVT shift extracts carry exactly. Products above the output width are discarded, giving wrapping multiplication.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
       elif op is Ops.ADD: out = subtract(a, negate(b))[0]
       elif op is Ops.SUB: out = subtract(a, b)[0]
+      elif op is Ops.MUL:
+        out, carry = [], zero
+        for i in range(width):
+          total = carry
+          for j in range(i+1): total = add(total, mul(a[j], b[i-j]))
+          # No tie: round((2*total-255)/512) is floor(total/256).
+          carry = calc(4, add(total, total), const(255), shift=9)
+          out.append(sub(total, mul(radix, carry)))
```

For comparisons, the highest byte tells us each sign. If signs differ, the negative operand is smaller; otherwise unsigned subtraction's final borrow gives the ordering. Any nonzero difference means inequality.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
           carry = calc(4, add(total, total), const(255), shift=9)
           out.append(sub(total, mul(radix, carry)))
+      else:
+        sign_a, sign_b = (lt(const(127), x[-1]) if dtype in dtypes.sints else zero for x in (a, b))
+        different_sign = calc(5, sub(sign_a, sign_b))
+        if op in (*GroupOp.Comparison, Ops.MAX):
+          difference, unsigned_lt = subtract(a, b)
+          less = select(different_sign, sign_a, unsigned_lt)
+          if op is Ops.MAX: out = choose(less, b, a)
+          elif op is Ops.CMPLT: out = [less]
+          else:
+            unequal = nonzero(difference)
+            out = [unequal if op is Ops.CMPNE else sub(one, unequal)]
```

For division, keep a quotient q and remainder r. Read the magnitude of a from the most significant bit down:

```text
r = 2*r + next_input_bit
if r >= abs(b):
    r = r - abs(b)
    next_quotient_bit = 1
else:
    next_quotient_bit = 0
q = 2*q + next_quotient_bit
```

The NPU implements the condition with integer MIN + BINARY_EN and selects the limbs. There is no Python branch on a computed remainder. This is restoring division, not the old 32-bit-plane representation.

For `13 / 3`, ignore the leading zero bits and feed `1101`:

| Input bit | 2*r + bit | Take subtraction? | New r | New q |
| --------: | --------: | ----------------- | ----: | ----: |
|         1 |         1 | no                |     1 |     0 |
|         1 |         3 | yes               |     0 |     1 |
|         0 |         0 | no                |     0 |     2 |
|         1 |         1 | no                |     1 |     4 |

So CDIV gives 4 and CMOD gives 1. Both come from the same loop.

An extra remainder byte keeps the carry, so doubling a large unsigned remainder cannot overflow the word. Afterwards restore q's sign from both inputs and r's sign from a. CDIV by zero returns 0 and CMOD returns a, matching tinygrad's helper definitions.

Start with truncating division. Floor division's sign correction comes after this loop:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
+        else:
+          assert op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD)
+          numerator, divisor = choose(sign_a, negate(a), a), choose(sign_b, negate(b), b)
+          quotient, remainder = [zero]*width, [zero]*(width+1)
+          # Restoring division: feed one numerator bit into the remainder, subtract if it fits.
+          # The extra byte retains the carry; there is no array of bit planes or host arithmetic.
```

Feed the numerator from its highest byte and highest bit. Each iteration doubles the remainder, subtracts the divisor, and uses the borrow mask to select whether the subtraction fits.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
           # Restoring division: feed one numerator bit into the remainder, subtract if it fits.
           # The extra byte retains the carry; there is no array of bit planes or host arithmetic.
+          for word in reversed(numerator):
+            for _ in range(8):
+              bit = lt(const(127), word)
+              rest = sub(word, mul(const(128), bit))
+              word = add(rest, rest)
+              remainder = double(remainder, bit)
+              candidate, borrow = subtract(remainder, divisor+[zero])
+              take = sub(one, borrow)
+              remainder = choose(take, candidate, remainder)
+              quotient = double(quotient, take)
```

Restore the signs after unsigned division. The quotient sign depends on both operands; the remainder sign follows the numerator. A zero divisor selects quotient zero.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
               remainder = choose(take, candidate, remainder)
               quotient = double(quotient, take)
+          valid = nonzero(divisor)
+          quotient = choose(valid, choose(different_sign, negate(quotient), quotient), [zero]*width)
+          remainder = choose(sign_a, negate(remainder[:width]), remainder[:width])
```

### Ops.FLOORDIV and Ops.FLOORMOD

CDIV truncates toward zero. For a negative non-integer quotient, floor is one smaller. So the correction is needed only when signs differ and the remainder is nonzero:

| Inputs | CDIV | CMOD | FLOORDIV | FLOORMOD |
| ------ | ---: | ---: | -------: | -------: |
| 13, 3  |    4 |    1 |        4 |        1 |
| -13, 3 |   -4 |   -1 |       -5 |        2 |
| 13, -3 |   -4 |    1 |       -5 |       -2 |

Subtract one from the quotient and add the original divisor to the remainder. Keep zero divisors outside this correction.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
           quotient = choose(valid, choose(different_sign, negate(quotient), quotient), [zero]*width)
           remainder = choose(sign_a, negate(remainder[:width]), remainder[:width])
+          if op in (Ops.FLOORDIV, Ops.FLOORMOD):
+            correction = mul(mul(different_sign, nonzero(remainder)), valid)
+            quotient = subtract(quotient, [correction]+[zero]*(width-1))[0]
+            remainder = subtract(remainder, negate(choose(correction, b, [zero]*width)))[0]
+          out = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else remainder
```

Finally copy the low byte from each NPU-produced INT32 limb, lowest limb first. The arithmetic has already finished; this only restores the tensor's storage layout.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
             remainder = subtract(remainder, negate(choose(correction, b, [zero]*width)))[0]
           out = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else remainder
+      # Rejoin the low bytes of the NPU-produced limbs. This is a storage copy, not numeric evaluation.
+      out_dtype = dtypes.bool if op in GroupOp.Comparison else dtype
+      raw_out = [bytes(to_mv(self.dev.input_buf+x-base, 32)) for x in out]
+      result.extend(typed_view(b"".join(x[i*4:i*4+1] for x in raw_out), out_dtype) for i in range(count))
+    return result
```

Remove the earlier lossy INT32 → FP16 → INT32 matchers for ADD/MUL and comparisons. Keep the float/bool comparison matchers and restrict the arithmetic WHERE matcher to half. Route integer operations before the FP16 gate:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
@@
-    # Lossy INT32 comparison: FP16 conversion can make distinct integers equal.
-    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=dtypes.int32), UPat(dtype=dtypes.int32)), name="u"),
-     lambda u: RockchipRenderer._pm_lower_compare(u)),
@@
-    # Experimental: FP16 arithmetic is not exact for arbitrary INT32 values.
-    (UPat((Ops.MUL, Ops.ADD), dtypes.int32, name="u"),
-     lambda u: u.src[0].cast(dtypes.half).alu(u.op, u.src[1].cast(dtypes.half)).cast(dtypes.int32)),
```

Remove INT32 from the less-than and WHERE patterns too. Otherwise those operations would still be converted to FP16 before reaching our integer dispatch:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
@@
-    (UPat(Ops.CMPLT, src=(UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.int32, dtypes.bool)),
-                         UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.int32, dtypes.bool))), name="u"),
+    (UPat(Ops.CMPLT, src=(UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.bool)),
+                         UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.bool))), name="u"),
@@
-    # Arithmetic selection; INT32 casts are lossy, and non-finite values/signed zero need separate handling.
-    (UPat(Ops.WHERE, (dtypes.half, dtypes.int32), src=(UPat(dtype=dtypes.bool),
-      UPat(dtype=(dtypes.half, dtypes.int32, dtypes.weakint, dtypes.weakfloat)),
-      UPat(dtype=(dtypes.half, dtypes.int32, dtypes.weakint, dtypes.weakfloat))), name="u"),
+    # Integer WHERE now selects exact limbs; keep this arithmetic rule for FP16 only.
+    (UPat(Ops.WHERE, dtypes.half, src=(UPat(dtype=dtypes.bool), UPat(dtype=dtypes.half), UPat(dtype=dtypes.half)), name="u"),
      lambda u: RockchipRenderer._pm_lower_where(u)),
```

Now enable CDIV and the exact integer helper. Keep CMOD gated until its own step:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          if u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
+          integer_dtype = src_dtypes[1] if u.op is Ops.WHERE else src_dtypes[0]
+          if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
+                                                      Ops.CDIV, Ops.FLOORDIV, *GroupOp.Comparison):
+            values[u] = self.run_integer(u.op, src_values, integer_dtype)
+          elif u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
```

tinygrad already lowers FLOORDIV/FLOORMOD using CDIV/CMOD plus sign correction in codegen/decomp/op.py. We do not need to change that file. The direct helper also handles both, so we can check the formula independently.

Check direct CDIV with `TestNonFloatUOps.test_div_int32`; `test_div_int` also needs CMOD, as the initial trace showed.

The current-runtime check of direct CDIV passed; this is not a reconstruction of the helper from these diffs:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestNonFloatUOps.test_div_int32

Ran 1 test in 1.649s
OK
```

## Ops.CMOD

TOREVIEW1: CDIV returns the quotient, but its restoring-division loop also keeps the remainder. CMOD returns that remainder after restoring the numerator's sign. This is why the shared helper above contains both paths; CMOD does not need a second division algorithm.

The direct remainder test with the later division/remainder dispatch disabled in memory stops here:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestNonFloatUOps.test_mod_int32

NotImplementedError: ROCKCHIP NPU does not support Ops.CMOD with dtypes.int
Ran 1 test in 0.062s
FAILED (errors=1)
```

Enable the remaining path:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
-                                                      Ops.CDIV, Ops.FLOORDIV, *GroupOp.Comparison):
+                                                      Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, *GroupOp.Comparison):
```

The saved combined-helper run now belongs here:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

test_div_int (__main__.TestOps.test_div_int) ... ok
Ran 1 test in 3.505s
OK
```

Check remainder first, then floor-modulo separately:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_fmod

test_fmod (__main__.TestOps.test_fmod) ... ok

Ran 1 test in 5.890s

OK

$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mod

test_mod (__main__.TestOps.test_mod) ... ok

Ran 1 test in 13.169s

OK
```

The smaller run_integer diffs reconstruct the runtime helper exactly. The fmod and mod runs above also used that reconstructed helper.

The focused checks passed 88 subtests across signed and unsigned 8/16/32/64-bit arithmetic, comparisons and division. This includes full-width random inputs, wrapping results, negative remainders and zero divisors. It is not exhaustive over every pair. INT32 division took about 0.25s for eight lanes; correctness first, not a speed claim.

## Ops.THREEFRY

TOREVIEW1: No THREEFRY entry is needed in ops_map: leaving it unsupported lets tinygrad expand it into simpler UOps. But those UOps still need working primitives. With the later 64-bit shift dispatch disabled in memory, the existing test fails at the split, before the rounds:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_threefry

NotImplementedError: ROCKCHIP NPU does not support Ops.SHR with dtypes.ulong
Ran 1 test in 0.180s
FAILED (errors=1)
```


This disabled-feature probe reuses the rest of the current runtime, including storage handling. It establishes the missing UINT64 SHR path, not a complete replay of the tutorial. The earlier current-runtime pass belongs after the helpers, not before them.

Check what tinygrad already lowers. `codegen/decomp/op.py` supplies the rounds using wrapping ADD, shifts and XOR. We have those UINT32 operations now, so leave THREEFRY out of code_for_op and inspect what its split/join needs.

One round mixes two UINT32 words:

```text
sum = x0 + x1                         # wrap to 32 bits
rotated = (x1 << r) + (x1 >> (32-r))   # the two pieces occupy different bits
x0, x1 = sum, sum XOR rotated
```

1. Split the UINT64 counter and key into low/high UINT32 words.
2. Run 20 rounds, with the rotation counts and key additions supplied by tinygrad's existing decomposition.
3. Join the final words as `(high << 32) OR low`.

We have the round operations already. What is missing is the 64-bit split/join path, not a new THREEFRY register algorithm.

The remaining storage operations are UINT64 split/join. Narrowing keeps the low bytes; unsigned widening inserts zero bytes:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
         elif u.op is Ops.CAST:
@@
+          elif src_dtypes[0] in dtypes.ints and u.dtype in dtypes.ints and \
+              (u.dtype.itemsize <= src_dtypes[0].itemsize or src_dtypes[0] in dtypes.uints):
+            # Integer narrowing drops upper bytes; unsigned widening adds zero bytes, without float conversion.
+            values[u] = [typed_view(bytes(raw16(x, src_dtypes[0]))[:u.dtype.itemsize].ljust(u.dtype.itemsize, b"\0"), u.dtype)
+                         for x in src_values[0]]
```

A 64-bit shift uses two 32-bit words. For a shift n between 1 and 31:

```text
SHL: low' = low << n
     high' = (high << n) | (low >> (32-n))

SHR: low' = (low >> n) | (high << (32-n))
     high' = high >> n
```

For signed SHR, shift the high word arithmetically. At n>=32, only the high word contributes to SHR and only the low word contributes to SHL. The existing convolution helpers do the bit movement:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_u64_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
+    counts = list(map(scalar16, b))
+    if not counts or not all_same(counts) or not 0 <= counts[0] < 64:
+      raise NotImplementedError("ROCKCHIP 64-bit shift requires one uniform count in 0..63")
+    amount = counts[0]
+    raw = [bytes(raw16(x, dtype)) for x in a]
+    low, high = ([typed_view(x[i:i+4], dtypes.uint) for x in raw] for i in (0, 4))
+    zero = [typed_view(bytes(4), dtypes.uint)]*len(a)
+    def left(x:list, count:int) -> list: return self.run_u32_shift(Ops.SHL, x, [count]*len(x), dtypes.uint) if count else x
+    def right(x:list, count:int, signed:bool=False) -> list:
+      return self.run_u32_shift(Ops.SHR, x, [count]*len(x), dtypes.int if signed else dtypes.uint) if count else x
```

First handle left shift. Below 32 bits, combine the shifted high word with the low word's carry. At 32 or more, the low output is zero and only the original low word contributes to the high output.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_u64_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
@@
     def right(x:list, count:int, signed:bool=False) -> list:
       return self.run_u32_shift(Ops.SHR, x, [count]*len(x), dtypes.int if signed else dtypes.uint) if count else x
+    if op is Ops.SHL:
+      lo = left(low, amount) if amount < 32 else zero
+      hi = self.run_u32_bitwise(Ops.OR, left(high, amount), right(low, 32-amount), dtypes.uint) if 0 < amount < 32 else \
+        high if amount == 0 else left(low, amount-32)
```

For right shift, the direction is reversed. The high word supplies the low word's incoming bits. Signed inputs use arithmetic SHR for the high word, including the sign-filled case at counts 32..63.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_u64_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
@@
       hi = self.run_u32_bitwise(Ops.OR, left(high, amount), right(low, 32-amount), dtypes.uint) if 0 < amount < 32 else \
         high if amount == 0 else left(low, amount-32)
+    else:
+      signed = dtype in dtypes.sints
+      hi = right(high, min(amount, 31), signed) if signed or amount < 32 else zero
+      lo = self.run_u32_bitwise(Ops.OR, right(low, amount), left(high, 32-amount), dtypes.uint) if 0 < amount < 32 else \
+        low if amount == 0 else right(high, amount-32, signed)
```

Join the two completed four-byte results, low word first. The NPU has already moved the bits; Python only concatenates their storage.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_u64_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
@@
       lo = self.run_u32_bitwise(Ops.OR, right(low, amount), left(high, 32-amount), dtypes.uint) if 0 < amount < 32 else \
         low if amount == 0 else right(high, amount-32, signed)
+    return [typed_view(bytes(x)+bytes(y), dtype) for x,y in zip(lo, hi)]
```

AND/OR/XOR can process each 32-bit chunk independently:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_integer_bitwise(self, op:Ops, a:list, b:list, dtype:DType) -> list:
+    # Each 32-bit chunk is independent. Narrow integers use the low bytes of the same convolution.
+    result = [bytearray() for _ in a]
+    for offset in range(0, dtype.itemsize, 4):
+      lhs, rhs = ([typed_view(bytes(raw16(x, dtype))[offset:offset+4].ljust(4, b"\0"), dtypes.uint) for x in values]
+                  for values in (a, b))
+      for out,value in zip(result, self.run_u32_bitwise(op, lhs, rhs, dtypes.uint)): out.extend(value.cast("B"))
+    return [typed_view(bytes(x[:dtype.itemsize]), dtype) for x in result]
```

Add the dispatch:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op in (Ops.AND, Ops.XOR, Ops.OR) and u.dtype in (dtypes.int, dtypes.uint):
-            values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
+          elif u.op in (Ops.SHL, Ops.SHR) and u.dtype in (dtypes.int64, dtypes.uint64):
+            values[u] = self.run_u64_shift(u.op, src_values[0], src_values[1], u.dtype)
+          elif u.op in (Ops.AND, Ops.XOR, Ops.OR) and u.dtype in dtypes.ints:
+            values[u] = self.run_integer_bitwise(u.op, src_values[0], src_values[1], u.dtype)
```

The 32-bit wrappers already accept typed storage from the BITCAST step. Reuse them here; no second definition or register change is needed.

The direct THREEFRY check matched the CPU backend on five counter/key pairs, including full-width values. The JAX vector already recorded in test_randomness.py also matched all 20 words. That module could not collect here because hypothesis is missing; the same reference vector is checked in test/device/test_rockchip_integer.py without adding a dependency.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_threefry_jax_reference

1 passed in 10.12s
```

The 18 earlier arithmetic/comparison/shift regressions passed again in 35.57s.

The storage paths use typed memoryviews, not a wrapper class. BITCAST changes the view format; raw16 copies its bytes. These tutorial helpers copy completed results before reusing scratch memory; they do not yet reuse intermediate DMA addresses.

Check the storage and integer helpers reconstructed from these diffs:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_raw_bitcast \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_integer_boundaries \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_signed_division \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_wide_shifts \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_threefry_jax_reference

5 passed, 124 subtests passed in 34.43s
```

The raw-storage check round-trips half and wider encodings without unpacking/repacking NaN payloads. MULACC's rounding and signed-zero checks are recorded in its own step above.

Progress: **25 / 30** paths covered. Remaining: SQRT, EXP2, LOG2, POW and SIN. This is not every dtype or edge case, and the full test_ops.py sweep is still pending.

## Ops.SQRT

TOREVIEW1: `TRANSCENDENTAL=2` tells tinygrad to use its common math decompositions even when the renderer advertises a native handler. It also decomposes dependent math ops. At a genuinely earlier tutorial stage, leaving SQRT out of code_for_op selects that fallback without this override; here we use it to test the current runtime without its custom SQRT.

```bash
$ TRANSCENDENTAL=2 TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sqrt
```
TOREVIEW1: The separate forced-decomposition run, without TRACE, did not finish within 150 seconds. The diagnostic runner stopped it:

```text
TIMEOUT after 150s; no unittest result
```


This is a wall-time limit, not an observed wrong square root. Profile the decomposed path before blaming precision. The Newton probe below tests a separate candidate, not tinygrad's SQRT lowering.

Inspect the result before using the custom candidate below. A missing primitive is not evidence that the decomposition is inaccurate.

The 1500 branch uses 14 FP16 Newton steps. Repeating a rounded step may stop changing the answer before it reaches the correctly rounded root. Lets check that before copying the approach.

A bit-based starting estimate followed by three FP16 Newton steps still gave 8,030 wrong answers across all 31,743 positive finite FP16 values. Each was one ULP away. Repeating the half-precision arithmetic does not guarantee the correctly rounded root.

The one-ULP errors suggest checking the two neighbouring answers directly. Positive FP16 encodings have the same order as their values, so we can search for the largest y with y² <= x. The BS product mode tested for MULACC gives exact FP32 products of two half inputs.

Finding that y is not enough: we must decide whether y or its next neighbour is nearer. Compare x with the square of their midpoint, then use the even encoding for a tie:

| Stage          | Operation                                                           |
| -------------- | ------------------------------------------------------------------- |
| Input          | Convert FP16 x to FP32, retaining the original half bits            |
| Search         | Find the largest FP16 y with y² ≤ x                                 |
| Neighbour      | z is the next FP16 value after y; g = z - y                         |
| Midpoint       | m² = y² + y*g + (g/2)²                                              |
| Round          | x < m² selects y; x > m² selects z; a tie selects the even encoding |
| Special values | Keep ±0 and +inf; negative nonzero inputs return NaN                |

1. Search encodings from 0 to 0x5c00, which represents 256. sqrt(65504) is smaller than 256. Fifteen steps cover the interval.
2. Each search step runs an exact BS square, an INT32 comparison of the positive FP32 encodings, and INT32 selection. No Python comparison chooses a lane's answer.
3. The midpoint itself may not fit in FP16. Expand its square instead: y² + y*g + (g/2)². For positive finite half inputs, these products and their sum fit in FP32 exactly.
4. The final NPU selection also handles signed zero, infinities and NaNs. Python only copies bytes between packed-half and padded-word layouts.

Add the helper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_sqrt(self, a:list) -> list:
+    # Search ordered positive FP16 encodings; half*half is exact in the BS FP32 output.
+    base = self.dev.input_mem.dma_addr
+    result:list = []
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      slot, constants = 0, {}
+      def alloc(raw:bytes|None=None) -> int:
+        nonlocal slot
+        addr = base + slot*64
+        slot += 1
+        assert slot*64 <= self.dev.input_mem.size
+        if raw is not None: to_mv(self.dev.input_buf+addr-base, len(raw))[:] = raw
+        return addr
```

Cache constants in the same scratch area. calc uses the private task builder from MULACC; its arguments are DMA addresses, not lane values.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
         if raw is not None: to_mv(self.dev.input_buf+addr-base, len(raw))[:] = raw
         return addr
+      def const(value:int) -> int:
+        if value not in constants: constants[value] = alloc(struct.pack("<i", value)*8)
+        return constants[value]
+      def read(addr:int, size:int=32) -> bytes: return bytes(to_mv(self.dev.input_buf+addr-base, size))
+      def calc(algo:int, x:int, y:int|None=None, precision:int=4, output:int=4, **kw) -> int:
+        out = alloc()
+        self.mulacc_stage(algo, x, const(0) if y is None else y, out, precision, output, **kw)
+        return out
```

Use INT32 ADD/SUB/MUL and MIN+BINARY_EN for encoding arithmetic and selection. The search bounds fit in INT32, so selection cannot overflow.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
         self.mulacc_stage(algo, x, const(0) if y is None else y, out, precision, output, **kw)
         return out
+      def add(x:int, y:int) -> int: return calc(2, x, y)
+      def sub(x:int, y:int) -> int: return calc(4, x, y)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
```

The candidate encoding occupies a padded INT32 lane. half_bits copies its low two bytes into packed FP16 storage for BS MUL. half_output removes the output surface padding after a numeric FP32 → FP16 conversion.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
       def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def half_bits(bits:int) -> int:
+        raw = read(bits)
+        return alloc(b"".join(raw[i:i+2] for i in range(0, 32, 4)))
+      def half_output(x:int) -> int:
+        out = calc(-1, x, precision=5, output=2)
+        return alloc(read(out, 8)+read(out+16, 8))
+      def square(x:int) -> int: return calc(-1, x, precision=2, output=5, bs_mul=x)
```

Upload the original FP16 values and retain a second, padded copy of their encodings. Convert x to FP32 once for the square comparisons.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
         return alloc(read(out, 8)+read(out+16, 8))
       def square(x:int) -> int: return calc(-1, x, precision=2, output=5, bs_mul=x)
+      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8]) + bytes(2*(8-count))
+      source = alloc(raw)
+      source_bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = calc(-1, source, precision=2, output=5)
```

Search the encoding interval. Each midpoint is rounded upward; if its square exceeds x, lower the upper bound. Otherwise keep it as the new lower bound. Fifteen iterations find the largest representable y with y² ≤ x.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       source_bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
       x = calc(-1, source, precision=2, output=5)
+      lo, hi = const(0), const(0x5c00)  # sqrt(max finite half) < 256.
+      for _ in range(15):
+        mid = calc(2, lo, hi, shift=1)  # Positive ties round up: ceil((lo+hi)/2).
+        above = lt(x, square(half_bits(mid)))
+        lo, hi = select(above, lo, mid), select(above, sub(mid, const(1)), hi)
```

Now get y and its next representable neighbour z. Compute their gap in FP32, then use the exact expansion of the midpoint square:

```text
((y+z)/2)² = y² + y*(z-y) + ((z-y)/2)²
```

Only these small gap operands are converted to half for BS MUL. The square and sums stay FP32.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
         above = lt(x, square(half_bits(mid)))
         lo, hi = select(above, lo, mid), select(above, sub(mid, const(1)), hi)
+      lower, upper = half_bits(lo), half_bits(add(lo, const(1)))
+      lower32 = calc(-1, lower, precision=2, output=5)
+      upper32 = calc(-1, upper, precision=2, output=5)
+      gap = half_output(calc(4, upper32, lower32, precision=5, output=5))
+      # midpoint² = lower² + lower*gap + (gap/2)²; each product is exact FP32.
+      half_scale = alloc(struct.pack("<e", 0.5)*8)
+      half_gap = half_output(calc(-1, gap, precision=2, output=5, bs_mul=half_scale))
+      cross = calc(-1, lower, precision=2, output=5, bs_mul=gap)
+      midpoint2 = calc(2, calc(2, square(lower), cross, precision=5, output=5), square(half_gap), precision=5, output=5)
```

Compare x with the midpoint square. Below selects y, above selects z. At an exact tie, add y's low encoding bit: an odd y rounds up to the even z.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       cross = calc(-1, lower, precision=2, output=5, bs_mul=gap)
       midpoint2 = calc(2, calc(2, square(lower), cross, precision=5, output=5), square(half_gap), precision=5, output=5)
+      below, above = lt(x, midpoint2), lt(midpoint2, x)
+      tie = sub(sub(const(1), below), above)
+      parity = sub(lo, mul(calc(4, lo, const(1), shift=1), const(2)))
+      rounded = add(lo, add(above, mul(tie, parity)))
```

Handle special values with masks on the original encoding. Keep both zeros and +inf; negative nonzero values give NaN. These selections do not branch in Python.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       parity = sub(lo, mul(calc(4, lo, const(1), shift=1), const(2)))
       rounded = add(lo, add(above, mul(tie, parity)))
+      # Preserve signed zeros/+inf; negative nonzero inputs produce NaN. NaN payloads may pass through.
+      negative = lt(const(32767), source_bits)
+      magnitude = sub(source_bits, mul(const(32768), negative))
+      special = select(negative, const(0x7e00), source_bits)
+      special = select(lt(const(0), magnitude), special, source_bits)
+      valid = mul(sub(const(1), negative), mul(lt(const(0), magnitude), lt(magnitude, const(0x7c00))))
+      out = read(select(valid, rounded, special))
```

Read the low two bytes of each selected encoding. The NPU has already chosen the correctly rounded result.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       valid = mul(sub(const(1), negative), mul(lt(const(0), magnitude), lt(magnitude, const(0x7c00))))
       out = read(select(valid, rounded, special))
+      result.extend(typed_view(out[i*4:i*4+2], dtypes.half) for i in range(count))
+    return result
```

Advertise SQRT and dispatch it before the ordinary FP16 gate:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP, Ops.SQRT: CMP}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.MULACC and u.dtype == dtypes.half:
             values[u] = self.run_mulacc(*src_values)
+          elif u.op is Ops.SQRT and u.dtype == dtypes.half:
+            values[u] = self.run_sqrt(src_values[0])
```

This uses more tasks than Newton iteration. All 65,536 FP16 bit patterns passed: finite results matched the rounded reference bit-for-bit, including -0, and NaN classification matched too. The expanded midpoint square was also exact at all 20,480 relevant rounding boundaries.

The exhaustive test takes more than the repository's default two-minute timeout, so use TEST_TIMEOUT=600. It still runs serially; no NPU tests run together.

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_sqrt_all_half_patterns \
    test/backend/test_ops.py::TestOps::test_sqrt \
    test/backend/test_ops.py::TestOps::test_rsqrt

3 passed in 227.83s (0:03:47)
```

The ADD, MUL, maximum, MULACC and raw-NaN regressions also passed: 6 passed in 6.46s.

Progress: **26 / 30** paths covered. EXP2, LOG2, POW and SIN remain. The full test_ops.py sweep is still pending.

## Ops.EXP2

```bash
$ TRANSCENDENTAL=2 TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp2
```
TOREVIEW1: The fresh forced-decomposition run, without TRACE, passed:

```text
test_exp2 (__main__.TestOps.test_exp2) ... ok
Ran 1 test in 119.321s
OK
```

TOREVIEW1: This passed on the current runtime, not the runtime built so far in this tutorial. A four-input EXP2 probe called run_half_compare 24 times, but we only introduce that helper later under POW. It also used Python CAST fallback for half → INT16 and INT16 → half, four lanes each. No run_math calls were made: tinygrad's decomposition really ran, but with later support and CPU conversions. We still need to run it at this tutorial step before calling it an NPU-only baseline. The custom helper below is an alternative to investigate, not a required fix established by this pass.

Inspect the result before using the custom candidate below. A missing primitive is not evidence that the decomposition is inaccurate.

Next Ops.EXP2. A polynomial over the whole half range would be awkward. Split x into an integer n and a small fraction r instead: the exponent bits can supply 2^n, leaving only 2^r to approximate.

Use the FP32 stages from MULACC to avoid rounding every polynomial step to half. The recorded BS layout probe found that an FP32 main input needs four half multipliers at offset 0 and four at offset 16; a contiguous eight-half pack only produced the first four lanes.

```text
x = n + r
n = floor(x + 0.5)
2^x = 2^n * 2^r
```

1. Keep r in [-0.5, 0.5]. For a half input this reduced fraction fits in half exactly.
2. Evaluate the degree-eight exponential polynomial in FP32. Each BS MUL takes r as its half operand; EW ADD adds the next FP32 coefficient.
3. Add n*2^23 to the positive FP32 result's bits. This scales by 2^n without rounding to half early.
4. Convert to half once. Inputs at or below -25 round to zero; -24.5 still gives the smallest subnormal. Inputs at or above 16 overflow to infinity.

These steps give a candidate, not a rounding guarantee. Check all half encodings before calling it done; the correction found by that sweep is introduced after the initial output conversion below.

Start the shared helper with EXP2 only:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_math(self, op:Ops, a:list) -> list:
+    assert op is Ops.EXP2
+    base = self.dev.input_mem.dma_addr
+    coefficients = tuple(math.log(2)**k/math.factorial(k) for k in range(9))
+    result:list = []
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      slot, constants = 0, {}
+      def alloc(raw:bytes|None=None) -> int:
+        nonlocal slot
+        addr = base+slot*64
+        slot += 1
+        assert slot*64 <= self.dev.input_mem.size
+        if raw is not None: to_mv(self.dev.input_buf+addr-base, len(raw))[:] = raw
+        return addr
```

Cache each constant in its requested format. calc defaults to FP32 arithmetic through mulacc_stage; the output address belongs to that one task.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         if raw is not None: to_mv(self.dev.input_buf+addr-base, len(raw))[:] = raw
         return addr
+      def const(value:int|float, fmt:str="i") -> int:
+        key = (value, fmt)
+        if key not in constants: constants[key] = alloc(struct.pack("<"+fmt, value)*8)
+        return constants[key]
+      def read(addr:int, size:int=32) -> bytes: return bytes(to_mv(self.dev.input_buf+addr-base, size))
+      def calc(algo:int, x:int, y:int|None=None, precision:int=5, output:int=5, **kw) -> int:
+        out = alloc()
+        self.mulacc_stage(algo, x, const(0) if y is None else y, out, precision, output, **kw)
+        return out
```

The encoding calculations use INT32. exponent_scale multiplies n by 2²³, the FP32 exponent-field unit. These helpers submit NPU tasks; their arguments are addresses.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         self.mulacc_stage(algo, x, const(0) if y is None else y, out, precision, output, **kw)
         return out
+      def add(x:int, y:int) -> int: return calc(2, x, y, 4, 4)
+      def sub(x:int, y:int) -> int: return calc(4, x, y, 4, 4)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, 4, 4, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, 4, 4, binary=True)
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def exponent_scale(x:int) -> int: return mul(mul(mul(x, const(128)), const(256)), const(256))
```

BS reads four half multipliers from each 16-byte surface. Convert the reduced fraction, then copy it into that layout.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
       def exponent_scale(x:int) -> int: return mul(mul(mul(x, const(128)), const(256)), const(256))
+      def half_weights(value:int) -> int:
+        half = calc(-1, value, output=2)
+        return alloc(read(half, 8)+bytes(8)+read(half+16, 8))
```

Horner's method rewrites a polynomial as nested multiply-add steps. For example:
TOREVIEW1: No precision failure in tinygrad's EXP2 decomposition has been established; its test passed. Horner is just how this alternative evaluates its polynomial without separately building every power of r. tinygrad also uses polynomial evaluation, so Horner itself is not a reason to replace the default path. Compare the coefficient choice, arithmetic precision, accuracy and task cost first.

```text
c0 + c1*r + c2*r*r = (c2*r + c1)*r + c0
```

So we dont need to compute each power of r separately. Start at the highest coefficient, multiply by r, then add the next coefficient. Repeat down to c0, keeping every accumulated result in FP32.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         half = calc(-1, value, output=2)
         return alloc(read(half, 8)+bytes(8)+read(half+16, 8))
+      def polynomial(weights:int, cs:tuple[float, ...]) -> int:
+        value = const(cs[-1], "f")
+        for c in reversed(cs[:-1]):
+          value = calc(-1, value, bs_mul=weights)
+          value = calc(2, value, const(c, "f"))
+        return value
```

Upload x and save its original encoding for special-value checks. Convert the numeric input to FP32 once.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
           value = calc(2, value, const(c, "f"))
         return value
+      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8])+bytes(2*(8-count))
+      source = alloc(raw)
+      bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = calc(-1, source, precision=2)
```

Bound the working input to [-25, 16], then split it into integer n and fraction r. The final masks below handle values outside this interval.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
       x = calc(-1, source, precision=2)
+      bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
+      integer = calc(7, calc(2, bounded, const(0.5, "f")))
+      fraction = calc(4, bounded, integer)
+      exponent = calc(-1, integer, output=4)
+      weights = half_weights(fraction)
+      poly = polynomial(weights, coefficients)
+      scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
```

Convert the scaled FP32 answer to half only once. Keep its output encoding in INT32 lanes for the final rounding correction and special-value masks.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       poly = polynomial(weights, coefficients)
       scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
+      half = calc(-1, scaled, output=2)
+      packed = read(half, 8)+read(half+16, 8)
+      output_bits = alloc(b"".join(packed[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      negative = lt(const(32767), bits)
+      magnitude = sub(bits, mul(const(32768), negative))
```

The recorded full NPU sweep of this candidate found one wrong rounding: input 0x11c5 gave 0x3c00 instead of 0x3c01. So the FP32 polynomial is not enough at that boundary. The earlier investigation also linked LLVM's [exp2f16 exception table](https://raw.githubusercontent.com/llvm/llvm-project/main/libc/src/__support/math/exp2f16.h).

Add one output bit for that input, using NPU integer comparisons. Then select zero or infinity at the range limits. These are masks on the original input bits, not a Python calculation of EXP2.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       negative = lt(const(32767), bits)
       magnitude = sub(bits, mul(const(32768), negative))
+      # FP32 Horner lands on a half midpoint for this input; the exact exponential rounds upward.
+      exceptional = sub(sub(const(1), lt(bits, const(0x11c5))), lt(const(0x11c5), bits))
+      output_bits = add(output_bits, exceptional)
+      overflow = mul(sub(const(1), negative), lt(const(0x4bff), magnitude))
+      underflow = mul(negative, lt(const(0x4e3f), magnitude))
+      output_bits = select(overflow, const(0x7c00), select(underflow, const(0), output_bits))
```

NaN inputs stay NaN. Read the low two bytes of each selected encoding.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       underflow = mul(negative, lt(const(0x4e3f), magnitude))
       output_bits = select(overflow, const(0x7c00), select(underflow, const(0), output_bits))
+      output_bits = select(lt(const(0x7c00), magnitude), const(0x7e00), output_bits)
+      out = read(output_bits)
+      result.extend(typed_view(out[i*4:i*4+2], dtypes.half) for i in range(count))
+    return result
```

Advertise EXP2 and dispatch its FP16 input:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP, Ops.SQRT: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP, Ops.SQRT: CMP, Ops.EXP2: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.SQRT and u.dtype == dtypes.half:
             values[u] = self.run_sqrt(src_values[0])
+          elif u.op is Ops.EXP2 and u.dtype == dtypes.half:
+            values[u] = self.run_math(u.op, src_values[0])
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp2

test_exp2 (__main__.TestOps.test_exp2) ... ok

Ran 1 test in 14.065s

OK
```

The EXP2-only helper reconstructed from these diffs also passed all 65,536 half encodings and test_exp2: `2 passed in 54.35s`.

## Ops.LOG2

```bash
$ TRANSCENDENTAL=2 TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log2
```
TOREVIEW1: This time the separate forced-decomposition run, without TRACE, finished and exceeded the test tolerance. Excerpt:

```text
 [10, 18]: -0.225830078125 (ACTUAL), -0.22607421875 (DESIRED)
Max absolute difference among violations: 0.0007324
Max relative difference among violations: 0.00156
Ran 1 test in 126.416s
FAILED (errors=1)
```

The earlier pytest batch stopped at its 120-second timeout; this standalone run was allowed 150 seconds. We now have an accuracy failure in the decomposed path on this backend, not just a timeout. Trace the intermediate operations or compare another backend before attributing it to the polynomial itself. This probe uses current primitive support, so it is not a promise that the earlier tutorial code reaches the same result.

Inspect the result before using the custom candidate below. A missing primitive is not evidence that the decomposition is inaccurate.

Next add Ops.LOG2. We can reverse the exponent split: FP32 conversion normalizes half subnormals, so x becomes m*2^n with m in [1, 2). The exponent gives n exactly. Only log2(m) needs approximation.

The series around 1 converges faster if m stays near 1. Halve the upper part of [1, 2) and increment n; the value of x stays unchanged:

```text
x = m * 2^n
log2(x) = n + log2(m)
r = m - 1
log2(m) = r * (1 - r/2 + r²/3 - ...) / ln(2)
```

1. Convert to FP32 first, which also normalizes half subnormals. Read the exponent through the INT32 datapath.
2. Bring m into [1/sqrt(2), sqrt(2)] by halving the upper part of [1, 2) and adding one to n.
3. Evaluate the 16-term polynomial in FP32, then add n. The reduced r fits in half exactly, so BS does not lose its bits.
4. Select -inf for either zero, NaN for negative nonzero inputs, and +inf for +inf. These selections also run on the NPU.

Select the logarithm coefficients. The shared polynomial evaluator stays unchanged.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
-    assert op is Ops.EXP2
+    assert op in (Ops.EXP2, Ops.LOG2)
     base = self.dev.input_mem.dma_addr
-    coefficients = tuple(math.log(2)**k/math.factorial(k) for k in range(9))
+    coefficients = (tuple(math.log(2)**k/math.factorial(k) for k in range(9)) if op is Ops.EXP2 else
+                    tuple((-1)**k/((k+1)*math.log(2)) for k in range(16)))
     result:list = []
     for start in range(0, len(a), 8):
```

Extract the exponent from the positive FP32 encoding, restore the mantissa, and reduce it around 1. After evaluating the polynomial, multiply by r and add the exponent.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
       x = calc(-1, source, precision=2)
-      bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
-      integer = calc(7, calc(2, bounded, const(0.5, "f")))
-      fraction = calc(4, bounded, integer)
-      exponent = calc(-1, integer, output=4)
+      if op is Ops.EXP2:
+        bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
+        integer = calc(7, calc(2, bounded, const(0.5, "f")))
+        fraction = calc(4, bounded, integer)
+        exponent = calc(-1, integer, output=4)
+      else:
+        bounded = calc(1, calc(0, calc(5, x), const(2**-24, "f")), const(65504., "f"))
+        # FP32 conversion normalizes half subnormals. Extract exponent and restore a [1,2) mantissa.
+        exponent = calc(4, bounded, const(4194304), 4, 4, shift=23)
+        mantissa = add(sub(bounded, exponent_scale(exponent)), const(1., "f"))
+        upper = lt(const(math.sqrt(2), "f"), mantissa)
+        mantissa = sub(mantissa, mul(const(8388608), upper))
+        exponent = add(sub(exponent, const(127)), upper)
+        fraction = calc(4, mantissa, const(1., "f"))
+      # The EXP2/LOG2 reduced fraction is exactly representable in half.
       weights = half_weights(fraction)
       poly = polynomial(weights, coefficients)
```

The reduced fraction uses the same half-weight layout and Horner evaluator. LOG2 multiplies that polynomial by r, then adds n after converting the integer exponent to FP32:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       weights = half_weights(fraction)
       poly = polynomial(weights, coefficients)
-      scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
+      if op is Ops.EXP2:
+        scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
+      else:
+        scaled = calc(2, calc(-1, poly, bs_mul=weights), calc(-1, exponent, precision=4))
       half = calc(-1, scaled, output=2)
       packed = read(half, 8)+read(half+16, 8)
```

Keep EXP2's masks in its branch. LOG2 selects +inf for +inf, NaN for negative inputs, and -inf for either zero. The common final mask handles input NaNs.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       negative = lt(const(32767), bits)
       magnitude = sub(bits, mul(const(32768), negative))
-      # FP32 Horner lands on a half midpoint for this input; the exact exponential rounds upward.
-      exceptional = sub(sub(const(1), lt(bits, const(0x11c5))), lt(const(0x11c5), bits))
-      output_bits = add(output_bits, exceptional)
-      overflow = mul(sub(const(1), negative), lt(const(0x4bff), magnitude))
-      underflow = mul(negative, lt(const(0x4e3f), magnitude))
-      output_bits = select(overflow, const(0x7c00), select(underflow, const(0), output_bits))
+      if op is Ops.EXP2:
+        # FP32 Horner lands on a half midpoint for this input; the exact exponential rounds upward.
+        exceptional = sub(sub(const(1), lt(bits, const(0x11c5))), lt(const(0x11c5), bits))
+        output_bits = add(output_bits, exceptional)
+        overflow = mul(sub(const(1), negative), lt(const(0x4bff), magnitude))
+        underflow = mul(negative, lt(const(0x4e3f), magnitude))
+        output_bits = select(overflow, const(0x7c00), select(underflow, const(0), output_bits))
+      else:
+        output_bits = select(lt(const(0x7bff), magnitude), const(0x7c00), output_bits)
+        output_bits = select(negative, const(0x7e00), output_bits)
+        output_bits = select(lt(const(0), magnitude), output_bits, const(0xfc00))
       output_bits = select(lt(const(0x7c00), magnitude), const(0x7e00), output_bits)
       out = read(output_bits)
```

Enable LOG2:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP, Ops.SQRT: CMP, Ops.EXP2: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP,
+           Ops.SQRT: CMP, Ops.EXP2: CMP, Ops.LOG2: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op is Ops.EXP2 and u.dtype == dtypes.half:
+          elif u.op in (Ops.EXP2, Ops.LOG2) and u.dtype == dtypes.half:
             values[u] = self.run_math(u.op, src_values[0])
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log2

test_log2 (__main__.TestOps.test_log2) ... ok

Ran 1 test in 18.448s

OK
```

The EXP2+LOG2 stage passed all 65,536 LOG2 inputs and test_log2: `2 passed in 72.62s`.

## Ops.SIN

```bash
$ TRANSCENDENTAL=2 TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin
```
TOREVIEW1: The separate forced-decomposition run, without TRACE, gives:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.SHR with dtypes.ushort
Ran 1 test in 0.233s
FAILED (errors=1)
```



So first investigate UINT16 SHR support, then rerun the existing decomposition. We have not reached a sine accuracy failure; this does not justify replacing it with the custom polynomial or a LUT yet.

Inspect the result before using the custom candidate below. A missing primitive is not evidence that the decomposition is inaccurate.

The custom candidate below uses range reduction and polynomials. The checked 1500 branch also uses a polynomial in `_dpu_sin`, not a LUT, and clamps input to ±10000. That clamp changes larger finite half inputs, but does not prove tinygrad's own decomposition needs replacing. Compare the baseline first; a LUT is another candidate to probe, not an approach already ruled out.

Subtract the nearest multiple of pi/2. The remaining angle is small enough for sine/cosine polynomials; the multiple tells us the quadrant:

```text
q = floor(x * 2/pi + 0.5)
r = x - q * pi/2
q mod 4 selects sin(r), cos(r), -sin(r), or -cos(r)
```

1. Split q into a multiple of 256 and its remainder. Both pieces fit exactly in half.
2. Split pi/2 into three constants and subtract their products in FP32. Splitting q keeps the first two products exact.
3. Evaluate sine and cosine polynomials on the reduced half angle. Keep the remaining FP32 angle error e, then correct with sin(r+e) ≈ sin(r)+e*cos(r) and cos(r+e) ≈ cos(r)-e*sin(r).
4. Select the quadrant and sign on the NPU. Keep sin(-0) = -0; infinities and NaNs produce NaN.

Large angles make range reduction the risky part: rounding q or q*pi/2 too early loses the small remainder. Split q and pi/2 before multiplication, retain the reduction error, then check every half encoding. The recorded boundary failures and their correction come after this candidate below.

Allow SIN through the same helper.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
-    assert op in (Ops.EXP2, Ops.LOG2)
+    assert op in (Ops.EXP2, Ops.LOG2, Ops.SIN)
     base = self.dev.input_mem.dma_addr
     coefficients = (tuple(math.log(2)**k/math.factorial(k) for k in range(9)) if op is Ops.EXP2 else
```

SIN also needs multipliers made from NPU-produced half encodings. Add word_weights for that layout, and let Horner multiply twice per step for polynomials in r².

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         half = calc(-1, value, output=2)
         return alloc(read(half, 8)+bytes(8)+read(half+16, 8))
-      def polynomial(weights:int, cs:tuple[float, ...]) -> int:
+      def word_weights(value:int) -> int:
+        raw = read(value)
+        return alloc(b"".join(raw[i:i+2] for i in range(0, 16, 4))+bytes(8)+
+                     b"".join(raw[i:i+2] for i in range(16, 32, 4)))
+      def polynomial(weights:int, cs:tuple[float, ...], squared:bool=False) -> int:
         value = const(cs[-1], "f")
         for c in reversed(cs[:-1]):
           value = calc(-1, value, bs_mul=weights)
+          if squared: value = calc(-1, value, bs_mul=weights)
           value = calc(2, value, const(c, "f"))
         return value
```

Reduce x by q*pi/2, using the split q and split constants. Then evaluate the sine/cosine polynomials, correct the reduced-angle residual, and select the quadrant.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
       x = calc(-1, source, precision=2)
-      if op is Ops.EXP2:
+      if op is Ops.SIN:
+        multiple = calc(7, calc(2, calc(-1, const(2/math.pi, "f"), bs_mul=half_weights(x)), const(0.5, "f")))
+        q = calc(-1, multiple, output=4)
+        # Split the integer multiple so the first two pi/2 products are exact FP32.
+        qhi = mul(calc(4, mul(q, const(2)), const(255), 4, 4, shift=9), const(256))
+        qlo = sub(q, qhi)
+        qweights = [half_weights(calc(-1, part, precision=4)) for part in (qhi, qlo)]
+        reduced = x
+        for c in (1.5703125, 0.0004837512969970703125, math.pi/2-1.5703125-0.0004837512969970703125):
+          for w in qweights: reduced = calc(4, reduced, calc(-1, const(c, "f"), bs_mul=w))
+      elif op is Ops.EXP2:
         bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
         integer = calc(7, calc(2, bounded, const(0.5, "f")))
```

Next evaluate sin(r) and cos(r), then restore the FP32 reduction residual e. Rounding r to half alone loses useful bits; the first-order corrections are sin(r)+e*cos(r) and cos(r)-e*sin(r).

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         for c in (1.5703125, 0.0004837512969970703125, math.pi/2-1.5703125-0.0004837512969970703125):
           for w in qweights: reduced = calc(4, reduced, calc(-1, const(c, "f"), bs_mul=w))
+        weights = half_weights(reduced)
+        residual = calc(4, reduced, calc(-1, const(1., "f"), bs_mul=weights))
+        sine = calc(-1, polynomial(weights, (1., -1/6, 1/120, -1/5040, 1/362880, -1/39916800), True), bs_mul=weights)
+        cosine = polynomial(weights, (1., -1/2, 1/24, -1/720, 1/40320, -1/3628800), True)
+        # Keep the full FP32 reduction residual instead of rounding the angle to half.
+        sin_r = calc(2, sine, calc(-1, residual, bs_mul=half_weights(cosine)))
+        cos_r = calc(4, cosine, calc(-1, residual, bs_mul=half_weights(sine)))
       elif op is Ops.EXP2:
```

Finally select the quadrant on the NPU:

| q mod 4 | Result    |
| ------: | --------- |
|       0 | sin(r+e)  |
|       1 | cos(r+e)  |
|       2 | -sin(r+e) |
|       3 | -cos(r+e) |

The odd-quadrant mask selects sine or cosine. A half multiplier with the selected sign then applies +1 or -1.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         sin_r = calc(2, sine, calc(-1, residual, bs_mul=half_weights(cosine)))
         cos_r = calc(4, cosine, calc(-1, residual, bs_mul=half_weights(sine)))
+        odd = sub(q, mul(calc(4, mul(q, const(2)), const(1), 4, 4, shift=2), const(2)))
+        quadrant = sub(q, mul(calc(4, mul(q, const(2)), const(3), 4, 4, shift=3), const(4)))
+        sin_weight = word_weights(mul(const(0x3c00), sub(const(1), odd)))
+        cos_weight = word_weights(mul(const(0x3c00), odd))
+        scaled = calc(2, calc(-1, sin_r, bs_mul=sin_weight), calc(-1, cos_r, bs_mul=cos_weight))
+        sign_weight = word_weights(add(const(0x3c00), mul(const(32768), lt(const(1), quadrant))))
+        scaled = calc(-1, scaled, bs_mul=sign_weight)
       elif op is Ops.EXP2:
```

SIN has already produced scaled. Keep the EXP2/LOG2 polynomial path separate.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
         exponent = add(sub(exponent, const(127)), upper)
         fraction = calc(4, mantissa, const(1., "f"))
-      # The EXP2/LOG2 reduced fraction is exactly representable in half.
-      weights = half_weights(fraction)
-      poly = polynomial(weights, coefficients)
-      if op is Ops.EXP2:
-        scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
-      else:
-        scaled = calc(2, calc(-1, poly, bs_mul=weights), calc(-1, exponent, precision=4))
+      if op is not Ops.SIN:
+        # The EXP2/LOG2 reduced fraction is exactly representable in half.
+        weights = half_weights(fraction)
+        poly = polynomial(weights, coefficients)
+        if op is Ops.EXP2:
+          scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
+        else:
+          scaled = calc(2, calc(-1, poly, bs_mul=weights), calc(-1, exponent, precision=4))
       half = calc(-1, scaled, output=2)
       packed = read(half, 8)+read(half+16, 8)
```

The recorded sweep of this candidate still found six one-ULP errors: magnitudes 0x32b3, 0x51f5 and 0x5cb0, and their negatives. The earlier investigation linked LLVM's [sinf16 exception table](https://raw.githubusercontent.com/llvm/llvm-project/main/libc/src/__support/math/sinf16.h) for the latter two.

Correct those observed boundaries symmetrically by magnitude. Keep negative zero unchanged; infinities and NaNs give NaN. The exhaustive check below is what validates the resulting half path, not the polynomial formula alone.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       negative = lt(const(32767), bits)
       magnitude = sub(bits, mul(const(32768), negative))
-      if op is Ops.EXP2:
+      if op is Ops.SIN:
+        # Three rounding boundaries remain after FP32 reduction; the correction is symmetric in the sign.
+        for code, correction in ((0x32b3, 1), (0x51f5, -1), (0x5cb0, -1)):
+          equal = sub(sub(const(1), lt(magnitude, const(code))), lt(const(code), magnitude))
+          output_bits = add(output_bits, mul(const(correction), equal))
+        output_bits = select(lt(const(0), magnitude), output_bits, bits)  # Keep sin(-0) = -0.
+        output_bits = select(lt(const(0x7bff), magnitude), const(0x7e00), output_bits)
+      elif op is Ops.EXP2:
         # FP32 Horner lands on a half midpoint for this input; the exact exponential rounds upward.
         exceptional = sub(sub(const(1), lt(bits, const(0x11c5))), lt(const(0x11c5), bits))
```

Enable SIN:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.SQRT: CMP, Ops.EXP2: CMP, Ops.LOG2: CMP}
+           Ops.SQRT: CMP, Ops.EXP2: CMP, Ops.LOG2: CMP, Ops.SIN: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op in (Ops.EXP2, Ops.LOG2) and u.dtype == dtypes.half:
+          elif u.op in (Ops.EXP2, Ops.LOG2, Ops.SIN) and u.dtype == dtypes.half:
             values[u] = self.run_math(u.op, src_values[0])
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin

test_sin (__main__.TestOps.test_sin) ... ok

Ran 1 test in 28.918s

OK
```

The helper reconstructed through this SIN step passed all 65,536 SIN inputs and test_sin: `2 passed in 117.79s`.

The earlier EXP2/LOG2 run passed both exhaustive checks and test_exp2/test_log2, but test_exp and test_log failed: **2 failed, 4 passed in 138.55s**. test_exp reaches unsupported FP32 MUL because Tensor.exp promotes its calculation. test_log loses accuracy when the rounded half LOG2 result is multiplied by the half ln(2) constant. Passing the primitive tests does not mean these composed functions pass.

Rerun all three primitive checks after the rounding corrections:

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_exp2_all_half_patterns \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_log2_all_half_patterns \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_sin_all_half_patterns \
    test/backend/test_ops.py::TestOps::test_exp2 \
    test/backend/test_ops.py::TestOps::test_log2 \
    test/backend/test_ops.py::TestOps::test_sin

6 passed in 234.77s (0:03:54)
```

Each exhaustive check covers all 65,536 FP16 bit patterns. Non-NaN outputs match the reference bits, including signed zeros and infinities; NaN classification matches too. Arithmetic, range reduction and rounding corrections run on the NPU. Python only sets constants, submits tasks and copies storage layouts.

Progress: **29 / 30** paths covered. POW remains, followed by the full test_ops.py sweep. FP32 promotion and the composed log accuracy issue above are still open.

## Ops.POW

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow
```
TOREVIEW1: The current-runtime run, without TRACE, did not finish within the diagnostic's 150-second limit:

```text
TIMEOUT after 150s; no unittest result
```


This run includes the later LOG2/EXP2 and WHERE fixes, so it is not a pre-change baseline. It gives no new accuracy result. The small recorded probe below remains the evidence for the old WHERE problem; do not claim this timed-out test reproduced it.

POW is already expanded by tinygrad's symbolic rewrite into LOG2, MUL, EXP2 and WHERE. Before adding a POW handler, try that existing path. The recorded small probe gave NaN even for 2^3 instead of 8.

Why? The old arithmetic WHERE multiplies the unused branch by zero, but 0*NaN is still NaN. A NaN exponent also reached the Python float-to-integer CAST and raised ValueError. Fix selection first, then rerun; adding POW to ops_map would not repair either problem.

First fix WHERE. Reinterpret each half as UINT16, select its storage bits with our existing integer WHERE, then reinterpret the selected bits as half. There is no floating-point arithmetic on the two branches:

```diff
 class RockchipRenderer(Renderer):
@@
   @staticmethod
   def _pm_lower_where(u:UOp) -> UOp:
     x, a, b = u.src
     a, b = a.cast(dtypes.half), b.cast(dtypes.half)
-    mask = x.cast(dtypes.half)
-    positive = a.alu(Ops.MUL, mask)
-    inverse = b.alu(Ops.MUL, mask.const_like(1).alu(Ops.SUB, mask))
-    return positive.alu(Ops.ADD, inverse).cast(u.dtype)
+    # Select storage bits: an unselected NaN must not contaminate the chosen value.
+    return x.where(a.bitcast(dtypes.uint16), b.bitcast(dtypes.uint16)).bitcast(u.dtype)
@@
-    # Integer WHERE now selects exact limbs; keep this arithmetic rule for FP16 only.
+    # FP16 raw-bit selection reuses the exact integer WHERE path.
-    (UPat(Ops.WHERE, dtypes.half, src=(UPat(dtype=dtypes.bool), UPat(dtype=dtypes.half), UPat(dtype=dtypes.half)), name="u"),
+    (UPat(Ops.WHERE, dtypes.half, src=(UPat(dtype=dtypes.bool),
+      UPat(dtype=(dtypes.half, dtypes.weakfloat)), UPat(dtype=(dtypes.half, dtypes.weakfloat))), name="u"),
```

Keep weakfloat constants in the matcher too: _pm_lower_where casts both branches to half before reinterpreting their bits.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_where_special_bits \
    test/backend/test_ops.py::TestOps::test_where

2 passed in 4.63s
```

The storage check includes both signed zeros, infinities and NaN payloads in either branch. The same positive-power probe now gives [8, 0.25, 1.732, 2]. Negative bases and -inf with a fractional exponent also gave the expected values in the small probe. The NaN-exponent CAST error is still open.

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/backend/test_ops.py::TestOps::test_pow_full \
    test/backend/test_ops.py::TestOps::test_pow_neg_inf_frac_exponent \
    test/backend/test_ops.py::TestOps::test_pow_zero_exponent

FAILED test/backend/test_ops.py::TestOps::test_pow_full
Mismatched elements: 17 / 2925 (0.581%)
Max relative difference among violations: 0.002638
1 failed, 2 passed in 96.24s (0:01:36)
```

WHERE no longer contaminates the result with an unused NaN. The full POW case now reaches a precision failure: separately rounded half LOG2 and MUL lose too much accuracy before EXP2. We need to retain wider intermediates for this composition; the tolerance is unchanged.

### Keep the POW intermediates in FP32

The existing xpow formula can stay. Match its magnitude expression after the general rewrites:

```text
LOG2(half) → FP32 result
          → BS MUL by the half exponent
          → EXP2 with an FP32 input
          → half result
```

1. LOG2 keeps its FP32 result instead of rounding to half.
2. BS multiplies that FP32 value by the original half exponent.
3. EXP2 still reduces its input to n+r. Now r may not fit in half, so keep the discarded FP32 residual and apply the polynomial derivative correction.
4. Only the final result is rounded to half. The surrounding WHERE expressions still implement the original xpow special cases.

This is tagged EXP2_MUL_LOG2, not a new hardware POW instruction. Constants are prepared on the host; all three operations run on the NPU.

Allow only the two wider modes needed here: LOG2 may return FP32, and EXP2 may accept FP32. Other calls still use FP16.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_math(self, op:Ops, a:list) -> list:
+  def run_math(self, op:Ops, a:list, input_dtype:DType=dtypes.half, dtype:DType=dtypes.half) -> list:
     assert op in (Ops.EXP2, Ops.LOG2, Ops.SIN)
+    assert input_dtype == dtypes.half or (op is Ops.EXP2 and input_dtype == dtypes.float)
+    assert dtype == dtypes.half or (op is Ops.LOG2 and dtype == dtypes.float)
     base = self.dev.input_mem.dma_addr
     coefficients = (tuple(math.log(2)**k/math.factorial(k) for k in range(9)) if op is Ops.EXP2 else
```

Selection now has to handle full FP32 encodings. A signed INT32 subtraction can overflow between encodings of different signs. Select the magnitude and sign separately, then rejoin them.

Removing a sign bit uses two additions of 2³⁰. This avoids subtracting INT32_MIN, which the probe found did not behave as ordinary subtraction on this datapath.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list, input_dtype:DType=dtypes.half, dtype:DType=dtypes.half) -> list:
@@
       def mul(x:int, y:int) -> int: return calc(0, x, y, 4, 4, mul=True)
       def lt(x:int, y:int) -> int: return calc(1, x, y, 4, 4, binary=True)
-      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def select(mask:int, yes:int, no:int) -> int:
+        if dtype != dtypes.float: return add(no, mul(sub(yes, no), mask))
+        # Select full FP32 bit patterns without overflowing a signed INT32 difference.
+        sy, sn = lt(yes, const(0)), lt(no, const(0))
+        hy, hn = mul(const(1073741824), sy), mul(const(1073741824), sn)
+        my, mn = add(add(yes, hy), hy), add(add(no, hn), hn)
+        magnitude = add(mn, mul(sub(my, mn), mask))
+        sign = add(sn, mul(sub(sy, sn), mask))
+        return add(magnitude, mul(const(-2147483648), sign))
       def exponent_scale(x:int) -> int: return mul(mul(mul(x, const(128)), const(256)), const(256))
       def half_weights(value:int) -> int:
```

FP32 input is already in the word layout. Keep it there; half input still uses the existing conversion and padded encoding copy.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list, input_dtype:DType=dtypes.half, dtype:DType=dtypes.half) -> list:
@@
           value = calc(2, value, const(c, "f"))
         return value
-      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8])+bytes(2*(8-count))
+      raw = b"".join(raw16(x, input_dtype) for x in a[start:start+8])+bytes(input_dtype.itemsize*(8-count))
       source = alloc(raw)
-      bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
-      x = calc(-1, source, precision=2)
+      bits = source if input_dtype == dtypes.float else alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = source if input_dtype == dtypes.float else calc(-1, source, precision=2)
       if op is Ops.SIN:
         multiple = calc(7, calc(2, calc(-1, const(2/math.pi, "f"), bs_mul=half_weights(x)), const(0.5, "f")))
```

For a wider EXP2 input, the fraction may lose bits when converted to a half multiplier. Keep the residual e and add `e * P'(r)` to the FP32 polynomial result before exponent scaling.

LOG2's FP32 output must skip the final half conversion. Its sign/magnitude masks and special-value encodings also need the 32-bit format.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list, input_dtype:DType=dtypes.half, dtype:DType=dtypes.half) -> list:
@@
         poly = polynomial(weights, coefficients)
         if op is Ops.EXP2:
+          if input_dtype == dtypes.float:
+            # A composed POW exponent is FP32: keep the residual discarded by the half multiplier.
+            residual = calc(4, fraction, calc(-1, const(1., "f"), bs_mul=weights))
+            derivative = polynomial(weights, tuple(k*coefficients[k] for k in range(1, len(coefficients))))
+            poly = calc(2, poly, calc(-1, residual, bs_mul=half_weights(derivative)))
           scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
         else:
           scaled = calc(2, calc(-1, poly, bs_mul=weights), calc(-1, exponent, precision=4))
-      half = calc(-1, scaled, output=2)
-      packed = read(half, 8)+read(half+16, 8)
-      output_bits = alloc(b"".join(packed[i:i+2]+bytes(2) for i in range(0, 16, 2)))
-      negative = lt(const(32767), bits)
-      magnitude = sub(bits, mul(const(32768), negative))
+      if dtype == dtypes.float: output_bits = scaled
+      else:
+        half = calc(-1, scaled, output=2)
+        packed = read(half, 8)+read(half+16, 8)
+        output_bits = alloc(b"".join(packed[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      negative = lt(bits, const(0)) if input_dtype == dtypes.float else lt(const(32767), bits)
+      if input_dtype == dtypes.float:
+        sign_half = mul(const(1073741824), negative)
+        magnitude = add(add(bits, sign_half), sign_half)
+      else: magnitude = sub(bits, mul(const(32768), negative))
+      infinity, nan = (0x7f800000, 0x7fc00000) if dtype == dtypes.float else (0x7c00, 0x7e00)
       if op is Ops.SIN:
         # Three rounding boundaries remain after FP32 reduction; the correction is symmetric in the sign.
```

Use the appropriate input encoding for the EXP2 correction and overflow boundaries. LOG2 chooses infinity, NaN and negative infinity in its requested output format. Finally copy dtype.itemsize bytes per result.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list, input_dtype:DType=dtypes.half, dtype:DType=dtypes.half) -> list:
@@
       elif op is Ops.EXP2:
         # FP32 Horner lands on a half midpoint for this input; the exact exponential rounds upward.
-        exceptional = sub(sub(const(1), lt(bits, const(0x11c5))), lt(const(0x11c5), bits))
+        code = 0x3a38a000 if input_dtype == dtypes.float else 0x11c5
+        exceptional = sub(sub(const(1), lt(bits, const(code))), lt(const(code), bits))
         output_bits = add(output_bits, exceptional)
-        overflow = mul(sub(const(1), negative), lt(const(0x4bff), magnitude))
-        underflow = mul(negative, lt(const(0x4e3f), magnitude))
+        overflow = mul(sub(const(1), negative), lt(const(0x417fffff if input_dtype == dtypes.float else 0x4bff), magnitude))
+        underflow = mul(negative, lt(const(0x41c7ffff if input_dtype == dtypes.float else 0x4e3f), magnitude))
         output_bits = select(overflow, const(0x7c00), select(underflow, const(0), output_bits))
       else:
-        output_bits = select(lt(const(0x7bff), magnitude), const(0x7c00), output_bits)
-        output_bits = select(negative, const(0x7e00), output_bits)
-        output_bits = select(lt(const(0), magnitude), output_bits, const(0xfc00))
-      output_bits = select(lt(const(0x7c00), magnitude), const(0x7e00), output_bits)
+        output_bits = select(lt(const(0x7bff), magnitude), const(infinity), output_bits)
+        output_bits = select(negative, const(nan), output_bits)
+        output_bits = select(lt(const(0), magnitude), output_bits, const(-8388608 if dtype == dtypes.float else 0xfc00))
+      output_bits = select(lt(const(0x7f800000 if input_dtype == dtypes.float else 0x7c00), magnitude), const(nan), output_bits)
       out = read(output_bits)
-      result.extend(typed_view(out[i*4:i*4+2], dtypes.half) for i in range(count))
+      result.extend(typed_view(out[i*4:i*4+dtype.itemsize], dtype) for i in range(count))
     return result
```

Add the composed helper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_exp2_mul_log2(self, a:list, b:list) -> list:
+    assert len(a) == len(b)
+    logarithms = self.run_math(Ops.LOG2, a, dtype=dtypes.float)
+    products:list = []
+    base = self.dev.input_mem.dma_addr
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      to_mv(self.dev.input_buf, 32)[:] = b"".join(raw16(x, dtypes.float) for x in logarithms[start:start+8])+bytes(4*(8-count))
+      weights = b"".join(raw16(x, dtypes.half) for x in b[start:start+8])+bytes(2*(8-count))
+      to_mv(self.dev.input_buf+64, 24)[:] = weights[:8]+bytes(8)+weights[8:]
+      self.mulacc_stage(-1, base, base+64, base+128, bs_mul=base+64)
+      raw = bytes(to_mv(self.dev.input_buf+128, 32))
+      products.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
+    return self.run_math(Ops.EXP2, products, input_dtype=dtypes.float)
```

The matcher accepts either MUL operand order:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Retain FP32 intermediates for the magnitude expression generated by xpow.
+    (UPat(Ops.EXP2, dtypes.half, src=(UPat(Ops.MUL, dtypes.half,
+      src=[UPat(Ops.LOG2, dtypes.half, src=(UPat.var("a", dtypes.half),)), UPat.var("b", dtypes.half)]),)),
+     lambda a,b: UOp(Ops.CUSTOM, src=(a,b), arg=("EXP2_MUL_LOG2", dtypes.half))),
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
         elif u.op is Ops.CUSTOM:
+          if u.arg == ("EXP2_MUL_LOG2", dtypes.half) and u.dtype == dtypes.half and src_dtypes == [dtypes.half]*2:
+            values[u] = self.run_exp2_mul_log2(*src_values)
+            i += 1
+            continue
```

### Ops.CAST: remove the Python integer conversion

POW checks whether its exponent is an integer. That CAST still used Python and raised on NaN.

The FP32→INT32 converter rounds to nearest. Toggling CVT_TYPE and CVT_ROUND did not change it. For finite values, remove the fraction first:

```text
trunc(x) = max(floor(x), min(ceil(x), 0))
         → FP32-to-INT32 converter
```

FLOOR, CEIL, MIN and MAX run in FP32 here. Converting the resulting integer-valued float no longer changes the answer. Nonfinite/out-of-range conversion still follows the hardware converter's saturation behavior; this is not a claim that every backend defines those CASTs identically.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
+    assert src_dtype in (dtypes.half, dtypes.float, dtypes.int) and dtype in (dtypes.half, dtypes.float, dtypes.int)
+    result:list = []
+    base = self.dev.input_mem.dma_addr
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      to_mv(self.dev.input_buf, 8*src_dtype.itemsize)[:] = b"".join(raw16(x, src_dtype) for x in a[start:start+8]) + \
+        bytes(src_dtype.itemsize*(8-count))
+      to_mv(self.dev.input_buf+64, 32)[:] = bytes(32)
```

Convert the source to FP32 first. The precision number selects how to read the input bytes: 2 for FP16, 4 for INT32, 5 for FP32.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
@@
         bytes(src_dtype.itemsize*(8-count))
       to_mv(self.dev.input_buf+64, 32)[:] = bytes(32)
+      precision = {dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[src_dtype]
+      self.mulacc_stage(-1, base, base+64, base+128, precision=precision)
+      value = base+128
```

For an INT32 destination, calculate truncation in FP32 first. Algorithm 7 is FLOOR, 8 is CEIL, 1 is MIN, and 0 is MAX; the expression is the same TRUNC formula from above.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
@@
       self.mulacc_stage(-1, base, base+64, base+128, precision=precision)
       value = base+128
+      if dtype == dtypes.int:
+        # The converter rounds to nearest. Remove the fraction before converting to get truncation toward zero.
+        self.mulacc_stage(7, value, base+64, base+192)
+        self.mulacc_stage(8, value, base+64, base+256)
+        self.mulacc_stage(1, base+256, base+64, base+320)
+        self.mulacc_stage(0, base+192, base+320, base+384)
+        value = base+384
```

Now convert the integer-valued float to the requested output format. FP16 keeps the two-surface layout; INT32/FP32 each occupy 32 contiguous output bytes.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
@@
         self.mulacc_stage(0, base+192, base+320, base+384)
         value = base+384
+      self.mulacc_stage(-1, value, base+64, base+448, output={dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[dtype])
+      raw = (bytes(to_mv(self.dev.input_buf+448, 8))+bytes(to_mv(self.dev.input_buf+464, 8)) if dtype == dtypes.half else
+             bytes(to_mv(self.dev.input_buf+448, 32)))
+      result.extend(typed_view(raw[i*dtype.itemsize:(i+1)*dtype.itemsize], dtype) for i in range(count))
+    return result
```

Route the matching UOps to this helper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
         elif u.op is Ops.CAST:
@@
                                      dtype=u.dtype)
+          elif src_dtypes[0] in (dtypes.half, dtypes.float, dtypes.int) and u.dtype in (dtypes.half, dtypes.float, dtypes.int):
+            values[u] = self.run_cast(src_values[0], src_dtypes[0], u.dtype)
```

### Ops.CMPEQ, Ops.CMPNE and Ops.CMPLT: handle NaNs

The old floating-point comparison formula still rejects NaN inputs. Releasing that gate alone would not give the required unordered results.

What changed since we wrote it? We now have exact integer comparison and preserved storage bits. Positive half encodings increase with the value; negative encodings run in the opposite direction. That gives a candidate comparison on encodings, with explicit masks for NaNs and signed zero:

| Input property | NPU check                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------- |
| Sign           | bits > 32767                                                                              |
| Magnitude      | bits - sign*32768                                                                         |
| NaN            | magnitude > 0x7c00                                                                        |
| Both zero      | magnitude_a + magnitude_b = 0                                                             |
| Equal          | Same bits or both zero, and neither input is NaN                                          |
| Less           | Different signs select the negative input; same negative signs reverse the bit comparison |

Equality of integer encodings is 1 - (a < b) - (b < a), using native integer comparisons. NaN makes CMPEQ and CMPLT false; CMPNE is the inverse of equality. This also makes +0 and -0 compare equal.

The final 0/1 mask goes through the NPU half-to-byte converter, so the result is still a bool byte:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_half_compare(self, op:Ops, a:list, b:list) -> list:
+    assert op in (Ops.CMPEQ, Ops.CMPNE, Ops.CMPLT) and len(a) == len(b)
+    result:list = []
+    base = self.dev.input_mem.dma_addr
+    for start in range(0, len(a), 8):
+      count, slot = min(8, len(a)-start), 0
+      def alloc(raw:bytes|None=None) -> int:
+        nonlocal slot
+        addr = base+64*slot
+        slot += 1
+        if raw is not None: to_mv(self.dev.input_buf+addr-base, len(raw))[:] = raw
+        return addr
```

Build the comparisons with the private INT32 task helper. Each constant and result gets its own scratch slot.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_half_compare(self, op:Ops, a:list, b:list) -> list:
@@
         if raw is not None: to_mv(self.dev.input_buf+addr-base, len(raw))[:] = raw
         return addr
+      def const(value:int) -> int: return alloc(struct.pack("<i", value)*8)
+      zero, one = const(0), const(1)
+      def calc(algo:int, x:int, y:int=zero, **kw) -> int:
+        out = alloc()
+        self.mulacc_stage(algo, x, y, out, precision=4, output=4, **kw)
+        return out
+      def sub(x:int, y:int) -> int: return calc(4, x, y)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
+      def select(mask:int, yes:int, no:int) -> int: return calc(2, no, mul(sub(yes, no), mask))
```

Copy each half encoding into a zero-padded INT32 lane. Detect signs, remove the sign bits, reject NaN magnitudes, and detect the both-zero case.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_half_compare(self, op:Ops, a:list, b:list) -> list:
@@
       def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
       def select(mask:int, yes:int, no:int) -> int: return calc(2, no, mul(sub(yes, no), mask))
+      lhs, rhs = (alloc(b"".join(bytes(raw16(x, dtypes.half))+bytes(2) for x in xs[start:start+8])+bytes(4*(8-count)))
+                  for xs in (a, b))
+      sa, sb = lt(const(32767), lhs), lt(const(32767), rhs)
+      ma, mb = sub(lhs, mul(const(32768), sa)), sub(rhs, mul(const(32768), sb))
+      valid = mul(sub(one, lt(const(0x7c00), ma)), sub(one, lt(const(0x7c00), mb)))
+      both_zero = sub(one, lt(zero, calc(2, ma, mb)))
+      ab, ba = lt(lhs, rhs), lt(rhs, lhs)
```

For CMPLT, different signs select the negative operand. If both inputs are negative, reverse the encoding comparison. Clear the result for NaNs and for either combination of signed zeros.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_half_compare(self, op:Ops, a:list, b:list) -> list:
@@
       both_zero = sub(one, lt(zero, calc(2, ma, mb)))
       ab, ba = lt(lhs, rhs), lt(rhs, lhs)
+      if op is Ops.CMPLT:
+        less = select(calc(5, sub(sa, sb)), sa, select(sa, ba, ab))
+        mask = mul(mul(valid, sub(one, both_zero)), less)
```

For equality, neither encoding is less than the other, or both are zero. Clear equality for NaNs. CMPNE inverts this final equality mask, so it is true for unordered inputs too.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_half_compare(self, op:Ops, a:list, b:list) -> list:
@@
         less = select(calc(5, sub(sa, sb)), sa, select(sa, ba, ab))
         mask = mul(mul(valid, sub(one, both_zero)), less)
+      else:
+        equal = sub(one, mul(calc(2, ab, ba), sub(one, both_zero)))
+        mask = mul(valid, equal)
+        if op is Ops.CMPNE: mask = sub(one, mask)
```

The mask is INT32 0/1. Convert it to FP32, then FP16, and feed our existing byte-output mode. Reuse the packed eight-half atom for both inputs; read the first count bool bytes.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_half_compare(self, op:Ops, a:list, b:list) -> list:
@@
         mask = mul(valid, equal)
         if op is Ops.CMPNE: mask = sub(one, mask)
+      wide, half = alloc(), alloc()
+      self.mulacc_stage(-1, mask, zero, wide, precision=4)
+      self.mulacc_stage(-1, wide, zero, half, output=2)
+      packed = alloc(bytes(to_mv(self.dev.input_buf+half-base, 8))+bytes(to_mv(self.dev.input_buf+half-base+16, 8)))
+      out = alloc()
+      self.build_registers(Ops.MUL, byte_output=True, input_addr=packed, weight_addr=packed, output_addr=out)
+      self.submit()
+      raw = bytes(to_mv(self.dev.input_buf+out-base, count))
+      result.extend(typed_view(raw[i:i+1], dtypes.bool) for i in range(count))
+    return result
```

Route the matching UOps to this helper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.SQRT and u.dtype == dtypes.half:
             values[u] = self.run_sqrt(src_values[0])
+          elif u.op in GroupOp.Comparison and src_dtypes == [dtypes.half, dtypes.half]:
+            values[u] = self.run_half_compare(u.op, *src_values)
```

Stop rewriting half comparisons into the older formula. Bool inputs can keep their existing 0/1 lowering:

```diff
 class RockchipRenderer(Renderer):
@@
-    # Scale before subtracting epsilon so the smallest positive FP16 delta stays positive.
-    (UPat(Ops.CMPLT, src=(UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.bool)),
-                         UPat(dtype=(dtypes.half, dtypes.weakfloat, dtypes.bool))), name="u"),
+    # Bool comparison can still use the FP16 0/1 lowering; half inputs use raw-bit comparison in the runtime.
+    (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.bool), UPat(dtype=dtypes.bool)), name="u"),
@@
-    # Lower FP16 comparisons after general rewrites, preserving the final mask CAST.
-    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=(dtypes.half, dtypes.weakfloat)),
-                                     UPat(dtype=(dtypes.half, dtypes.weakfloat))), name="u"),
-     lambda u: RockchipRenderer._pm_lower_compare(u)),
```

The focused checks passed: 4,096 POW magnitude pairs, all 65,536 half bit patterns for EQ/NE/LT with the other operand cycling through eight special values, and all 63,488 finite half values cast to INT32. That comparison check is not every possible pair of half values.

Reconstructing run_math, run_exp2_mul_log2, run_cast and run_half_compare from these diffs gave the same helpers as the runtime. Their focused checks, including special-value WHERE, gave `4 passed in 74.35s`.

The combined run gave **7 passed, 1 failed in 557.49s**. `test_pow_full`, `test_pow`, `test_pow_neg_inf_frac_exponent` and `test_pow_zero_exponent` passed. `test_pow_const` failed at `x ** 8.0`: 617 / 2925 values exceeded the tolerance. The same test on tinygrad CPU, with the same HALF/NOOPT settings, failed at the same expression with the same 617 mismatches. This expression becomes repeated FP16 MULs, so the wider LOG2/EXP2 path does not apply.

This is not a claim of every POW edge case. For example, `Tensor([1.0]) ** Tensor([nan])` returns NaN on tinygrad CPU too. A Python scalar NaN exponent instead fails in the common `simplify_pow` rewrite before reaching either backend.

The saved full-runtime sweep ran the whole file, serially: **433 tests**. The longer timeout lets slower NPU decompositions finish; it does not change comparisons or skip tests. This is runtime coverage, not a fresh reconstruction of the whole tutorial.

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest -n0 -vv --tb=short test/backend/test_ops.py
```

`test_all_large` hit the 600-second per-test timeout; its log contains the timeout dump, and the process exited with signal 11 while dumping it. We continued the unattempted cases in separate serial pytest processes, using the same settings and unchanged source. Each process kept the 600-second timeout, with a 660-second external limit. Timeouts are not passes or skips.

The complete sweep gave **209 passed, 195 failed, 21 timed out, 8 skipped — 433 methods total**. This is the combined result, not one pytest summary. All collected methods were accounted for, and the runtime and test-file hashes stayed unchanged. The eight skips came from existing test decorators.

Count collected test methods, not the green parent line alone: pytest can print `PASSED` for a method whose subtests failed, then exit with code 1. The summary checks the exit code and `SUBFAILED` entries too. Both asymmetric-padding convolution methods failed this way on unsupported FP32 ADD; they are not passes.

## Progress and remaining limits

Progress so far

| Group             | Paths implemented                      |
| ----------------- | -------------------------------------- |
| `GroupOp.Unary`   | `NEG`, `RECIPROCAL`, `TRUNC`           |
|                   | `SQRT`, `EXP2`, `LOG2`, `SIN`          |
| `GroupOp.Binary`  | `ADD`, `MUL`, `SUB`, `FDIV`, `MAX`     |
|                   | `CMPEQ`, `CMPNE`, `CMPLT`              |
|                   | `AND`, `OR`, `XOR`, `SHL`, `SHR`       |
|                   | `CDIV`, `CMOD`, `FLOORDIV`, `FLOORMOD` |
|                   | `THREEFRY`, `POW`                      |
| `GroupOp.Ternary` | `WHERE`, `MULACC`                      |
| Extras            | `CAST`, `BITCAST`                      |
| **Total**         | **30 / 30 paths**                      |

FLOORDIV, FLOORMOD, THREEFRY and POW use tinygrad's existing decompositions. This count is not 30 fully supported ops for every dtype. General FP32 arithmetic is still gated; the FP32 stages above are private helpers. INT16 SHL still saturates on overflow, and vector shift calls require a uniform count. Some CAST combinations still use the Python fallback. BITCAST only reinterprets storage; it does not calculate new values.

The full sweep uses forward-only FP16 with NOOPT=1. It does not establish backward, default-FP32 or optimized-kernel coverage.

The saved runtime sweep exposes limits beyond the primitive checks:

| Limit                               | Examples                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------- |
| FP32 arithmetic is gated            | test_cumsum, backward comparison tests, test_pow_const_direct              |
| Some output dtypes are missing      | test_masked_select reaches WHERE(bool)                                     |
| Signed-zero/infinity division       | test_div_naninf and test_copysign_exact fail                               |
| FP16 accuracy in composed functions | test_asin, test_gelu, test_tanh and test_pow_const                         |
| Runtime scratch exhaustion          | test_any, test_simple_cummin and fancy indexing                            |
| Reference/configuration failures    | test_bitcast's odd half-to-INT32 shape; test_stack's FP16 scalar tolerance |

The scratch exhaustion belongs to the current runtime's mapped-result allocator. The tutorial above copies each completed result instead; it does not introduce that allocator. Do not add its page-lifetime handling just to reproduce this failure.

CPU baselines under the same HALF settings also failed some accuracy cases. For example, test_pow_const had the same 617 / 2925 mismatches on CPU. Other cases differed: test_asin passed on CPU, and GELU/TANH had more mismatches on Rockchip. A shared CPU failure does not make an NPU result correct.

The five forward comparison methods passed, but their shared special-value loop leaves NaN commented out. The separate raw-bit checks above cover NaNs. FORWARD_ONLY does not disable methods that explicitly call backward().

Likewise, test_cast passing does not prove every conversion ran on the NPU: bool → FP32 still uses the Python fallback. The NPU CAST modes introduced here are listed explicitly in their dispatch.

The saved focused run was:

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest -n0 -v --tb=short test/device/test_rockchip_integer.py \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc

18 passed, 124 subtests passed in 434.68s
```

These checks cover every half encoding for SQRT/EXP2/LOG2/SIN, raw comparison and CAST checks, integer boundaries, wide shifts, THREEFRY and special-value WHERE. They do not replace the full sweep: its result remains **209 / 433 passed**.

The recorded targeted lint check passed. The same review's whole-tree mypy and ruff checks reported 47 and 142 errors respectively, in reference/generated files. Those were not green; these are saved results, not checks rerun during this writing review.
