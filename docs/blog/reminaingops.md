# Remaining ops

TOREVIEW1: The shared output converter setup section below extracts the repeated offset/shift writes after both callers have been introduced. Precision, scale and layout remain explicit in each task.

Start with the smaller changes, then build the helpers needed by the later ops. This order starts from the completed SHL/SHR sections in blog.md:

| Order | Ops / support                         | Prerequisite                                    |
| ----- | ------------------------------------- | ----------------------------------------------- |
| 1     | TRUNC                                 | Existing unary EW setup                         |
| 2     | AND → XOR → OR                        | Convolution shifts and shared digit tables      |
| 3     | BITCAST → MULACC                      | Raw storage, then private FP32/INT32 stages      |
| 4     | CDIV → CMOD; floor division/remainder  | Exact integer limbs and sign correction         |
| 5     | THREEFRY                              | Integer arithmetic, bitwise ops and shifts      |
| 6     | Shared comparisons and WHERE          | Raw encodings and integer selection             |
| 7     | Half FDIV fixes; scratch reuse        | Sign handling and stable result storage         |
| 8     | FP32 ADD/SUB/NEG → MUL → FDIV          | Private stages, then exact significand arithmetic |
| 9     | Numeric CAST; FP32 WHERE               | FP32 converters and raw-word selection          |
| 10    | SQRT → EXP2 → LOG2 → SIN → POW        | Shared arithmetic, comparisons and conversions  |
| 11    | Full-suite sweep and remaining limits | All preceding implementations                   |

The sections now follow dependencies rather than discovery dates. Recorded commands, failures and timings below come from the earlier investigation order unless explicitly rerun at the new checkpoint. In particular, moving CAST and FP32 support earlier changes the math baselines; the old failures do not establish what the new stage will do.

At each step, run the code built so far first. A missing prerequisite gets fixed where the test exposes it; a later implementation is not silently enabled for that run.

TOREVIEW1: Replay commands now include TRACE=1 to show the UOps. Adding the flag does not rerun the old measurements: retained results and timings are historical summaries unless the text explicitly says the traced command was rerun. Do not read an old summary as a newly captured TRACE log.

## Ops.TRUNC

First run test_trunc 

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc

0 Ops.PARAM dtypes.half ParamArg(0, dtypes.half, 1575, device='ROCKCHIP') [] []
1 Ops.PARAM dtypes.half ParamArg(1, dtypes.half, 1575, device='ROCKCHIP') [] []
2 Ops.CONST dtypes.weakint 1575 [] []
3 Ops.CAST dtypes.int dtypes.int [[1575]] [dtypes.weakint]
4 Ops.SPECIAL dtypes.int gidx0 [[1575]] [dtypes.int]
5 Ops.INDEX dtypes.half ... [dtypes.half, dtypes.int]
6 Ops.LOAD dtypes.half ... [dtypes.half]
7 Ops.INDEX dtypes.half ... [dtypes.half, dtypes.int]
8 Ops.TRUNC dtypes.half None [[0.1953125]] [dtypes.half]

NotImplementedError: ROCKCHIP NPU does not support Ops.TRUNC with dtypes.half
Ran 1 test in 0.200s
FAILED (errors=1)
```
TOREVIEW1: TRACE reaches the unary TRUNC itself; no decomposition has replaced it. The fresh run above stopped before an NPU submission for TRUNC.

Lets add TRUNC to the supported-op map and temporarily let it reach run_npu. CMP is still only our composite-op marker, not a register algorithm:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
-           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0}
+           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
+           Ops.TRUNC: CMP}
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

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-          if u.op not in self.ops_map or self.ops_map[u.op] == CMP or \
+          if u.op not in self.ops_map or (self.ops_map[u.op] == CMP and u.op is not Ops.TRUNC) or \
```

TOREVIEW1: Run again after opening this gate:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc

8 Ops.TRUNC dtypes.half None [[0.1953125]] [dtypes.half]

  assert (b is None and op in (Ops.NEG, Ops.CAST, Ops.CUSTOM)) or (b is not None and len(a) == len(b))
AssertionError
Ran 1 test in 0.199s
FAILED (failures=1)
```

The earlier UOps are unchanged; the first trace abbreviates memory addresses. Opening the gate only reaches run_npu's unary-op assertion. It does not implement truncation, and this attempt stops before submission. Restore the composite-op gate before implementing TRUNC:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-          if u.op not in self.ops_map or (self.ops_map[u.op] == CMP and u.op is not Ops.TRUNC) or \
+          if u.op not in self.ops_map or self.ops_map[u.op] == CMP or \
```

Ops.TRUNC. Does the CVT right shift help here? 
Not directly: `2.9 >> 1` would scale the number, while TRUNC needs 2. We need to remove the fraction towards zero.

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

TRUNC is already advertised as a composite operation. Now supply its lowering.

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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc

test_trunc (__main__.TestOps.test_trunc) ... ok

Ran 1 test in 1.174s

OK
```

All 65,536 FP16 encodings passed the direct NPU check against numpy.trunc. Non-NaN outputs matched bit for bit, including negative zero, subnormals and both infinities. NaN inputs remained NaN; their payload bits are not preserved. This is FP16 TRUNC only, with six NPU stages per atom, not a fused single-task implementation.

Progress: **16 / 30** paths covered, **14 / 30** remaining. TRUNC joins NEG and RECIPROCAL; AND/XOR are next.
TOREVIEW1: The blog's comparison recap has 12 paths. WHERE, SHL and SHR bring it to 15; TRUNC makes 16. Signed shift cases extend SHL/SHR, not two extra ops. AND/XOR then make 18, BITCAST 19 and MULACC 20. CDIV/CMOD/FLOORDIV/FLOORMOD and THREEFRY make 25; SQRT/EXP2/LOG2/SIN/POW bring the total to 30. Extending an existing op to another dtype does not increase this count.

## Ops.AND

Next we will do Ops.AND. And lets try to run it directly

Run the code built through blog.md and TRUNC above, before applying the AND diffs:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.int
Ran 1 test in 0.154s
FAILED (errors=1)
```

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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

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

Lets try the INT16 CAST before rejecting it:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Try narrower integer operands; this changes their dtype, not the AND operation.
+    (UPat(Ops.AND, dtypes.int32, name="u"),
+     lambda u: u.src[0].cast(dtypes.int16).alu(Ops.AND, u.src[1].cast(dtypes.int16)).cast(dtypes.int32)),
```

Run the same test after this INT16 matcher diff:

```text
NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.short
Ran 1 test in 0.157s
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
Ran 1 test in 0.159s
FAILED (errors=1)
```

Both runs follow the diffs above, using the CAST support built so far. Neither supplies a bitwise AND implementation.

Remove the unsuccessful rule before continuing:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
-    # Try narrower integer operands; this changes their dtype, not the AND operation.
-    (UPat(Ops.AND, dtypes.int32, name="u"),
-     lambda u: u.src[0].cast(dtypes.half).alu(Ops.AND, u.src[1].cast(dtypes.half)).cast(dtypes.int32)),
```

Removing that matcher brings us back to the INT32 AND error. The cast did not give us a working AND path.

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
This is plan step 3, the eight threshold convolutions (Tasks 5–12). Only these calls enable relux; Task 13 combines their step masks with the table-difference weights.

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

TOREVIEW1: No second number is needed for dispatch: CMP only means "not a direct EW algorithm". The path is decided separately:

| Path             | Where it happens      | Example at this step                         |
| ---------------- | --------------------- | -------------------------------------------- |
| UOp lowering     | Renderer matcher      | Bool AND becomes FP16 MUL and CAST           |
| Multi-task helper | Program dispatch     | INT32/UINT32 AND calls the convolution helper |

Both must stay out of the generic EW path. A second numeric marker would still need the same matcher and dtype-specific dispatch; it would not select either path by itself. The comments below distinguish the paths without inventing another register-looking number.

```diff
-CMP = 9  # Internal multi-stage comparison marker, not an EW algorithm.
+CMP = 9  # Internal lowering/dispatch marker, not an EW algorithm.
@@
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
            Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
-           Ops.TRUNC: CMP}
+           Ops.AND: CMP, Ops.TRUNC: CMP}
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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

test_and (__main__.TestOps.test_and) ... ok

Ran 1 test in 16.470s

OK
```

## Ops.XOR

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_xor

10 Ops.XOR dtypes.int None [[1], [4919]] [dtypes.int, dtypes.int]
NotImplementedError: ROCKCHIP NPU does not support Ops.XOR with dtypes.int
Ran 1 test in 0.150s
FAILED (errors=1)
```
TOREVIEW1: Reran this checkpoint with TRACE=1. This excerpt shows the first unsupported XOR; earlier constant, index and copy lines are omitted.

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
TOREVIEW1: Yes, flatten the table row by row: a=0 gives 0,1,2,3; a=1 gives 1,0,3,2; then append the a=2 and a=3 rows. Each row has four b values, so lookup[4*a+b] is a XOR b. For a=2,b=3, lookup[11]=1. These are table outputs, not the convolution weights themselves; the existing helper derives the difference weights from adjacent entries.

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
            Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
-           Ops.AND: CMP, Ops.TRUNC: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.TRUNC: CMP}
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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_xor

test_xor (__main__.TestOps.test_xor) ... ok

Ran 1 test in 0.275s

OK
```

Both operations use 14 convolution tasks per word pair. Python builds constant weights and copies original/final storage; the NPU calculates the digits, table selection and result bytes.

The helper diffs also passed 196 INT32/UINT32 AND/XOR pairs, including negative values and word boundaries. All four bool XOR pairs passed too.

Progress: **18 / 30** paths covered. AND/XOR cover bool and INT32/UINT32 here, not every integer width.

## Ops.OR

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or

10 Ops.OR dtypes.int None [[1], [4919]] [dtypes.int, dtypes.int]
NotImplementedError: ROCKCHIP NPU does not support Ops.OR with dtypes.int
Ran 1 test in 0.154s
FAILED (errors=1)
```
TOREVIEW1: Reran this tutorial checkpoint with TRACE=1. The excerpt shows the failing OR with 0x1337 (4919); earlier LOAD/STORE and index lines are omitted.

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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or

test_or (__main__.TestOps.test_or) ... ok

Ran 1 test in 0.459s

OK
```

OR was already counted for bool, so progress stays **18 / 30**. Its coverage now includes INT32/UINT32, with the same 14 convolution tasks per word pair and no host arithmetic on intermediate values.

Four additional INT32/UINT32 Tensor checks passed (144 lanes), including broadcasting, random full-width values, sign-bit boundaries and alternating-bit patterns. Every check submitted NPU work.

## Ops.BITCAST

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitcast

RuntimeError: self.size(-1) must be divisible by 2 to view Half as Int (different element sizes), but got 3
Ran 1 test in 0.005s
FAILED (errors=1)
```
TOREVIEW1: Reran this checkpoint with TRACE=1. The reference raises before the Rockchip interpreter, so there are no NPU UOp lines for this failure.

TOREVIEW1: Yes, it fails in the first (and only) case in test_bitcast:

```python
helper_test_op([(3, 3)], lambda x: x.view(torch.int32), lambda x: x.bitcast(dtypes.int32), forward_only=True)
```

With DEFAULT_FLOAT=HALF, each row has three 2-byte values: six bytes cannot form a whole number of 4-byte INT32 values. helper_test_op runs Torch's x.view(torch.int32) first, which raises before tinygrad's x.bitcast runs. This is a reference-input shape error, not evidence that NPU BITCAST failed. The raw-payload checks below test storage preservation separately.

BITCAST keeps the same bytes and changes their dtype. In this interpreter the host copies storage; it does not calculate a converted value or submit a fake identity operation.

For example, FP16 1.0 and UINT16 15360 have the same two bytes, `00 3c`. BITCAST changes which dtype reads them; CAST would calculate a new numeric representation.

Keep those bytes in a memoryview. No new wrapper class is needed. These helpers use the storage-format functions already imported by our starting Python interpreter:

```diff
@@
+# Interpret the same bytes with a dtype's storage format; do not numerically convert them.
+def typed_view(raw:bytes|memoryview, dtype:DType) -> memoryview:
+  return memoryview(raw).cast("B").cast(storage_fmt_for_dtype(dtype))
+
+def raw16(value, dtype:DType) -> bytes|memoryview:
+  # Preserve stored bits; only Python constants need packing into the requested storage format.
+  if isinstance(value, memoryview): return value.cast("B")
+  return struct.pack("<" + storage_fmt_for_dtype(dtype), to_storage_scalar(value, dtype))
+
 def _load(m, i, dtype: DType):
```
TOREVIEW1: Added comments at the two helpers, keeping storage reinterpretation separate from numeric conversion.

raw16 copies a view's bytes unchanged. Only constants that are still Python values need packing. Despite the old name, it handles the dtype's full size, not only 16 bits.

Some interpreter checks still need a scalar, such as an index or loop condition. Decode only at those boundaries:

```diff
@@
+def scalar16(value, dtype:DType|None=None):
+  # Decode a single lane for validation/indexing; leave its original stored bytes untouched.
+  if not isinstance(value, memoryview): return value
+  return from_storage_scalar(value[0], dtype) if dtype is not None else value[0]
+
 def _load(m, i, dtype: DType):
```
TOREVIEW1: Added the scalar-decoding comment in the helper and runtime.

LOAD must keep the original bits, including NaN payloads. STORE copies them back without a numeric conversion:

```diff
 def _load(m, i, dtype: DType):
@@
   if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
+  if m.itemsize == dtype.itemsize:
+    # Copy one lane as raw storage so loading a NaN does not canonicalize its payload.
+    return typed_view(bytes(m.cast("B")[i*m.itemsize:(i+1)*m.itemsize]), dtype)
```
TOREVIEW1: Added the raw LOAD comment in the diff and runtime.

```diff
 def _store(m, i, v, dtype: DType):
   if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
+  if isinstance(v, memoryview):
+    # Store the selected bytes directly, without converting through a Python scalar.
+    assert v.nbytes == dtype.itemsize
+    m.cast("B")[i*m.itemsize:i*m.itemsize+dtype.itemsize] = v.cast("B")
+    return
```
TOREVIEW1: Added the raw STORE comment in the diff and runtime.

Now BITCAST is just a different view of the same bytes:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-        elif u.op is Ops.BITCAST: values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
+        elif u.op is Ops.BITCAST:
+          assert src_dtypes[0].itemsize == u.dtype.itemsize, "bitcast itemsize mismatch"
+          values[u] = [typed_view(raw16(x, src_dtypes[0]), u.dtype) for x in src_values[0]]
```

Remove the old scalar-based BITCAST helper from the import. Keep the storage-format helpers:

```diff
-from tinygrad.dtype import bitcast, DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
+from tinygrad.dtype import DType, dtypes, AddrSpace, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
```
TOREVIEW1: Not just because it uses the CPU: both paths do host-side storage handling. The old helper packs a Python scalar, unpacks it as the new dtype, then returns another scalar. That can lose a NaN's original payload when the float is repacked. Our values now carry their original bytes, so typed_view changes how those bytes are read without decoding and rebuilding a float. This is bit preservation, not an NPU arithmetic operation.

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

TOREVIEW1: map(scalar16, a) calls scalar16 on each input as the guard reads it, like `(scalar16(x) for x in a)`. It gives the guard numeric values instead of memoryviews; it does not replace a or rewrite its stored bytes. any() stops reading once it finds a rejected value. This is CPU validation, not the NPU arithmetic.
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
SHL already works; BITCAST changes what its inputs look like. LOAD and the output loop above now return typed memoryviews instead of Python numbers. The old SHL count check and `powers[shift]` lookup need integers, so decode b before using them. RECIPROCAL's sign guard and CAST's 0/1 guard need numbers for the same reason. The SHL formula and NPU registers are unchanged.

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

The bit-pattern checks passed all 65,536 FP16 encodings, all 65,536 BF16 encodings, and selected FP32/FP64 zeros, infinities and NaN payloads. The existing test_bitcast passed with its default FP32 inputs.

Use FLOAT for that existing test; the HALF reference-shape error above is not fixed by our runtime changes:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=FLOAT DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitcast

Ran 1 test in 0.132s
OK
```

Progress: **19 / 30** paths covered. BITCAST adds storage reinterpretation, not a numeric CAST.

## Ops.MULACC

For Ops.MULACC, which is `a*b+c`. We can try normal MUL then ADD? 

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_uops.py TestFloatUOps.test_mulacc

test_mulacc (__main__.TestFloatUOps.test_mulacc) ... skipped 'only python supports MULACC'

Ran 1 test in 0.000s
OK (skipped=1)
```

TOREVIEW1: This is the complete test output with TRACE=1: there are no numbered Ops lines to show because the decorator skips the test before the runtime executes. We got OK (skipped=1), not an NPU pass. Check the test case:

```python
@unittest.skipUnless(Device.DEFAULT == "PYTHON", "only python supports MULACC")
def test_mulacc(self):
  self._test_top_fxn(Ops.MULACC, lambda a,b,c: a*b+c, (dtypes.float, dtypes.float, dtypes.float))
```

Why does it say "only python supports MULACC"? Check its history:

```bash
$ git blame ebf1636 -L 157,159 -- test/backend/test_uops.py
```

TOREVIEW1: blame finds the last change to each line, not necessarily why the skip was introduced. The decorator points to bc180a963c. Inspect that patch, then search for when the skip text was added, following the file rename:

```bash
$ git show bc180a963c -- test/backend/test_uops.py

-  @unittest.skipUnless(getenv("PYTHON"), "only python supports MULACC")
+  @unittest.skipUnless(Device.DEFAULT == "PYTHON", "only python supports MULACC")

$ git log --follow --format='%h %s' -S 'only python supports MULACC' ebf1636 -- test/backend/test_uops.py
c13da83f1 tests from lowerer branch (#5339)
```

The show excerpt above is only the decorator change; the full command also prints the commit message and import change.

The decorator's blame leads to the device-selection change; following the earlier version finds the commit that introduced the skip:

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

Rerun the existing test after the test-file diff, before adding MULACC support:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_uops.py TestFloatUOps.test_mulacc

13 Ops.MULACC dtypes.half

NotImplementedError: ROCKCHIP NPU does not support Ops.MULACC with dtypes.half
Ran 1 test in 0.057s
FAILED (errors=1)
```

This excerpt keeps the failing UOp; preceding PARAM/LOAD lines, memory addresses and argument lists are omitted. Unlike the skipped run, TRACE now reaches MULACC.

The test now reaches the missing MULACC. The separate MUL→ADD idea below is a candidate, not evidence that tinygrad lowers a direct MULACC that way.

TOREVIEW1: Also try the existing Tensor test at this same step:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mulacc_with_zero_strides

Exception: forward pass failed shape (2, 4): dtype mismatch: tinygrad=float32 | torch=float16
Ran 1 test in 0.164s
FAILED (errors=1)
```

This test uses expanded inputs and a reduction. It stops at a dtype mismatch, so it cannot yet tell us whether the proposed fused task rounds correctly. Keep that failure; use test_uops.py for the direct MULACC gate and probe the candidate registers separately before integrating them.

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

The targeted probe source is included in [extra/rockchip/probe_fp16_mulacc.py](../../extra/rockchip/probe_fp16_mulacc.py). An earlier 4,096-triple random check had no numeric mismatches, but these targeted halfway cases did. Passing ordinary random tests is not enough to claim correctly rounded MULACC. The included probe reproduces the rounding and negative-zero failures before adding the correction.

```bash
$ TRACE=1 PYTHONPATH=$PWD .venv/bin/python extra/rockchip/probe_fp16_mulacc.py
```
TOREVIEW1: This probe tests the proposed BS→EW task directly. Neither test_ops.py nor test_uops.py can dispatch that task yet; the failures above are our normal-test baseline.

This prints the native and reference results; it is a diagnostic probe, not a passing correctness test. Compare lane 0 and the sign of zero in lane 3 in its final output.

Both probe sources are included with this draft under extra/rockchip; they are not in the upstream commit checked out at the start of blog.md. Copy the two linked files into extra/rockchip in your tutorial checkout before running these commands. Run from that checkout's root so the probes import the backend built so far, not a later installed version. The correction probe uses typed views from the BITCAST step; it does not require RockchipValue.

Dont add MULACC to ops_map yet. The native task loses the rounding information in the third case. First correct that on the NPU; progress stays **19 / 30**.

So we need to recover the lost rounding information

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

The correction probe is [extra/rockchip/probe_fp16_mulacc_correct.py](../../extra/rockchip/probe_fp16_mulacc_correct.py). Intermediate values stay in DMA storage; Python only uploads inputs/constants, submits tasks and reads the final result. It uses 32 NPU tasks per eight lanes, so this fixes the tested numerical limits but is much more expensive than the native task. It is still a probe, not advertised MULACC support in ops_rockchip.py.

```bash
$ TRACE=1 PYTHONPATH=$PWD .venv/bin/python extra/rockchip/probe_fp16_mulacc_correct.py

4096 triples PASS
8192 triples PASS
12288 triples PASS
16384 triples PASS
17723 triples PASS; non-NaN results bit-exact, including signed zero
```
TOREVIEW1: The correction is still only in this standalone probe, not the runtime. Running test_ops.py now would exercise the old implementation, not the candidate correction. After the diffs below integrate it, rerun the existing MULACC tests as well; the probe is not a replacement for them.

This was rerun with the code built so far, before adding mulacc_stage or run_mulacc below. The standalone probe supplies its own task sequence.

Run from the tinygrad repo root. This covers eight targeted cases, 16,384 random triples and 1,331 special-value combinations against a float64 multiply-add rounded to FP16. NaNs are checked as NaNs, not by payload. This is not an exhaustive test of all FP16 triples; progress stays **19 / 30** until the corrected path is integrated.

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
+    if algo < 0:  # Conversion/product-only task: bypass EW; BS and the output converter still run.
+      ew |= (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT)
+    else:
+      ew |= ((1 << rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT) | (3 << rk.DPU_EW_CFG_EDATA_SIZE__SHIFT) |
+             (algo << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
+             (binary << rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT) | (mul << rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT))
```
TOREVIEW1: algo=-1 is our helper's bypass marker, not a hardware ALU selector.

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
+        to_mv(self.dev.input_buf+i*64, 16)[:] = b"".join(raw16(x, dtypes.half) for x in values[start:start+8]) + bytes(2*(8-count))
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
+      result.extend(typed_view(raw[i*2:i*2+2], dtypes.half) for i in range(count))
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

Keep a small regression here too. Create test/device/test_rockchip_integer.py with the halfway-rounding, negative-zero and overflow-cancellation cases:

```diff
--- /dev/null
+++ b/test/device/test_rockchip_integer.py
@@
+import struct, unittest
+import numpy as np
+from tinygrad import Device, Tensor, dtypes
+from tinygrad.runtime.ops_rockchip import RockchipProgram, ops_map, scalar16, typed_view, _load, _store
+from tinygrad.uop.ops import Ops, GroupOp, exec_alu
+
+@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "serial RK3588 hardware tests; use -n0")
+class TestRockchipInteger(unittest.TestCase):
+  def setUp(self):
+    self.program = object.__new__(RockchipProgram)
+    self.program.dev, self.program.ops_map = Device["ROCKCHIP"], ops_map
+    self.program.npu_regs = []
+
+  def test_mulacc_storage(self):
+    a, b, c = [11.8125, -0., 65504.], [-2688., 1., 2.], [-0.00023484230041503906, -0., -65504.]
+    result = self.program.run_mulacc(a, b, c)
+    self.assertEqual([bytes(x) for x in result], [struct.pack("<e", x) for x in [-31760., -0., 65504.]])
```

Run this small MULACC regression first, then the larger checks below:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.device.test_rockchip_integer.TestRockchipInteger.test_mulacc_storage

Ran 1 test in 0.008s
OK
```

This run used the reconstructed code at this step, before CDIV or the later math helpers.

Now run the same targeted, random and special-value triples through run_mulacc, rather than the probe's own register sequence:

```bash
$ TRACE=1 PYTHONPATH=$PWD .venv/bin/python extra/rockchip/probe_fp16_mulacc_correct.py --backend

4096 triples PASS
8192 triples PASS
12288 triples PASS
16384 triples PASS
17723 triples PASS; non-NaN results bit-exact, including signed zero
```

This passed with the code reconstructed through this step, before CDIV or the later math helpers.

The saved larger integrated-backend sweeps went further. The first included all FP16 encodings in identity and negation; the second added halfway products, overflow boundaries and all encodings multiplied by 0.5. Historical summary lines:

```bash
$ TRACE=1 PYTHONPATH=$PWD .venv/bin/python extra/rockchip/probe_fp16_mulacc_correct.py --backend --exhaustive-unary
148795 triples PASS; non-NaN results bit-exact, including signed zero

$ TRACE=1 PYTHONPATH=$PWD .venv/bin/python extra/rockchip/probe_fp16_mulacc_correct.py --backend --rounding-boundaries
125665 triples PASS; non-NaN results bit-exact, including signed zero
```

The integration probe also checks integer MULACC decomposition. Run it after CDIV/CMOD below, once that integer arithmetic exists.

The existing test_mulacc_with_zero_strides still fails under DEFAULT_FLOAT=HALF:

```text
dtype mismatch: tinygrad=float32 | torch=float16
```

It fails the same way with MULACC fusion disabled, before testing this path. We have not changed its dtype assertions or counted it as passing.

Rerun the existing arithmetic, comparison, selection and integer paths:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
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

Progress: **20 / 30** paths covered. MULACC now has an NPU-only FP16 implementation, including the tested rounding, overflow, underflow, infinity and signed-zero cases. NaNs remain NaNs; payload bits are not guaranteed. It costs 32 submissions per atom of up to eight lanes, and the NOOPT interpreter can submit only one lane at a time. This is not general FP32 MULACC support or an exhaustive check of every possible input triple.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc

.                                                                        [100%]
1 passed in 0.68s
```

This test now passes on ROCKCHIP, not skips. It checks both buffer inputs and constants; the larger correction probes above still cover rounding and special values. This is separate from test_ops.test_mulacc_with_zero_strides, whose dtype mismatch remains.


## Ops.CDIV

There is no test_cdiv in test_ops.py. Here are the division and remainder tests in test_ops.py and test_uops.py.

TOREVIEW1:

| File         | Test                               | What it checks                                        |
| ------------ | ---------------------------------- | ----------------------------------------------------- |
| test_ops.py  | `test_div`                         | Tensor and scalar floating division                   |
|              | `test_div_rounding_mode`           | True, truncating and floor division; invalid mode     |
|              | `test_div_int`                     | Integer inputs with `/`, `//` and truncating division |
|              | `test_scalar_div`                  | Scalar numerator or denominator                       |
|              | `test_div_naninf`                  | Division involving infinity and NaN                   |
|              | `test_idiv_shift_rewrite_negative` | Negative truncating division after shift rewriting    |
|              | `test_mod`                         | Remainder with floor-division semantics               |
|              | `test_fmod`                        | Remainder with truncating-division semantics          |
| test_uops.py | `TestNonFloatUOps.test_div_int32`  | Direct INT32 CDIV, buffer inputs and constants        |
|              | `TestNonFloatUOps.test_mod_int32`  | Direct INT32 CMOD, buffer inputs and constants        |

TOREVIEW1: We do not need to run every test to find CDIV: test_div_int32 explicitly passes Ops.CDIV and two dtypes.int32 inputs to _test_bop_fxn. It checks buffer inputs and constants against int(a/b), including negative values, with zero divisors excluded. That makes it the direct starting check for CDIV; the Tensor tests mix division with promotion, rounding and remainder. We still need those broader tests afterwards, and this small input set does not cover integer boundaries.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

NotImplementedError: ROCKCHIP NPU does not support Ops.CMOD with dtypes.int
Ran 1 test in 0.169s
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

The quotient must stay exact, so narrowing these integers to FP16 is not enough. The BITCAST step already keeps the original bytes through LOAD and STORE. Use those bytes for exact integer arithmetic instead.

So we need to handle Exact integer arithmetic

There is no `TestOps.test_cdiv`; the existing direct test is `TestNonFloatUOps.test_div_int32`. Run it with the BITCAST changes above, before adding integer division:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestNonFloatUOps.test_div_int32

NotImplementedError: ROCKCHIP NPU does not support Ops.CDIV with dtypes.int
Ran 1 test in 0.062s
FAILED (errors=1)
```

Implement the quotient path first, then enable CMOD in its separate section.

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
         for x in a: total = add(total, x)
         return lt(zero, total)
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
             unequal = nonzero(difference)
             out = [unequal if op is Ops.CMPNE else sub(one, unequal)]
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

Replaying the diffs up to here, direct CDIV passes:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestNonFloatUOps.test_div_int32

Ran 1 test in 1.598s
OK
```

## Ops.CMOD

CDIV returns the quotient, but its restoring-division loop also keeps the remainder. CMOD returns that remainder after restoring the numerator's sign. This is why the shared helper above contains both paths; CMOD does not need a second division algorithm.

At this tutorial step, the CMOD dispatch has not been added yet. The direct remainder test stops here:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestNonFloatUOps.test_mod_int32

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

Now test the quotient and remainder paths together:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

test_div_int (__main__.TestOps.test_div_int) ... ok
Ran 1 test in 3.505s
OK
```

Check remainder first, then floor-modulo separately:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_fmod

test_fmod (__main__.TestOps.test_fmod) ... ok

Ran 1 test in 5.890s

OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mod

test_mod (__main__.TestOps.test_mod) ... ok

Ran 1 test in 12.833s

OK
```

The smaller run_integer diffs reconstruct the runtime helper exactly. Replaying the whole tutorial through this section passed test_div_int and test_fmod together: 2 tests in 9.978s. The separate test_mod result above is from the same reconstructed stage. No THREEFRY or later math implementation was included.

The focused checks passed 88 subtests across signed and unsigned 8/16/32/64-bit arithmetic, comparisons and division. This includes full-width random inputs, wrapping results, negative remainders and zero divisors. It is not exhaustive over every pair. INT32 division took about 0.25s for eight lanes; correctness first, not a speed claim.

### MULACC integration with integer arithmetic

The non-FP16 MULACC matcher emits integer MUL and ADD. Those now have a dispatch, so we can run the full integration probe without borrowing later support.

The bundled integration probe checks actual UOp dispatch and Tensor fusion, including a CNA task before MULACC and ordinary EW ADD after it. At this reconstructed checkpoint it also confirmed that scratch reuse did not overwrite earlier results:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP PYTHONPATH=$PWD \
    .venv/bin/python extra/rockchip/probe_fp16_mulacc_correct.py --integration-only

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

## Ops.THREEFRY

Add this direct check to test/device/test_rockchip_integer.py, created in the MULACC section. The CPU result is the reference, while the other call uses ROCKCHIP:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_threefry(self):
+    values = np.array([0, 1, 5, 2**64-1, 0x123456789abcdef0], dtype=np.uint64)
+    keys = np.array([0, 1337, 10, 0x123456789abcdef0, 2**64-1], dtype=np.uint64)
+    def run(device): return Tensor(values, device=device).threefry(Tensor(keys, device=device)).numpy()
+    np.testing.assert_array_equal(run("ROCKCHIP"), run("CPU"))
```

No THREEFRY entry is needed in ops_map: leaving it unsupported lets tinygrad expand it into simpler UOps. But those UOps still need working primitives. With the code built so far, the new test fails at the split, before the rounds:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_threefry

NotImplementedError: ROCKCHIP NPU does not support Ops.SHR with dtypes.ulong
Ran 1 test in 0.183s
FAILED (errors=1)
```


The first missing primitive is UINT64 SHR. No THREEFRY handler has been added.

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
             values[u] = self.run_npu(Ops.CAST, [scalar16(x) for x in src_values[0]] if src_dtypes[0] == dtypes.bool else src_values[0],
                                      dtype=u.dtype)
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

Add the JAX reference vector from test_randomness.py to the same test file:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_threefry_jax_reference(self):
+    # The JAX reference from test/backend/test_randomness.py, without that module's optional hypothesis dependency.
+    expected = [2221762175, 1752107825, 653745012, 1967534793, 1395205442, 3840423848, 2159346757,
+                603508235, 3319473678, 3363866483, 3544324138, 1436466838, 2169858556, 2570072943,
+                2387150698, 3678370550, 2911697663, 403244401, 2560861638, 1692360114]
+    counts0, counts1 = Tensor.arange(20, dtype=dtypes.uint32).chunk(2)
+    got = Tensor._threefry_random_bits(Tensor([0, 1337], dtype=dtypes.uint32, device="ROCKCHIP"), counts0, counts1).numpy()
+    np.testing.assert_array_equal(got, np.array(expected, dtype=np.uint32))
+
```

The direct THREEFRY check matched the CPU backend on five counter/key pairs, including full-width values. The JAX vector already recorded in test_randomness.py also matched all 20 words. That module could not collect here because hypothesis is missing; the same reference vector is checked in test/device/test_rockchip_integer.py without adding a dependency.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_threefry_jax_reference

1 passed in 10.12s
```

The 18 earlier arithmetic/comparison/shift regressions passed again in 35.57s.

The storage paths use typed memoryviews, not a wrapper class. BITCAST changes the view format; raw16 copies its bytes. These tutorial helpers copy completed results before reusing scratch memory; they do not yet reuse intermediate DMA addresses.

Add the storage and integer boundary checks before running them. exec_alu is the reference here, not a runtime fallback:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_raw_bitcast(self):
+    for src,dst in ((dtypes.uint16, dtypes.half), (dtypes.uint16, dtypes.bfloat16),
+                    (dtypes.uint32, dtypes.float), (dtypes.uint64, dtypes.double)):
+      patterns = range(65536) if src.itemsize == 2 else (0, 1, src.max, 2**(src.bitsize-1),
+                                                       0x7f800001 if src.itemsize == 4 else 0x7ff0000000000001)
+      raw = b"".join(struct.pack("<"+src.fmt, x) for x in patterns)
+      inp, out = memoryview(raw).cast(src.fmt), memoryview(bytearray(len(raw))).cast(src.fmt)
+      for i in range(len(inp)): _store(out, i, typed_view(typed_view(_load(inp, i, src), dst), src), src)
+      self.assertEqual(out.cast("B").tobytes(), raw)
+
+  def test_npu_raw_nan(self):
+    product = self.program.run_npu(Ops.MUL, [0.0]*8, [float("inf")]*8)
+    self.assertTrue(all(isinstance(x, memoryview) and bytes(x) == b"\x01\x7c" for x in product))
+    tag = self.program.run_npu(Ops.CUSTOM, product, custom="fp16_exponent_shift_minus(16)")
+    self.assertEqual([bytes(x) for x in tag], [b"\x01\x3c"]*8)
+
+  def test_integer_boundaries(self):
+    rng = np.random.default_rng(42)
+    for dt in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64):
+      a = [dt.min, dt.max, 0, 1, 17, 127, 64, 5]
+      b = [dt.max, 1, 0, 3, 5, 7, 2, 8]
+      a += list(map(int, np.frombuffer(rng.bytes(8*dt.itemsize), dtype=dt.fmt)))
+      b += list(map(int, np.frombuffer(rng.bytes(8*dt.itemsize), dtype=dt.fmt)))
+      for op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE,
+                 Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD):
+        with self.subTest(dtype=dt, op=op):
+          out_dt = dtypes.bool if op in GroupOp.Comparison else dt
+          expected = [exec_alu(op, out_dt, pair) for pair in zip(a, b)]
+          self.assertEqual(list(map(scalar16, self.program.run_integer(op, [a, b], dt))), expected)
+
+  def test_signed_division(self):
+    a = [4094, 2049, -4094, -2**31, 2**31-1, -17, 17, -1]
+    b = [3, 1, 3, -1, 3, 5, -5, 0]
+    for op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD):
+      self.assertEqual(list(map(scalar16, self.program.run_integer(op, [a, b], dtypes.int32))),
+                       [exec_alu(op, dtypes.int32, pair) for pair in zip(a, b)])
+
+  def test_tensor_division(self):
+    a = np.array([4094, 2049, -4094, -17, 17, 2147483647], dtype=np.int32)
+    b = np.array([3, 1, 3, 5, -5, 3], dtype=np.int32)
+    ta, tb = Tensor(a, device="ROCKCHIP"), Tensor(b, device="ROCKCHIP")
+    np.testing.assert_array_equal(ta.div(tb, rounding_mode="trunc").numpy(), [1364, 2049, -1364, -3, -3, 715827882])
+    np.testing.assert_array_equal((ta//tb).numpy(), a//b)
+    np.testing.assert_array_equal((ta%tb).numpy(), a%b)
+
+  def test_wide_shifts(self):
+    raw = np.array([0, 1, 2**63, 2**64-1, 0x123456789abcdef0], dtype=np.uint64)
+    for dt,npdt in ((dtypes.uint64, np.uint64), (dtypes.int64, np.int64)):
+      values = list(map(int, raw.view(npdt)))
+      for op in (Ops.SHL, Ops.SHR):
+        for count in (0, 1, 4, 8, 16, 31, 32, 33, 63):
+          with self.subTest(dtype=dt, op=op, count=count):
+            self.assertEqual(list(map(scalar16, self.program.run_u64_shift(op, values, [count]*len(values), dt))),
+                             [exec_alu(op, dt, (x, count)) for x in values])
+
```

Check storage, division and shifts, then run the slower integer boundary check separately:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_raw_bitcast \
    test.device.test_rockchip_integer.TestRockchipInteger.test_signed_division \
    test.device.test_rockchip_integer.TestRockchipInteger.test_wide_shifts \
    test.device.test_rockchip_integer.TestRockchipInteger.test_threefry_jax_reference

Ran 4 tests in 11.441s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest test.device.test_rockchip_integer.TestRockchipInteger.test_integer_boundaries

Ran 1 test in 21.985s
OK
```

The raw-storage check round-trips half and wider encodings without unpacking/repacking NaN payloads. MULACC's rounding and signed-zero checks are recorded in its own step above.

Progress: **25 / 30** paths covered. Remaining: SQRT, EXP2, LOG2, POW and SIN. This is not every dtype or edge case, and the full test_ops.py sweep is still pending.

## Shared math prerequisites: SHR, comparisons and WHERE

The original SQRT baseline exposed these shared prerequisites before we added a SQRT implementation. Keep that investigation here, before the wider math support:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sqrt

NotImplementedError: ROCKCHIP NPU does not support Ops.SHR with dtypes.short
Ran 1 test in 0.202s
FAILED (errors=1)
```

The decomposition needs signed INT16 SHR. We already have 32-bit SHR. Put the original two bytes in the upper half of its input, shift by 16+n, then keep the low two output bytes. Signed 32-bit SHR supplies the sign fill; UINT16 uses unsigned SHR:

```text
INT16 -2, bytes FE FF → input bytes 00 00 FE FF
INT32 0xFFFE0000 >> 17 → 0xFFFFFFFF → low bytes FF FF = INT16 -1
```

Add a wrapper for uniform counts 0..15. The byte copies prepare/read the layout; conv_shift does the variable shift on the NPU:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_u16_shr(self, a:list, b:list, dtype:DType) -> list:
+    counts = list(map(scalar16, b))
+    if not counts or not all_same(counts) or not 0 <= counts[0] < 16:
+      raise NotImplementedError("ROCKCHIP 16-bit SHR requires one uniform count in 0..15")
+    wide = dtypes.int if dtype == dtypes.int16 else dtypes.uint
+    words = [typed_view(bytes(2)+bytes(raw16(x, dtype)), wide) for x in a]
+    shifted = self.run_u32_shift(Ops.SHR, words, [16+counts[0]]*len(a), wide)
+    return [typed_view(bytes(x)[:2], dtype) for x in shifted]
+
@@
           elif u.op is Ops.SHR and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_shift(Ops.SHR, src_values[0], src_values[1], u.dtype)
+          elif u.op is Ops.SHR and u.dtype in (dtypes.int16, dtypes.uint16):
+            values[u] = self.run_u16_shr(src_values[0], src_values[1], u.dtype)
```

Rerun SQRT after adding this primitive:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sqrt

NotImplementedError: ROCKCHIP NPU FP16 comparisons do not support NaN inputs
Ran 1 test in 0.245s
FAILED (errors=1)
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
           elif u.op is Ops.MULACC and u.dtype == dtypes.half:
             values[u] = self.run_mulacc(*src_values)
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
-    # Lower comparisons after general rewrites so the final mask CAST stays a CAST.
-    (UPat((Ops.CMPEQ, Ops.CMPNE), src=(UPat(dtype=(dtypes.half, dtypes.weakfloat)),
-                                     UPat(dtype=(dtypes.half, dtypes.weakfloat))), name="u"),
-     lambda u: RockchipRenderer._pm_lower_compare(u)),
```

Rerun SQRT with the corrected comparison:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sqrt

ValueError: cannot convert float NaN to integer
Ran 1 test in 0.256s
FAILED (errors=1)
```

TRACE shows the arithmetic WHERE multiplying an unselected NaN by zero, then passing the NaN into an integer CAST. Even 0*NaN is NaN; this is a selection bug, not a square-root accuracy result.

Reuse integer WHERE to select the FP16 storage bits instead. BITCAST changes their interpretation without numeric conversion:

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

## Ops.WHERE: boolean output

Start with one of the six boolean WHERE failures:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select

NotImplementedError: ROCKCHIP NPU does not support Ops.WHERE with dtypes.bool
Ran 1 test in 2.674s
FAILED (errors=1)
```

The missing dtype is the result of WHERE, not its condition. We already select FP16 branches through their integer storage bits, and both bool → FP16 and normalized FP16 → bool CAST run on the NPU. Can we reuse those paths instead of adding more registers?

```text
WHERE(condition, yes_bool, no_bool)
  → WHERE(condition, CAST(yes_bool, half), CAST(no_bool, half))
  → CAST(selected_0_or_1, bool)
```

Bool inputs convert exactly to 0.0 or 1.0. Selection keeps one of those values, so the last CAST meets our 0/1 requirement. The condition is unchanged. Unlike general floating arithmetic, there are no NaN branches to handle here.

Put this in comparison_matcher, after the general rewrites, so the final bool CAST stays a CAST. The new FP16 WHERE then matches our existing raw-bit selection rule:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
@@
+    # Bool branches become exact FP16 0/1; reuse selection and the NPU mask CAST.
+    (UPat(Ops.WHERE, dtypes.bool, name="u"),
+     lambda u: u.src[0].where(u.src[1].cast(dtypes.half), u.src[2].cast(dtypes.half)).cast(dtypes.bool)),
     # FP16 raw-bit selection reuses the exact integer WHERE path.
```

No dtype gate is relaxed. Boolean WHERE must lower to the supported selection and CAST operations; it cannot fall through to Python arithmetic.

The first retest of test_masked_select reached the 30-second command limit without finishing. It no longer stopped at the bool WHERE gate, but that is not a pass. First check all eight combinations of condition, yes and no:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_bool_where(self):
+    condition, yes, no = [False]*4+[True]*4, [False, False, True, True]*2, [False, True]*4
+    tensors = [Tensor(x, dtype=dtypes.bool, device="ROCKCHIP") for x in (condition, yes, no)]
+    actual = tensors[0].where(tensors[1], tensors[2]).numpy()
+    self.assertEqual(actual.dtype, np.dtype(np.bool_))
+    np.testing.assert_array_equal(actual, np.where(condition, yes, no))
+
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_bool_where

Ran 1 test in 0.163s
OK
```

The existing smaller nonzero test also reaches boolean selection:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_nonzero_size

Ran 1 test in 5.699s
OK
```

test_masked_select still did not finish within a 120-second limit with the code built from these diffs. We have fixed the bool WHERE gate and checked its truth table, but have not passed the full masked-select case. Keep that as a timeout to investigate, not another green test. The 209/433 figure in the final sweep section remains the earlier sweep, not a new total after this change.

## Ops.FDIV: signs and invalid results

Two saved failures involve division:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_copysign_exact TestOps.test_div_naninf

test_copysign_exact ... ERROR
test_div_naninf ... ERROR
Ran 2 tests in 1.856s
FAILED (errors=2)
```

copysign returned +1 where the reference expected -1. Its trace contains FDIV(1, x): tinygrad uses the reciprocal to distinguish -0 from +0. test_div_naninf also found wrong signs for an infinite numerator. So changing only the copysign matcher would leave the division problem.

The 1500 branch handles a constant infinite numerator as `(signed_one / denominator) / 0`. That does not cover variable numerators or signed-zero division. Lets first keep the hardware quotient's magnitude and replace its sign.

For raw FP16 words, the sign bit is bit 15. Unlike `x < 0`, reading that bit distinguishes -0:

```text
sign(word) = word > 0x7fff
result_sign = ABS(sign(a) - sign(b))
result_bits = magnitude_bits(quotient) + result_sign * 32768
```

We already have private INT32 ADD/SUB, ABS and binary MIN. Copy each two-byte FP16 word into a zero-extended INT32 lane; these are storage bits, not a numeric FP16-to-INT32 CAST. Use the NPU for the comparisons and arithmetic, then copy the low two output bytes back.

First add the sign repair before run_npu:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_fdiv(self, a:list, b:list) -> list:
+    quotient = self.run_npu(Ops.FDIV, a, b)
+    result:list = []
+    base = self.dev.input_mem.dma_addr
+    for start in range(0, len(a), 8):
+      count, slot = min(8, len(a)-start), 0
+      def put(raw:bytes) -> int:
+        nonlocal slot
+        addr = base+64*slot
+        slot += 1
+        to_mv(self.dev.input_buf+addr-base, 32)[:] = raw+bytes(32-len(raw))
+        return addr
+      def const(x:int) -> int: return put(struct.pack("<i", x)*8)
+      # Zero-extend raw FP16 words into INT32 lanes; do not numerically convert floats.
+      lhs, rhs, out = (put(b"".join(bytes(raw16(x, dtypes.half))+bytes(2) for x in xs[start:start+8]))
+                       for xs in (a, b, quotient))
+      sign, threshold = const(0x8000), const(0x7fff)
+      def calc(algo:int, x:int, y:int=threshold, **kw) -> int:
+        dst = put(bytes(32))
+        self.mulacc_stage(algo, x, y, dst, precision=4, output=4, **kw)
+        return dst
+      def signbit(x:int) -> int: return calc(1, threshold, x, binary=True)
+      # XOR of 0/1 signs is ABS(sa-sb). Replace the quotient sign, including signed zero.
+      desired = calc(5, calc(4, signbit(lhs), signbit(rhs)))
+      magnitude = calc(4, out, calc(0, signbit(out), sign, mul=True))
+      corrected = calc(2, magnitude, calc(0, desired, sign, mul=True))
+      raw = bytes(to_mv(self.dev.input_buf+corrected-base, 32))
+      result.extend(typed_view(raw[i*4:i*4+2], dtypes.half) for i in range(count))
+    return result
+
   def run_npu(self, op:Ops, a:list, b:list|None=None, custom:str|None=None, dtype:DType=dtypes.half) -> list:
     if op is Ops.RECIPROCAL:
-      # Decode each typed view for the numeric guard; keep the original input bytes for the NPU.
-      if any(x == -math.inf or (x == 0 and math.copysign(1.0, x) < 0) for x in map(scalar16, a)):
-        raise NotImplementedError("ROCKCHIP NPU RECIPROCAL does not preserve the sign of negative zero or negative infinity")
-      return self.run_npu(Ops.FDIV, [1.0] * len(a), a)
+      return self.run_fdiv([1.0] * len(a), a)
@@
           elif u.op is Ops.MULACC and u.dtype == dtypes.half:
             values[u] = self.run_mulacc(*src_values)
+          elif u.op is Ops.FDIV and u.dtype == dtypes.half:
+            values[u] = self.run_fdiv(*src_values)
```

run_fdiv calls the raw run_npu(FDIV) path, not itself. Both FDIV and RECIPROCAL now use its correction.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_copysign_exact TestOps.test_div_naninf

Ran 2 tests in 11.547s
OK
```

Those cases passed, but they do not cross every special numerator with every denominator. Add that check before calling this complete. Compare NaN classification separately; for other results compare the bits, so a wrong zero sign cannot pass:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_fdiv_specials(self):
+    values = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan, 2**-24, 65504.], dtype=np.float16)
+    a, b = np.repeat(values, len(values)), np.tile(values, len(values))
+    actual = np.frombuffer(b"".join(self.program.run_fdiv(list(a), list(b))), dtype=np.float16)
+    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
+      expected = (a.astype(np.float64)/b.astype(np.float64)).astype(np.float16)
+    nan = np.isnan(expected)
+    np.testing.assert_array_equal(np.isnan(actual), nan)
+    np.testing.assert_array_equal(actual.view(np.uint16)[~nan], expected.view(np.uint16)[~nan])
+
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fdiv_specials

Mismatched elements: 7 / 81
Ran 1 test in 0.040s
FAILED (failures=1)
```

The wrong cases were +0/-0, -0/+0, -0/-0 and all four infinity/infinity sign combinations. They returned infinities instead of NaNs. Sign repair cannot fix that.

After removing the sign bits, FP16 magnitude words have useful integer ordering:

| Magnitude bits | Meaning |
| -------------- | ------- |
| 0              | Zero, either sign |
| 1..0x7bff      | Finite nonzero |
| 0x7c00         | Infinity |
| Above 0x7c00   | NaN |

Let lo and hi be the MIN and MAX of the two magnitude words. We need NaN when hi=0 (both zero), lo>=0x7c00 (both nonfinite), or hi>0x7c00 (either NaN). Each comparison gives an integer 0/1 mask. MAX combines those masks, then integer selection supplies canonical NaN bits:

```text
invalid = MAX(both_zero, both_nonfinite, has_nan)
result_bits = corrected_bits + invalid * (0x7e00 - corrected_bits)
```

Add this after the sign repair:

```diff
   def run_fdiv(self, a:list, b:list) -> list:
@@
-      desired = calc(5, calc(4, signbit(lhs), signbit(rhs)))
+      lhs_sign, rhs_sign = signbit(lhs), signbit(rhs)
+      desired = calc(5, calc(4, lhs_sign, rhs_sign))
       magnitude = calc(4, out, calc(0, signbit(out), sign, mul=True))
       corrected = calc(2, magnitude, calc(0, desired, sign, mul=True))
+      lhs_mag = calc(4, lhs, calc(0, lhs_sign, sign, mul=True))
+      rhs_mag = calc(4, rhs, calc(0, rhs_sign, sign, mul=True))
+      lower, upper = calc(1, lhs_mag, rhs_mag), calc(0, lhs_mag, rhs_mag)
+      zero, one, infinity = const(0), const(1), const(0x7c00)
+      both_zero = calc(4, one, calc(1, zero, upper, binary=True))
+      both_nonfinite = calc(4, one, calc(1, lower, infinity, binary=True))
+      has_nan = calc(1, infinity, upper, binary=True)
+      invalid = calc(0, both_zero, calc(0, both_nonfinite, has_nan))
+      # Integer selection supplies a canonical NaN for 0/0, inf/inf or a NaN operand.
+      corrected = calc(2, corrected, calc(0, invalid, calc(4, const(0x7e00), corrected), mul=True))
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fdiv_specials

Mismatched elements: 33 / 81
Ran 1 test in 0.107s
FAILED (failures=1)
```

More failures! Inspecting the intermediate INT32 lanes found `1 * 0x8000 = -32768`, not +32768. EW MUL's operand is signed INT16 in this setup. The sign-only version still produced the right low 16 bits, but the new magnitude checks used the full wrong INT32 value. For -1, magnitude extraction produced 80896 instead of 15360, so it looked like NaN.

Build the positive sign weight using `part = sign * 16384; part + part`. Both MUL operands now fit INT16, while ADD produces +32768 in INT32:

```diff
   def run_fdiv(self, a:list, b:list) -> list:
@@
-      sign, threshold = const(0x8000), const(0x7fff)
+      half_sign, threshold = const(0x4000), const(0x7fff)
@@
       def signbit(x:int) -> int: return calc(1, threshold, x, binary=True)
+      def signword(x:int) -> int:
+        # EW MUL's operand is signed INT16: build +32768 without multiplying by 0x8000.
+        part = calc(0, x, half_sign, mul=True)
+        return calc(2, part, part)
@@
-      magnitude = calc(4, out, calc(0, signbit(out), sign, mul=True))
-      corrected = calc(2, magnitude, calc(0, desired, sign, mul=True))
-      lhs_mag = calc(4, lhs, calc(0, lhs_sign, sign, mul=True))
-      rhs_mag = calc(4, rhs, calc(0, rhs_sign, sign, mul=True))
+      magnitude = calc(4, out, signword(signbit(out)))
+      corrected = calc(2, magnitude, signword(desired))
+      lhs_mag = calc(4, lhs, signword(lhs_sign))
+      rhs_mag = calc(4, rhs, signword(rhs_sign))
```

The last selection only needs the low 16 bits. Its difference may exceed signed INT16, but sign-extending those same low bits changes the integer by a multiple of 65536; multiplying by 0/1 and copying the low word preserves the selected encoding. Unlike magnitude classification, we do not compare that intermediate as an INT32 value.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fdiv_specials

Ran 1 test in 0.101s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_copysign_exact TestOps.test_div_naninf

Ran 2 tests in 25.877s
OK
```

All 81 pairs passed, including zero signs and invalid divisions. This adds NPU tasks; it is not a speed improvement or proof of correctly rounded FDIV for every finite pair. Python copies storage and submits tasks; sign correction and invalid-result selection run on the NPU.

Replaying the documents applied all 290 hunks and compiled all 95 command checkpoints. The reconstructed backend also passed the 81-pair check in 0.054s. Its test_div_naninf printed OK in 27.096s, but the command reached the 30-second limit during shutdown (exit 124). That is a completed test with an unclean command exit, not a clean end-to-end rerun. The two-test OK above is from the working runtime. The full-sweep total has not been updated.

## Scratch reuse

The saved sweep has three scratch-exhaustion failures. Before changing an allocator, check whether the tutorial actually builds that allocator. Here it does not: the working runtime had retained mapped output views and advanced one page per atom, while our earlier diffs copy completed output bytes.

The working runtime still reproduced the error:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_any

RuntimeError: ROCKCHIP intermediate buffer exhausted
Ran 1 test in 1.778s
FAILED (errors=1)
```

Its 4 MiB buffer held only 1024 separate 4096-byte pages. Repeated operations in one workgroup consumed those pages even when an atom contained only 16 useful bytes. Increasing the buffer would only move the limit.

The 1500 branch tracks live scratch storage with `_reuse_linear_scratch` and RKPlan. We do not need that machinery for the copied-result path already used here. Synchronize the working runtime instead: stop advancing the page after each atom, and copy `bytes(to_mv(..., 16))` after the blocking submit. Subsequent tasks may overwrite the scratch page but cannot overwrite those copied bytes. This is a storage copy, not CPU tensor arithmetic; it also preserves NaN payload bits.

There is no new runtime diff for a reader following this tutorial: its result-copy line is already present. Add a regression which crosses the old 1024-atom limit, keeps the first result alive, runs a second operation, and only then checks both:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_scratch_reuse(self):
+    # More atoms than the old 4 MiB scratch buffer held as separate pages.
+    values = [float(i % 16) for i in range(8200)]
+    first = self.program.run_npu(Ops.ADD, values, [1.0]*len(values))
+    second = self.program.run_npu(Ops.MUL, values, [2.0]*len(values))
+    for actual,expected in ((first, np.array(values)+1), (second, np.array(values)*2)):
+      np.testing.assert_array_equal(np.frombuffer(b"".join(actual), dtype=np.float16), expected)
+
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_scratch_reuse \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fdiv_specials

Ran 2 tests in 0.434s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_any

Ran 1 test in 2.092s
OK
```

test_simple_cummin and test_slice_fancy_indexing_tuple_indices each reached a separate 30-second command limit (exit 124). Neither printed the old scratch error before stopping, but neither completed. The indexing run also printed a multiprocessing semaphore-cleanup warning when terminated. Keep both as timeouts; the storage regression and test_any do not establish that these larger methods pass.

The backend reconstructed from these diffs passed both storage/FDIV checks in 0.305s. The working runtime also passed the earlier shift and selection cases after the storage change:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_lshift TestOps.test_lshift_signed TestOps.test_rshift TestOps.test_rshift_signed \
    TestOps.test_where TestOps.test_maximum

Ran 6 tests in 3.594s
OK
```

Next investigate the FP32 dtype gates. None of these sign or storage fixes provides general FP32 arithmetic, and the historical 209/433 total is still not a new sweep result.

## FP32 ADD, SUB and NEG

Start with a small reduction from the FP32 failure group:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sum_tiny

NotImplementedError: ROCKCHIP NPU does not support Ops.ADD with dtypes.float
Ran 1 test in 0.192s
FAILED (errors=1)
```

The input is half, but SUM accumulates in FP32. We already used private FP32 stages for MULACC. That does not release the public FP32 ADD gate or give run_npu the right four-byte packing.

The 1500 branch's _fp32_expr_to_half narrows some expressions at a half-storage boundary. That is not a general solution for FP32 inputs. For example, 1+2^-20 and 2^100 must not become half values before adding.

Probe mulacc_stage directly with eight packed FP32 lanes, precision=5 and output=5. On 1024 pairs (random raw words plus chosen boundaries), the native operations gave:

| Operation | Mismatches | Observed problem |
| --------- | ---------: | ---------------- |
| ADD       | 1 / 1024   | -0 + -0 returned +0 |
| SUB       | 3 / 1024   | inf - inf returned inf; opposite infinities returned NaN |

So we cannot just remove the gate. Next probe algorithm 6, NEG. It flipped the sign of zero, 1+2^-23, infinity and the smallest subnormal correctly. Algorithm 5, ABS, cleared their sign bits. These probes also kept the original FP32 precision.

Lets use ADD(a, NEG(b)) for SUB. That avoids the native SUB infinity behavior. ADD still needs its -0 result fixed:

```text
FP32 -0 bits = 0x80000000 = INT32_MIN
both_negative_zero = MAX(raw_a, raw_b) < INT32_MIN + 1
result_bits = raw_ADD_result + both_negative_zero * INT32_MIN
```

Here MAX and the comparison use signed INT32 words, not floating values. MAX can equal INT32_MIN only when both inputs have that exact encoding. Every other result is unchanged. Put the full-width constant on MUL's main-input side and the 0/1 mask on its operand side; the previous FDIV probe showed why that matters.

| Stage | Dtype | Work |
| ----- | ----- | ---- |
| NEG, for SUB only | FP32 | Flip b's sign |
| ADD | FP32 | Add a and the selected b |
| MAX, then binary MIN | INT32 bits | Detect two negative-zero encodings |
| MUL, then ADD | INT32 bits | Restore the negative-zero result |
| Readback | FP32 storage | Copy four bytes per lane |

Add the helper before run_fdiv. It reuses mulacc_stage; no new register sequence or lossy CAST is needed:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_float_alu(self, op:Ops, a:list, b:list|None=None) -> list:
+    assert op in (Ops.ADD, Ops.SUB, Ops.NEG)
+    assert (op is Ops.NEG and b is None) or (b is not None and len(a) == len(b))
+    base = self.dev.input_mem.dma_addr
+    to_mv(self.dev.input_buf+192, 32)[:] = struct.pack("<i", -2147483647)*8
+    to_mv(self.dev.input_buf+448, 32)[:] = struct.pack("<i", -2147483648)*8
+    result:list = []
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      for offset,values in ((0, a), (64, b)):
+        raw = b"".join(raw16(x, dtypes.float) for x in values[start:start+8]) if values is not None else b""
+        to_mv(self.dev.input_buf+offset, 32)[:] = raw+bytes(32-len(raw))
+      if op is Ops.NEG:
+        self.mulacc_stage(6, base, base+64, base+128)
+        output = 128
+      else:
+        rhs = base+64
+        if op is Ops.SUB:
+          self.mulacc_stage(6, rhs, base, base+256)
+          rhs = base+256
+        self.mulacc_stage(2, base, rhs, base+128)
+        # ADD loses -0 + -0. As signed INT32 bits, only -0 equals INT32_MIN.
+        self.mulacc_stage(0, base, rhs, base+320, precision=4, output=4)
+        self.mulacc_stage(1, base+320, base+192, base+384, precision=4, output=4, binary=True)
+        self.mulacc_stage(0, base+448, base+384, base+512, precision=4, output=4, mul=True)
+        self.mulacc_stage(2, base+128, base+512, base+576, precision=4, output=4)
+        output = 576
+      raw = bytes(to_mv(self.dev.input_buf+output, 32))
+      result.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
+    return result
+
   def run_fdiv(self, a:list, b:list) -> list:
```

Release only these three FP32 operations:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.MULACC and u.dtype == dtypes.half:
             values[u] = self.run_mulacc(*src_values)
+          elif u.op in (Ops.ADD, Ops.SUB, Ops.NEG) and u.dtype == dtypes.float:
+            values[u] = self.run_float_alu(u.op, src_values[0], src_values[1] if len(src_values) > 1 else None)
           elif u.op is Ops.FDIV and u.dtype == dtypes.half:
```

Check random raw FP32 words and all pairs of the selected special values. Compare non-NaN bits, not just a tolerance, so this also checks signed zeros and subnormal results:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_fp32_add_sub_neg(self):
+    rng = np.random.default_rng(42)
+    a, b = (rng.integers(0, 2**32, 1024, dtype=np.uint32).view(np.float32) for _ in range(2))
+    special = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan, 2**-149, 2**100], dtype=np.float32)
+    a, b = np.concatenate((a, np.repeat(special, len(special)))), np.concatenate((b, np.tile(special, len(special))))
+    for op,fn in ((Ops.ADD, np.add), (Ops.SUB, np.subtract), (Ops.NEG, np.negative)):
+      with self.subTest(op=op):
+        actual = np.frombuffer(b"".join(self.program.run_float_alu(op, list(a), None if op is Ops.NEG else list(b))), dtype=np.float32)
+        with np.errstate(all="ignore"): expected = fn(a) if op is Ops.NEG else fn(a, b)
+        nan = np.isnan(expected)
+        np.testing.assert_array_equal(np.isnan(actual), nan)
+        np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])
+
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_add_sub_neg

Ran 1 test in 0.250s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_sum_tiny TestOps.test_sum_simple TestOps.test_sum_relu

Ran 3 tests in 0.740s
OK
```

The direct check covers 1105 lanes for each of ADD, SUB and NEG, including a one-lane tail. It does not exhaust every FP32 pair. Now remove DEFAULT_FLOAT=HALF from the existing arithmetic tests:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_add TestOps.test_sub TestOps.test_neg

Ran 3 tests in 8.838s
OK
```

FP32 MUL and FDIV remain gated. Passing these ADD/SUB cases does not mean all 154 saved FP32-gate failures are fixed; the next operation in a test may still be unsupported.

The reconstructed tutorial backend also passed the direct FP32 check in 0.200s and the three sum tests in 0.489s. All 294 diffs replayed. The working runtime's targeted lint check passed.

## FP32 MUL: operand conversion still unresolved

Keep the default FP32 dtype and try the existing small multiply test:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tiny_mul

NotImplementedError: ROCKCHIP NPU does not support Ops.MUL with dtypes.float
Ran 1 test in 0.119s
FAILED (errors=1)
```

ADD working in FP32 does not establish that MUL reads its operand the same way. The earlier SIN probe already found that mulacc_stage with mul=True consumed FP16-looking operand bits. Recheck the converter controls without changing the known working 32-byte operand layout:

| EW_OP_CVT_BYPASS | EW_CVT_TYPE | Observed result |
| --------------: | ----------: | --------------- |
| 1               | 0           | 1 * 2 returned 0; (1+2^-23) squared returned about 5.960465e-8 |
| 1               | 1           | Same incorrect results |
| 0               | 0           | DRM_IOCTL_RKNPU_SUBMIT timed out, errno 110 |
| 0               | 1           | Not attempted after the timeout |

The NVDLA reference gives us a reason to inspect the operand converter rather than assume the multiplier cannot do FP32. In hw/cmod/hls/sdp/sdp_y_core.cpp, Y_mul multiplies two internal FP32 values. But sdp_y_cvt.cpp converts its floating operand from FP16 to FP32. That is reference evidence, not proof of the RK3588 wiring.

Could a register operand skip that conversion? With EW_OP_SRC=0 and ERDMA disabled, the first output lane for input 1 gave:

| REG_DPU_EW_OP_VALUE_0 | Interpretation we wanted | Observed first lane |
| -------------------- | ------------------------ | ------------------: |
| 0x40000000           | FP32 2                   | 0 |
| 0x00004000           | FP16 2                   | 2 |
| 0x3f800001           | FP32 1+2^-23             | 2^-24 |

That also follows the low FP16 operand bits. This probe only set operand register 0; it is not a complete eight-lane constant-MUL implementation.

The separate FDIV ALU probe with packed FP32 inputs was wrong too: 1 / 2 produced about 4.448422e-41 instead of 0.5. Do not route FP32 FDIV through the half helper or release either dtype gate yet.

After the converter timeout, the known-good FP32 ADD/SUB/NEG check passed again in 0.192s. No speculative register change was kept in the runtime. We still need a verified full-width operand path or a decomposition that preserves FP32 precision and range; narrowing both inputs to half is not a fix.

### FP32 MUL through integer significands

We can avoid that operand converter by doing the multiplication as integers. This is more work than native EW MUL, but it need not lose FP32 precision.

For a normal FP32 value, the significand has 24 bits including its hidden leading 1. Split it into two 12-bit pieces:

```text
ma = ah * 4096 + al
mb = bh * 4096 + bl

ma * mb = ah*bh * 2^24 + (ah*bl + al*bh) * 4096 + al*bl
```

Each small product is at most 4095*4095, which fits INT32. The cross sum also fits. Carry its low part into the low 24-bit word, then carry from there into the high word. We never ask one INT32 lane to hold the full 48-bit product.

| Stage | Storage / arithmetic | Work |
| ----- | -------------------- | ---- |
| Unpack | FP32 bytes → two zero-extended 16-bit words | Copy storage; do not numerically cast to half |
| Decode | INT32 | Extract sign, exponent and significand |
| Normalize | INT32 | Shift subnormal significands and adjust their exponents |
| Multiply | INT32 | Four 12-bit products; retain both 24-bit result words |
| Round | INT32 | Guard, round and sticky bits; ties go to even |
| Encode | INT32 bits → FP32 bytes | Handle zero, overflow, infinity and NaN |

First add the private helper's scratch allocation and integer operations. We will not dispatch FP32 MUL until the helper is finished and checked. floor_shift reuses the measured converter formula `round((2*x - (2^n-1))/2^(n+1)) = floor(x/2^n)`. Its inputs below stay small enough that doubling does not overflow.

jam keeps the low bit set if any discarded bit was nonzero. That remembers whether later rounding is an exact tie:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_float_mul(self, a:list, b:list) -> list:
+    assert len(a) == len(b)
+    base, result = self.dev.input_mem.dma_addr, []
+    for start in range(0, len(a), 8):
+      count, slot, constants = min(8, len(a)-start), 0, {}
+      def alloc(raw:bytes|None=None) -> int:
+        nonlocal slot
+        addr = base+64*slot
+        slot += 1
+        assert slot*64 <= self.dev.input_mem.size
+        if raw is not None: to_mv(self.dev.input_buf+addr-base, 32)[:] = raw+bytes(32-len(raw))
+        return addr
+      def const(x:int) -> int:
+        if x not in constants: constants[x] = alloc(struct.pack("<i", x)*8)
+        return constants[x]
+      def calc(algo:int, x:int, y:int, **kw) -> int:
+        out = alloc()
+        self.mulacc_stage(algo, x, y, out, precision=4, output=4, **kw)
+        return out
+      def add(x:int, y:int) -> int: return calc(2, x, y)
+      def sub(x:int, y:int) -> int: return calc(4, x, y)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
+      def maximum(x:int, y:int) -> int: return calc(0, x, y)
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      zero, one = const(0), const(1)
+      def floor_shift(x:int, n:int) -> int: return calc(4, add(x, x), const(2**n-1), shift=n+1)
+      def parity(x:int) -> int: return sub(x, mul(floor_shift(x, 1), const(2)))
+      def jam(x:int, lost:int) -> int:
+        odd = parity(x)
+        return add(sub(x, odd), maximum(odd, lt(zero, lost)))
+    return result
+
   def run_float_alu(self, op:Ops, a:list, b:list|None=None) -> list:
```

The high storage word contains the sign, eight exponent bits and seven fraction bits. Remove the sign, divide by 128 to get the exponent, and combine the remaining fraction bits with the low word. Add 2^23 for a normal value.

Subnormals have no hidden bit. Normalize with conditional shifts of 16, 8, 4, 2 and 1, subtracting each shift from the exponent. The 16-bit move uses two multiplications by 256 so neither multiplier exceeds signed INT16. All decisions are NPU masks:

```diff
   def run_float_mul(self, a:list, b:list) -> list:
@@
+      operands = []
+      for values in (a, b):
+        raw = [bytes(raw16(x, dtypes.float)) for x in values[start:start+8]]
+        lo, hi = (alloc(b"".join(x[i:i+2]+bytes(2) for x in raw)) for i in (0, 2))
+        sign = lt(const(32767), hi)
+        hi = sub(hi, mul(const(32768), sign))
+        magnitude = add(lo, mul(const(65536), hi))
+        exponent = floor_shift(hi, 7)
+        mantissa = add(lo, mul(const(65536), sub(hi, mul(exponent, const(128)))))
+        mantissa = add(mantissa, mul(const(8388608), lt(zero, exponent)))
+        exponent = maximum(exponent, one)
+        # Normalize subnormals without discarding their low significand bits.
+        for n in (16, 8, 4, 2, 1):
+          take = lt(mantissa, const(2**(24-n)))
+          factor = add(one, mul(take, const(2**min(n, 8)-1)))
+          mantissa = mul(mantissa, factor)
+          if n == 16: mantissa = mul(mantissa, factor)
+          exponent = sub(exponent, mul(take, const(n)))
+        upper = floor_shift(mantissa, 12)
+        operands.append((sub(mantissa, mul(upper, const(4096))), upper, exponent, sign, magnitude))
+      (al, ah, ae, sa, ma), (bl, bh, be, sb, mb) = operands
     return result
```

Now form the 48-bit product as high/low 24-bit words. The top product bit decides whether the normalized result needs one more exponent increment. Keep 24 significand bits plus three rounding bits; jam any further discarded bits into the last one:

```diff
   def run_float_mul(self, a:list, b:list) -> list:
@@
+      cross = add(mul(ah, bl), mul(al, bh))
+      cross_hi = floor_shift(cross, 12)
+      low = add(mul(al, bl), mul(sub(cross, mul(cross_hi, const(4096))), const(4096)))
+      carry = floor_shift(low, 24)
+      high = add(add(mul(ah, bh), cross_hi), carry)
+      low = sub(low, mul(const(16777216), carry))
+      top = lt(const(8388607), high)
+      exponent = add(sub(add(ae, be), const(127)), top)
+      # Retain 24 significand bits plus guard/round/sticky; normalize the 48-bit product.
+      r0, r1 = floor_shift(low, 20), floor_shift(low, 21)
+      extended = select(top, add(mul(high, const(8)), r1), add(mul(high, const(16)), r0))
+      lost = select(top, sub(low, mul(const(2097152), r1)), sub(low, mul(const(1048576), r0)))
+      extended = jam(extended, lost)
     return result
```

For an exponent below the normal range, shift right again before rounding. Clamp the shift distance to 31: this intermediate has at most 27 useful bits, so larger shifts also leave only a sticky bit and round to zero. The loop handles different distances per lane without a CPU shift of tensor values.

After shifting, divide by eight. A remainder above four rounds up; a remainder of four rounds up only when the retained significand is odd. Encoding `(exponent-1)*2^23 + significand` includes the hidden bit, and a carry from rounding advances the exponent naturally. A subnormal uses exponent contribution zero.

Finally select zeros, infinities and invalid products (NaN inputs or zero times infinity), and apply sign(a) XOR sign(b):

```diff
   def run_float_mul(self, a:list, b:list) -> list:
@@
+      distance = calc(1, maximum(sub(one, exponent), zero), const(31))
+      # Variable right shift with sticky bits, including gradual underflow.
+      for n in (16, 8, 4, 2, 1):
+        take = sub(one, lt(distance, const(n)))
+        shifted = floor_shift(extended, n)
+        restored = mul(const(65536), shifted) if n == 16 else mul(shifted, const(2**n))
+        extended = select(take, jam(shifted, sub(extended, restored)), extended)
+        distance = sub(distance, mul(take, const(n)))
+      significand = floor_shift(extended, 3)
+      remainder = sub(extended, mul(significand, const(8)))
+      tie = sub(one, add(lt(remainder, const(4)), lt(const(4), remainder)))
+      significand = add(significand, add(lt(const(4), remainder), mul(tie, parity(significand))))
+      biased = maximum(sub(calc(1, exponent, const(254)), one), zero)
+      bits = add(mul(const(8388608), biased), significand)
+      infinity = const(0x7f800000)
+      bits = select(lt(const(254), exponent), infinity, bits)
+      any_zero = sub(one, lt(zero, calc(1, ma, mb)))
+      any_inf = sub(one, lt(maximum(ma, mb), infinity))
+      invalid = maximum(lt(infinity, maximum(ma, mb)), mul(any_zero, any_inf))
+      bits = select(any_inf, infinity, select(any_zero, zero, bits))
+      sign = calc(5, sub(sa, sb), zero)
+      bits = select(invalid, const(0x7fc00000), add(bits, mul(const(-2147483648), sign)))
+      raw = bytes(to_mv(self.dev.input_buf+bits-base, 32))
+      result.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
     return result
```

Start with 128 random raw pairs and the 81 special-value pairs. Only the reference calculation uses NumPy arithmetic:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_fp32_mul(self):
+    rng = np.random.default_rng(42)
+    a, b = (rng.integers(0, 2**32, 128, dtype=np.uint32).view(np.float32) for _ in range(2))
+    special = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan, 2**-149, 2**100], dtype=np.float32)
+    a, b = np.concatenate((a, np.repeat(special, len(special)))), np.concatenate((b, np.tile(special, len(special))))
+    actual = np.frombuffer(b"".join(self.program.run_float_mul(list(a), list(b))), dtype=np.float32)
+    with np.errstate(all="ignore"): expected = a*b
+    nan = np.isnan(expected)
+    np.testing.assert_array_equal(np.isnan(actual), nan)
+    np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])
+
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_mul

Mismatched elements: 10 / 209
Ran 1 test in 0.679s
FAILED (failures=1)
```

The first direct probe found **10 wrong results out of 209**. For example, 0 * -inf returned 0x7f7fffff (the largest finite FP32 value), not NaN. The finite products matched.

The final select was subtracting a negative signed encoding from positive NaN bits. That difference exceeds INT32_MAX and saturates. This is the same reason we must check integer intermediate ranges, not just the final four bytes. Select NaN while both alternatives are nonnegative magnitude encodings, then add the sign:

```diff
   def run_float_mul(self, a:list, b:list) -> list:
@@
-      bits = select(invalid, const(0x7fc00000), add(bits, mul(const(-2147483648), sign)))
+      # Select nonnegative encodings before adding the sign, avoiding signed INT32 overflow.
+      bits = add(select(invalid, const(0x7fc00000), bits), mul(const(-2147483648), sign))
```

Increase the random coverage and add exact halfway cases. In particular, 1+2^-23 and 1+3*2^-23 multiplied by 1.5 exercise opposite tie-to-even decisions; the smallest subnormals multiplied by 0.5 check gradual underflow.

```diff
 class TestRockchipInteger(unittest.TestCase):
@@
   def test_fp32_mul(self):
     rng = np.random.default_rng(42)
-    a, b = (rng.integers(0, 2**32, 128, dtype=np.uint32).view(np.float32) for _ in range(2))
+    a, b = (rng.integers(0, 2**32, 4096, dtype=np.uint32).view(np.float32) for _ in range(2))
@@
     a, b = np.concatenate((a, np.repeat(special, len(special)))), np.concatenate((b, np.tile(special, len(special))))
+    # Even/odd rounding ties, gradual underflow, normal/overflow boundaries and both NaN signs.
+    edges = np.array([
+      [0x3f800001, 0x3fc00000], [0x3f800003, 0x3fc00000], [1, 0x3f000000], [3, 0x3f000000],
+      [0x80000001, 0x3f000000], [0x80000003, 0x3f000000], [0x007fffff, 0x3f800001], [0x00800000, 0x3f7fffff],
+      [0x7f7fffff, 0x3f800000], [0x7f7fffff, 0x3f800001], [0x00800001, 0x3f000000], [0x00800000, 1],
+      [0x7f800001, 0x3f800000], [0xff800001, 0x3f800000], [0x7fc12345, 0], [0xffc12345, 0x80000000],
+    ], dtype=np.uint32).view(np.float32)
+    a, b = np.concatenate((a, edges[:, 0])), np.concatenate((b, edges[:, 1]))
     actual = np.frombuffer(b"".join(self.program.run_float_mul(list(a), list(b))), dtype=np.float32)
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_mul

Ran 1 test in 12.746s
OK
```

All 4193 pairs passed, including the one-lane tail. This is not an exhaustive FP32-pair test. NaN classification is checked, not its payload. Now dispatch FP32 MUL:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op in (Ops.ADD, Ops.SUB, Ops.NEG) and u.dtype == dtypes.float:
             values[u] = self.run_float_alu(u.op, src_values[0], src_values[1] if len(src_values) > 1 else None)
+          elif u.op is Ops.MUL and u.dtype == dtypes.float:
+            values[u] = self.run_float_mul(*src_values)
           elif u.op is Ops.FDIV and u.dtype == dtypes.half:
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tiny_mul

Ran 1 test in 1.660s
OK
```

This test uses default FP32, not DEFAULT_FLOAT=HALF. Tensor multiplication and rounding run on the NPU; Python packs the raw words and dispatches the fixed stages.

### Fill the eight lanes

test_tiny_mul passes, but the first default-FP32 test_mul run reached the 30-second limit. There was no mismatch reported before the timeout; that is not a pass.

Counting calls to run_float_mul in test_tiny_mul showed 64 calls with one lane each. Our integer stages can process eight lanes, but NOOPT gives this kernel one local lane and the interpreter runs one workgroup at a time. Can we put eight independent workgroups into those eight lanes?

Only batch straight-line elementwise kernels with local_size=(1,1,1). Leave loops, local memory and the other operations on the original path. This changes dispatch, not the arithmetic or dtype gates.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
     warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
-    warp_size = len(warp)
-    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
+    # Batch only independent straight-line workgroups; retain the original path for control flow and local memory.
+    batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
+                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.SINK, Ops.NOOP, Ops.AFTER}
+    batch = 8 if local_size == (1,1,1) and all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
+    groups = itertools.product(*[range(x) for x in global_size[::-1]])
+    while group := list(itertools.islice(groups, batch)):
+      warp_size = len(warp)*len(group)
+      self.output_offset = 0
       values: dict[UOp, Any] = {}
```

Each lane now needs its own global index. Repeat the local indices for each workgroup; when batch=1 this is the old ordering. Scratch can restart for each batch because completed outputs are copied before reuse.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
         elif u.op is Ops.SPECIAL:
-          if u.arg[0] == 'g': values[u] = [idxs[2-int(u.arg[-1])]] * warp_size
-          elif u.arg[0] == 'l': values[u] = [x[2-int(u.arg[-1])] for x in warp]
+          if u.arg[0] == 'g': values[u] = [idxs[2-int(u.arg[-1])] for idxs in group for _ in warp]
+          elif u.arg[0] == 'l': values[u] = [x[2-int(u.arg[-1])] for _ in group for x in warp]
```

The same counted tiny test now made eight calls with eight lanes each. Check a partial batch too: 17 values should give 8, 8, 1, with no padded values stored.

```diff
+from unittest.mock import patch
 import numpy as np
 from tinygrad import Device, Tensor, dtypes
+from tinygrad.helpers import Context
@@
 class TestRockchipInteger(unittest.TestCase):
+  def test_elementwise_batch_tail(self):
+    sizes, original = [], RockchipProgram.run_float_mul
+    def counted(program, a, b):
+      sizes.append(len(a))
+      return original(program, a, b)
+    a, b = np.arange(17, dtype=np.float32)-8, np.full(17, 1.5, dtype=np.float32)
+    with Context(NOOPT=1), patch.object(RockchipProgram, "run_float_mul", counted):
+      actual = (Tensor(a, device="ROCKCHIP")*Tensor(b, device="ROCKCHIP")).numpy()
+    np.testing.assert_array_equal(actual, a*b)
+    self.assertEqual(sizes, [8, 8, 1])
+
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_elementwise_batch_tail \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_add_sub_neg \
    test.device.test_rockchip_integer.TestRockchipInteger.test_scratch_reuse

Ran 3 tests in 0.719s
OK
```

Check reductions and the unbatched shift/selection paths too:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_sum_tiny TestOps.test_sum_simple \
    TestOps.test_lshift TestOps.test_lshift_signed TestOps.test_rshift TestOps.test_rshift_signed \
    TestOps.test_where TestOps.test_maximum

Ran 8 tests in 4.458s
OK
```

Now retry the full default-FP32 multiplication test:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mul

Ran 1 test in 25.539s
OK
```

This command exited normally within the 30-second limit. The three small checks also passed with the code reconstructed from the tutorial, in 0.843s. Reconstructed test_mul printed OK in 24.528s, but its process then reached the 30-second limit during shutdown (exit 124). Its assertions passed; that reconstructed command did not finish cleanly. All 307 tutorial diff hunks replayed and compiled.

FP32 FDIV and the other saved failure groups still need work. These targeted passes do not replace the full-suite baseline.

## FP32 FDIV

Run the existing division test without DEFAULT_FLOAT=HALF:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div

NotImplementedError: ROCKCHIP NPU does not support Ops.FDIV with dtypes.float
Ran 1 test in 0.151s
FAILED (errors=1)
```

The earlier native FP32 FDIV probe did not give 0.5 for 1/2. Can we reuse the integer encoding work from MUL instead of narrowing to FP16?

After normalization, each nonzero finite significand is an integer in [2^23, 2^24). Let them be A and B. A/B is in (0.5, 2). If A < B, double A and subtract one from the result exponent, so the ratio is in [1, 2). Its biased exponent is now ea - eb + 127 - int(A < B).

Generate one quotient bit at a time:

| Step           | NPU integer operation                         |
| -------------- | --------------------------------------------- |
| Next bit       | bit = 1 - CMPLT(remainder, B)                  |
| Append it      | quotient = 2*quotient + bit                    |
| Next remainder | remainder = 2*(remainder - B*bit)              |
| Repeat         | 27 bits: 24 significand bits and 3 extra bits  |
| Sticky         | OR any remaining nonzero remainder into bit 0 |

These are arithmetic masks, not Python decisions on tensor values. The quotient stays below 2^27 and the remainder below 2^25. Both fit the INT32 stages we already measured. Our existing jam, gradual-underflow and ties-to-even rounding can consume this result just like the MUL result.

Zero divisors need a harmless denominator during the loop. Use max(B, 2^23); this leaves normalized nonzero B unchanged. Afterwards select zero, infinity or NaN from the original operand encodings, then apply the XOR of their signs. In particular, 0/0 and inf/inf are NaN, finite nonzero/0 is infinity, and finite/inf is zero.

Rename the shared helper to run_float_binary. Keep the old MUL calculation in its own branch; only FDIV generates quotient bits:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_float_mul(self, a:list, b:list) -> list:
-    assert len(a) == len(b)
+  def run_float_binary(self, op:Ops, a:list, b:list) -> list:
+    assert op in (Ops.MUL, Ops.FDIV) and len(a) == len(b)
     base, result = self.dev.input_mem.dma_addr, []
     for start in range(0, len(a), 8):
@@
           if n == 16: mantissa = mul(mantissa, factor)
           exponent = sub(exponent, mul(take, const(n)))
-        upper = floor_shift(mantissa, 12)
-        operands.append((sub(mantissa, mul(upper, const(4096))), upper, exponent, sign, magnitude))
-      (al, ah, ae, sa, ma), (bl, bh, be, sb, mb) = operands
-      cross = add(mul(ah, bl), mul(al, bh))
-      cross_hi = floor_shift(cross, 12)
-      low = add(mul(al, bl), mul(sub(cross, mul(cross_hi, const(4096))), const(4096)))
-      carry = floor_shift(low, 24)
-      high = add(add(mul(ah, bh), cross_hi), carry)
-      low = sub(low, mul(const(16777216), carry))
-      top = lt(const(8388607), high)
-      exponent = add(sub(add(ae, be), const(127)), top)
-      # Retain 24 significand bits plus guard/round/sticky; normalize the 48-bit product.
-      r0, r1 = floor_shift(low, 20), floor_shift(low, 21)
-      extended = select(top, add(mul(high, const(8)), r1), add(mul(high, const(16)), r0))
-      lost = select(top, sub(low, mul(const(2097152), r1)), sub(low, mul(const(1048576), r0)))
+        operands.append((mantissa, exponent, sign, magnitude))
+      (am, ae, sa, ma), (bm, be, sb, mb) = operands
+      if op is Ops.FDIV:
+        below = lt(am, bm)
+        exponent = sub(add(sub(ae, be), const(127)), below)
+        remainder = select(below, add(am, am), am)
+        denominator = maximum(bm, const(8388608))  # Keep zero-divisor lanes bounded until special-value selection.
+        extended = zero
+        # Binary long division: 24 significand bits and three rounding bits.
+        for _ in range(27):
+          bit = sub(one, lt(remainder, denominator))
+          extended = add(add(extended, extended), bit)
+          remainder = mul(sub(remainder, mul(denominator, bit)), const(2))
+        lost = remainder
+      else:
+        ah, bh = floor_shift(am, 12), floor_shift(bm, 12)
+        al, bl = sub(am, mul(ah, const(4096))), sub(bm, mul(bh, const(4096)))
+        cross = add(mul(ah, bl), mul(al, bh))
+        cross_hi = floor_shift(cross, 12)
+        low = add(mul(al, bl), mul(sub(cross, mul(cross_hi, const(4096))), const(4096)))
+        carry = floor_shift(low, 24)
+        high = add(add(mul(ah, bh), cross_hi), carry)
+        low = sub(low, mul(const(16777216), carry))
+        top = lt(const(8388607), high)
+        exponent = add(sub(add(ae, be), const(127)), top)
+        # Retain 24 significand bits plus guard/round/sticky; normalize the 48-bit product.
+        r0, r1 = floor_shift(low, 20), floor_shift(low, 21)
+        extended = select(top, add(mul(high, const(8)), r1), add(mul(high, const(16)), r0))
+        lost = select(top, sub(low, mul(const(2097152), r1)), sub(low, mul(const(1048576), r0)))
       extended = jam(extended, lost)
       distance = calc(1, maximum(sub(one, exponent), zero), const(31))
@@
       infinity = const(0x7f800000)
       bits = select(lt(const(254), exponent), infinity, bits)
-      any_zero = sub(one, lt(zero, calc(1, ma, mb)))
-      any_inf = sub(one, lt(maximum(ma, mb), infinity))
-      invalid = maximum(lt(infinity, maximum(ma, mb)), mul(any_zero, any_inf))
-      bits = select(any_inf, infinity, select(any_zero, zero, bits))
+      if op is Ops.FDIV:
+        az, bz = sub(one, lt(zero, ma)), sub(one, lt(zero, mb))
+        ai, bi = sub(one, lt(ma, infinity)), sub(one, lt(mb, infinity))
+        invalid = maximum(lt(infinity, maximum(ma, mb)), maximum(mul(az, bz), mul(ai, bi)))
+        bits = select(maximum(ai, bz), infinity, select(maximum(az, bi), zero, bits))
+      else:
+        any_zero = sub(one, lt(zero, calc(1, ma, mb)))
+        any_inf = sub(one, lt(maximum(ma, mb), infinity))
+        invalid = maximum(lt(infinity, maximum(ma, mb)), mul(any_zero, any_inf))
+        bits = select(any_inf, infinity, select(any_zero, zero, bits))
       sign = calc(5, sub(sa, sb), zero)
       # Select nonnegative encodings before adding the sign, avoiding signed INT32 overflow.
@@
             values[u] = self.run_float_alu(u.op, src_values[0], src_values[1] if len(src_values) > 1 else None)
           elif u.op is Ops.MUL and u.dtype == dtypes.float:
-            values[u] = self.run_float_mul(*src_values)
+            values[u] = self.run_float_binary(u.op, *src_values)
           elif u.op is Ops.FDIV and u.dtype == dtypes.half:
             values[u] = self.run_fdiv(*src_values)
```

Update the MUL checks for the shared name, and add a direct FDIV check. This compares raw bits for non-NaN results, including signed zero; NaN payloads are not promised.

```diff
@@
 @unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "serial RK3588 hardware tests; use -n0")
 class TestRockchipInteger(unittest.TestCase):
+  def test_fp32_div(self):
+    rng = np.random.default_rng(43)
+    a, b = (rng.integers(0, 2**32, 512, dtype=np.uint32).view(np.float32) for _ in range(2))
+    special = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan, 2**-149, 2**100], dtype=np.float32)
+    a, b = np.concatenate((a, np.repeat(special, len(special)))), np.concatenate((b, np.tile(special, len(special))))
+    edges = np.array([
+      [1, 0x40000000], [3, 0x40000000], [0x80000001, 0x40000000], [0x80000003, 0x40000000],
+      [0x00800000, 0x3f800001], [0x007fffff, 0x3f7fffff], [0x7f7fffff, 0x3f7fffff], [1, 1],
+      [0x3f800000, 0x40400000], [0x3f800001, 0x40400000], [0x7f800001, 0], [0xff800001, 0x7f800000],
+    ], dtype=np.uint32).view(np.float32)
+    a, b = np.concatenate((a, edges[:, 0])), np.concatenate((b, edges[:, 1]))
+    actual = np.frombuffer(b"".join(self.program.run_float_binary(Ops.FDIV, list(a), list(b))), dtype=np.float32)
+    with np.errstate(all="ignore"): expected = a/b
+    nan = np.isnan(expected)
+    np.testing.assert_array_equal(np.isnan(actual), nan)
+    np.testing.assert_array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])
+
   def test_elementwise_batch_tail(self):
-    sizes, original = [], RockchipProgram.run_float_mul
-    def counted(program, a, b):
+    sizes, original = [], RockchipProgram.run_float_binary
+    def counted(program, op, a, b):
       sizes.append(len(a))
-      return original(program, a, b)
+      return original(program, op, a, b)
     a, b = np.arange(17, dtype=np.float32)-8, np.full(17, 1.5, dtype=np.float32)
-    with Context(NOOPT=1), patch.object(RockchipProgram, "run_float_mul", counted):
+    with Context(NOOPT=1), patch.object(RockchipProgram, "run_float_binary", counted):
       actual = (Tensor(a, device="ROCKCHIP")*Tensor(b, device="ROCKCHIP")).numpy()
     np.testing.assert_array_equal(actual, a*b)
@@
     ], dtype=np.uint32).view(np.float32)
     a, b = np.concatenate((a, edges[:, 0])), np.concatenate((b, edges[:, 1]))
-    actual = np.frombuffer(b"".join(self.program.run_float_mul(list(a), list(b))), dtype=np.float32)
+    actual = np.frombuffer(b"".join(self.program.run_float_binary(Ops.MUL, list(a), list(b))), dtype=np.float32)
     with np.errstate(all="ignore"): expected = a*b
     nan = np.isnan(expected)
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_div

Ran 1 test in 2.876s
OK
```

All 605 pairs passed: 512 random raw pairs, 81 special-value pairs and 12 boundary pairs. This is not an exhaustive FP32 division test. Now allow FP32 FDIV and include independent FDIV workgroups in the eight-lane batch:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
     # Batch only independent straight-line workgroups; retain the original path for control flow and local memory.
     batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
-                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.SINK, Ops.NOOP, Ops.AFTER}
+                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.SINK, Ops.NOOP, Ops.AFTER}
     batch = 8 if local_size == (1,1,1) and all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
     groups = itertools.product(*[range(x) for x in global_size[::-1]])
@@
           elif u.op in (Ops.ADD, Ops.SUB, Ops.NEG) and u.dtype == dtypes.float:
             values[u] = self.run_float_alu(u.op, src_values[0], src_values[1] if len(src_values) > 1 else None)
-          elif u.op is Ops.MUL and u.dtype == dtypes.float:
+          elif u.op in (Ops.MUL, Ops.FDIV) and u.dtype == dtypes.float:
             values[u] = self.run_float_binary(u.op, *src_values)
           elif u.op is Ops.FDIV and u.dtype == dtypes.half:
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div

Ran 1 test in 25.259s
OK
```

The process exited normally within the 30-second limit. No FP16 narrowing or Python quotient calculation is used. This fixes this FP32 primitive; it does not establish that every division-based expression or saved failure now passes.

Rerun the shared MUL helper and the batch-tail check with FDIV:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_mul \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_div \
    test.device.test_rockchip_integer.TestRockchipInteger.test_elementwise_batch_tail

Ran 3 tests in 15.403s
OK
```

The half division path now batches independent FDIV workgroups too. Check it still handles signs and special values:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_div TestOps.test_div_naninf TestOps.test_copysign_exact

Ran 3 tests in 7.270s
OK
```

All 315 tutorial hunks replayed and compiled. The reconstructed code passed the FDIV, MUL and tail checks in 15.951s too. The full-suite count is still the saved baseline; no new complete sweep has run.

## Ops.CAST: remove the Python integer conversion

First run the existing CAST test before adding the numeric NPU conversion. This checkpoint was reconstructed and rerun in the new order, with FP32 arithmetic and eight-lane batching already present. The output below omits the other TRACE lines:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cast

8 Ops.CAST dtypes.float dtypes.float ... [dtypes.half]
9 Ops.STORE dtypes.void ... [dtypes.float, dtypes.float]

Ran 1 test in 0.233s
OK
```

It passes, but the numeric CAST branch still converts these values in Python. Seeing Ops.CAST in TRACE identifies the UOp, not where its arithmetic ran. This is not NPU CAST support or a check of NaN exponents.

The later POW investigation also found this fallback: checking whether an exponent is an integer raised on NaN. We introduce the shared numeric converter here, before the math sections that use it.

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

Rerun the same test after the dispatch change:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cast

Ran 1 test in 0.253s
OK
```

This rerun also counted run_cast calls without changing their inputs or results: two half → float, one int → float and two half → int. Batching packs multiple values into each call; the earlier unbatched count was 22. Both checkpoint commands exited normally within 30 seconds. This verifies those NPU paths, not every CAST: bool → float still uses the Python fallback.

## Ops.WHERE: FP32 storage

The saved convolution failure still reaches FP32 WHERE:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_strided_conv_transpose2d

NotImplementedError: ROCKCHIP NPU does not support Ops.WHERE with dtypes.float
Ran 1 test in 1.069s
FAILED (errors=1)
```

Before retrying the whole convolution, isolate selection with the existing direct UOp test. Running test_uops.py by file path failed to import test.helpers in this shell; use the module command from the repo root:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestFloatUOps.test_where

NotImplementedError: ROCKCHIP NPU does not support Ops.WHERE with dtypes.float
Ran 1 test in 0.067s
FAILED (errors=1)
```

Our FP16 WHERE already selects raw UINT16 storage. FP32 needs the same idea with UINT32, not a CAST to half:

```text
bool condition
    → WHERE(condition, BITCAST(a, uint32), BITCAST(b, uint32))
    → BITCAST(selected_word, float32)
```

The integer helper selects each byte with no + (yes-no)*mask. Bytes are in 0..255, so these INT32 intermediates cannot overflow. Rejoining the selected bytes preserves the original FP32 encoding; no floating multiply touches an unselected infinity or NaN. The 1500 branch's _raw_where uses the same raw-storage principle, but we can reuse our existing integer helper.

Change only the branch dtype and storage width in the matcher. Constants are cast to the WHERE result dtype before BITCAST; half branches still use UINT16.

```diff
 class RockchipRenderer(Renderer):
@@
   def _pm_lower_where(u:UOp) -> UOp:
     x, a, b = u.src
-    a, b = a.cast(dtypes.half), b.cast(dtypes.half)
+    a, b = a.cast(u.dtype), b.cast(u.dtype)
+    storage = dtypes.uint16 if u.dtype == dtypes.half else dtypes.uint32
     # Select storage bits: an unselected NaN must not contaminate the chosen value.
-    return x.where(a.bitcast(dtypes.uint16), b.bitcast(dtypes.uint16)).bitcast(u.dtype)
+    return x.where(a.bitcast(storage), b.bitcast(storage)).bitcast(u.dtype)
@@
-    # FP16 raw-bit selection reuses the exact integer WHERE path.
-    (UPat(Ops.WHERE, dtypes.half, src=(UPat(dtype=dtypes.bool),
-      UPat(dtype=(dtypes.half, dtypes.weakfloat)), UPat(dtype=(dtypes.half, dtypes.weakfloat))), name="u"),
+    # FP16/FP32 raw-bit selection reuses the exact integer WHERE path.
+    (UPat(Ops.WHERE, (dtypes.half, dtypes.float), src=(UPat(dtype=dtypes.bool),
+      UPat(dtype=(dtypes.half, dtypes.float, dtypes.weakfloat)), UPat(dtype=(dtypes.half, dtypes.float, dtypes.weakfloat))), name="u"),
      lambda u: RockchipRenderer._pm_lower_where(u)),
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest test.backend.test_uops.TestFloatUOps.test_where

Ran 1 test in 0.153s
OK
```

The direct test uses small values. Add raw-bit checks for signed zeros, subnormals, infinities and NaNs. Also select a broadcast constant just above 1.0, which would be lost by narrowing to half:

```diff
@@
 class TestRockchipInteger(unittest.TestCase):
+  def test_fp32_where_bits(self):
+    words = np.array([0, 0x80000000, 1, 0x80000001, 0x3f800001, 0x7f7fffff,
+                      0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f800001], dtype=np.uint32)
+    yes, no = np.resize(words, 257), np.resize(words[::-1], 257)
+    mask = np.arange(257) % 2 == 0
+    condition = Tensor(mask, device="ROCKCHIP")
+    a, b = (Tensor(x.view(np.float32), device="ROCKCHIP") for x in (yes, no))
+    actual = condition.where(a, b).numpy()
+    self.assertEqual(actual.dtype, np.dtype(np.float32))
+    np.testing.assert_array_equal(actual.view(np.uint32), np.where(mask, yes, no))
+    actual = condition.where(a, 1.0000001192092896).numpy()
+    np.testing.assert_array_equal(actual.view(np.uint32), np.where(mask, yes, np.uint32(0x3f800001)))
+
   def test_fp32_div(self):
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_where_bits

Ran 1 test in 0.809s
OK
```

Both 257-lane checks passed. The second branch is selected on alternating lanes; the first check compares every output bit, including NaN payloads. Python still copies storage bytes; the NPU selects them.

The same check passed with TRACE=1 using the reconstructed tutorial code (exit 0). Its final lane shows the intended path; argument lists are omitted here:

```text
13 Ops.BITCAST dtypes.uint
14 Ops.BITCAST dtypes.uint
15 Ops.WHERE dtypes.uint
16 Ops.BITCAST dtypes.float
17 Ops.STORE dtypes.void
```

The existing half WHERE, permuted WHERE, NaN-condition WHERE and maximum tests also passed together: 4 tests in 3.572s, recorded without TRACE. All 318 tutorial hunks replayed and compiled.

Retrying test_strided_conv_transpose2d reached the 30-second limit without a result. It no longer stopped at FP32 WHERE, but that is not a convolution pass. Keep it unresolved until a complete run finishes.
## Ops.SQRT

The following is the recorded FP16-decomposition accuracy investigation, before the FP32 and numeric CAST extensions now introduced above. It is not a new baseline in this order.

Rerun the decomposition after fixing selection.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sqrt

Not equal to tolerance rtol=0.001, atol=1e-06
Mismatched elements: 12 / 2925 (0.41%)
Max absolute difference among violations: 0.0004883
Max relative difference among violations: 0.00176
Ran 1 test in 262.923s
FAILED (errors=1)
```

Now we have an accuracy failure. For example, one result is 0.219482421875 instead of 0.2198486328125. 
This measured path uses our FP16 arithmetic and still has Python numeric CAST fallbacks; it is not evidence that tinygrad's decomposition fails on every backend.

Lets try a different SQRT implementation and compare its rounding. The following probes are separate candidates.

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

For this exhaustive check, enumerate every FP16 encoding and compare the NPU result with a double-precision reference rounded to half. Keep raw NaN inputs; compare NaN classification rather than payload:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def _check_half_unary(self, run, reference):
+    # A double-precision reference rounds once to FP16; compare every finite result bit, including -0.
+    for first in range(0, 65536, 512):
+      bits = np.arange(first, first+512, dtype=np.uint16)
+      values = bits.view(np.float16)
+      inputs = [typed_view(bits[i:i+1].tobytes(), dtypes.half) for i in range(len(bits))]
+      actual = np.frombuffer(b"".join(run(inputs)), dtype=np.float16)
+      with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
+        expected = reference(values.astype(np.float64)).astype(np.float16)
+      nan = np.isnan(expected)
+      np.testing.assert_array_equal(np.isnan(actual), nan)
+      np.testing.assert_array_equal(actual.view(np.uint16)[~nan], expected.view(np.uint16)[~nan])
+
+  def test_sqrt_all_half_patterns(self):
+    self._check_half_unary(self.program.run_sqrt, np.sqrt)
+
```

The exhaustive test takes more than the repository's default two-minute timeout, so use TEST_TIMEOUT=600. It still runs serially; no NPU tests run together.

```bash
$ TRACE=1 TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_sqrt_all_half_patterns \
    test/backend/test_ops.py::TestOps::test_sqrt \
    test/backend/test_ops.py::TestOps::test_rsqrt

3 passed in 227.83s (0:03:47)
```

The ADD, MUL, maximum, MULACC and raw-NaN regressions also passed: 6 passed in 6.46s.

Progress: **26 / 30** paths covered. EXP2, LOG2, POW and SIN remain. The full test_ops.py sweep is still pending.

## Ops.EXP2

With the SHR, comparison and WHERE fixes already in place, run the existing EXP2 decomposition before advertising EXP2:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp2

test_exp2 (__main__.TestOps.test_exp2) ... ok
Ran 1 test in 119.310s
OK
```

This is the reconstructed tutorial stage, with no custom EXP2 handler. Its generic CAST fallback still handles half ↔ INT16 in Python, so this numerical pass is not an NPU-only claim. The direct EXP2 candidate below avoids those conversions; it is not a fix for a measured precision failure in tinygrad's decomposition.

So test_exp2 already passes, but its numeric CASTs still run on CPU. We will explore a direct NPU path below, not an accuracy fix for that passing test. LOG2 and SIN will extend the same run_math helper, so apply these diffs before continuing to those sections.

Could LUT help instead? `~/npu/include/rknnops.h` has bounded SIN/EXP2 tables, and `~/rk3588/experimental/silu.py` shows table upload and reuse. But the SiLU example divides the result by its scale on the CPU; ops_rockchip_ref.py also decodes its LUT output on the CPU. We would need to probe NPU output scaling and accuracy before using either here. The 1500 branch uses polynomials for EXP2, LOG2 and SIN, not LUTs. A LUT could replace a reduced-range polynomial; it would not remove range reduction or special-value handling.



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
Horner evaluates the polynomial without building each power of r separately. Tinygrad also uses a polynomial; using Horner alone does not make our candidate more accurate. We still need to test its FP32 arithmetic and final FP16 rounding.

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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp2

test_exp2 (__main__.TestOps.test_exp2) ... ok

Ran 1 test in 13.945s

OK
```

Reuse the same all-encodings check for EXP2:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_exp2_all_half_patterns(self):
+    self._check_half_unary(lambda a: self.program.run_math(Ops.EXP2, a), np.exp2)
+
```

test_exp2 passes with the code built here, before LOG2. The all-encodings check also passed in an earlier run: `2 passed in 54.35s`. That longer sweep was not rerun for this revision.

## Ops.LOG2

First use the code built so far, without adding LOG2 support or forcing a different decomposition:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log2

Mismatched elements: 22 / 2925 (0.752%)
 [10, 18]: -0.225830078125 (ACTUAL), -0.22607421875 (DESIRED)
Max absolute difference among violations: 0.0007324
Max relative difference among violations: 0.00156
Ran 1 test in 127.103s
FAILED (errors=1)
```

The existing decomposition exceeds the tolerance on this backend. This run uses the code built so far, without TRANSCENDENTAL=2 or a custom LOG2 handler. The result shows an accuracy problem, but does not yet tell us whether it comes from the polynomial or rounding in our primitives.

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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log2

test_log2 (__main__.TestOps.test_log2) ... ok

Ran 1 test in 18.127s

OK
```

Add LOG2 to the all-encodings checks:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_log2_all_half_patterns(self):
+    self._check_half_unary(lambda a: self.program.run_math(Ops.LOG2, a), np.log2)
+
```

test_log2 passes with the code built here, before SIN. The all-encodings check also passed in an earlier run: `2 passed in 72.62s`. That longer sweep was not rerun for this revision.

## Ops.SIN

We already added UINT16 SHR for SQRT. Lets try the existing SIN decomposition next:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin

NotImplementedError: ROCKCHIP NPU does not support Ops.MUL with dtypes.float
Ran 1 test in 0.385s
FAILED (errors=1)
```

This is the reconstructed tutorial stage. SIN's decomposition reaches FP32 MUL, but our public arithmetic gate still accepts only half here. Private FP32 register helpers do not automatically enable that UOp. We need to handle its FP32 inputs, output and dispatch before judging the decomposition's accuracy. The candidate below has separate checks; those do not establish that tinygrad's decomposition needs replacing.

Can we just use mulacc_stage with mul=True? A direct probe packed eight FP32 values in each input, selected precision=5/output=5 and called mulacc_stage(0, lhs, rhs, out, mul=True). It did not produce FP32 multiplication:

| Lane | Input a             | Input b             | Expected FP32       | Observed             |
| ---- | ------------------- | ------------------- | ------------------- | -------------------- |
| 0    | 1.0000001192092896  | 1.0000001192092896  | 1.000000238418579   | 5.960465188081798e-8  |
| 2    | -2.5               | 4                   | -10                 | -0                   |

The input buffers were 64 bytes apart, with the 32-byte output another 64 bytes after them. This tests the existing helper's layout, not every possible FP32 mode. Do not release the FP32 MUL gate yet: we still need a working operand layout or a decomposition using the supported datapaths. A LUT would not repair this missing primitive in native SIN.

Changing ERDMA_DATA_SIZE to 1 and clearing EW_OP_CVT_BYPASS then timed out in DRM_IOCTL_RKNPU_SUBMIT (errno 110). The sweep stopped there; the other combinations were not tested. This is not evidence that all FP32 MUL modes are unsupported.

The native route is stopped at FP32 MUL, not at a measured SIN accuracy failure. Below we take an alternative route using the private run_math helper from EXP2/LOG2; we do not enable general FP32 MUL. The checked 1500 branch also uses a polynomial in `_dpu_sin`, not a LUT, and clamps input to ±10000. That clamp changes larger finite half inputs. Our candidate uses range reduction instead; its tests below check that separate implementation, not tinygrad's native decomposition.

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
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin

test_sin (__main__.TestOps.test_sin) ... ok

Ran 1 test in 30.043s

OK
```

This passes with the code built through this SIN step. Allow more than 30 seconds for this test; the earlier 30-second run timed out without a result.

Add SIN to the all-encodings checks:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_sin_all_half_patterns(self):
+    self._check_half_unary(lambda a: self.program.run_math(Ops.SIN, a), np.sin)
+
```

The helper reconstructed through this SIN step passed all 65,536 SIN inputs and test_sin: `2 passed in 117.79s`.

The earlier EXP2/LOG2 run passed both exhaustive checks and test_exp2/test_log2, but test_exp and test_log failed: **2 failed, 4 passed in 138.55s**. test_exp reaches unsupported FP32 MUL because Tensor.exp promotes its calculation. test_log loses accuracy when the rounded half LOG2 result is multiplied by the half ln(2) constant. Passing the primitive tests does not mean these composed functions pass.

Rerun all three primitive checks after the rounding corrections:

```bash
$ TRACE=1 TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_exp2_all_half_patterns \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_log2_all_half_patterns \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_sin_all_half_patterns \
    test/backend/test_ops.py::TestOps::test_exp2 \
    test/backend/test_ops.py::TestOps::test_log2 \
    test/backend/test_ops.py::TestOps::test_sin

6 passed in 234.77s (0:03:54)
```

Each exhaustive check covers all 65,536 FP16 bit patterns. Non-NaN outputs match the reference bits, including signed zeros and infinities; NaN classification matches too. Arithmetic, range reduction and rounding corrections run on the NPU. Python only sets constants, submits tasks and copies storage layouts.

Progress: **29 / 30** paths covered. POW remains, followed by the full test_ops.py sweep. General FP32 arithmetic is now introduced above; the composed log accuracy issue still needs its own check.

## Ops.POW

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow

test_pow (__main__.TestOps.test_pow) ... ok

Ran 1 test in 108.073s

OK
```
test_pow already passes using tinygrad's decomposition, before any POW-specific changes. It checks fixed exponents; we still need test_pow_full for tensor exponents. The existing Python CAST fallback also means this pass is not yet an NPU-only result.

Check both mask directions, including signed zero and NaN payloads:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_where_special_bits(self):
+    a = np.array([0, 0x8000, 0x7c01, 0xfc01, 0x7e00, 0x7c00, 0xfc00, 0x3c00], dtype=np.uint16)
+    b = a[::-1].copy()
+    for mask in (np.array([True, False]*4), np.array([False, True]*4)):
+      actual = Tensor(mask, device="ROCKCHIP").where(Tensor(a.view(np.float16), device="ROCKCHIP"),
+                                                    Tensor(b.view(np.float16), device="ROCKCHIP")).numpy().view(np.uint16)
+      np.testing.assert_array_equal(actual, np.where(mask, a, b))
+
```

Keep weakfloat constants in the matcher too: _pm_lower_where casts both branches to half before reinterpreting their bits.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_where_special_bits \
    test/backend/test_ops.py::TestOps::test_where

2 passed in 4.63s
```

The storage check includes both signed zeros, infinities and NaN payloads in either branch. A separate unittest rerun of test_where_special_bits passed in 0.219s using the reconstructed pre-CAST stage. The earlier positive-power probe gave [8, 0.25, 1.732, 2]. Negative bases and -inf with a fractional exponent also gave the expected values in that small probe. The NaN-exponent CAST error is still open.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_full

Not equal to tolerance rtol=0.001, atol=1e-06
Mismatched elements: 17 / 2925 (0.581%)
Max relative difference among violations: 0.002638

Ran 1 test in 107.463s
FAILED (errors=1)
```

This failure was reproduced before the wider POW changes below. Unlike test_pow's fixed exponents, test_pow_full supplies a tensor of exponents. It reaches a precision failure rather than the earlier NaN-selection problem. The composed path rounds LOG2 and MUL to half before EXP2; next we try retaining wider intermediates, without changing the tolerance.

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

Add the wider POW, comparison and CAST checks before running the final focused suite:

```diff
 class TestRockchipInteger(unittest.TestCase):
+  def test_pow_wide_intermediates(self):
+    rng = np.random.default_rng(61)
+    a = rng.integers(1, 0x7c00, 4096, dtype=np.uint16).view(np.float16)
+    b = rng.uniform(-4, 4, len(a)).astype(np.float16)
+    actual = np.frombuffer(b"".join(self.program.run_exp2_mul_log2(list(a), list(b))), dtype=np.float16)
+    with np.errstate(over="ignore", under="ignore"):
+      expected = np.power(a.astype(np.float64), b.astype(np.float64)).astype(np.float16)
+    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
+    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
+
+  def test_half_compare_patterns(self):
+    # Every input encoding participates, including all NaN payloads and both zeros.
+    raw = np.arange(65536, dtype=np.uint16)
+    rhs = np.resize(np.array([0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7c01, 0xfc01], dtype=np.uint16), len(raw))
+    a, b = ([typed_view(x[i:i+1].tobytes(), dtypes.half) for i in range(len(x))] for x in (raw, rhs))
+    for op, reference in ((Ops.CMPEQ, np.equal), (Ops.CMPNE, np.not_equal), (Ops.CMPLT, np.less)):
+      actual = np.frombuffer(b"".join(self.program.run_half_compare(op, a, b)), dtype=np.bool_)
+      with np.errstate(invalid="ignore"): expected = reference(raw.view(np.float16), rhs.view(np.float16))
+      np.testing.assert_array_equal(actual, expected)
+
+  def test_half_int_cast(self):
+    bits = np.arange(65536, dtype=np.uint16)
+    values = bits.view(np.float16)
+    values = values[np.isfinite(values)]
+    actual = np.frombuffer(b"".join(self.program.run_cast(list(values), dtypes.half, dtypes.int)), dtype=np.int32)
+    np.testing.assert_array_equal(actual, np.trunc(values.astype(np.float64)).astype(np.int32))
+
```

First check the wider POW helper on its own:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_pow_wide_intermediates

Ran 1 test in 8.507s
OK
```

This rerun used the code built through the diffs above and passed all 4,096 magnitude pairs. It checks run_exp2_mul_log2 directly; it does not replace the full Tensor POW test.

The earlier focused checks also passed all 65,536 half bit patterns for EQ/NE/LT with the other operand cycling through eight special values, and all 63,488 finite half values cast to INT32. That comparison check is not every possible pair of half values.

Reconstructing run_math, run_exp2_mul_log2, run_cast and run_half_compare from these diffs gave the same helpers as the runtime. Their focused checks, including special-value WHERE, gave `4 passed in 74.35s`.

The saved combined run gave **7 passed, 1 failed in 557.49s**. `test_pow_full`, `test_pow`, `test_pow_neg_inf_frac_exponent` and `test_pow_zero_exponent` passed. `test_pow_const` failed at `x ** 8.0`: 617 / 2925 values exceeded the tolerance. The same test on tinygrad CPU, with the same HALF/NOOPT settings, failed at the same expression with the same 617 mismatches. This expression becomes repeated FP16 MULs, so the wider LOG2/EXP2 path does not apply.

Now rerun the same full Tensor test that failed before widening the intermediates:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_full

test_pow_full (__main__.TestOps.test_pow_full) ... ok
Ran 1 test in 293.597s
OK
```

This run used the backend built from these diffs, not the later runtime. Allow at least five minutes: the first attempt stopped at a 180-second command limit without a result; the 300-second rerun finished. The inputs and tolerance are unchanged from the failing run.

The code built here does pass these shorter special-value checks:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_pow_neg_inf_frac_exponent TestOps.test_pow_zero_exponent TestOps.test_pow_zero_tensor

Ran 3 tests in 0.636s
OK
```

This is not a claim of every POW edge case. For example, `Tensor([1.0]) ** Tensor([nan])` returns NaN on tinygrad CPU too. A Python scalar NaN exponent instead fails in the common `simplify_pow` rewrite before reaching either backend.

## Shared output converter setup

TOREVIEW1: We now repeat the same output offset/shift writes in convolution and mulacc_stage. Extract those two writes; keep scale, precision and byte layout in their task builders. The emitted words and their order must stay unchanged.

Both callers set CVT_TYPE=1. Convolution leaves CVT_ROUND=0; mulacc_stage sets it only for a nonzero shift. Pass that difference explicitly rather than assuming the modes are interchangeable:

```diff
@@
 class RockchipProgram(Program['RockchipDevice']):
+  def output_cvt_registers(self, offset:int=0, shift:int=0, rounding:bool=False) -> list[int]:
+    E = self.EMIT
+    return [
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, offset),
+      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
+        (1 << rk.DPU_OUT_CVT_SHIFT_CVT_TYPE__SHIFT) | (rounding << rk.DPU_OUT_CVT_SHIFT_CVT_ROUND__SHIFT) |
+        (shift << rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT)),
+    ]
+
@@
     self.build_conv_uint8_registers(1, out_addr)
-    self.npu_regs += [
-      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, offset),
-      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, (1 << rk.DPU_OUT_CVT_SHIFT_CVT_TYPE__SHIFT) | (shift << rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT)),
-    ]
+    self.npu_regs += self.output_cvt_registers(offset, shift)
@@
       E(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
         ((output == 2) << rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT) | (1 << rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT)),
-      E(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0),
-      E(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
-        (1 << rk.DPU_OUT_CVT_SHIFT_CVT_TYPE__SHIFT) | ((shift != 0) << rk.DPU_OUT_CVT_SHIFT_CVT_ROUND__SHIFT) |
-        (shift << rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT)),
+    ]
+    self.npu_regs += self.output_cvt_registers(shift=shift, rounding=shift != 0)
+    self.npu_regs += [
```

The helper emitted identical register words for 256 combinations of offset, shift and rounding. The current-runtime regression below also exited 0 within the 30-second limit; its full TRACE is omitted:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m unittest \
    test.backend.test_ops.TestOps.test_lshift \
    test.backend.test_ops.TestOps.test_rshift \
    test.device.test_rockchip_integer.TestRockchipInteger.test_fp32_add_sub_neg
```

This is a register-construction refactor, not a new arithmetic path. All 323 tutorial hunks replay and compile; the test above used the current runtime, not a newly reconstructed hardware checkpoint.

## Full-suite sweep

The full-runtime run reached **209 / 433 passed**. It ran the whole file serially, with a longer timeout for slow NPU decompositions. No comparisons were relaxed or cases removed. This is the earlier runtime sweep; the step-by-step reruns above did not repeat the whole file.

```bash
$ TRACE=1 TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
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

Group the 195 failures by the error reached in the saved logs. Each method is counted once; fixing its first error may expose another:

| Failure group                       | Count | Examples / observed error                                                |
| ----------------------------------- | ----: | ----------------------------------------------------------------------- |
| Unsupported FP32 operations          |   154 | ADD: 105, MUL: 45, SUB: 2, FDIV: 1, WHERE: 1                              |
| Unsupported boolean WHERE           |     6 | masked_select, nonzero, scatter, scatter_add, scatter_mul                 |
| Finite-value accuracy mismatch       |    17 | asin, asinh, atan, atanh, cumprod, gelu, log, pow_const, sigmoid, tanh     |
| Sign / infinity mismatch             |     2 | copysign_exact returns the wrong sign; div_naninf has wrong infinity signs |
| Boolean result mismatch              |     1 | isclose returns False where the reference returns True                   |
| Runtime scratch exhaustion           |     3 | any, simple_cummin, slice_fancy_indexing_tuple_indices                    |
| Reference / dtype / configuration    |     7 | avg_pool3d, bitcast, mulacc_with_zero_strides, normalize_int,             |
|                                     |       | pad_replicate_mode, scatter_reduce_prod_zeros, stack                     |
| Expected exception not raised        |     1 | scatter_reduce_errors                                                   |
| Missing traceback                    |     4 | 9_gemm, acos, acosh, all                                                 |
| **Total failed**                     | **195** |                                                                       |

The FP32 rows are dtype gates, not proof that those operations are numerically wrong. Here dtypes.float is FP32, even though the test command uses DEFAULT_FLOAT=HALF: promotion or explicit dtypes can still produce FP32 UOps. WHERE's bool/float error names its result dtype, not necessarily the condition's dtype.

The accuracy group means a finite result exceeded the test tolerance; it does not establish rounding as the root cause of every case. Keep isclose and the sign/infinity failures separate from ordinary precision errors.

The seven reference/configuration cases include Torch's missing Half avg_pool3d, an invalid Half-to-Int view shape, two dtype mismatches, invalid pooling dimensions, scatter dtype disagreement, and stack comparing FP16 3.140625 against 3.14 at rtol=1e-7. These need individual investigation, not a blanket NPU precision fix.

The first process recorded four FAILED statuses before crashing during test_all_large, without printing their tracebacks. Their causes remain unclassified. The **21 timeouts** and **8 skips** are separate from these 195 failures; a timeout alone does not tell us whether the cause is slow execution or a hang.

The saved scratch-exhaustion failures came from the working runtime's mapped-result allocator. The tutorial above copies each completed result instead; it does not introduce that allocator. Do not add its page-lifetime handling just to reproduce this failure.

CPU baselines under the same HALF settings also failed some accuracy cases. For example, test_pow_const had the same 617 / 2925 mismatches on CPU. Other cases differed: test_asin passed on CPU, and GELU/TANH had more mismatches on Rockchip. A shared CPU failure does not make an NPU result correct.

The five forward comparison methods passed, but their shared special-value loop leaves NaN commented out. The separate raw-bit checks above cover NaNs. FORWARD_ONLY does not disable methods that explicitly call backward().

Likewise, test_cast passing does not prove every conversion ran on the NPU: bool → FP32 still uses the Python fallback. The NPU CAST modes introduced here are listed explicitly in their dispatch.

Before the longer checks, rerun the earlier shift and selection cases with all the diffs applied:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_lshift TestOps.test_lshift_signed TestOps.test_rshift TestOps.test_rshift_signed \
    TestOps.test_where TestOps.test_maximum

Ran 6 tests in 3.567s
OK
```

This run used the backend built from the tutorial. The later helpers did not break these six earlier cases.

The saved longer focused run was:

```bash
$ TRACE=1 TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest -n0 -v --tb=short test/device/test_rockchip_integer.py \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc

18 passed, 124 subtests passed in 434.68s
```

These checks cover every half encoding for SQRT/EXP2/LOG2/SIN, raw comparison and CAST checks, integer boundaries, wide shifts, THREEFRY and special-value WHERE. They do not replace the full sweep: its result remains **209 / 433 passed**.

The earlier targeted lint check passed, but whole-tree mypy and ruff reported 47 and 142 errors respectively in reference/generated files. Those whole-tree checks have not been rerun here, so we cannot call them clean.

### What to fix next

Lets use rockchip-2608-1500 as a reference, not copy everything from it. The version checked here is commit `078eee94d8f1912bb63ce786c250d2c24984844a`. Most of these helpers are in tinygrad/renderer/rockchip.py; buffer allocation and task submission are in tinygrad/runtime/ops_rockchip.py. These are proposed next steps, not fixes tested in this tutorial yet.

| Failure group | Next step | Reference in 1500 |
| ------------- | --------- | ----------------- |
| FP32 ops | Probe FP32 ADD/SUB first, then MUL and FDIV. Check input packing, register precision and output layout before releasing each gate. | `_convert`, `emit_ew_stage` |
| Boolean WHERE | Cast the bool branches to FP16 0/1, reuse our raw-bit WHERE, then CAST the selected mask back to bool. Check all eight input combinations first. | `_raw_where`, `_i16_select` select storage without floating-branch contamination |
| Accuracy | Trace the failing expression first. Keep wider intermediates where rounding loses information; use a stable formula where cancellation is the problem. | `_precise_add_parts`, `_precise_mul_sum`, `_accurate_add_recipe`, `_fold_quadratic` |
| Sign / infinity | Preserve the stored sign bit for copysign, including -0. Probe division's zero, infinity and NaN combinations before adding special-value handling. | `_alu` has a constant-infinite-numerator workaround |
| isclose | Inspect the tolerance expression and its intermediate dtypes. Check comparison boundaries, equal infinities and NaNs separately. | Raw comparison handling is relevant; no dedicated isclose fix was found |
| Scratch exhaustion | Reuse storage after its last consumer and allocate enough for the peak live values. | `_reuse_linear_scratch`, `RKPlan`, runtime `_ensure_buffer` |
| Reference / configuration | Check each test's intended dtypes and shapes. Fix invalid reference setup only when justified; do not loosen assertions to get a pass. | Compare test source and configuration, not just the branch's pass count |
| Missing exception | Find which invalid scatter call failed to raise, then add the required validation before dispatch. | Start with shared Tensor validation, not registers |
| Missing traceback | Rerun the four cases individually with bounded time and captured output. | No fix can be selected from the FAILED status alone |
| Timeouts | Distinguish too many small tasks from a driver hang. Batch lanes for the former; inspect submissions and mode transitions for the latter. | `_tile` and grouped EW submission |

There are some traps in copying the reference:

1. Its `_FIXED_LAYOUTS` maps FP32 values to half storage, and `_fp32_expr_to_half` narrows expressions. That is not general FP32 arithmetic support. We need to verify the real FP32 datapath, not just remove our dtype check.
2. Its division workaround only recognizes a constant infinite numerator: `(signed_one / denominator) / 0`. It is a useful candidate, not a complete fix for every sign, zero or NaN case.
3. Its `_raw_where` selects integer storage rather than multiplying floating branches, so an unused NaN cannot contaminate the result. Reuse that idea with our existing NPU bool-byte conversion, not a new CPU conversion.
4. Scratch lifetime tracking applies to the mapped-result runtime. The tutorial currently copies completed results, so it does not need that allocator just to follow these steps.

Boolean WHERE, sign handling, scratch reuse and FP32 arithmetic now appear earlier in the tutorial. Next verify FP32 comparisons, then revisit the accuracy failures with those primitives available. Investigate the reference errors and missing tracebacks separately.

FP32 support is the biggest group, but opening those gates does not mean another 154 tests will pass. Each test may expose another missing operation or accuracy problem after its first error is fixed.

For each group, keep a small check before rerunning the larger cases:

| Group                 | What would show the proposed fix works?                                  |
| --------------------- | ----------------------------------------------------------------------- |
| FP32 gates            | Values not representable in FP16 survive ADD/SUB/MUL/FDIV; then rerun the failing methods without narrowing their dtypes. |
| Bool WHERE            | All eight bool combinations, then nonzero and masked_select.             |
| Finite accuracy       | Capture the first wrong intermediate and compare it with the CPU backend under the same settings; retest the original tolerance. |
| Sign / infinity       | Cross zero, -0, finite values, both infinities and NaN; compare zero signs and NaN classification as well as numeric values. |
| isclose               | Just below/at/above the tolerance boundary, equal infinities and NaNs, with both equal_nan settings. |
| Scratch exhaustion    | Measure peak live storage, verify reused pages have no remaining readers, then rerun the three failing methods. |
| Reference / config    | Reproduce the reference-side error independently before deciding whether the test or backend needs changing. |
| Missing exception     | Identify the exact invalid scatter input and preserve the expected exception type. |
| Missing traceback     | Obtain each individual traceback before assigning a cause or proposing code. |
| Timeouts              | Record task count and progress under a time limit; a faster helper alone does not establish a full-test pass. |

The 1500 helpers are candidates for these checks, not evidence that our failures are already solved. Keep tinygrad's native decomposition first; use a different formula or LUT only after tracing an actual missing primitive or accuracy failure. The eight skips also need their decorators checked separately: skipped is neither passed nor a hardware failure.

The earlier sections have targeted checks for boolean/FP32 WHERE, division sign handling, scratch reuse and FP32 ADD/SUB/NEG/MUL/FDIV. That does not clear their whole failure groups: masked_select still needs a completed run, the tutorial has not yet added FP32 comparisons, and the accuracy cases need their original tests rerun. Until another complete sweep finishes, **209 / 433 is the saved baseline, not the current pass count**.

## FP32 comparisons

The tutorial still only dispatches comparisons with two half inputs. The earlier default-FP32 run stopped here:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_lt

11 Ops.CMPLT dtypes.bool ... [dtypes.float, dtypes.float]
NotImplementedError: ROCKCHIP NPU does not support Ops.CMPLT with dtypes.bool
Ran 1 test in 0.120s
FAILED (errors=1)
```

Bool is the result dtype; the missing inputs are FP32. Can we extend the raw-encoding comparison instead of casting them to half?

Read each FP32 word as signed INT32. Its sign is word < 0. Clear that sign to get the magnitude encoding, then reuse the existing ordering, zero and NaN masks:

| Step      | FP16                          | FP32                                |
| --------- | ----------------------------- | ----------------------------------- |
| Sign      | zero-extended word > 32767     | signed word < 0                     |
| Magnitude | word - sign*32768             | word + sign*2^30 + sign*2^30         |
| Infinity  | 0x7c00                        | 0x7f800000                          |
| Both zero | both magnitudes are zero      | MAX(magnitude_a, magnitude_b) = 0    |

Why two additions? The first candidate subtracted sign*INT32_MIN. The probe returned INT32_MIN for SUB(-1082130432, INT32_MIN), instead of 1065353216, so -1.0 and -2.0 incorrectly compared equal. The initial FP32 check had 271 wrong equality masks and 129 wrong less-than masks out of 1073 pairs. Two additions of 2^30 keep each intermediate within signed INT32. MAX also avoids overflowing a sum of two magnitude encodings.

Rename the helper and keep its FP16 default for existing callers:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_float_compare(self, op:Ops, a:list, b:list, dtype:DType=dtypes.half) -> list:
-  def run_half_compare(self, op:Ops, a:list, b:list) -> list:
     assert op in (Ops.CMPEQ, Ops.CMPNE, Ops.CMPLT) and len(a) == len(b)
+    assert dtype in (dtypes.half, dtypes.float)
     result:list = []
     base = self.dev.input_mem.dma_addr
@@
       def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
       def select(mask:int, yes:int, no:int) -> int: return calc(2, no, mul(sub(yes, no), mask))
+      lhs, rhs = (alloc(b"".join(bytes(raw16(x, dtype))+bytes(4-dtype.itemsize) for x in xs[start:start+8])+bytes(4*(8-count)))
-      lhs, rhs = (alloc(b"".join(bytes(raw16(x, dtypes.half))+bytes(2) for x in xs[start:start+8])+bytes(4*(8-count)))
                   for xs in (a, b))
+      # FP16 words are zero-extended; full FP32 words are read as signed INT32.
+      sa, sb = (lt(const(32767), x) if dtype == dtypes.half else lt(x, zero) for x in (lhs, rhs))
+      if dtype == dtypes.half:
+        ma, mb = sub(lhs, mul(const(32768), sa)), sub(rhs, mul(const(32768), sb))
+      else:
+        # SUB with INT32_MIN as its operand failed the probe; two bounded additions clear the sign instead.
+        ma, mb = (calc(2, calc(2, x, offset), offset) for x, offset in
+                  ((lhs, mul(const(1073741824), sa)), (rhs, mul(const(1073741824), sb))))
+      infinity = const(0x7c00 if dtype == dtypes.half else 0x7f800000)
+      valid = mul(sub(one, lt(infinity, ma)), sub(one, lt(infinity, mb)))
+      both_zero = sub(one, lt(zero, calc(0, ma, mb)))  # MAX avoids overflowing ma+mb for FP32 encodings.
-      sa, sb = lt(const(32767), lhs), lt(const(32767), rhs)
-      ma, mb = sub(lhs, mul(const(32768), sa)), sub(rhs, mul(const(32768), sb))
-      valid = mul(sub(one, lt(const(0x7c00), ma)), sub(one, lt(const(0x7c00), mb)))
-      both_zero = sub(one, lt(zero, calc(2, ma, mb)))
       ab, ba = lt(lhs, rhs), lt(rhs, lhs)
       if op is Ops.CMPLT:
```

Now allow either input width:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-          elif u.op in GroupOp.Comparison and src_dtypes == [dtypes.half, dtypes.half]:
-            values[u] = self.run_half_compare(u.op, *src_values)
+          elif u.op in GroupOp.Comparison and src_dtypes in ([dtypes.half]*2, [dtypes.float]*2):
+            values[u] = self.run_float_compare(u.op, *src_values, dtype=src_dtypes[0])
```

Check 1024 random raw pairs and all 49 pairs of zero, negative zero, ±1, ±infinity and NaN, for each input width and each comparison. Update the older half-only check for the renamed helper too:

```diff
@@
 class TestRockchipInteger(unittest.TestCase):
+  def test_float_compare_patterns(self):
+    rng = np.random.default_rng(44)
+    for dtype, word_dtype, float_dtype in ((dtypes.half, np.uint16, np.float16), (dtypes.float, np.uint32, np.float32)):
+      raw, rhs = (rng.integers(0, 2**(8*dtype.itemsize), 1024, dtype=word_dtype) for _ in range(2))
+      special = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan], dtype=float_dtype).view(word_dtype)
+      raw, rhs = np.concatenate((raw, np.repeat(special, 7))), np.concatenate((rhs, np.tile(special, 7)))
+      a, b = ([typed_view(x[i:i+1].tobytes(), dtype) for i in range(len(x))] for x in (raw, rhs))
+      for op, reference in ((Ops.CMPEQ, np.equal), (Ops.CMPNE, np.not_equal), (Ops.CMPLT, np.less)):
+        with self.subTest(dtype=dtype, op=op):
+          actual = np.frombuffer(b"".join(self.program.run_float_compare(op, a, b, dtype)), dtype=np.bool_)
+          with np.errstate(invalid="ignore"): expected = reference(raw.view(float_dtype), rhs.view(float_dtype))
+          np.testing.assert_array_equal(actual, expected)
+
@@
-      actual = np.frombuffer(b"".join(self.program.run_half_compare(op, a, b)), dtype=np.bool_)
+      actual = np.frombuffer(b"".join(self.program.run_float_compare(op, a, b)), dtype=np.bool_)
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python -m unittest \
    test.device.test_rockchip_integer.TestRockchipInteger.test_float_compare_patterns

Ran 1 test in 1.901s
OK
```

This fresh current-runtime check passed all six dtype/op combinations. It is not exhaustive over FP32 pairs and does not replace the full Tensor comparison tests. No Python floating comparison chooses the result; the reference calculation only checks the NPU output.
