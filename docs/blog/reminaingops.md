# Remaining ops

Start with the smaller changes, then build the helpers needed by the later ops. This order starts from the completed SHL/SHR sections in blog.md:

Accuracy first: skip cases already known to reach the 30-second limit. Keep their recorded timeouts as unresolved, not passes, and do not rerun them or add performance fixes in this pass. Existing setup diffs remain where later code depends on them.

| Order | Ops / support                         | Prerequisite                                      |
| ----- | ------------------------------------- | ------------------------------------------------- |
| 1     | TRUNC                                 | Existing unary EW setup                           |
| 2     | AND → XOR → OR                        | Convolution shifts and shared digit tables        |
| 3     | BITCAST → MULACC                      | Raw storage, then private FP32/INT32 stages       |
| 4     | CDIV → CMOD; floor division/remainder | Exact integer limbs and sign correction           |
| 5     | THREEFRY                              | Integer arithmetic, bitwise ops and shifts        |
| 6     | Shared comparisons and WHERE          | Raw encodings and integer selection               |
| 7     | Half FDIV fixes; scratch reuse        | Sign handling and stable result storage           |
| 8     | FP32 ADD/SUB/NEG → MUL → FDIV         | Private stages, then exact significand arithmetic |
| 9     | FP32 comparisons; WHERE; numeric CAST | Raw encodings, word selection and converters      |
| 10    | SQRT → EXP2 → LOG2 → SIN → POW        | Shared arithmetic, comparisons and conversions    |
| 11    | Full-suite sweep and remaining limits | All preceding implementations                     |

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

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

NotImplementedError: ROCKCHIP NPU does not support Ops.AND with dtypes.int
Ran 1 test in 0.154s
FAILED (errors=1)
```

TOREVIEW1: Try advertising AND through ew_algo and opening its INT32 gate. None is a temporary unset selector, not an algorithm number; it stops register construction before submission. The real composite implementation will go in lowered_ops instead.
```diff
-ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0}
+ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: None}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           if u.op not in self.ew_algo or \
-             u.dtype not in ((dtypes.int16, dtypes.int, dtypes.uint) if u.op is Ops.SHL else
+             u.dtype not in ((dtypes.int,) if u.op is Ops.AND else (dtypes.int16, dtypes.int, dtypes.uint) if u.op is Ops.SHL else
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_and

  ((rounding if rounding is not None else self.ew_algo[op]) << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT)
TypeError: unsupported operand type(s) for <<: 'NoneType' and 'int'
Ran 1 test in 0.300s
FAILED (errors=1)
```
TOREVIEW1: rounding was introduced in the preceding TRUNC section to select native FLOOR/CEIL. For this AND trial custom is None, so rounding is None and the builder falls back to ew_algo[Ops.AND], which we deliberately set to None.

The gate opens, but the register builder requires a numeric EW selector. None deliberately stops this trial before submission. This exposes the missing implementation; it does not prove the hardware lacks AND. Replacing None with an arbitrary number would test a different operation, not implement AND.

Restore the AND experiment before adding its real implementation:

```diff
-ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: None}
+ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           if u.op not in self.ew_algo or \
-             u.dtype not in ((dtypes.int,) if u.op is Ops.AND else (dtypes.int16, dtypes.int, dtypes.uint) if u.op is Ops.SHL else
+             u.dtype not in ((dtypes.int16, dtypes.int, dtypes.uint) if u.op is Ops.SHL else
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
+  def conv_and_u32(self, raw:bytes) -> bytes:
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
TOREVIEW1: Use conv_and_u32: the helper handles one pair of 32-bit values, not an unspecified word size. The later shared helper is conv_bitwise_u32.

The wrapper only packs the original words and reads the result. Unlike signed SHR, INT32 AND needs no sign-fill task: signed and unsigned words use the same 32 bits.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_u32_and(self, a:list, b:list, dtype:DType) -> list:
+    assert len(a) == len(b)
+    fmt = "<I" if dtype == dtypes.uint else "<i"
+    result:list = []
+    for x,y in zip(a,b):
+      raw = self.conv_and_u32(struct.pack(fmt, x) + struct.pack(fmt, y))
+      result.append(struct.unpack(fmt, raw)[0])
+    return result
```

Advertise AND as a composite operation, not an EW algorithm number. Bool inputs still use the matcher; INT32/UINT32 use the convolution helper:

Composite helpers now dispatch before the ordinary EW gate; otherwise AND would be rejected because it has no entry in ew_algo.

TOREVIEW1: lowered_ops advertises support without assigning a register number. The path is decided separately:

| Path             | Where it happens      | Example at this step                         |
| ---------------- | --------------------- | -------------------------------------------- |
| UOp lowering     | Renderer matcher      | Bool AND becomes FP16 MUL and CAST           |
| Multi-task helper | Program dispatch     | INT32/UINT32 AND calls the convolution helper |

Both must stay out of the generic EW path; each still needs its matcher or dtype-specific dispatch.

TOREVIEW1: Add AND to lowered_ops now that its handlers are ready.
```diff
@@
-lowered_ops = {Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.TRUNC, Ops.WHERE}
+lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.TRUNC, Ops.WHERE}
@@
 class RockchipProgram(Program['RockchipDevice']):
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          if u.op not in self.ew_algo or \
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
+          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
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
TOREVIEW1: Advertise XOR and allow INT32 through first. As with AND, None means no known EW selector; do not submit an invented algorithm number.

```diff
-ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0}
+ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0, Ops.XOR: None}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
+          elif u.op not in self.ew_algo or u.dtype != (dtypes.int if u.op is Ops.XOR else dtypes.int16 if u.op is Ops.SHL else dtypes.half):
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_xor

TypeError: unsupported operand type(s) for <<: 'NoneType' and 'int'
Ran 1 test in 0.150s
FAILED (errors=1)
```

The register builder reaches the unset EW selector and stops before submission, just like AND. This is a construction failure in the trial, not evidence of hardware XOR behavior.

TOREVIEW1: Restore the gate-only trial before implementing XOR:

```diff
-ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0, Ops.XOR: None}
+ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0}
@@
-          elif u.op not in self.ew_algo or u.dtype != (dtypes.int if u.op is Ops.XOR else dtypes.int16 if u.op is Ops.SHL else dtypes.half):
+          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
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

Flatten the table row by row, so lookup[4*a+b] gives a XOR b. These are the output values; the helper turns their adjacent differences into convolution weights, just like AND.
TOREVIEW1: Shortened the explanation; the example is already in step 3 above.

Rename the helper now that it handles two operations:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def conv_and_u32(self, raw:bytes) -> bytes:
-    assert len(raw) == 8
+  def conv_bitwise_u32(self, op:Ops, raw:bytes) -> bytes:
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
-      raw = self.conv_and_u32(struct.pack(fmt, x) + struct.pack(fmt, y))
+      raw = self.conv_bitwise_u32(op, struct.pack(fmt, x) + struct.pack(fmt, y))
```

Then advertise XOR and extend dispatch:

```diff
-lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.TRUNC, Ops.WHERE}
+lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.TRUNC, Ops.WHERE, Ops.XOR}
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

TOREVIEW1: test_ops.py has no test_xor_* methods. test_xor checks tensor XOR, a scalar on the right, and a scalar on the left. The related test_bitwise_not checks both bitwise_not() and ~ on INT32 and bool inputs. Integer NOT lowers to XOR with -1; bool NOT uses logical_not.

Run that related test at this checkpoint, before adding OR:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitwise_not

Ran 1 test in 0.263s
OK
```

Both operations use 14 convolution tasks per word pair. Python builds constant weights and copies original/final storage; the NPU calculates the digits, table selection and result bytes.

The helper diffs also passed 196 INT32/UINT32 AND/XOR pairs, including negative values and word boundaries. All four bool XOR pairs passed too.

AND/XOR have tested bool and INT32/UINT32 paths here, not every integer width. Their complete variant coverage is not established.

## Ops.OR

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or

8 Ops.CONST dtypes.weakint 4919 [] []
9 Ops.CAST dtypes.int dtypes.int [[4919]] [dtypes.weakint]
10 Ops.OR dtypes.int None [[1], [4919]] [dtypes.int, dtypes.int]
NotImplementedError: ROCKCHIP NPU does not support Ops.OR with dtypes.int
Ran 1 test in 0.232s
FAILED (errors=1)
```
TOREVIEW1: Command and fresh checkpoint output shown above. The preceding tensor|tensor case only copies values; the scalar 0x1337 case reaches OR and fails. Earlier LOAD/STORE and index lines are omitted.

The probe establishes this missing path, not an accuracy failure in a built-in decomposition.

First advertise OR and let its INT32/UINT32 inputs reach the existing bitwise helper:

```diff
-lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op in (Ops.AND, Ops.XOR) and u.dtype in (dtypes.int, dtypes.uint):
+          elif u.op in (Ops.AND, Ops.XOR, Ops.OR) and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or

  assert op in (Ops.AND, Ops.XOR) and len(raw) == 8
AssertionError
Ran 1 test in 0.152s
FAILED (failures=1)
```

The dispatch now reaches conv_bitwise_u32, which only supports AND/XOR. Opening the public gate did not add an OR table. The assertion stops before its tasks are submitted.

Now extend the OR helper. We already split and rejoin the digits for AND/XOR. OR changes which of the two bits are kept, not their positions, so only the two-bit table should change:

| a   |  b=0 |  b=1 |  b=2 |  b=3 |
| --- | ---: | ---: | ---: | ---: |
| 0   |    0 |    1 |    2 |    3 |
| 1   |    1 |    1 |    3 |    3 |
| 2   |    2 |    3 |    2 |    3 |
| 3   |    3 |    3 |    3 |    3 |

The same `index = 4*a+b`, RELUX1 thresholds and table-difference weights now give a OR b. Signed top-digit handling and byte assembly stay the same. Bool OR still uses its existing CAST → MAX path.

```diff
   def conv_bitwise_u32(self, op:Ops, raw:bytes) -> bytes:
-    assert op in (Ops.AND, Ops.XOR) and len(raw) == 8
+    assert op in (Ops.AND, Ops.XOR, Ops.OR) and len(raw) == 8
@@
-    lookup = ((0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3) if op is Ops.AND else
-              (0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0))
+    lookup = {Ops.AND: (0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3),
+              Ops.XOR: (0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0),
+              Ops.OR:  (0,1,2,3,1,1,3,3,2,3,2,3,3,3,3,3)}[op]
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_or

test_or (__main__.TestOps.test_or) ... ok

Ran 1 test in 0.459s

OK
```

OR now has an INT32/UINT32 path in addition to bool, with the same 14 convolution tasks per word pair and no host arithmetic on intermediate values. This is partial coverage, not a completed-op count.

Four additional INT32/UINT32 Tensor checks passed (144 lanes), including broadcasting, random full-width values, sign-bit boundaries and alternating-bit patterns. Every check submitted NPU work.

## Ops.BITCAST

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitcast

RuntimeError: self.size(-1) must be divisible by 2 to view Half as Int (different element sizes), but got 3
Ran 1 test in 0.005s
FAILED (errors=1)
```
TOREVIEW1: Rerunning with TRACE=1 printed no UOps. Torch raises before the tinygrad call, so there is no ROCKCHIP Ops list to include at this step.

The failed case is 

```python
helper_test_op([(3, 3)], lambda x: x.view(torch.int32), lambda x: x.bitcast(dtypes.int32), forward_only=True)
```

With DEFAULT_FLOAT=HALF, each row has three 2-byte values: six bytes cannot form a whole number of 4-byte INT32 values. helper_test_op runs Torch's x.view(torch.int32) first, which raises before tinygrad's x.bitcast runs. This is a reference-input shape error, not evidence that NPU BITCAST failed. The raw-payload checks below test storage preservation separately.

BITCAST keeps the same bytes and changes their dtype. In this interpreter the host copies storage; it does not calculate a converted value or submit a fake identity operation.

Check the gate before adding one: BITCAST already has an explicit __call__ branch before GroupOp.ALU. It does not use ops_map or the ALU dtype gate. So there is no missing advertisement to release here; the next diff replaces that existing storage handler. The HALF test above fails in Torch before either handler runs.

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

LOAD must keep the original bits, including NaN payloads. STORE copies them back without a numeric conversion:

```diff
 def _load(m, i, dtype: DType):
@@
   if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
+  if m.itemsize == dtype.itemsize:
+    # Copy one lane as raw storage so loading a NaN does not canonicalize its payload.
+    return typed_view(bytes(m.cast("B")[i*m.itemsize:(i+1)*m.itemsize]), dtype)
```

```diff
 def _store(m, i, v, dtype: DType):
   if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
+  if isinstance(v, memoryview):
+    # Store the selected bytes directly, without converting through a Python scalar.
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
TOREVIEW1: The old helper goes through a Python number, which can change a NaN's bits when packed again. Keep the bytes instead and only change their view. This is still host-side storage handling, not NPU arithmetic.

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

TOREVIEW1: The guards need numbers, but a now contains memoryviews. Use scalar16 for these checks; leave the original bytes unchanged for the NPU.

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
-      raw = self.conv_bitwise_u32(op, struct.pack(fmt, x) + struct.pack(fmt, y))
-      result.append(struct.unpack(fmt, raw)[0])
+      raw = self.conv_bitwise_u32(op, bytes(raw16(x, dtype)) + bytes(raw16(y, dtype)))
+      result.append(typed_view(raw, dtype))
```

The bit-pattern checks passed all 65,536 FP16 encodings, all 65,536 BF16 encodings, and selected FP32/FP64 zeros, infinities and NaN payloads. The existing test_bitcast passed with its default FP32 inputs.

Use FLOAT for that existing test; the HALF reference-shape error above is not fixed by our runtime changes:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=FLOAT DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitcast

Ran 1 test in 0.132s
OK
```

BITCAST adds storage reinterpretation, not a numeric CAST. This alone does not complete all BITCAST variants.

## Ops.MULACC

For Ops.MULACC, which is `a*b+c`. We can try normal MUL then ADD?

Run the existing Tensor test first:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mulacc_with_zero_strides

Exception: forward pass failed shape (2, 4): dtype mismatch: tinygrad=float32 | torch=float16
Ran 1 test in 0.164s
FAILED (errors=1)
```

This test uses expanded inputs and a reduction. It stops at a dtype mismatch, so it cannot yet tell us whether a fused task rounds correctly. Keep that failure while investigating the arithmetic.

The old "only python supports MULACC" skip is not a hardware support list. [c13da83f1](https://github.com/tinygrad/tinygrad/commit/c13da83f128fba7afce2c00a23a086fee45d021b) introduced it; [bc180a963c](https://github.com/tinygrad/tinygrad/commit/bc180a963c264aa426ce2a0aa6bee097c3e5e597) only changed device selection. Neither explains a Rockchip limitation. We do not need to edit another test file to continue this Tensor test.

Advertise MULACC and temporarily bypass its composite/dtype gate, without adding the three-input implementation yet:

```diff
-lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
+          elif u.op is not Ops.MULACC and (u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half)):
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mulacc_with_zero_strides

Exception: forward pass failed shape (2, 4): dtype mismatch: tinygrad=float32 | torch=float16
Ran 1 test in 0.166s
FAILED (errors=1)
```

The result is unchanged. Advertising MULACC and opening its gate does not resolve the reduction's output dtype mismatch. This test still does not establish whether the three-input NPU path works; do not describe it as a fused-arithmetic pass.

Restore the MULACC trial before implementing its arithmetic:

```diff
-lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op is not Ops.MULACC and (u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half)):
+          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
```

MULACC should round the combined result once; our two FP16 tasks would round the product first.

TOREVIEW1: Separate FP16 MUL and ADD round twice: round16(round16(a*b)+c). MULACC needs round16(a*b+c), so this is about rounding, not just saving a task. Can BS MUL followed by EW ADD keep enough precision? BRDMA supplies b and ERDMA supplies c. The recorded native-task probe gave:

| Inputs                                      | Separate FP16 MUL then ADD | One BS → EW task | Correctly rounded FP16 MULACC |
| ------------------------------------------- | -------------------------: | ---------------: | ----------------------------: |
| `1.0009765625 * 1.0009765625 - 1.001953125` |                          0 |          `2^-20` |                       `2^-20` |
| `65504 * 2 - 65504`                         |                        inf |            65504 |                         65504 |
| `11.8125 * -2688 - 0.00023484230041503906`  |                     -31744 |           -31744 |                        -31760 |
| `(-0) * 1 + (-0)`                           |                         -0 |               +0 |                            -0 |

The first two cases show why keeping the product in FP32 helps. But the third exposes another rounding step: the exact product is -31752, halfway between two FP16 values. The small negative addend should move it towards -31760. FP32 addition loses that small amount, so the final FP16 conversion rounds the halfway value to -31744 instead.

The converter controls we tried did not fix this. Bypassing EW operand conversion misreads FP16 addends; changing its offset to negative zero or changing CVT_TYPE also did not preserve the negative-zero case.

An earlier 4,096-triple random probe had no numeric mismatches, but these targeted halfway cases did. Passing ordinary random inputs is not enough to claim correctly rounded MULACC.

TOREVIEW1: The test above only reached a dtype mismatch; it did not verify MULACC. In the separate native-task probe, the third row returns -31744 instead of -31760, and the last row loses negative zero. So that candidate is not enough yet. These are probe results, not a passing test_ops result.

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

The correction probe kept intermediates in DMA storage; Python uploaded inputs/constants, submitted tasks and read the final result. It used 32 NPU tasks per eight lanes, much more than the native task. The runtime diff below implements that correction.

This was rerun with the code built so far, before adding mulacc_stage or run_mulacc below. The standalone probe supplies its own task sequence.

Run from the tinygrad repo root. This covers eight targeted cases, 16,384 random triples and 1,331 special-value combinations against a float64 multiply-add rounded to FP16. NaNs are checked as NaNs, not by payload. This is not an exhaustive test of all FP16 triples; completion is not established until the corrected path is integrated.

Now port the correction to ops_rockchip.py. The input and result are still FP16; FP32 and INT32 are private intermediate modes, not general dtype support.

First add a task helper. It starts from build_registers, then selects precision 5 for FP32 or 4 for INT32. Passing None bypasses EW for the two input conversions and the final output conversion. BS MUL reads b through BRDMA only for the product task; every other task disables BRDMA again.

The stride fields start at bit 4: `1 << ...__SHIFT` writes a 16-byte stride. Shifting 16 instead wrote 256 bytes and left the last four output lanes missing in the first port. The final FP32 → FP16 task writes four half lanes at offset 0 and four at offset 16.

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
+                   shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
+    # Private FP32/INT32 stages: precision 2 = FP16, 4 = INT32, 5 = FP32; algo None bypasses EW.
+    self.build_registers(Ops.ADD, input_addr=lhs, weight_addr=rhs, output_addr=out)
+    E = self.EMIT
+    ew = ((1 << rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT) |
+          (1 << rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT))
+    if algo is None:  # Conversion/product-only task: bypass EW; BS and the output converter still run.
+      ew |= (1 << rk.DPU_EW_CFG_EW_BYPASS__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT)
+    else:
+      ew |= ((1 << rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT) | (3 << rk.DPU_EW_CFG_EDATA_SIZE__SHIFT) |
+             (algo << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) | (1 << rk.DPU_EW_CFG_EW_OP_SRC__SHIFT) |
+             (binary << rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT) | (mul << rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT))
```
TOREVIEW1: Pass algo=None for conversion or product-only tasks. There is no EW algorithm to select; bypass EW and disable its operand DMA. Numeric values select an actual ALU algorithm.

Select input/output precision. Keep the converter scale at 1; only the final FP16 output enables FP32TOFP16.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
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
TOREVIEW1: Extract the shared OUT_CVT writes below, before using this helper for the arithmetic stages. Convolution and carry extraction can call it too.

Now set the output layout. One stride unit is 16 bytes. FP32 has four lanes per surface; FP16 output still keeps that surface stride.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
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
TOREVIEW1: Keep this precision-dependent layout in mulacc_stage, shared by its callers. The convolution byte layout is different; do not copy this layout into each op or substitute the convolution layout here.


Configure the input streams for the same precision. Disable BRDMA/NRDMA by default, and disable ERDMA when EW is bypassed.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
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
+        (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT) if algo is None else
+        (1 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT) | (3 << rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT)),
+    ]
```
TOREVIEW1: These input-stream settings also belong to the shared mulacc_stage. Later ops pass precision and operand addresses rather than repeating the RDMA setup. CNA tasks keep using build_conv_uint8_registers.

The product task alone enables BS operand DMA. Point BRDMA at b, keep BS ALU/ReLU bypassed, then submit. The next task starts from the shared initialization again.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
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

The helper emitted identical register words for 256 combinations of offset, shift and rounding. This checks the refactor, not an additional arithmetic capability.

This is a register-construction refactor, not a new arithmetic path.

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
+      def calc(algo:int|None, x:int, y:int=zero, precision:int=5, output:int=5, **kw) -> int:
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
+      product = calc(None, base, precision=2, bs_mul=base+64)
+      addend = calc(None, base+128, precision=2)
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
+      out = calc(None, signed, output=2) - base
+      # Four FP32 lanes per surface: half output retains the 16-byte surface stride.
+      raw = bytes(to_mv(self.dev.input_buf+out, 8)) + bytes(to_mv(self.dev.input_buf+out+16, 8))
+      result.extend(typed_view(raw[i*2:i*2+2], dtypes.half) for i in range(count))
+    return result
```

The helpers reconstructed from these smaller diffs passed 1,030 triples on the NPU: 1,024 seeded random triples plus six rounding, signed-zero and special-value cases. Non-NaN output bits matched the FP16 reference; NaNs were checked as NaNs. This checks these tutorial helpers, not every possible triple.

Now advertise MULACC and dispatch its three inputs. tinygrad also fuses integer MUL+ADD when MULACC is advertised, so decompose those non-FP16 cases again in the late matcher. This does not add general INT32 or FP32 MULACC support.

```diff
-lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
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

The existing test_mulacc_with_zero_strides still fails under DEFAULT_FLOAT=HALF:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mulacc_with_zero_strides

Exception: forward pass failed shape (2, 4): dtype mismatch: tinygrad=float32 | torch=float16
Ran 1 test in 0.166s
FAILED (errors=1)
```
TOREVIEW1: This checkpoint rerun prints no ROCKCHIP UOps despite TRACE=1. It reaches the output dtype comparison, but gives no trace evidence of an NPU MULACC task. Do not count it as a MULACC pass.

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

MULACC now has an NPU-only FP16 implementation, including the tested rounding, overflow, underflow, infinity and signed-zero cases. NaNs remain NaNs; payload bits are not guaranteed. It costs 32 submissions per atom of up to eight lanes, and the NOOPT interpreter can submit only one lane at a time. This is not general FP32 MULACC support or an exhaustive check of every possible input triple.

## Ops.CDIV

There is no test_cdiv in test_ops.py. Use its existing division and remainder cases:

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
TOREVIEW1: Choose test_div_int because it exercises integer inputs and truncating division, which needs CDIV. It also includes floor division, so run it first and inspect which unsupported op appears; this is not an isolated CDIV test.


```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

NotImplementedError: ROCKCHIP NPU does not support Ops.CMOD with dtypes.int
Ran 1 test in 0.169s
FAILED (errors=1)
```

`test_div_int` includes floor division too. Its first unsupported UOp was CMOD, so it does not isolate CDIV. Implement the shared quotient/remainder arithmetic, then return to this same test after enabling CMOD.

```diff
-ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0}
+ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0, Ops.CDIV: None}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
+          elif u.op not in self.ew_algo or u.dtype != (dtypes.int if u.op is Ops.CDIV else dtypes.int16 if u.op is Ops.SHL else dtypes.half):
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

NotImplementedError: ROCKCHIP NPU does not support Ops.CMOD with dtypes.int
Ran 1 test in 0.165s
FAILED (errors=1)
```

Still CMOD: opening CDIV does not get past the first missing remainder operation. This test cannot tell us anything new about CDIV's arithmetic yet. Keep the shared quotient/remainder implementation next rather than claiming the gate fixed division.

Restore the CDIV experiment before adding the shared arithmetic:

```diff
-ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0, Ops.CDIV: None}
+ew_algo = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3, Ops.SHL: 0, Ops.SHR: 0}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          elif u.op not in self.ew_algo or u.dtype != (dtypes.int if u.op is Ops.CDIV else dtypes.int16 if u.op is Ops.SHL else dtypes.half):
+          elif u.op not in self.ew_algo or u.dtype != (dtypes.int16 if u.op is Ops.SHL else dtypes.half):
```

For CDIV, we already have TRUNC, so first check the obvious candidate, FDIV → TRUNC:

```text
4094 / 3 = 1364.666...
FP16 FDIV rounds it to 1365
TRUNC(1365) = 1365, but CDIV should be 1364
```

Both inputs fit FP16 exactly. This is quotient rounding, not just an input CAST problem. The direct NPU probe returned 1365 too.

The quotient must stay exact, so narrowing these integers to FP16 is not enough. The BITCAST step already keeps the original bytes through LOAD and STORE. Use those bytes for exact integer arithmetic instead.

So we need to handle Exact integer arithmetic

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
+      def calc(algo:int|None, x:int, y:int=zero, **kw) -> int:
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
+      # Arguments are DMA addresses: each 0/1 mask lane selects no/yes using NPU arithmetic.
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
```
TOREVIEW1: select returns the result's DMA address, not a Python-selected value.

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
TOREVIEW1: A limb is one piece of a wider integer. Here each limb holds one byte, 0..255, in an INT32 NPU lane. Four limbs represent a UINT32 value: b0 + 256*b1 + 65536*b2 + 16777216*b3, low byte first.

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

Finally copy the low byte from each NPU-produced INT32 limb, lowest limb first. The arithmetic has already finished; this only restores the tensor's storage layout.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
           remainder = choose(sign_a, negate(remainder[:width]), remainder[:width])
+          out = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else remainder
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
@@
-lowered_ops = {Ops.AND, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          if u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
+          integer_dtype = src_dtypes[1] if u.op is Ops.WHERE else src_dtypes[0]
+          if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
+            values[u] = self.run_integer(u.op, src_values, integer_dtype)
+          elif u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
```

The original test_div_int still needs the CMOD gate. Add that next, then retest the complete method.

## Ops.CMOD

CDIV returns the quotient, but its restoring-division loop also keeps the remainder. CMOD returns that remainder after restoring the numerator's sign. This is why the shared helper above contains both paths; CMOD does not need a second division algorithm.

The earlier test_div_int trace stopped at CMOD. Its dispatch has not been added yet.

Enable the remaining path:

```diff
@@
-lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
```

Now test the quotient and remainder paths together, before adding floor correction:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int TestOps.test_fmod

test_div_int (__main__.TestOps.test_div_int) ... ok
test_fmod (__main__.TestOps.test_fmod) ... ok

Ran 2 tests in 9.528s

OK
```

This run used the reconstructed CMOD checkpoint in the new order, without the floor correction or later support.

## Ops.FLOORDIV and Ops.FLOORMOD

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
           out = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else remainder
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
```

tinygrad already lowers FLOORDIV/FLOORMOD using CDIV/CMOD plus sign correction in codegen/decomp/op.py. Keep that native decomposition; the direct helper uses the same correction. No core change is needed.

Check floor-modulo separately:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mod

test_mod (__main__.TestOps.test_mod) ... ok

Ran 1 test in 12.264s

OK
```

The reordered diffs reconstruct the same final helper. test_div_int and test_fmod passed before the floor step; test_mod passed after it. No THREEFRY or later math implementation was included in either checkpoint.

The focused checks passed 88 subtests across signed and unsigned 8/16/32/64-bit arithmetic, comparisons and division. This includes full-width random inputs, wrapping results, negative remainders and zero divisors. It is not exhaustive over every pair. INT32 division took about 0.25s for eight lanes; correctness first, not a speed claim.

### MULACC integration with integer arithmetic

The non-FP16 MULACC matcher emits integer MUL and ADD. Those now have a dispatch, so we can run the full integration probe without borrowing later support.

A saved integration probe exercised Tensor fusion, with a CNA task before MULACC and ordinary EW ADD after it. It also checked that scratch reuse did not overwrite earlier results.

The Tensor checks also passed with NOOPT=0, with 67 MULACC dispatches. These checks count calls to run_mulacc, so a constant-folded expression cannot silently pass as an NPU MULACC test.

## Ops.THREEFRY

TOREVIEW1: There is no test_threefry method in test_ops.py. Do not substitute a separate test file or count the helper probe as acceptance coverage. This section keeps native decomposition: advertising THREEFRY would suppress the very decomposition we want to reuse, so it is not a gate-only experiment for a direct THREEFRY implementation.

No THREEFRY entry is needed in ops_map: leaving it unsupported lets tinygrad expand it into simpler UOps. The saved decomposition probe stopped at UINT64 SHR, before the rounds. This is probe evidence, not a test_ops.py pass.

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

The 18 earlier arithmetic/comparison/shift regressions passed again in 35.57s.

The storage paths use typed memoryviews, not a wrapper class. BITCAST changes the view format; raw16 copies its bytes. These tutorial helpers copy completed results before reusing scratch memory; they do not yet reuse intermediate DMA addresses.

The raw-storage check round-trips half and wider encodings without unpacking/repacking NaN payloads. MULACC's rounding and signed-zero checks are recorded in its own step above.

Still to introduce: SQRT, EXP2, LOG2, POW and SIN. This is not every dtype or edge case, and the full test_ops.py sweep is still pending.

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

TOREVIEW1: Yes, this came from test_sqrt above. After adding INT16 SHR, tinygrad's existing SQRT decomposition reached a comparison with NaN and hit our old guard. We have not added a SQRT handler yet; first fix the comparison primitive used by that decomposition.

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
+      def calc(algo:int|None, x:int, y:int=zero, **kw) -> int:
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
+      self.mulacc_stage(None, mask, zero, wide, precision=4)
+      self.mulacc_stage(None, wide, zero, half, output=2)
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

107 Ops.CAST dtypes.half dtypes.half [...] [dtypes.bool]
108 Ops.MUL dtypes.half None [[nan], [...]] [dtypes.half, dtypes.half]
109 Ops.SUB dtypes.half None [[1.0], [...]] [dtypes.half, dtypes.half]
110 Ops.MUL dtypes.half None [...] [dtypes.half, dtypes.half]
111 Ops.ADD dtypes.half None [...] [dtypes.half, dtypes.half]
...
148 Ops.CAST dtypes.short dtypes.short [...] [dtypes.half]

ValueError: cannot convert float NaN to integer
Ran 1 test in 0.252s
FAILED (errors=1)
```
TOREVIEW1: This is the fresh trace at the earlier failing checkpoint, not the later SQRT implementation. The excerpt omits memoryview addresses and intervening UOps. CAST → MUL → SUB → MUL → ADD is our arithmetic WHERE lowering; it evaluates the NaN branch even when the mask is zero.

Even 0*NaN is NaN. The final integer CAST raises while converting that NaN, so this is a selection bug, not a square-root accuracy result.

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

The first retest of test_masked_select reached the 30-second command limit without finishing. It no longer stopped at the bool WHERE gate, but that is not a pass.

## Recorded WHERE timeout investigation (deferred)

The following changes were investigated before the accuracy-first pass. Keep their diffs because later sections extend these helpers, but skip their timed-out test commands for now. They do not establish a full masked_select pass.

### Fill the existing lanes first

A 15-second probe at this checkpoint records 2,398 integer ADD calls, 1,384 CMPLT calls and 692 WHERE calls, all with one lane. That is a partial profile, not a passing test. We extracted an eight-lane task, but the NOOPT interpreter feeds it one workgroup at a time.

First batch independent straight-line workgroups. Keep loops and local memory on the old path; they need separate checks before batching. This changes dispatch, not arithmetic or dtype gates.

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

This fills the existing eight lanes where the allowlist permits it. It does not make every kernel batchable, and it does not yet increase the hardware task width.

### Increase the task width

The eight lanes came from our initial task setup, not a measured hardware maximum. The reference elementwise.py uses `dataout_width = (tile_n + 7) // 8 - 1`. Check the layout before changing the loop size.

In the separate width probe, changing only DPU/RDMA widths wrote 12 of 16 words, 20 of 32, and 68 of 128; the remaining sentinel bytes were untouched. Surface strides alone were not enough either. These five fields together passed INT32 ADD, FP32 ADD and INT32 binary MIN for 16, 32 and 128 lanes:

| Field                   | N eight-lane atoms |
| ----------------------- | -----------------: |
| DPU cube width          |              N - 1 |
| RDMA cube width         |              N - 1 |
| Destination surf stride |                  N |
| EW operand surf stride  |                  N |
| Output surface add      |                  N |

Keep one atom as the default. Wider tasks here only support equal-width INT32/FP32 modes without BS operand DMA:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
-                   shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None) -> None:
+                   shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None, atoms:int=1) -> None:
+    assert 1 <= atoms <= 16 and (atoms == 1 or (precision in (4, 5) and output == precision and bs_mul is None))
@@
-      E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, 1 << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
+      E(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, atoms-1),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, atoms-1),
+      E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, atoms << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
       E(rk.DPU, rk.REG_DPU_BS_OW_CFG,
@@
-      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, 1 << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
+      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, atoms << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
@@
-      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 1 << rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT),
+      E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, atoms << rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT),
```

Now widen integer WHERE's scratch slots and constants with the task. Its raw-word selection helper already exists at this step. Other integer operations keep eight lanes for now:

```diff
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
     assert dtype in dtypes.ints and all(len(x) == len(inputs[0]) for x in inputs)
     base = self.dev.input_mem.dma_addr
+    batch = 128 if op is Ops.WHERE else 8
+    stride = max(64, batch*4)
     result:list = []
-    for start in range(0, len(inputs[0]), 8):
-      count = min(8, len(inputs[0])-start)
+    for start in range(0, len(inputs[0]), batch):
+      count = min(batch, len(inputs[0])-start)
+      atoms = (count+7)//8
       slot = 0
       def allocate() -> int:
         nonlocal slot
-        addr = base + slot*64
+        addr = base + slot*stride
         slot += 1
-        if slot*64 > self.dev.input_mem.size: raise RuntimeError("ROCKCHIP integer scratch exhausted")
+        if slot*stride > self.dev.input_mem.size: raise RuntimeError("ROCKCHIP integer scratch exhausted")
         return addr
       def put(raw:bytes) -> int:
         addr = allocate()
-        to_mv(self.dev.input_buf+addr-base, 32)[:] = raw + bytes(32-len(raw))
+        to_mv(self.dev.input_buf+addr-base, atoms*32)[:] = raw + bytes(atoms*32-len(raw))
         return addr
       constants:dict[int, int] = {}
       def const(value:int) -> int:
-        if value not in constants: constants[value] = put(struct.pack("<i", value)*8)
+        if value not in constants: constants[value] = put(struct.pack("<i", value)*(atoms*8))
         return constants[value]
       zero, one, radix = const(0), const(1), const(256)
       def calc(algo:int|None, x:int, y:int=zero, **kw) -> int:
         out = allocate()
-        self.mulacc_stage(algo, x, y, out, precision=4, output=4, **kw)
+        self.mulacc_stage(algo, x, y, out, precision=4, output=4, atoms=atoms, **kw)
         return out
       def add(x:int, y:int) -> int: return calc(2, x, y)
@@
       for index,values in enumerate(inputs):
         dt = dtypes.bool if op is Ops.WHERE and index == 0 else dtype
-        raw = [bytes(raw16(x, dt)) for x in values[start:start+8]]
+        raw = [bytes(raw16(x, dt)) for x in values[start:start+count]]
         operands.append([put(b"".join(x[i:i+1]+bytes(3) for x in raw)) for i in range(dt.itemsize)])
       a = operands[0]
@@
       # Rejoin the low bytes of the NPU-produced limbs. This is a storage copy, not numeric evaluation.
       out_dtype = dtypes.bool if op in GroupOp.Comparison else dtype
-      raw_out = [bytes(to_mv(self.dev.input_buf+x-base, 32)) for x in out]
+      raw_out = [bytes(to_mv(self.dev.input_buf+x-base, atoms*32)) for x in out]
       result.extend(typed_view(b"".join(x[i*4:i*4+1] for x in raw_out), out_dtype) for i in range(count))
     return result
```

These changes only help calls that contain multiple lanes; a one-lane call stays one lane. Retest before claiming the timeout is fixed. PC chaining belongs after this check if submission overhead remains, not before filling and widening the tasks.

The existing smaller nonzero test also reaches boolean selection:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_nonzero_size

Ran 1 test in 5.699s
OK
```

Retest the original method after widening the tasks:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
# 30-second command limit, exit 124. No test summary.
```

The wider helper works, but the full method still times out. The straight-line allowlist above does not yet batch comparisons, WHERE or loops, so not every call uses the available lanes. The old 120-second attempt predates this rearrangement; we did not rerun it. The 209/433 full-suite figure is also historical, not a new result.

### Chain the WHERE tasks
TOREVIEW1: WHERE already works. This is the recorded attempt to reduce submission overhead after masked_select timed out, not another WHERE implementation. Integer selection currently submits three tasks per byte; chaining keeps those operations but submits their register bodies together. This performance investigation remains deferred during the accuracy pass.

We still submit and wait after every arithmetic stage. For each byte of integer WHERE, we already do `no + (yes - no)*mask`. That is three INT32 tasks, with no CPU read between them. Can we submit them together?

The 1500 branch's `_submit_bodies` and `rk3588/examples/elementwise.py` link register bodies with a PC tail. This does not fuse their arithmetic: each task still runs, but its tail points to the next task's registers.

| Task | Operation          | Next body |
|------|--------------------|-----------|
| 1    | `yes - no`         | MUL       |
| 2    | `result * mask`    | ADD       |
| 3    | `no + result`      | guard, 0 registers |

First probe two dependent INT32 stages, ADD then MUL. With our three-write single-task tail, the expected `[-3, 0, 3, 6, 9, 12, 15, 18]` came back as `[-3, 0, 3, 6, 0, 0, 0, 0]`. Restoring the reference's VERSION entry made all eight lanes match. The earlier single-task removal test was not enough to remove it from chains.

Keep that entry for multi-task submissions. Align each body to 16 bytes, give every body a task descriptor, and point the last tail into the existing zero-filled guard page. `next_amount` counts the next body's registers, without its tail:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def pc_tail(self, next_addr:int|None, next_amount:int=0, cna:bool=False) -> list[int]:
+    E = self.EMIT
+    return [
+      E(rk.PC, rk.REG_PC_BASE_ADDRESS,
+        0 if next_addr is None else next_addr & rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK),
+      E(rk.PC, rk.REG_PC_REGISTER_AMOUNTS, 0 if next_addr is None else next_amount),
+      E(0x80, rk.REG_PC_OPERATION_ENABLE,
+        rk.GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK | (rk.GLOBAL_OPERATION_ENABLE_CNA_OP_EN__MASK |
+        rk.GLOBAL_OPERATION_ENABLE_CORE_OP_EN__MASK if cna else rk.GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK)),
+    ]
@@
-  def submit(self, cna:bool=False) -> None:
+  def submit(self, cna:bool=False, bodies:list[list[int]]|None=None) -> None:
     E = self.EMIT
-    guard_offset = (len(self.npu_regs) + 3 + 1) // 2 * 16
+    bodies = [self.npu_regs] if bodies is None else bodies
+    assert bodies and all(bodies) and (not cna or len(bodies) == 1)
+    offsets = [0]
+    tail_size = 4 if len(bodies) > 1 else 3
+    for body in bodies: offsets.append(offsets[-1] + (len(body)+tail_size+1)//2*16)
+    guard_offset = offsets[-1]
     assert guard_offset + mmap.PAGESIZE <= self.dev.regcmd_mem.size
-    regs = self.npu_regs + [
-      E(rk.PC, rk.REG_PC_BASE_ADDRESS, (self.dev.regcmd_mem.dma_addr + guard_offset) & rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK),
-      E(rk.PC, rk.REG_PC_REGISTER_AMOUNTS, 0),
-      E(0x80, rk.REG_PC_OPERATION_ENABLE,
-        rk.GLOBAL_OPERATION_ENABLE_DPU_OP_EN__MASK | (rk.GLOBAL_OPERATION_ENABLE_CNA_OP_EN__MASK |
-        rk.GLOBAL_OPERATION_ENABLE_CORE_OP_EN__MASK if cna else rk.GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN__MASK)),
-    ]
-    # copy the registers to the C array regcmd
     ctypes.memset(self.dev.regcmd_buf, 0, self.dev.regcmd_mem.size)
-    regcmd = (ctypes.c_uint64 * len(regs)).from_address(self.dev.regcmd_buf)
-    regcmd[:] = list(regs)
-
-    # set the regcmd DMA address in a task
+    assert len(bodies)*ctypes.sizeof(rk.struct_rknpu_task) <= self.dev.task_mem.size
     tasks = ctypes.cast(self.dev.task_buf, ctypes.POINTER(rk.struct_rknpu_task))
-    tasks[0] = rk.struct_rknpu_task(
-      flags=0,
-      op_idx=1 if cna else 4,
-      enable_mask=0xd if cna else 0x18,
-      int_mask=0x300,
-      int_clear=0x1ffff,
-      int_status=0,
-      regcfg_amount=len(regs),
-      regcfg_offset=0,
-      regcmd_addr=self.dev.regcmd_mem.dma_addr
-    )
+    for i,body in enumerate(bodies):
+      next_amount = len(bodies[i+1]) if i+1 < len(bodies) else 0
+      tail = self.pc_tail(self.dev.regcmd_mem.dma_addr + offsets[i+1], next_amount, cna=cna)
+      if len(bodies) > 1: tail.insert(2, E(0x40, 0, 0)) # EMIT adds 1: VERSION target 0x41.
+      regs = body + tail
+      regcmd = (ctypes.c_uint64 * len(regs)).from_address(self.dev.regcmd_buf + offsets[i])
+      regcmd[:] = list(regs)
+      tasks[i] = rk.struct_rknpu_task(
+        flags=0, op_idx=1 if cna else 4, enable_mask=0xd if cna else 0x18, int_mask=0x300, int_clear=0x1ffff, int_status=0,
+        regcfg_amount=len(regs), regcfg_offset=0, regcmd_addr=self.dev.regcmd_mem.dma_addr + offsets[i])
@@
-      task_number=1,
+      task_number=len(bodies),
@@
-    submit.subcore_task[0] = rk.struct_rknpu_subcore_task(task_start=0, task_number=1) # assigned to core 0
-    submit.subcore_task[1] = rk.struct_rknpu_subcore_task(task_start=1, task_number=0)
-    submit.subcore_task[2] = rk.struct_rknpu_subcore_task(task_start=2, task_number=0)
+    submit.subcore_task[0] = rk.struct_rknpu_subcore_task(task_start=0, task_number=len(bodies)) # assigned to core 0
+    submit.subcore_task[1] = rk.struct_rknpu_subcore_task(task_start=len(bodies), task_number=0)
+    submit.subcore_task[2] = rk.struct_rknpu_subcore_task(task_start=len(bodies), task_number=0)
```

Let `mulacc_stage` collect a body instead of submitting it when a list is supplied. Other callers still submit immediately:

```diff
   def mulacc_stage(self, algo:int|None, lhs:int, rhs:int, out:int, precision:int=5, output:int=5,
-                   shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None, atoms:int=1) -> None:
+                   shift:int=0, binary:bool=False, mul:bool=False, bs_mul:int|None=None, atoms:int=1,
+                   bodies:list[list[int]]|None=None) -> None:
@@
         E(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BS_BASE_ADDR, bs_mul),
       ]
-    self.submit()
+    if bodies is None: self.submit()
+    else: bodies.append(self.npu_regs)
```

Only collect the integer WHERE stages for now. All three use INT32 input/output and separate scratch slots. Submit before returning the selected value; do not defer past the CPU output read or scratch reuse:

```diff
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
       zero, one, radix = const(0), const(1), const(256)
+      bodies:list[list[int]]|None = [] if op is Ops.WHERE else None
       def calc(algo:int|None, x:int, y:int=zero, **kw) -> int:
         out = allocate()
-        self.mulacc_stage(algo, x, y, out, precision=4, output=4, atoms=atoms, **kw)
+        self.mulacc_stage(algo, x, y, out, precision=4, output=4, atoms=atoms, bodies=bodies, **kw)
@@
-      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def select(mask:int, yes:int, no:int) -> int:
+        out = add(no, mul(sub(yes, no), mask))
+        if bodies is not None:
+          self.submit(bodies=bodies)
+          bodies.clear()
+        return out
```

Retest this checkpoint before moving on. A working chain does not by itself prove `test_masked_select` meets the time limit.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
# 30-second command limit, exit 124. No test summary.
```

masked_select still times out. We reduced WHERE submissions, not every other operation or interpreter iteration in this test. Keep it incomplete; the next investigation needs to measure the remaining work, not increase the timeout.

### Avoid byte decomposition for bounded indices

A 15-second profile of this masked_select checkpoint was deliberately interrupted. It recorded 1,348 calls to run_integer, taking 6.52 seconds, with subtraction/borrow work underneath ADD. This is profiling evidence, not a new numerical failure.

The 1500 branch's `_int_info` uses conservative UOp bounds before choosing a narrower arithmetic recipe. We can use those bounds without narrowing storage:

Our general integer helper splits values into bytes to preserve overflow behavior. Index expressions often have much smaller bounds. Tinygrad's `UOp.vmin/vmax` already computes static bounds for ADD, MUL, RANGE and SPECIAL. If both operands fit INT16, their sum and product fit INT32, so these two operations need no byte decomposition. This is a static proof, not a host calculation on tensor values.

A saved native INT32 ADD/MUL probe covered all pairs from `[-32768, -32767, -1, 0, 1, 32766, 32767]`. Both operations passed all 49 pairs, including `-32768 * -32768 = 1073741824`. Keep full INT32 input/output storage; the bound also avoids the EW multiplier's signed-INT16 operand limitation found earlier.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_bounded_int32(self, op:Ops, a:list, b:list) -> list:
+    # Caller proves both inputs fit INT16; their ADD/MUL results fit INT32 without overflow.
+    assert op in (Ops.ADD, Ops.MUL) and len(a) == len(b)
+    base = self.dev.input_mem.dma_addr
+    result:list = []
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      for offset,values in ((0, a), (64, b)):
+        raw = b"".join(raw16(x, dtypes.int) for x in values[start:start+8])
+        to_mv(self.dev.input_buf+offset, 32)[:] = raw+bytes(32-len(raw))
+      self.mulacc_stage(2 if op is Ops.ADD else 0, base, base+64, base+128, precision=4, output=4, mul=op is Ops.MUL)
+      raw = bytes(to_mv(self.dev.input_buf+128, 32))
+      result.extend(typed_view(raw[i*4:i*4+4], dtypes.int) for i in range(count))
+    return result
+
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
```

Select this path only for proven bounded INT32 ADD/MUL. Wider or unknown ranges still use the original integer helper:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
     st = time.perf_counter()
+    bounded_int32 = {u for u in self.uops if u.op in (Ops.ADD, Ops.MUL) and u.dtype == dtypes.int and
+                     all(s.dtype == dtypes.int and -32768 <= s.vmin <= s.vmax <= 32767 for s in u.src)}
@@
-          if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
+          if u in bounded_int32: values[u] = self.run_bounded_int32(u.op, src_values[0], src_values[1])
+          elif integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
+                                                      Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, *GroupOp.Comparison):
```


Retest the same method with only these earlier steps reconstructed:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
# 30-second command limit, exit 124. No test summary.
```

Still too slow. A second interrupted profile recorded 508 bounded calls taking 0.099 seconds, but 1,333 general integer calls taking 5.797 seconds and 574 half-comparison calls taking 2.615 seconds. These samples cover different amounts of work; they are not a whole-test speedup ratio.

Inspect the kernels before changing another register. A separate 12-second diagnostic recorded:

| Kernel | Global size | Local size | Loop/storage UOps          |
|--------|-------------|------------|----------------------------|
| 1      | `(1,1,1)`   | `(1,1,1)`  | `BUFFER`, `RANGE`, `END`  |
| 2      | `(320,1,1)` | `(1,1,1)`  | `BUFFER`, `RANGE`, `END`  |

The second kernel also contains CMPLT and WHERE. These are outside our current batching allowlist, so its 320 workgroups still run separately. The diagnostic counted 349 general ADD calls, 60 integer CMPLT calls and 29 integer WHERE calls, all with one lane, before stopping. It deliberately raised SystemExit(124); that is not a numerical test failure.

Wider tasks and PC chaining do not remove this restriction. Next check whether the loop bounds are uniform and the accumulator storage is private to each workgroup before batching these loops. Do not just admit RANGE and BUFFER without preserving that storage.

### Batch the fixed loops too

Inspecting both kernels before executing the second one shows the same loop bound, `CAST(CONST(320), int)`. Their BUFFERs use AddrSpace.REG, and each END has two sources: STORE and RANGE. The existing REG allocator already creates one accumulator per lane.

So we can batch independent workgroups without sharing their accumulators or changing the reduction order. Keep dynamic loop bounds, conditional backedges and LOCAL buffers on the old path. Include the existing elementwise comparison, WHERE and TRUNC paths in the allowlist as well:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-    # Batch only independent straight-line workgroups; retain the original path for control flow and local memory.
+    # Batch independent workgroups, including fixed loops with per-thread accumulators.
     batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
-                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.SINK, Ops.NOOP, Ops.AFTER}
-    batch = 8 if local_size == (1,1,1) and all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
+                 Ops.BUFFER, Ops.RANGE, Ops.END,
+                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE,
+                 *GroupOp.Comparison, Ops.SINK, Ops.NOOP, Ops.AFTER}
+    uniform_loops = all(u.op is not Ops.RANGE or (u.src and (u.src[0].op is Ops.CONST or
+                        (u.src[0].op is Ops.CAST and u.src[0].src[0].op is Ops.CONST))) for u in self.uops)
+    private_buffers = all(u.op is not Ops.BUFFER or u.addrspace is AddrSpace.REG for u in self.uops)
+    simple_ends = all(u.op is not Ops.END or len(u.src) == 2 for u in self.uops)
+    batch = 8 if uniform_loops and private_buffers and simple_ends and local_size == (1,1,1) and \
+      all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
```

Run masked_select again with these changes, before adding any later support:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
test_masked_select (__main__.TestOps.test_masked_select) ...
# 30-second command limit, exit 124. No test summary.
```

Still timed out. The 1500 branch has the same two `(32, 10)` cases here; smaller inputs do not explain its reported pass. Batching independent workgroups still leaves each 320-iteration reduction sequential. We have not yet shown which remaining operation dominates after this change, so do not call the timeout solved or increase the limit.

Check the original WHERE test at the same checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_where
test_where (__main__.TestOps.test_where) ... ok

Ran 1 test in 0.504s

OK
```

WHERE still passes; masked_select remains incomplete. This regression check does not cover every fixed-loop kernel.

### The first reduction still has one workgroup

A fresh 15-second interrupted profile recorded 1,315 run_integer calls taking 5.747s, 569 run_half_compare calls taking 2.558s, and 498 bounded calls taking 0.136s. There were 54,976 submissions. The interruption raised SystemExit(124), so the printed test ERROR is a profiling stop, not a wrong-output result.

Checking the first kernel confirms `batch=8`, all three loop/storage checks true, and no excluded UOps. But its global size is `(1,1,1)`: there is only one workgroup to put into that batch. This change can batch the later independent groups, not the iterations of this first reduction.

Why is masked_select reducing before selecting? Read its existing implementation in `tinygrad/mixin/op.py`: it computes `mask.cumsum()`, reads the last count to size the result, then scatters and computes another cumulative sum. We have not changed that shared code. Passing the elementwise WHERE test does not cover this work.

The 1500 renderer's `_lower_mapped_reduce` handles this differently. It identifies the reduction axes, materializes the mapped inputs, then calls a physical reduction helper. Our interpreter still executes one accumulator update per loop iteration. Reusing that idea needs a separate reduction-lowering step with its own checks; simply increasing the workgroup batch or adding more PC tails cannot make these dependent updates independent.

### Shorten each integer ADD first

Before adding a reduction compiler, look at the accumulator update we already have. ADD currently computes `a - (-b)`: two subtraction chains, ten NPU tasks per byte. We can add the bytes directly:

```text
total = a_byte + b_byte + carry_in     # 0..511
carry_out = 255 < total               # 0 or 1, using MIN+BINARY_EN
result_byte = total - 256*carry_out   # 0..255
```

Start with carry 0 and visit the low byte first. Discard the last carry for fixed-width wrapping, including signed integers. This uses five tasks per byte with the existing ADD, comparison, MUL and SUB modes; Python still only copies the bytes.

```diff
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
-      elif op is Ops.ADD: out = subtract(a, negate(b))[0]
+      elif op is Ops.ADD:
+        out, carry = [], zero
+        for x,y in zip(a, b):
+          total = add(add(x, y), carry)
+          carry = lt(const(255), total)
+          out.append(sub(total, mul(radix, carry)))
```

Run the existing fixed-size variant first. It covers truncating and padding the selection, empty input, and preserving the output dtype:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select_size
test_masked_select_size (__main__.TestOps.test_masked_select_size) ... ok

Ran 1 test in 1.779s

OK
```

This is not a full overflow test for integer ADD, or a replacement for the original masked_select test.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
test_masked_select (__main__.TestOps.test_masked_select) ...
# 30-second command limit, exit 124. No test summary.
```

Still too slow. The ADD recipe now needs half as many tasks per byte, but this does not establish a whole-test speedup or fix the sequential reductions. Keep masked_select incomplete.

### Reuse repeated integer results

A 15-second diagnostic kept executing every NPU call while counting raw input batches. It saw 535 integer ADD calls with 194 distinct batches and 215 integer WHERE calls with one distinct batch. The diagnostic stopped deliberately; these counts are not a completed test or a speedup measurement.

Keep up to 4096 results within one kernel call. These helpers return copies backed by immutable bytes, so later scratch writes cannot overwrite them. A hit reuses the exact NPU result; a miss still submits the arithmetic. Use complete input bytes rather than numeric equality, and discard the cache after the call:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
     st = time.perf_counter()
+    alu_cache:dict[tuple, list] = {}
@@
         elif u.op in GroupOp.ALU:
           assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {u.op}"
           assert all_same([u.dtype] + src_dtypes) or u.op in {*GroupOp.Comparison, Ops.WHERE, Ops.SHL, Ops.SHR}, f"dtype mismatch on {u.op}"
+          # Reuse only exact NPU-produced words, within this kernel call; NaN payloads and signed zeros stay distinct.
+          cache_key = (u.op, u.dtype, tuple(tuple(bytes(raw16(x, dt)) for x in xs) for dt,xs in zip(src_dtypes, src_values))) \
+            if (u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int)) else None
+          if cache_key is not None and cache_key in alu_cache:
+            values[u] = alu_cache[cache_key]
+            i += 1
+            continue
           integer_dtype = src_dtypes[1] if u.op is Ops.WHERE else src_dtypes[0]
@@
+          if cache_key is not None and len(alu_cache) < 4096: alu_cache[cache_key] = values[u]
         assert u in values, u
```

Retest with the cache enabled:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
test_masked_select (__main__.TestOps.test_masked_select) ...
# 30-second command limit, exit 124. No test summary.
```

Still incomplete. Repeated results can avoid submissions, but the cache does not change the reduction algorithm or eliminate every comparison.

The same checkpoint still passes the existing WHERE and fixed-size selection checks:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_where TestOps.test_masked_select_size
test_where (__main__.TestOps.test_where) ... ok
test_masked_select_size (__main__.TestOps.test_masked_select_size) ... ok

Ran 2 tests in 2.200s

OK
```

### Inspect the reduction before lowering it

Inspecting the first kernel before execution gives this shortened UOp list. This diagnostic stopped before submitting the kernel; it is not another test pass or arithmetic error:

```text
BUFFER int REG[1]
STORE accumulator, 0
RANGE 320
  LOAD accumulator
  LOAD half_input[range]
  CMPLT half(0.5), input
  CAST bool -> int
  ADD accumulator, mask
  STORE accumulator
END
LOAD accumulator
STORE output[0]
```

So this first kernel counts the selected inputs. The mapped input is `CAST(CMPLT(0.5, input), int)`, independent of the accumulator. Only ADD carries a dependency between iterations.

For this integer reduction, the candidate is:

| Step | Work                                         | Dtype          |
|------|----------------------------------------------|----------------|
| 1    | Compare the independent input lanes with 0.5 | FP16 → bool    |
| 2    | Convert each mask to 0/1                     | bool → INT32   |
| 3    | Add pairs, then repeat on the partial sums   | INT32 → INT32  |
| 4    | Store the remaining count                   | INT32          |

A balanced tree takes 9 levels for 320 terms. Each level still needs enough NPU tasks for its lane count; this does not mean nine submissions. The count stays within 0..320. More generally, fixed-width wrapping integer ADD is associative, so regrouping its additions preserves the bits. Floating-point ADD does not have that guarantee and must not enter this path.

Before enabling this, the matcher must prove a fixed loop bound, one private accumulator, an ADD update, and a mapped expression independent of that accumulator. Other stores, nested loops or conditional backedges need the existing interpreter. This lowering is not implemented yet; the timeout above remains open.

### Map the fixed integer sum

Match only one fixed loop of at most 1024 terms, with a private integer ADD accumulator and no other stores. Reject terms that read register storage, other non-global loads, conditional execution, or loop intermediates consumed after END. Leave SHL/SHR out too: batching loop coordinates could turn a supported task-wide shift count into varying counts. Unsupported shapes keep the original interpreter.

At RANGE, replicate each workgroup's existing inputs across the loop coordinates. The body then evaluates independent terms together. At the accumulator STORE, pairwise NPU ADD reduces those terms, carrying an odd last term forward unchanged. Add the initial accumulator once, restore the original workgroup lanes, and resume after END. Python rearranges lists and copies storage; it never sums the values.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def mapped_sum_loop(self) -> tuple[UOp, UOp, UOp, UOp, int]|None:
+    if any(u.op in (Ops.IF, Ops.ENDIF) for u in self.uops): return None
+    ranges = [u for u in self.uops if u.op is Ops.RANGE]
+    if len(ranges) != 1: return None
+    loop = ranges[0]
+    if loop.dtype not in dtypes.ints or not loop.src: return None
+    bound = loop.src[0]
+    if bound.op is Ops.CAST: bound = bound.src[0]
+    if bound.op is not Ops.CONST or not isinstance(bound.arg, int) or not 1 < bound.arg <= 1024: return None
+    end = self.uops[self.loop_ends[loop]]
+    if len(end.src) != 2 or end.src[0].op is not Ops.STORE: return None
+    store = end.src[0]
+    index, update = store.src
+    if index.addrspace is not AddrSpace.REG or update.op is not Ops.ADD or update.dtype not in dtypes.ints: return None
+    if index.op is not Ops.INDEX or len(index.src) != 2 or not index.src[1].vmin == index.src[1].vmax == 0: return None
+    buffer = index.src[0]
+    while buffer.op is Ops.AFTER: buffer = buffer.src[0]
+    if buffer.op is not Ops.BUFFER or buffer.max_numel() != 1: return None
+    acc = next((s for s in update.src if s.op is Ops.LOAD and s.src == (index,)), None)
+    if acc is None: return None
+    term = update.src[1] if update.src[0] is acc else update.src[0]
+    if any(u.addrspace is AddrSpace.REG for u in term.toposort()): return None
+    body = self.uops[self.uop_to_index[loop]+1:self.loop_ends[loop]]
+    pure = {Ops.CONST, Ops.CAST, Ops.BITCAST, Ops.INDEX, Ops.LOAD, Ops.AFTER, *GroupOp.ALU} - {Ops.SHL, Ops.SHR}
+    if any(u is not store and u.op not in pure for u in body): return None
+    if any(u.op is Ops.LOAD and u is not acc and u.src[0].addrspace is not AddrSpace.GLOBAL for u in body): return None
+    if any(acc in u.src and u is not update for u in body): return None
+    if any(s in body for u in self.uops[self.loop_ends[loop]+1:] for s in u.src): return None
+    return loop, store, term, acc, bound.arg
+
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
     warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
+    mapped_sum = self.mapped_sum_loop() if local_size == (1,1,1) else None
@@
         if getenv("TRACE"): print(i, u.op, u.dtype, u.arg, src_values, src_dtypes)
+        if mapped_sum is not None and u is mapped_sum[1].src[1]:
+          # Reduce the independent terms below; add the original accumulator only once.
+          values[u] = values[mapped_sum[2]]
+          i += 1
+          continue
+        if mapped_sum is not None and u is mapped_sum[1]:
+          loop, _, term, acc, count = mapped_sum
+          rows = [values[term][j:j+count] for j in range(0, warp_size, count)]
+          while len(rows[0]) > 1:
+            pairs = len(rows[0])//2
+            reduced = self.run_integer(Ops.ADD, [[x for row in rows for x in row[:pairs*2:2]],
+                                                [x for row in rows for x in row[1:pairs*2:2]]], term.dtype)
+            rows = [reduced[j*pairs:(j+1)*pairs] + (row[-1:] if len(row)%2 else []) for j,row in enumerate(rows)]
+          totals = self.run_integer(Ops.ADD, [[row[0] for row in rows], values[acc][::count]], term.dtype)
+          values = {node:lanes[::count] for node,lanes in values.items()}
+          warp_size //= count
+          for (mem,offset),value in zip(values[u.src[0]], totals): _store(mem, offset, value, term.dtype)
+          del values[loop]
+          i = self.loop_ends[loop]+1
+          continue
         if u.op is Ops.END:
@@
         elif u.op is Ops.RANGE:
+          if mapped_sum is not None and u is mapped_sum[0]:
+            count = mapped_sum[4]
+            values = {node:[x for x in lanes for _ in range(count)] for node,lanes in values.items()}
+            values[u] = list(range(count))*warp_size
+            warp_size *= count
+            i += 1
+            continue
           if u not in values: values[u] = [0] * warp_size
```

Retest the existing small selection and nonzero cases, plus WHERE:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select_size TestOps.test_nonzero_size TestOps.test_where
test_masked_select_size (__main__.TestOps.test_masked_select_size) ... ok
test_nonzero_size (__main__.TestOps.test_nonzero_size) ... ok
test_where (__main__.TestOps.test_where) ... ok

Ran 3 tests in 4.053s

OK
```

A separate interrupted diagnostic confirmed that both 320-term loops in masked_select select the mapped path. It stopped inside run_half_compare, not at the matcher gate. That establishes dispatch, not completion or correctness of the unfinished large test.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
test_masked_select (__main__.TestOps.test_masked_select) ...
# 30-second command limit, exit 124. No test summary.
```

The full test still times out. We removed the serial accumulator dependency, but the overlapping prefixes still map many repeated inputs. Measure that remaining comparison work before changing another register or claiming the reduction fixed the test.

### Compare repeated lanes once

A 15-second interrupted diagnostic counted 2,880 FP16 CMPLT lanes but only 326 distinct operand pairs. It still executed every comparison; this is evidence of repetition, not a passing run.

The existing arithmetic cache matches a whole vector. Here the same pair can appear at different positions in different prefixes. Key each comparison by its operation, dtype and both raw inputs. Submit the distinct missing pairs together, then copy their NPU-produced bool bytes back to the original lane order. Keep at most 4096 pairs per kernel call. Signed zeros and NaN payloads remain distinct; Python does not compare the numeric values.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
     alu_cache:dict[tuple, list] = {}
+    comparison_cache:dict[tuple, Any] = {}
@@
           elif u.op in GroupOp.Comparison and src_dtypes == [dtypes.half, dtypes.half]:
-            values[u] = self.run_half_compare(u.op, *src_values)
+            keys = [(u.op, src_dtypes[0], bytes(raw16(a, src_dtypes[0])), bytes(raw16(b, src_dtypes[0])))
+                    for a,b in zip(*src_values)]
+            missing = {key:j for j,key in enumerate(keys) if key not in comparison_cache}
+            fresh = dict(zip(missing, self.run_half_compare(u.op, [src_values[0][j] for j in missing.values()],
+                              [src_values[1][j] for j in missing.values()]))) if missing else {}
+            for key,value in fresh.items():
+              if len(comparison_cache) < 4096: comparison_cache[key] = value
+            values[u] = [comparison_cache[key] if key in comparison_cache else fresh[key] for key in keys]
```

The comparison cache alone still hit the 30-second limit, with exit 124 and no test summary. The interrupted diagnostic also stopped in general integer CMPLT. Our bounded ADD/MUL gate already uses static input bounds, following the 1500 branch's range-admission idea. Apply the same gate to integer comparisons before decomposing their bytes.

We already use MIN+BINARY_EN for integer comparisons. Probe it with the same 49 INT16-boundary pairs and INT32 storage: every result matches `int(a < b)`. Extend the bounded path to CMPLT. Its NPU result is an INT32 0/1 word; copying its low byte gives the bool representation without a numeric host comparison:

```diff
   def run_bounded_int32(self, op:Ops, a:list, b:list) -> list:
-    # Caller proves both inputs fit INT16; their ADD/MUL results fit INT32 without overflow.
-    assert op in (Ops.ADD, Ops.MUL) and len(a) == len(b)
+    # Caller proves both inputs fit INT16; arithmetic fits INT32 and comparisons produce 0/1.
+    assert op in (Ops.ADD, Ops.MUL, Ops.CMPLT) and len(a) == len(b)
+    dtype = dtypes.bool if op is Ops.CMPLT else dtypes.int
@@
-      self.mulacc_stage(2 if op is Ops.ADD else 0, base, base+64, base+128, precision=4, output=4, mul=op is Ops.MUL)
+      self.mulacc_stage(1 if op is Ops.CMPLT else 2 if op is Ops.ADD else 0, base, base+64, base+128,
+                        precision=4, output=4, mul=op is Ops.MUL, binary=op is Ops.CMPLT)
       raw = bytes(to_mv(self.dev.input_buf+128, 32))
-      result.extend(typed_view(raw[i*4:i*4+4], dtypes.int) for i in range(count))
+      result.extend(typed_view(raw[i*4:i*4+dtype.itemsize], dtype) for i in range(count))
```

Keep the operand-bound check; only CMPLT's result dtype changes:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-    bounded_int32 = {u for u in self.uops if u.op in (Ops.ADD, Ops.MUL) and u.dtype == dtypes.int and
+    bounded_int32 = {u for u in self.uops if u.op in (Ops.ADD, Ops.MUL, Ops.CMPLT) and
+                     u.dtype == (dtypes.bool if u.op is Ops.CMPLT else dtypes.int) and
                      all(s.dtype == dtypes.int and -32768 <= s.vmin <= s.vmax <= 32767 for s in u.src)}
```

The bounded CMPLT checkpoint also timed out at 30 seconds. The mapped sum still calls general byte-wise ADD at every tree level, even for mask terms.

Tinygrad's static CAST bounds preserve bool's 0..1 range when converting to INT32. For N terms in [lo, hi], every partial sum lies in `[N*min(0, lo), N*max(0, hi)]`. If this whole interval fits INT16, the existing bounded INT32 ADD can handle every pair without splitting bytes. The initial accumulator is not covered by this proof, so its final addition keeps the general path.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           loop, _, term, acc, count = mapped_sum
           rows = [values[term][j:j+count] for j in range(0, warp_size, count)]
+          bounded = term.dtype == dtypes.int and -32768 <= count*min(0, term.vmin) and count*max(0, term.vmax) <= 32767
           while len(rows[0]) > 1:
             pairs = len(rows[0])//2
-            reduced = self.run_integer(Ops.ADD, [[x for row in rows for x in row[:pairs*2:2]],
-                                                [x for row in rows for x in row[1:pairs*2:2]]], term.dtype)
+            operands = [[x for row in rows for x in row[:pairs*2:2]], [x for row in rows for x in row[1:pairs*2:2]]]
+            reduced = self.run_bounded_int32(Ops.ADD, *operands) if bounded else self.run_integer(Ops.ADD, operands, term.dtype)
```

Retest this checkpoint, without later backend support:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select
test_masked_select (__main__.TestOps.test_masked_select) ...
# 30-second command limit, exit 124. No test summary.

$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_eq TestOps.test_cmp_lt TestOps.test_masked_select_size TestOps.test_where
test_cmp_eq (__main__.TestOps.test_cmp_eq) ... ok
test_cmp_lt (__main__.TestOps.test_cmp_lt) ... ok
test_masked_select_size (__main__.TestOps.test_masked_select_size) ... ok
test_where (__main__.TestOps.test_where) ... ok

Ran 4 tests in 2.741s

OK
```

The comparison and small selection regressions pass, but masked_select remains incomplete. These changes do not earn another completed op. Next locate which kernel consumes the remaining time; do not assume it is still the first reduction.

### Fill the bounded tasks too

A DEBUG=2 run completed the first `r_320` kernel in 218.23ms before the command timed out later. A separate interrupted diagnostic reached kernel 2 with global size `(320,1,1)`. Its mapped batches have 2,560 lanes. The stack stopped in run_bounded_int32, whose packing loop still handles only eight lanes per task.

We already widened mulacc_stage for integer WHERE. Use that same layout here instead of leaving the new helper at eight lanes:

Start with bounded integer arithmetic. Up to 128 INT32 lanes need 512 bytes per operand, so move the second operand to offset 512 and output to 1024. Pad only to the next eight-lane atom; return only the requested lanes.

```diff
   def run_bounded_int32(self, op:Ops, a:list, b:list) -> list:
@@
-    for start in range(0, len(a), 8):
-      count = min(8, len(a)-start)
-      for offset,values in ((0, a), (64, b)):
-        raw = b"".join(raw16(x, dtypes.int) for x in values[start:start+8])
-        to_mv(self.dev.input_buf+offset, 32)[:] = raw+bytes(32-len(raw))
-      self.mulacc_stage(1 if op is Ops.CMPLT else 2 if op is Ops.ADD else 0, base, base+64, base+128,
-                        precision=4, output=4, mul=op is Ops.MUL, binary=op is Ops.CMPLT)
-      raw = bytes(to_mv(self.dev.input_buf+128, 32))
+    for start in range(0, len(a), 128):
+      count = min(128, len(a)-start)
+      atoms = (count+7)//8
+      for offset,values in ((0, a), (512, b)):
+        raw = b"".join(raw16(x, dtypes.int) for x in values[start:start+count])
+        to_mv(self.dev.input_buf+offset, atoms*32)[:] = raw+bytes(atoms*32-len(raw))
+      self.mulacc_stage(1 if op is Ops.CMPLT else 2 if op is Ops.ADD else 0, base, base+512, base+1024,
+                        precision=4, output=4, mul=op is Ops.MUL, binary=op is Ops.CMPLT, atoms=atoms)
+      raw = bytes(to_mv(self.dev.input_buf+1024, atoms*32))
       result.extend(typed_view(raw[i*4:i*4+dtype.itemsize], dtype) for i in range(count))
```

Run with DEBUG=2 to see which kernels finish. These are excerpts from the bounded run, not a passing test summary:

```bash
$ DEBUG=2 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select

*** ROCKCHI    2 r_320       arg 2 ... tm 299.01ms
*** ROCKCHI    4 r_320_320   arg 2 ... tm 11.11s
# 30-second command limit, exit 124. No test summary.
```

The first count and the 320-output prefix kernel now finish within the command limit. The whole test does not: later selection work remains. Keep the measured kernel times separate from whole-test completion.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_eq TestOps.test_cmp_lt TestOps.test_masked_select_size TestOps.test_where
Ran 4 tests in 3.060s

OK
```

### Equality does not need signed ordering

The next inspected kernel has 118 output lanes and a 320-step reduction. It loads an INT32 prefix index, compares it with the output index using CMPNE, then adds `WHERE(unequal, 0, 1)` to the count. The diagnostic stopped before executing this kernel; its printed ERROR was that deliberate stop.

The 1500 branch's `_compare_int32_words` separates equality from signed ordering. Our general integer path still computes sign corrections and a borrow chain even for CMPNE. We can avoid those.

Copy each pair of bytes into a zero-extended INT32 lane. For integer equality, signedness does not change which bit patterns match:

```text
difference[i] = a_word[i] - b_word[i]       # -65535..65535
distance = SUM(ABS(difference[i]))
CMPNE = 0 < distance
CMPEQ = 1 - CMPNE
```

For INT64 there are four words, so distance is at most `4*65535 = 262140`, safely within INT32. INT8 uses one byte instead. No carry or sign correction is needed.

Probe the existing INT32 ABS and MIN+BINARY_EN modes before using them here:

| Input difference | -65535 | -32768 | -1 | 0 | 1 | 32768 | 65535 |
|------------------|-------:|-------:|---:|--:|--:|------:|------:|
| NPU ABS          |  65535 |  32768 |  1 | 0 | 1 | 32768 | 65535 |
| NPU 0 < ABS      |      1 |      1 |  1 | 0 | 1 |     1 |     1 |

These are measured register-probe results, not a test_ops.py pass. Keep the existing byte limbs for ordering, division and wrapping arithmetic; only EQ/NE switch to words. They can use the already-tested wider INT32 task layout:

```diff
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
-    batch = 128 if op is Ops.WHERE else 8
+    batch = 128 if op in (Ops.WHERE, Ops.CMPEQ, Ops.CMPNE) else 8
@@
-      # Only copy raw bytes into zero-padded INT32 lanes; all carries, signs and arithmetic are NPU operations.
+      # Equality needs raw words, not signed ordering. Other operations keep base-256 limbs.
+      limb_size = min(2, width) if op in (Ops.CMPEQ, Ops.CMPNE) else 1
       operands = []
@@
-        operands.append([put(b"".join(x[i:i+1]+bytes(3) for x in raw)) for i in range(dt.itemsize)])
+        operands.append([put(b"".join(x[i:i+limb_size]+bytes(4-limb_size) for x in raw)) for i in range(0, dt.itemsize, limb_size)])
@@
       if op is Ops.WHERE: out = choose(a[0], operands[1], operands[2])
+      elif op in (Ops.CMPEQ, Ops.CMPNE):
+        total = zero
+        for x,y in zip(a, b): total = add(total, calc(5, sub(x, y)))
+        unequal = lt(zero, total)
+        out = [sub(one, unequal) if op is Ops.CMPEQ else unequal]
       elif op is Ops.NEG: out = negate(a)
@@
-        if op in (*GroupOp.Comparison, Ops.MAX):
-          difference, unsigned_lt = subtract(a, b)
+        if op in (Ops.CMPLT, Ops.MAX):
+          _, unsigned_lt = subtract(a, b)
           less = select(different_sign, sign_a, unsigned_lt)
-          if op is Ops.MAX: out = choose(less, b, a)
-          elif op is Ops.CMPLT: out = [less]
-          else:
-            unequal = nonzero(difference)
-            out = [unequal if op is Ops.CMPNE else sub(one, unequal)]
+          out = choose(less, b, a) if op is Ops.MAX else [less]
```

Run the existing checks at this checkpoint, without the later FDIV or FP32 changes:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_eq TestOps.test_cmp_lt TestOps.test_masked_select_size TestOps.test_where

Ran 4 tests in 2.798s
OK
```

Now retry the original masked_select test with the same 30-second limit. DEBUG=2 shows which kernels finish (copy and compilation lines omitted):

```bash
$ DEBUG=2 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_masked_select

*** ROCKCHI    2 r_320                                          arg  2 mem   0.00 GB tm    128.86ms/   134.00ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    4 r_320_320                                      arg  2 mem   0.00 GB tm     11.32s / 11455.64ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    5 r_118_320                                      arg  2 mem   0.00 GB tm   1841.31ms/ 13296.95ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    6 r_118_118                                      arg  2 mem   0.00 GB tm   3663.49ms/ 16960.43ms (      0 GFLOPS    0|0      GB/s)
*** ROCKCHI    7 E_118                                          arg  3 mem   0.00 GB tm     87.88ms/ 17048.32ms (      0 GFLOPS    0|0      GB/s)
# Next kernel compiled: r_320_320_320_320
# 30-second command limit, exit 124. No test summary.
```

The equality-counting kernel now finishes in 1.84s. The first helper_test_op completes and the test reaches its second case, masked_select(Tensor(True)). That case still times out. So this fixes the earlier bottleneck, not the whole test; next inspect its nested reductions before changing another path.

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
+      def calc(algo:int|None, x:int, y:int=threshold, **kw) -> int:
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

Those cases passed, but a saved 81-pair probe crossed the special numerators and denominators and found failures. It checked NaN classification and the exact bits of other results, including zero signs.

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

There is no new runtime diff here: the tutorial already copies each completed result. Retest the existing Tensor cases.

```bash
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

```bash
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

Before moving to MUL, check the related ADD/SUB cases at this checkpoint. Test names are not all test_add_*: inspect test_add3, test_broadcasted_add and test_broadcasted_add_2 too. These fresh runs use default FP32, with no later MUL/FDIV support or batching changes. TRACE output is omitted below:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add3

Ran 1 test in 3.464s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_broadcasted_add

Ran 1 test in 22.075s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_broadcasted_add_2

Ran 1 test in 20.812s
OK

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_tiny_add TestOps.test_scalar_sub TestOps.test_scalar_rsub

Ran 3 tests in 4.218s
OK
```

All four commands exited normally within their separate 30-second limits. This adds three-input ADD, column/row/scalar broadcasting, tiny ADD and both scalar SUB directions to this section's coverage. Padding, scatter-add and logaddexp are composed operations with other dependencies, not extra coverage implied by these passes. They still belong in their own checks and the final sweep.

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
+      def calc(algo:int|None, x:int, y:int, **kw) -> int:
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

A saved register probe used 128 random raw pairs and 81 special-value pairs. NumPy supplied the reference, not the runtime arithmetic.

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

### Check the existing workgroup batching

The workgroup batching was introduced at the first bool-WHERE timeout above. Before batching, the counted tiny MUL probe made 64 one-lane calls. Now check that the new FP32 MUL helper uses those lanes too.

The same counted tiny test now made eight calls with eight lanes each. Check a partial batch too: 17 values should give 8, 8, 1, with no padded values stored.

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

Before leaving MUL, run its scalar and special-value variants at this reconstructed checkpoint, without DEFAULT_FLOAT=HALF:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mul_naninf
# 30-second command limit, exit 124. No test summary.

$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scalar_mul
# 30-second command limit, exit 124. No test summary.
```

These were separate serial runs, not a combined timeout. test_mul_naninf reached its NaN multiplication case after the two infinity cases, but the whole method did not finish. Neither method is counted as passed.

### Probe BS multiplication for scalar coefficients

The integer-significand implementation handles arbitrary FP32 operands, but these variants use constants such as 2, -1, 255 and infinity. Those constants fit FP16 exactly. Can our existing BS multiplier read an FP32 main input and an FP16 coefficient, avoiding the long integer multiplication sequence? This would not narrow the tensor input.

The first probe packed eight half coefficients into 16 bytes. With inputs 1..8 and coefficient 2, it returned:

```text
actual:   2, 4, 6, 8, 0, 0, 0, 0
expected: 2, 4, 6, 8, 10, 12, 14, 16
```

The missing lanes suggest a layout problem. In this mode the BS coefficients occupy two 16-byte surfaces: four useful half values at offset 0, then four at offset 16. Filling both surfaces with the constant fixed this probe. Padding each coefficient to a four-byte word instead gave alternating zeros, so that was not the right layout either.

With both surfaces populated, the direct mulacc_stage(None, ..., precision=5, output=5, bs_mul=...) probe compared 1032 FP32 input encodings per coefficient: 1024 random words plus zeros, subnormals, a normal boundary, maximum finite and infinities. It found zero mismatches for each of 2, -1, 255, +inf, -inf and NaN. Non-NaN results were checked bit-for-bit; NaN classification was checked, not its payload.

Keep the tensor input FP32. Reject a finite coefficient if packing it as half changes its value; that case falls back to our exact integer-significand helper. NaN and infinity remain special floating coefficients, not Python-computed results:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_float_scale(self, a:list, scale:float) -> list:
+    # Only the constant coefficient may use FP16; keep the main input and output FP32.
+    if math.isfinite(scale) and (abs(scale) > 65504 or struct.unpack("<e", struct.pack("<e", scale))[0] != scale):
+      return self.run_float_mul(a, [scale]*len(a))
+    base, result = self.dev.input_mem.dma_addr, []
+    to_mv(self.dev.input_buf+64, 32)[:] = bytes(32)
+    # Four half coefficients in each 16-byte surface; filling both also initializes the padding.
+    to_mv(self.dev.input_buf+128, 32)[:] = struct.pack("<e", scale)*16
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      to_mv(self.dev.input_buf, 32)[:] = b"".join(raw16(x, dtypes.float) for x in a[start:start+8])+bytes(4*(8-count))
+      self.mulacc_stage(None, base, base+64, base+192, precision=5, output=5, bs_mul=base+128)
+      raw = bytes(to_mv(self.dev.input_buf+192, 32))
+      result.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
+    return result
+
```

Select this helper only when one operand is a CONST, optionally behind its dtype CAST. Do not inspect a tensor to decide that it is uniform:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
           elif u.op is Ops.MUL and u.dtype == dtypes.float:
-            values[u] = self.run_float_mul(*src_values)
+            # Match compile-time coefficients, not coincidentally uniform tensor values.
+            constant = next((i for i,x in enumerate(u.src) if x.op is Ops.CONST or
+                             (x.op is Ops.CAST and x.src[0].op is Ops.CONST)), None)
+            values[u] = self.run_float_scale(src_values[1-constant], scalar16(src_values[constant][0])) if constant is not None else \
+              self.run_float_mul(*src_values)
```

Include a partial atom and two fallback coefficients: 1+2^-23 does not fit half exactly, and 65536 exceeds its finite range.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scalar_mul TestOps.test_mul_naninf

test_scalar_mul (__main__.TestOps.test_scalar_mul) ... ok
test_mul_naninf (__main__.TestOps.test_mul_naninf) ... ok

Ran 2 tests in 1.246s
OK
```

The helper check covers 105 input words for each of ten coefficients, including signed zeros and the two fallback cases. Both unchanged Tensor variants now finish within the 30-second limit. This improves constant MUL; arbitrary tensor-by-tensor FP32 MUL still uses the integer-significand path.

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
-      return self.run_float_mul(a, [scale]*len(a))
+      return self.run_float_binary(Ops.MUL, a, [scale]*len(a))
@@
-              self.run_float_mul(*src_values)
+              self.run_float_binary(u.op, *src_values)
```

Update the MUL checks for the shared name, and add a direct FDIV check. This compares raw bits for non-NaN results, including signed zero; NaN payloads are not promised.

All 605 pairs passed: 512 random raw pairs, 81 special-value pairs and 12 boundary pairs. This is not an exhaustive FP32 division test. Now allow FP32 FDIV; the batching allowlist already includes FDIV:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
           elif u.op in (Ops.ADD, Ops.SUB, Ops.NEG) and u.dtype == dtypes.float:
             values[u] = self.run_float_alu(u.op, src_values[0], src_values[1] if len(src_values) > 1 else None)
-          elif u.op is Ops.MUL and u.dtype == dtypes.float:
+          elif u.op in (Ops.MUL, Ops.FDIV) and u.dtype == dtypes.float:
             # Match compile-time coefficients, not coincidentally uniform tensor values.
             constant = next((i for i,x in enumerate(u.src) if x.op is Ops.CONST or
-                             (x.op is Ops.CAST and x.src[0].op is Ops.CONST)), None)
+                             (x.op is Ops.CAST and x.src[0].op is Ops.CONST)), None) if u.op is Ops.MUL else None
```

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div

Ran 1 test in 25.259s
OK
```

The process exited normally within the 30-second limit. No FP16 narrowing or Python quotient calculation is used. This fixes this FP32 primitive; it does not establish that every division-based expression or saved failure now passes.

The half division path now batches independent FDIV workgroups too. Check it still handles signs and special values:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_div TestOps.test_div_naninf TestOps.test_copysign_exact

Ran 3 tests in 7.270s
OK
```

All 315 tutorial hunks replayed and compiled. The reconstructed code passed the FDIV, MUL and tail checks in 15.951s too. The full-suite count is still the saved baseline; no new complete sweep has run.

Before moving to CAST, check the FP32 scalar and special-value variants too, without DEFAULT_FLOAT=HALF:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scalar_div
# 30-second command limit, exit 124. No test summary.

$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_naninf
# 30-second command limit, exit 124. No test summary.
```

Both were separate runs at this reconstructed FDIV checkpoint. Neither completed within the limit. The earlier half special-value pass does not cover these FP32 runs, and the direct 605-pair check does not make these Tensor variants passed.

Our scalar MUL shortcut does not cover FDIV. Each FP32 quotient still uses 27 long-division iterations before rounding and special-value selection. This suggests a performance problem, but these timeouts alone say nothing about whether the remaining outputs are correct. Keep both variants unresolved; do not replace division with multiplication by a rounded reciprocal without checking its accuracy.

The mixed integer division test finishes at the same checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

test_div_int (__main__.TestOps.test_div_int) ... ok

Ran 1 test in 4.141s
OK
```

This includes true division, floor division and truncating division. It is not proof that their input CASTs run on the NPU: the next section checks the numeric CAST fallback.

Before leaving division, try its rounding modes too:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode

NotImplementedError: ROCKCHIP NPU does not support Ops.TRUNC with dtypes.float
Ran 1 test in 5.985s
FAILED (errors=1)
```

The first missing primitive is FP32 TRUNC. Our earlier matcher only handles half. Reuse its formula without narrowing the quotient or converting it to INT32:

```text
FP32 x → FLOOR(x) ─────────────────┐
       → CEIL(x) → MIN(result, 0) ─┴→ MAX → FP32 trunc(x)
```

The direct FP32 register probe checked 265 inputs: nine explicit cases (signed zeros, ±0.9, ±2.9, infinities and NaN) plus 256 random encodings. All non-NaN result bits and NaN classifications matched NumPy truncation. In particular, negative values between -1 and 0 kept negative zero. Now add those four stages:

```diff
 class RockchipProgram(Program['RockchipDevice']):
+  def run_float_trunc(self, a:list) -> list:
+    base, result = self.dev.input_mem.dma_addr, []
+    to_mv(self.dev.input_buf+64, 32)[:] = bytes(32)
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      to_mv(self.dev.input_buf, 32)[:] = b"".join(raw16(x, dtypes.float) for x in a[start:start+8])+bytes(4*(8-count))
+      # trunc(x) = max(floor(x), min(ceil(x), 0)), keeping every stage FP32.
+      for algo,lhs,rhs,out in ((7, 0, 64, 128), (8, 0, 64, 192), (1, 192, 64, 256), (0, 128, 256, 320)):
+        self.mulacc_stage(algo, base+lhs, base+rhs, base+out)
+      raw = bytes(to_mv(self.dev.input_buf+320, 32))
+      result.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
+    return result
+
   def run_float_scale(self, a:list, scale:float) -> list:
```

Route FP32 TRUNC here; the half matcher remains unchanged:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+          elif u.op is Ops.TRUNC and u.dtype == dtypes.float:
+            values[u] = self.run_float_trunc(src_values[0])
           elif u.op in (Ops.MUL, Ops.FDIV) and u.dtype == dtypes.float:
```

Rerun the unchanged rounding-mode test from this checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode

NotImplementedError: ROCKCHIP NPU does not support Ops.CMPLT with dtypes.bool
Ran 1 test in 6.511s
FAILED (errors=1)
```

It gets past the floating truncation case and reaches rounding_mode="floor". That expression needs a comparison between FP32 operands. The bool in the error is the comparison's output dtype, not its input dtype. Our existing support is FP16, so extend it next.

## FP32 extension of existing comparisons

The blog already passed test_cmp_lt and test_cmp_eq with DEFAULT_FLOAT=HALF. This section extends those same operations to FP32; it does not add new ops to the progress count. The tutorial still only dispatches comparisons with two half inputs. The earlier default-FP32 run stopped here:

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
+          elif u.op in GroupOp.Comparison and src_dtypes in ([dtypes.half]*2, [dtypes.float]*2):
@@
-            fresh = dict(zip(missing, self.run_half_compare(u.op, [src_values[0][j] for j in missing.values()],
-                              [src_values[1][j] for j in missing.values()]))) if missing else {}
+            fresh = dict(zip(missing, self.run_float_compare(u.op, [src_values[0][j] for j in missing.values()],
+                              [src_values[1][j] for j in missing.values()], dtype=src_dtypes[0]))) if missing else {}
```

This reconstructed-checkpoint check passed all six dtype/op combinations. It is not exhaustive over FP32 pairs and does not replace the full Tensor comparison tests. No Python floating comparison chooses the result; the reference calculation only checks the NPU output.

Now reconstruct this earlier checkpoint, before numeric CAST and FP32 WHERE, and rerun the existing Tensor tests without DEFAULT_FLOAT=HALF:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_lt TestOps.test_cmp_eq

test_cmp_lt (__main__.TestOps.test_cmp_lt) ... ok
test_cmp_eq (__main__.TestOps.test_cmp_eq) ... ok

Ran 2 tests in 3.405s
OK
```

The command exited normally within 30 seconds. The unchanged tests cover broadcasting, constants, bool/int inputs and infinities. Their specials list excludes NaN, so keep the raw-pattern check above as well. These passes do not update the historical full-suite count.

Revisit the division rounding-mode test that previously stopped at FP32 CMPLT:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode

NotImplementedError: ROCKCHIP NPU does not support Ops.WHERE with dtypes.float
Ran 1 test in 6.387s
FAILED (errors=1)
```

The comparison now runs, but floor division also needs FP32 WHERE to choose its corrected quotient. Add that next. The timeout at the old, later checkpoint included this support and is not the result of this earlier stage.

## Ops.WHERE: FP32 storage

The rounding-mode test above now stops at FP32 WHERE. The saved convolution investigation found the same missing dispatch:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_strided_conv_transpose2d

NotImplementedError: ROCKCHIP NPU does not support Ops.WHERE with dtypes.float
Ran 1 test in 1.069s
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

A saved raw-bit probe also covered signed zeros, subnormals, infinities and NaNs, including a broadcast constant just above 1.0 that would be lost by narrowing to half.

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

At this earlier checkpoint, the raw FP32 WHERE and comparison checks both pass: 2 tests in 3.145s. No numeric CAST implementation from the next section was included.

Before retrying the long rounding-mode test, inspect one of its seven-value floor divisions. The resulting program contains:

```text
ADD, BITCAST, CAST, CMPLT, CONST, INDEX, LOAD, MUL,
PARAM, SINK, SPECIAL, STORE, TRUNC, WHERE
```

The small probe returns [-1, -1, -1, -0, 0, 0, 0] for [5, 6, 7, 0, -5, -6, -7] divided by -10. TRUNC, comparisons and WHERE were added to the batching allowlist at the first masked_select timeout; no second allowlist change is needed here.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode
# 30-second command limit, exit 124. No test summary.
```

The batching and raw-value checks pass, but the complete rounding-mode method still does not finish within the limit. It contains many separate integer/float and denominator cases, not just the seven-value probe. The TRUNC, comparison and WHERE gates are resolved; full-test completion remains unresolved.

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
+      self.mulacc_stage(None, base, base+64, base+128, precision=precision)
+      value = base+128
```

For an INT32 destination, calculate truncation in FP32 first. Algorithm 7 is FLOOR, 8 is CEIL, 1 is MIN, and 0 is MAX; the expression is the same TRUNC formula from above.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
@@
       self.mulacc_stage(None, base, base+64, base+128, precision=precision)
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
+      self.mulacc_stage(None, value, base+64, base+448, output={dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[dtype])
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

The bool → float case still takes the Python fallback. Can we reuse the two conversions we now have?

```text
bool 0/1 → NPU bool-to-half → FP16 0.0/1.0 → NPU numeric CAST → FP32 0.0/1.0
```

Both values are exact in either float format. A direct probe passed lengths 1, 7, 8, 9, 16 and 17 with alternating false/true values. Packing and copying the intermediate bytes stay on the host; both numeric conversions run on the NPU.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+          elif (src_dtypes[0], u.dtype) == (dtypes.bool, dtypes.float):
+            half = self.run_npu(Ops.CAST, [scalar16(x) for x in src_values[0]])
+            values[u] = self.run_cast(half, dtypes.half, dtypes.float)
           elif src_dtypes[0] in (dtypes.half, dtypes.float, dtypes.int) and u.dtype in (dtypes.half, dtypes.float, dtypes.int):
```

Rerun the unchanged CAST test at this reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cast

test_cast (__main__.TestOps.test_cast) ... ok

Ran 1 test in 0.220s
OK
```

Counting run_cast calls during this run now shows three half → float calls, rather than two: the extra call widens the bool case's NPU-produced half result. The int → float count stays one and half → int stays two. This closes this specific fallback, not every unsupported CAST dtype.

### Check the uint8 CAST variants

There is another CAST test under TestOpsUint8. Run it at the same checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOpsUint8.test_cast

test_cast (__main__.TestOpsUint8.test_cast) ... ok

Ran 1 test in 0.610s
OK
```

This is still a Python fallback: uint8 is not one of run_cast's destination dtypes. Do not count this pass as NPU conversion support.

Could we convert half to INT32 on the NPU, then keep its low byte? Every finite half fits INT32, and copying the low byte would reuse our integer-narrowing storage handling. Probe boundary values before changing the dispatch:

| Half input | NPU INT32 low byte | Torch uint8 |
| ---------- | ----------------: | ----------: |
| -65504     |                32 |          32 |
| -257.75    |               255 |         255 |
| -256       |                 0 |           0 |
| -255.5     |                 1 |           1 |
| -1.9       |               255 |         255 |
| 255.75     |               255 |         255 |
| 256        |                 0 |           0 |
| 65504      |               224 |         224 |

All 14 finite probe values matched, including fractions around zero and a partial atom. The NPU removes the fraction; keeping the low byte gives the observed wraparound for these finite values. But special values expose a missing case:

| Half input | NPU INT32 result | Low byte | Torch uint8 |
| ---------- | ---------------: | -------: | ----------: |
| +inf       |       2147483647 |      255 |         255 |
| -inf       |      -2147483648 |        0 |           0 |
| NaN        |       2147483647 |      255 |           0 |

These are observations from this machine, not a portable rule for out-of-range casts. The direct converter saturates NaN to INT32_MAX, so merely taking its low byte is wrong. This candidate needs an NPU NaN mask before replacing the fallback; it is not enabled yet.

The missing NaN case can be handled on the NPU. Keep the original converted FP32 input, take ABS, then compare its raw encoding as INT32:

```text
magnitude bits < 0x7f800001 → 1 for finite values and infinity, 0 for NaN
converted INT32 × mask    → keep the value, or clear NaN to zero
low output byte          → uint8 storage
```

Why 0x7f800001? Positive infinity is 0x7f800000; a positive NaN has a nonzero fraction above that encoding. ABS removes the sign before the integer comparison. The direct probe passed 523 half encodings, including positive/negative NaNs, infinities, random words and a partial atom. No Python comparison selects the output.

Extend only half → uint8 here; FP32 → uint8 needs its own range tests:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
-    assert src_dtype in (dtypes.half, dtypes.float, dtypes.int) and dtype in (dtypes.half, dtypes.float, dtypes.int)
+    assert src_dtype in (dtypes.half, dtypes.float, dtypes.int) and dtype in (dtypes.half, dtypes.float, dtypes.int, dtypes.uint8)
+    assert dtype != dtypes.uint8 or src_dtype == dtypes.half
@@
-      if dtype == dtypes.int:
+      if dtype in (dtypes.int, dtypes.uint8):
@@
-      self.mulacc_stage(None, value, base+64, base+448, output={dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[dtype])
+      self.mulacc_stage(None, value, base+64, base+448, output={dtypes.half: 2, dtypes.float: 5, dtypes.int: 4, dtypes.uint8: 4}[dtype])
+      if dtype == dtypes.uint8:
+        # ABS produces positive FP32 encodings. Only NaNs exceed the infinity encoding.
+        self.mulacc_stage(5, base+128, base+64, base+512)
+        to_mv(self.dev.input_buf+576, 32)[:] = struct.pack("<i", 0x7f800001)*8
+        self.mulacc_stage(1, base+512, base+576, base+640, precision=4, output=4, binary=True)
+        self.mulacc_stage(0, base+448, base+640, base+704, precision=4, output=4, mul=True)
+        raw = bytes(to_mv(self.dev.input_buf+704, 32))
+        result.extend(typed_view(raw[i*4:i*4+1], dtype) for i in range(count))
+        continue
@@
-          elif src_dtypes[0] in (dtypes.half, dtypes.float, dtypes.int) and u.dtype in (dtypes.half, dtypes.float, dtypes.int):
+          elif (src_dtypes[0], u.dtype) == (dtypes.half, dtypes.uint8):
+            values[u] = self.run_cast(src_values[0], src_dtypes[0], u.dtype)
+          elif src_dtypes[0] in (dtypes.half, dtypes.float, dtypes.int) and u.dtype in (dtypes.half, dtypes.float, dtypes.int):
```

Copying one byte from each completed INT32 lane is the same storage narrowing used for integer CASTs. Truncation and the NaN correction both run on the NPU; this does not claim the DMA already writes packed uint8 lanes.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOpsUint8.test_cast TestOps.test_cast

test_cast (__main__.TestOpsUint8.test_cast) ... ok
test_cast (__main__.TestOps.test_cast) ... ok

Ran 2 tests in 3.021s
OK
```

The first combined attempt also included TestOpsUint8.test_cast_relu between the two CAST tests. It printed ok for the standalone uint8 CAST, then reached the 30-second command limit inside test_cast_relu; the final test did not run. The clean two-test rerun above excludes that unfinished method. CAST-after-ReLU remains unresolved at this checkpoint; do not count it as passed.

The timeout may be unnecessary task overhead. ReLU lowers to MAX(x, 0), but MAX is absent from the straight-line batching allowlist. A 17-value ReLU → uint8 probe recorded these run_cast input sizes:

```text
before: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
target: 8, 8, 1
```

MAX is elementwise, so independent workgroups can share an eight-lane task. Keep the existing local_size, local-memory and control-flow restrictions; extend only this allowlist:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
     batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
                  Ops.BUFFER, Ops.RANGE, Ops.END,
-                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE,
+                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE, Ops.MAX,
                  *GroupOp.Comparison, Ops.SINK, Ops.NOOP, Ops.AFTER}
```

Now rerun the previously unfinished variant on the reconstructed backend:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOpsUint8.test_cast_relu

test_cast_relu (__main__.TestOpsUint8.test_cast_relu) ... ok

Ran 1 test in 3.386s
OK
```

The call-size check now sees 8, 8, 1. The unchanged full ReLU → CAST test finishes within the limit, while the NaN-pattern and existing MUL-tail checks still pass. This resolves the half-input timeout above; it says nothing about FP32 → uint8, which still has no NPU dispatch here.

## Ops.SQRT

TOREVIEW1: The separate ops_map/dtype-gate-only trial is still missing here. The full SQRT test is a known timeout and remains deferred. The implementation and saved probes below do not substitute for that trial.

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
+      def calc(algo:int|None, x:int, y:int|None=None, precision:int=4, output:int=4, **kw) -> int:
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
+        out = calc(None, x, precision=5, output=2)
+        return alloc(read(out, 8)+read(out+16, 8))
+      def square(x:int) -> int: return calc(None, x, precision=2, output=5, bs_mul=x)
```

Upload the original FP16 values and retain a second, padded copy of their encodings. Convert x to FP32 once for the square comparisons.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
         return alloc(read(out, 8)+read(out+16, 8))
       def square(x:int) -> int: return calc(None, x, precision=2, output=5, bs_mul=x)
+      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8]) + bytes(2*(8-count))
+      source = alloc(raw)
+      source_bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = calc(None, source, precision=2, output=5)
```

Search the encoding interval. Each midpoint is rounded upward; if its square exceeds x, lower the upper bound. Otherwise keep it as the new lower bound. Fifteen iterations find the largest representable y with y² ≤ x.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       source_bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
       x = calc(None, source, precision=2, output=5)
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
+      lower32 = calc(None, lower, precision=2, output=5)
+      upper32 = calc(None, upper, precision=2, output=5)
+      gap = half_output(calc(4, upper32, lower32, precision=5, output=5))
+      # midpoint² = lower² + lower*gap + (gap/2)²; each product is exact FP32.
+      half_scale = alloc(struct.pack("<e", 0.5)*8)
+      half_gap = half_output(calc(None, gap, precision=2, output=5, bs_mul=half_scale))
+      cross = calc(None, lower, precision=2, output=5, bs_mul=gap)
+      midpoint2 = calc(2, calc(2, square(lower), cross, precision=5, output=5), square(half_gap), precision=5, output=5)
```

Compare x with the midpoint square. Below selects y, above selects z. At an exact tie, add y's low encoding bit: an odd y rounds up to the even z.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_sqrt(self, a:list) -> list:
@@
       cross = calc(None, lower, precision=2, output=5, bs_mul=gap)
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
@@
-lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.MULACC and u.dtype == dtypes.half:
             values[u] = self.run_mulacc(*src_values)
+          elif u.op is Ops.SQRT and u.dtype == dtypes.half:
+            values[u] = self.run_sqrt(src_values[0])
```

This uses more tasks than Newton iteration. All 65,536 FP16 bit patterns passed: finite results matched the rounded reference bit-for-bit, including -0, and NaN classification matched too. The expanded midpoint square was also exact at all 20,480 relevant rounding boundaries.

The saved exhaustive probe enumerated every FP16 encoding against a double-precision reference rounded to half. It checked NaN classification rather than payload.

That historical exhaustive run is not part of the short test sequence here. Keep each new command within 30 seconds.

The ADD, MUL, maximum, MULACC and raw-NaN regressions also passed: 6 passed in 6.46s.

EXP2, LOG2, POW and SIN remain. The full test_ops.py sweep is still pending.

## Ops.EXP2

TOREVIEW1: The separate ops_map/dtype-gate-only trial is still missing. Do not infer its result from the direct implementation below. test_exp2 is a known timeout, so that full-test checkpoint remains deferred.

With the earlier fixes in place, run tinygrad's existing EXP2 decomposition before advertising EXP2. The fresh run at this reordered checkpoint was bounded to 30 seconds:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp2
# 30-second command limit, exit 124. No test summary.
```

The older checkpoint completed this test in 119.310s with OK. That is a historical result, not a pass for the new order or permission to run beyond the current limit.

A separate small probe used [-25, -24.5, -24, -15, -2, -0.5, -0, 0, 0.5, 1, 2, 10, 15, 16, +inf, -inf, NaN]. It passed rtol=1e-3, atol=1e-6 against double-precision exp2 rounded to half. This is a tolerance check, not bit-exact or full-test coverage. Inspecting its lowered program found no EXP2 UOp and these numeric CAST pairs:

```text
half → short
short → half
```

So tinygrad's decomposition is running, but half ↔ INT16 still takes the generic Python CAST fallback. The new numeric converter above covers INT32, not these INT16 conversions. Neither the timeout nor the small passing probe establishes an accuracy defect in the decomposition.

The direct NPU candidate below avoids those host conversions. It remains a separate implementation choice, not an accuracy fix justified by test_exp2 failing. LOG2 and SIN extend the same run_math helper, so apply these diffs before continuing to those sections.

Before replacing EXP2, try to remove those two CAST fallbacks.

For INT16 → half, first widen numerically to FP32 on the NPU, then convert once to half. The first precision=1 input probe returned only four correct lanes:

```text
input:    -32768, -32767, -2049, -1, 0, 1, 2049, 32767
actual:   -32768, -32767, -2049, -1, 0, 0,    0,     0
```

The half → FP32 path already enables BS_OW_CFG's three SIZE_E fields. Enabling them for INT16 input too fixed all 265 tested INT16 values, including a partial atom. The subsequent FP32 → half conversion also matched the reference for all 265 values.

For half → INT16, every finite half fits INT32. Reuse the NPU truncating conversion, then keep its low two bytes, as with integer narrowing. A separate 269-value finite probe matched, including values outside the INT16 range. Do not silently define nonfinite conversion here: reject those inputs for now.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
-        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) |
-        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT) | (1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT)),
+        ((precision in (1, 2) and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
+        ((precision in (1, 2) and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) |
+        ((precision in (1, 2) and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT) | (1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT)),
@@
   def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
-    assert src_dtype in (dtypes.half, dtypes.float, dtypes.int) and dtype in (dtypes.half, dtypes.float, dtypes.int, dtypes.uint8)
+    if (src_dtype, dtype) == (dtypes.half, dtypes.int16):
+      if any(not math.isfinite(scalar16(x)) for x in a):
+        raise NotImplementedError("ROCKCHIP half to INT16 CAST requires finite inputs")
+      # Numeric truncation runs on the NPU; narrowing retains the low two storage bytes.
+      return [typed_view(raw16(x, dtypes.int)[:2], dtype) for x in self.run_cast(a, src_dtype, dtypes.int)]
+    assert src_dtype in (dtypes.half, dtypes.float, dtypes.int, dtypes.int16) and dtype in (dtypes.half, dtypes.float, dtypes.int, dtypes.uint8)
+    assert src_dtype != dtypes.int16 or dtype == dtypes.half
@@
-      precision = {dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[src_dtype]
+      precision = {dtypes.half: 2, dtypes.float: 5, dtypes.int: 4, dtypes.int16: 1}[src_dtype]
@@
-          elif (src_dtypes[0], u.dtype) == (dtypes.half, dtypes.uint8):
+          elif (src_dtypes[0], u.dtype) in ((dtypes.half, dtypes.uint8), (dtypes.half, dtypes.int16), (dtypes.int16, dtypes.half)):
```

The CPU checks the finite-input restriction and copies storage bytes; it does not compute the converted lane values. Weak integer/float constants still use the existing constant CAST fallback.

Verify these conversions at the reconstructed checkpoint before adding custom EXP2:

The same 17-value native EXP2 probe still passes its tolerance check. Counting run_cast calls reports 17 half → INT16, 17 internal half → INT32 and 17 INT16 → half calls. The two data-conversion fallbacks found above are now replaced with NPU conversions. This does not turn the earlier full-test timeout into a pass.

The direct EXP2 implementation below is now an alternative to this repaired native path, not a prerequisite for removing those CAST fallbacks. Its later accuracy/performance checks must justify using it; the small native probe has not shown a precision failure.

The 17-value probe still used one lane per task. Inspecting its lowered program also finds SHL, SHR, MULACC and CUSTOM (the unary FLOOR/CEIL path). These are absent from the batching allowlist.

They are elementwise here, but the narrow shift helper has a restriction: its count must be uniform within a task. Batch constant-count narrow shifts only; leave dynamic counts on the old per-workgroup path. This avoids combining individually valid calls into an unsupported mixed-count task.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-                 *GroupOp.Comparison, Ops.SINK, Ops.NOOP, Ops.AFTER}
+                 *GroupOp.Comparison, Ops.SHL, Ops.SHR, Ops.MULACC, Ops.CUSTOM, Ops.SINK, Ops.NOOP, Ops.AFTER}
+    # Narrow shifts require one task-wide count. Do not batch dynamic counts across workgroups.
+    uniform_shifts = all(u.op not in (Ops.SHL, Ops.SHR) or u.dtype not in (dtypes.int16, dtypes.uint16) or
+                        u.src[1].op is Ops.CONST or (u.src[1].op is Ops.CAST and u.src[1].src[0].op is Ops.CONST) for u in self.uops)
@@
-    batch = 8 if uniform_loops and private_buffers and simple_ends and local_size == (1,1,1) and \
+    batch = 8 if uniform_shifts and uniform_loops and private_buffers and simple_ends and local_size == (1,1,1) and \
       all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
```

Now retry the unchanged EXP2 test before advertising a custom EXP2 handler:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp2

test_exp2 (__main__.TestOps.test_exp2) ... ok

Ran 1 test in 19.176s
OK
```

This reconstructed native-decomposition checkpoint exited normally within 30 seconds. The earlier timeout is resolved for test_exp2, and the half ↔ INT16 data conversions now run on the NPU. The custom implementation below is therefore an alternative optimization, not something needed to make this test pass. Passing this test is not a guarantee of correctly rounded results for every half encoding.

The reconstructed dynamic-shift, half/INT16 CAST and TRUNC-batching checks also passed together: 3 tests in 0.205s. Dynamic INT16 SHL kept 17 one-lane calls; the TRUNC check still used 8, 8, 1. No input or assertion was relaxed.

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
+      def calc(algo:int|None, x:int, y:int|None=None, precision:int=5, output:int=5, **kw) -> int:
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
+        half = calc(None, value, output=2)
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
         half = calc(None, value, output=2)
         return alloc(read(half, 8)+bytes(8)+read(half+16, 8))
+      def polynomial(weights:int, cs:tuple[float, ...]) -> int:
+        value = const(cs[-1], "f")
+        for c in reversed(cs[:-1]):
+          value = calc(None, value, bs_mul=weights)
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
+      x = calc(None, source, precision=2)
```

Bound the working input to [-25, 16], then split it into integer n and fraction r. The final masks below handle values outside this interval.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
   def run_math(self, op:Ops, a:list) -> list:
@@
       bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
       x = calc(None, source, precision=2)
+      bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
+      integer = calc(7, calc(2, bounded, const(0.5, "f")))
+      fraction = calc(4, bounded, integer)
+      exponent = calc(None, integer, output=4)
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
+      half = calc(None, scaled, output=2)
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
@@
-lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.MULACC, Ops.OR, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.EXP2, Ops.MULACC, Ops.OR, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
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

The saved EXP2 probe also covered all FP16 encodings.

test_exp2 passes with the code built here, before LOG2. The all-encodings check also passed in an earlier run: `2 passed in 54.35s`. That longer sweep was not rerun for this revision.

## Ops.LOG2

TOREVIEW1: The direct LOG2 candidate still needs its own gate-only trial before implementation. test_log2 is a known timeout; retain that gap rather than reporting the later helper result as the trial.

First use the code built so far, without adding LOG2 support or forcing a different decomposition. The earlier investigation recorded this failure:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log2

Mismatched elements: 22 / 2925 (0.752%)
 [10, 18]: -0.225830078125 (ACTUAL), -0.22607421875 (DESIRED)
Max absolute difference among violations: 0.0007324
Max relative difference among violations: 0.00156
Ran 1 test in 127.103s
FAILED (errors=1)
```

That long run is historical. At the reordered checkpoint, the fresh full test reached the 30-second command limit without a summary. A separate 17-value probe did reproduce the numerical problem:

```text
half input: 0.85498046875
actual:    -0.225830078125
reference: -0.22607421875
Mismatched elements: 1 / 17
```

The probe includes subnormal/normal boundaries, values near 1, both zeros, negative inputs, infinities and NaN. Its reference is double-precision log2 rounded to half, with the test's rtol=1e-3 and atol=1e-6. No custom LOG2 handler was present. This demonstrates an accuracy issue on this path, but does not yet locate it in the polynomial or our primitives.

Its lowered program also contains AND, which still excludes it from batching here. AND is already an elementwise NPU operation; add it to the same allowlist before retrying the full test:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE, Ops.MAX,
+                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE, Ops.MAX, Ops.AND,
                  *GroupOp.Comparison, Ops.SHL, Ops.SHR, Ops.MULACC, Ops.CUSTOM, Ops.SINK, Ops.NOOP, Ops.AFTER}
```

The full test still reaches the 30-second limit after allowing AND. Keep that run unresolved; the small probe provides the accuracy evidence.

Check the arithmetic feeding the polynomial before replacing it. The CPU's forced decomposition returns -0.2259521484375 for the same input, while this NPU path returns -0.225830078125. Tracing just the two FDIV calls identifies a different rounded quotient:

| Operation                       | NPU half result | Rounded reference      |
| ------------------------------- | --------------: | ---------------------: |
| -0.14501953125 / 1.85546875       |       -0.078125 | -0.07818603515625       |
| 1 / 0.85498046875                |   1.169921875   |  1.169921875            |

The first division forms the reduced variable used by the native LOG2 polynomial. Its result is already one half ULP away before the polynomial runs. The second division supplies the reciprocal used for special-value handling and matches in this probe. Both references were calculated in double precision and rounded to half; they were checks, not values supplied to the NPU.

So the next native-path fix is FDIV precision. Changing LOG2's polynomial would otherwise hide a measured input error. The direct LOG2 candidate below remains a separate approach; its success would not establish that the shared FDIV problem is fixed.

Before changing the polynomial, try the FP32 division helper we already built:

```text
half a, b → NPU CAST to FP32 → FP32 FDIV → NPU CAST to half
```

Every half input widens exactly to FP32. A 597-pair probe compared the final half result with double-precision division rounded to half: 512 random raw pairs, 81 special-value pairs and four boundary pairs, including the LOG2 quotient. All non-NaN bits and NaN classifications matched. This is not exhaustive over all half pairs, but it tests the candidate before replacing the shared path.

The FP32 helper already handles signs, zeros, infinities and NaNs. Replace the old sign-only repair wrapper with those conversions:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_fdiv(self, a:list, b:list) -> list:
-    quotient = self.run_npu(Ops.FDIV, a, b)
-    result:list = []
-    base = self.dev.input_mem.dma_addr
-    for start in range(0, len(a), 8):
-      count, slot = min(8, len(a)-start), 0
-      def put(raw:bytes) -> int:
-        nonlocal slot
-        addr = base+64*slot
-        slot += 1
-        to_mv(self.dev.input_buf+addr-base, 32)[:] = raw+bytes(32-len(raw))
-        return addr
-      def const(x:int) -> int: return put(struct.pack("<i", x)*8)
-      # Zero-extend raw FP16 words into INT32 lanes; do not numerically convert floats.
-      lhs, rhs, out = (put(b"".join(bytes(raw16(x, dtypes.half))+bytes(2) for x in xs[start:start+8]))
-                       for xs in (a, b, quotient))
-      half_sign, threshold = const(0x4000), const(0x7fff)
-      def calc(algo:int|None, x:int, y:int=threshold, **kw) -> int:
-        dst = put(bytes(32))
-        self.mulacc_stage(algo, x, y, dst, precision=4, output=4, **kw)
-        return dst
-      def signbit(x:int) -> int: return calc(1, threshold, x, binary=True)
-      def signword(x:int) -> int:
-        # EW MUL's operand is signed INT16: build +32768 without multiplying by 0x8000.
-        part = calc(0, x, half_sign, mul=True)
-        return calc(2, part, part)
-      # XOR of 0/1 signs is ABS(sa-sb). Replace the quotient sign, including signed zero.
-      lhs_sign, rhs_sign = signbit(lhs), signbit(rhs)
-      desired = calc(5, calc(4, lhs_sign, rhs_sign))
-      magnitude = calc(4, out, signword(signbit(out)))
-      corrected = calc(2, magnitude, signword(desired))
-      lhs_mag = calc(4, lhs, signword(lhs_sign))
-      rhs_mag = calc(4, rhs, signword(rhs_sign))
-      lower, upper = calc(1, lhs_mag, rhs_mag), calc(0, lhs_mag, rhs_mag)
-      zero, one, infinity = const(0), const(1), const(0x7c00)
-      both_zero = calc(4, one, calc(1, zero, upper, binary=True))
-      both_nonfinite = calc(4, one, calc(1, lower, infinity, binary=True))
-      has_nan = calc(1, infinity, upper, binary=True)
-      invalid = calc(0, both_zero, calc(0, both_nonfinite, has_nan))
-      # Integer selection supplies a canonical NaN for 0/0, inf/inf or a NaN operand.
-      corrected = calc(2, corrected, calc(0, invalid, calc(4, const(0x7e00), corrected), mul=True))
-      raw = bytes(to_mv(self.dev.input_buf+corrected-base, 32))
-      result.extend(typed_view(raw[i*4:i*4+2], dtypes.half) for i in range(count))
-    return result
+  def run_fdiv(self, a:list, b:list) -> list:
+    lhs, rhs = (self.run_cast(values, dtypes.half, dtypes.float) for values in (a, b))
+    return self.run_cast(self.run_float_binary(Ops.FDIV, lhs, rhs), dtypes.float, dtypes.half)
```

This uses more tasks than the native half divider. Keep the performance limitation visible; the change addresses the measured quotient error, not speed.

Verify the replacement at the reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div
# 30-second command limit, exit 124. No test summary.
```

The 17-value native LOG2 probe now passes rtol=1e-3, atol=1e-6. At input 0.85498046875 it returns -0.2259521484375, matching the CPU forced-decomposition probe; the rounded reference is -0.22607421875. The measured FDIV error is fixed, but this does not establish bit-exact LOG2 or a full test_log2 pass.

There is a performance cost: even the basic full division test reaches the limit with this general FP32 helper. Keep that regression visible. A next optimization can use the fact that nonzero finite half inputs widen to normal FP32 values, rather than running the generic FP32 subnormal-normalization steps. That must be tested before applying it.

For nonzero finite half inputs, the smallest magnitude is 2^-24 and the largest is below 2^16. Their quotient stays between 2^-40 and 2^40, well inside normal FP32. So this path does not need FP32 subnormal normalization or gradual underflow. Keep both for ordinary FP32 inputs; the final half CAST still handles half underflow.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_float_binary(self, op:Ops, a:list, b:list) -> list:
+  def run_float_binary(self, op:Ops, a:list, b:list, half_inputs:bool=False) -> list:
     assert op in (Ops.MUL, Ops.FDIV) and len(a) == len(b)
+    assert not half_inputs or op is Ops.FDIV
@@
-        # Normalize subnormals without discarding their low significand bits.
-        for n in (16, 8, 4, 2, 1):
-          take = lt(mantissa, const(2**(24-n)))
-          factor = add(one, mul(take, const(2**min(n, 8)-1)))
-          mantissa = mul(mantissa, factor)
-          if n == 16: mantissa = mul(mantissa, factor)
-          exponent = sub(exponent, mul(take, const(n)))
+        # Widened nonzero finite half inputs are normal FP32 values.
+        if not half_inputs:
+          for n in (16, 8, 4, 2, 1):
+            take = lt(mantissa, const(2**(24-n)))
+            factor = add(one, mul(take, const(2**min(n, 8)-1)))
+            mantissa = mul(mantissa, factor)
+            if n == 16: mantissa = mul(mantissa, factor)
+            exponent = sub(exponent, mul(take, const(n)))
@@
-      distance = calc(1, maximum(sub(one, exponent), zero), const(31))
-      # Variable right shift with sticky bits, including gradual underflow.
-      for n in (16, 8, 4, 2, 1):
-        take = sub(one, lt(distance, const(n)))
-        shifted = floor_shift(extended, n)
-        restored = mul(const(65536), shifted) if n == 16 else mul(shifted, const(2**n))
-        extended = select(take, jam(shifted, sub(extended, restored)), extended)
-        distance = sub(distance, mul(take, const(n)))
+      # Finite nonzero half quotients stay normal in FP32; final CAST handles half underflow.
+      if not half_inputs:
+        distance = calc(1, maximum(sub(one, exponent), zero), const(31))
+        for n in (16, 8, 4, 2, 1):
+          take = sub(one, lt(distance, const(n)))
+          shifted = floor_shift(extended, n)
+          restored = mul(const(65536), shifted) if n == 16 else mul(shifted, const(2**n))
+          extended = select(take, jam(shifted, sub(extended, restored)), extended)
+          distance = sub(distance, mul(take, const(n)))
@@
-    return self.run_cast(self.run_float_binary(Ops.FDIV, lhs, rhs), dtypes.float, dtypes.half)
+    return self.run_cast(self.run_float_binary(Ops.FDIV, lhs, rhs, half_inputs=True), dtypes.float, dtypes.half)
```

Run the unchanged division test again at this reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div

Ran 1 test in 17.661s
OK
```

The process now exits within 30 seconds. The current-runtime half rounding, division specials and FP32 division checks also passed: 3 tests in 4.890s. The generic FP32 path still includes its subnormal handling.

Check the related variants here too. A combined test_scalar_div/test_div_naninf command printed ok for test_scalar_div, then reached the 30-second limit inside test_div_naninf. That is only a scalar-variant pass, not a pass for the combined command.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_naninf
# 30-second command limit, exit 124. No test summary.
```

The separate special-value Tensor test also times out. The small special-value helper check passed, but this larger variant remains unresolved. test_div_rounding_mode and test_div_int are separate coverage too; the basic division pass does not replace them.

The bounded DEBUG rerun progresses through five kernels before timing out. We still calculate 27 quotient bits for every half division, although half has only 11 significand bits. Try 14 bits: 11 plus guard, round and sticky. The existing jam step sets the low bit when any discarded remainder is nonzero. This round-to-odd intermediate preserves which side of a half rounding boundary the quotient lies on, including half subnormal boundaries.

Multiply that 14-bit intermediate by 8192 (=2^13) to align it with the existing 27-bit encoding path. This is an NPU integer MUL, not a Python quotient calculation. Ordinary FP32 division keeps all 27 bits.

```diff
 class RockchipProgram(Program['RockchipDevice']):
   def run_float_binary(self, op:Ops, a:list, b:list, half_inputs:bool=False) -> list:
@@
-        # Binary long division: 24 significand bits and three rounding bits.
-        for _ in range(27):
+        # Half needs 11 significand bits plus guard/round/sticky; FP32 needs 24 plus three.
+        for _ in range(14 if half_inputs else 27):
@@
       extended = jam(extended, lost)
+      # Align the round-to-odd half intermediate with the existing FP32 encoding path.
+      if half_inputs: extended = mul(extended, const(8192))
```

The three current-runtime helper regressions passed in 5.467s. Now rerun the full special-value variant at this reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_naninf

Ran 1 test in 18.297s
OK
```

This command exits normally within 30 seconds; no cases or comparisons were removed.

A combined test_div/test_scalar_div rerun printed ok for test_div, then exceeded 30 seconds during test_scalar_div. Run these larger methods separately so the command budget is not shared; the interrupted combined command is not a two-test pass.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scalar_div

Ran 1 test in 13.166s
OK
```

The integer-input variant passes, but the rounding-mode variant still times out at this checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_int

Ran 1 test in 3.597s
OK

$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode
# 30-second command limit, exit 124. No test summary.
```

The batching allowlist omits integer division and remainder. run_integer already handles eight independent lanes, so check whether dispatch is feeding it only one:

At the reconstructed checkpoint the values match, but the batch-size assertion fails:

```text
AssertionError: Lists differ: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] != [8, 8, 1]
Ran 1 test in 4.566s
FAILED (failures=1)
```

Allow these independent operations to share a batch. Keep the existing local-memory and control-flow restrictions:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
     batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
                  Ops.BUFFER, Ops.RANGE, Ops.END,
+                 Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD,
```

Retest with the reconstructed code:

The quotient values are unchanged, and dispatch now sends 8, 8, 1 lanes. This checks batching, not the full rounding-mode test.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode

Ran 1 test in 17.762s
OK
```

The unchanged full rounding-mode test now finishes within 30 seconds too. It covers integer and float numerators, positive and negative denominators, truncation, floor rounding and rejection of an invalid rounding mode.

Retry native LOG2 with these dependency fixes before adding its custom path:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log2
# 30-second command limit, exit 124. No test summary.
```

The small accuracy probe passed earlier, but the full native test still does not finish within this limit. The following custom implementation is an alternative to measure, not evidence that the native decomposition is numerically wrong.

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
       x = calc(None, source, precision=2)
-      bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
-      integer = calc(7, calc(2, bounded, const(0.5, "f")))
-      fraction = calc(4, bounded, integer)
-      exponent = calc(None, integer, output=4)
+      if op is Ops.EXP2:
+        bounded = calc(1, calc(0, x, const(-25., "f")), const(16., "f"))
+        integer = calc(7, calc(2, bounded, const(0.5, "f")))
+        fraction = calc(4, bounded, integer)
+        exponent = calc(None, integer, output=4)
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
+        scaled = calc(2, calc(None, poly, bs_mul=weights), calc(None, exponent, precision=4))
       half = calc(None, scaled, output=2)
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
@@
-lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.EXP2, Ops.MULACC, Ops.OR, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.EXP2, Ops.LOG2, Ops.MULACC, Ops.OR, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
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

test_log2 passes with the code built here, before SIN. The all-encodings check also passed in an earlier run: `2 passed in 72.62s`. That longer sweep was not rerun for this revision.

## Ops.SIN

TOREVIEW1: The direct SIN candidate still needs a gate-only trial before implementation. test_sin is a known timeout, so this checkpoint is deferred, not verified by the later polynomial probes.

We already added UINT16 SHR, FP32 arithmetic and numeric CAST. Lets try the existing SIN decomposition with that code before adding SIN to ops_map:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin
# 30-second command limit, exit 124. No test summary.
```

This is the reconstructed pre-SIN checkpoint. The earlier FP32 MUL gate failure no longer applies after moving that prerequisite above this section. The full test times out, so it does not establish either an accuracy failure or a pass.

A smaller native-decomposition probe uses half inputs [-65504, -10000, -3.14, -1, -0.1, -0, 0, 0.1, 1, 3.14, 10000, 65504, inf, -inf, nan]. All 15 results pass rtol=1e-3, atol=1e-6 against NumPy sine evaluated in double precision and rounded to half. For example, sin(1) returns 0.841796875 versus 0.84130859375, and sin(65504) returns 0.97509765625 versus 0.9755859375. These are within that tolerance, not bit-exact matches. This small probe is not a full test_sin pass.

An earlier register probe tried mulacc_stage with mul=True directly: eight FP32 values in each input, precision=5/output=5 and mulacc_stage(0, lhs, rhs, out, mul=True). It did not produce FP32 multiplication:

| Lane | Input a             | Input b             | Expected FP32       | Observed             |
| ---- | ------------------- | ------------------- | ------------------- | -------------------- |
| 0    | 1.0000001192092896  | 1.0000001192092896  | 1.000000238418579   | 5.960465188081798e-8  |
| 2    | -2.5               | 4                   | -10                 | -0                   |

The input buffers were 64 bytes apart, with the 32-byte output another 64 bytes after them. This only tested that register layout. The earlier FP32 section now supplies multiplication through integer significands instead; this old probe is not a current missing primitive.

Changing ERDMA_DATA_SIZE to 1 and clearing EW_OP_CVT_BYPASS then timed out in DRM_IOCTL_RKNPU_SUBMIT (errno 110). The sweep stopped there; the other combinations were not tested. This is not evidence that all FP32 MUL modes are unsupported.

Below we measure an alternative using run_math from EXP2/LOG2. The checked 1500 branch also uses a polynomial in `_dpu_sin`, not a LUT, and clamps input to ±10000. That clamp changes larger finite half inputs. Our candidate uses range reduction instead; its tests below check that separate implementation, not tinygrad's native decomposition.

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
         half = calc(None, value, output=2)
         return alloc(read(half, 8)+bytes(8)+read(half+16, 8))
-      def polynomial(weights:int, cs:tuple[float, ...]) -> int:
+      def word_weights(value:int) -> int:
+        raw = read(value)
+        return alloc(b"".join(raw[i:i+2] for i in range(0, 16, 4))+bytes(8)+
+                     b"".join(raw[i:i+2] for i in range(16, 32, 4)))
+      def polynomial(weights:int, cs:tuple[float, ...], squared:bool=False) -> int:
         value = const(cs[-1], "f")
         for c in reversed(cs[:-1]):
           value = calc(None, value, bs_mul=weights)
+          if squared: value = calc(None, value, bs_mul=weights)
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
       x = calc(None, source, precision=2)
-      if op is Ops.EXP2:
+      if op is Ops.SIN:
+        multiple = calc(7, calc(2, calc(None, const(2/math.pi, "f"), bs_mul=half_weights(x)), const(0.5, "f")))
+        q = calc(None, multiple, output=4)
+        # Split the integer multiple so the first two pi/2 products are exact FP32.
+        qhi = mul(calc(4, mul(q, const(2)), const(255), 4, 4, shift=9), const(256))
+        qlo = sub(q, qhi)
+        qweights = [half_weights(calc(None, part, precision=4)) for part in (qhi, qlo)]
+        reduced = x
+        for c in (1.5703125, 0.0004837512969970703125, math.pi/2-1.5703125-0.0004837512969970703125):
+          for w in qweights: reduced = calc(4, reduced, calc(None, const(c, "f"), bs_mul=w))
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
           for w in qweights: reduced = calc(4, reduced, calc(None, const(c, "f"), bs_mul=w))
+        weights = half_weights(reduced)
+        residual = calc(4, reduced, calc(None, const(1., "f"), bs_mul=weights))
+        sine = calc(None, polynomial(weights, (1., -1/6, 1/120, -1/5040, 1/362880, -1/39916800), True), bs_mul=weights)
+        cosine = polynomial(weights, (1., -1/2, 1/24, -1/720, 1/40320, -1/3628800), True)
+        # Keep the full FP32 reduction residual instead of rounding the angle to half.
+        sin_r = calc(2, sine, calc(None, residual, bs_mul=half_weights(cosine)))
+        cos_r = calc(4, cosine, calc(None, residual, bs_mul=half_weights(sine)))
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
         sin_r = calc(2, sine, calc(None, residual, bs_mul=half_weights(cosine)))
         cos_r = calc(4, cosine, calc(None, residual, bs_mul=half_weights(sine)))
+        odd = sub(q, mul(calc(4, mul(q, const(2)), const(1), 4, 4, shift=2), const(2)))
+        quadrant = sub(q, mul(calc(4, mul(q, const(2)), const(3), 4, 4, shift=3), const(4)))
+        sin_weight = word_weights(mul(const(0x3c00), sub(const(1), odd)))
+        cos_weight = word_weights(mul(const(0x3c00), odd))
+        scaled = calc(2, calc(None, sin_r, bs_mul=sin_weight), calc(None, cos_r, bs_mul=cos_weight))
+        sign_weight = word_weights(add(const(0x3c00), mul(const(32768), lt(const(1), quadrant))))
+        scaled = calc(None, scaled, bs_mul=sign_weight)
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
-        scaled = calc(2, calc(None, poly, bs_mul=weights), calc(None, exponent, precision=4))
+      if op is not Ops.SIN:
+        # The EXP2/LOG2 reduced fraction is exactly representable in half.
+        weights = half_weights(fraction)
+        poly = polynomial(weights, coefficients)
+        if op is Ops.EXP2:
+          scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
+        else:
+          scaled = calc(2, calc(None, poly, bs_mul=weights), calc(None, exponent, precision=4))
       half = calc(None, scaled, output=2)
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
@@
-lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.EXP2, Ops.LOG2, Ops.MULACC, Ops.OR, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
+lowered_ops = {Ops.AND, Ops.CDIV, Ops.CMOD, Ops.CMPEQ, Ops.CMPLT, Ops.CMPNE, Ops.EXP2, Ops.LOG2, Ops.MULACC, Ops.OR, Ops.SIN, Ops.SQRT, Ops.TRUNC, Ops.WHERE, Ops.XOR}
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

This is a historical run through the SIN step, not a fresh bounded result. It exceeds our current 30-second limit; do not extend that limit to reproduce it. A new bounded run is still needed for this custom path.

The earlier helper run through this SIN step passed all 65,536 SIN inputs and test_sin: `2 passed in 117.79s`. That exhaustive run was not repeated in this revision.

The earlier EXP2/LOG2 run passed both exhaustive checks and test_exp2/test_log2, but test_exp and test_log failed: **2 failed, 4 passed in 138.55s**. In that older order test_exp reached unsupported FP32 MUL; that support is now introduced above. test_log lost accuracy when the rounded half LOG2 result was multiplied by the half ln(2) constant. Both composed tests need fresh checks at the reordered checkpoint. Passing the primitive tests does not mean these composed functions pass.

The earlier combined exhaustive run is historical evidence, not another command to run here.

Each exhaustive check covers all 65,536 FP16 bit patterns. Non-NaN outputs match the reference bits, including signed zeros and infinities; NaN classification matches too. Arithmetic, range reduction and rounding corrections run on the NPU. Python only sets constants, submits tasks and copies storage layouts.

POW remains, followed by the full test_ops.py sweep. General FP32 arithmetic is now introduced above; the composed log accuracy issue still needs its own check.

The fresh bounded custom-SIN run also times out:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin
# 30-second command limit, exit 124. No test summary.
```

run_math handles eight lanes, but EXP2, LOG2 and SIN are missing from the batching allowlist. Check the values and dispatch sizes before changing it:

All three value checks pass, but each batch-size assertion fails:

```text
AssertionError: Lists differ: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] != [8, 8, 1]
Ran 1 test in 0.546s
FAILED (failures=3)
```

Enable batching for those three independent operations, without changing their arithmetic:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
     batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
                  Ops.BUFFER, Ops.RANGE, Ops.END,
                  Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD,
+                 Ops.EXP2, Ops.LOG2, Ops.SIN,
```

Retest at this reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sin

Ran 1 test in 4.672s
OK
```

The custom SIN test now exits within 30 seconds. The small test checks exact half values and an 8, 8, 1 tail for all three math helpers; no exhaustive run was repeated.

Now check the composed functions at the same checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_exp

NotImplementedError: ROCKCHIP NPU does not support Ops.EXP2 with dtypes.float
Ran 1 test in 0.258s
FAILED (errors=1)

$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log

Not equal to tolerance rtol=0.001, atol=1e-06
Mismatched elements: 8 / 2925 (0.274%)
[1, 8]: -0.86962890625 (ACTUAL), -0.86865234375 (DESIRED)
Max relative difference among violations: 0.001294
Ran 1 test in 2.453s
FAILED (errors=1)
```

FP32 MUL is no longer the missing step in exp: the next UOp is FP32 EXP2. log has a different problem: its composed result fails tolerance even though the half LOG2 primitive passes. We need to inspect its intermediate rounding before changing the formula or tolerance. These failures remain open here; the later POW work also needs FP32 math intermediates, but its code is not available at this checkpoint yet.

The private POW helper later accepts an FP32 exponent but returns half. Using that as FP32 EXP2 would round too early. tinygrad already has xexp2 for FP32, so try its native decomposition first. Match only FP32 here; keep our half EXP2 helper:

```diff
 from tinygrad.renderer import Renderer
+from tinygrad.codegen.decomp.transcendental import xexp2
@@
 class RockchipRenderer(Renderer):
@@
   extra_matcher = PatternMatcher([
+    # Keep FP32 EXP2 in FP32 through tinygrad's native decomposition.
+    (UPat(Ops.EXP2, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xexp2(x)),
```

The full test_exp now runs past the FP32 EXP2 gate, but reaches the 30-second limit without a summary. A saved small FP32 probe covered subnormal outputs, overflow and special values.

This reconstructed-checkpoint test keeps FP32 output and passes its 17 inputs at rtol=2e-7, atol=0. It is not exhaustive FP32 coverage. Full test_exp remains a timeout, and test_log still has the measured composed-rounding failure above.

Before changing Rockchip LOG2, run the same log test on CPU:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=CPU python test/backend/test_ops.py TestOps.test_log

Mismatched elements: 28 / 2925 (0.957%)
[1, 8]: -0.86962890625 (ACTUAL), -0.86865234375 (DESIRED)
Ran 1 test in 0.185s
FAILED (errors=1)
```

CPU has additional mismatches, but shares this first one. Tensor.log is log2(x)*ln(2), with both intermediates in half here. For that input:

| Step                 | Value             |
|----------------------|------------------:|
| x                    | 0.41943359375     |
| half(log2(x))        | -1.25390625       |
| half(ln(2))          | 0.693359375       |
| half product        | -0.86962890625    |
| half(log(x))        | -0.86865234375    |

The NPU result matches the half-intermediate calculation. Changing its LOG2 result to compensate would make that primitive less accurate. Tensor.exp already widens its intermediate calculation; applying the same idea to log and log10 is a shared-method change, not an NPU register fix.

A CPU diagnostic evaluated cast-half(log2(cast-float(x))*ln(2)) on the same 2925 inputs. All values passed rtol=1e-3, atol=1e-6 against the double-precision reference rounded to half. The corresponding log10 expression passed too. These are candidate-expression checks, not passes of the unchanged test_log/test_log10 methods. The shared methods have not been edited here.

## Ops.POW

The following long result is historical. Fresh reconstructed pre-POW runs of test_pow and test_pow_full each reached the 30-second limit (exit 124), without a test summary. They are not new passes or reproduced accuracy failures.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow

test_pow (__main__.TestOps.test_pow) ... ok

Ran 1 test in 108.073s

OK
```
That earlier test_pow run passed using tinygrad's decomposition, before any POW-specific changes. It checks fixed exponents; test_pow_full uses tensor exponents. The Python CAST fallback still exists, so a pass alone does not prove every conversion ran on the NPU.

A fresh 17-value probe uses half bases linspace(0.25, 2, 17) and exponents linspace(-1, 2, 17). It passes rtol=1e-3, atol=1e-6 against double-precision power rounded to half. Its UOps include LOG2, MUL, EXP2, comparisons and WHERE; all are already allowed in batches. This does not replace either full test or establish exact rounding.

The storage check includes both signed zeros, infinities and NaN payloads in either branch. A separate unittest rerun of test_where_special_bits passed in 0.219s using the reconstructed pre-CAST stage. The earlier positive-power probe gave [8, 0.25, 1.732, 2]. Negative bases and -inf with a fractional exponent also gave the expected values in that small probe. The NaN-exponent CAST error is still open.

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_full

Not equal to tolerance rtol=0.001, atol=1e-06
Mismatched elements: 17 / 2925 (0.581%)
Max relative difference among violations: 0.002638

Ran 1 test in 107.463s
FAILED (errors=1)
```

This is the historical failure before the wider POW changes below; the fresh bounded run timed out before an assertion. Unlike test_pow's fixed exponents, test_pow_full supplies a tensor of exponents. The recorded failure was numerical rather than the earlier NaN-selection problem. The composed path rounds LOG2 and MUL to half before EXP2; next we try retaining wider intermediates, without changing the tolerance.

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
-      x = calc(None, source, precision=2)
+      bits = source if input_dtype == dtypes.float else alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = source if input_dtype == dtypes.float else calc(None, source, precision=2)
       if op is Ops.SIN:
         multiple = calc(7, calc(2, calc(None, const(2/math.pi, "f"), bs_mul=half_weights(x)), const(0.5, "f")))
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
+            residual = calc(4, fraction, calc(None, const(1., "f"), bs_mul=weights))
+            derivative = polynomial(weights, tuple(k*coefficients[k] for k in range(1, len(coefficients))))
+            poly = calc(2, poly, calc(None, residual, bs_mul=half_weights(derivative)))
           scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
         else:
           scaled = calc(2, calc(None, poly, bs_mul=weights), calc(None, exponent, precision=4))
-      half = calc(None, scaled, output=2)
-      packed = read(half, 8)+read(half+16, 8)
-      output_bits = alloc(b"".join(packed[i:i+2]+bytes(2) for i in range(0, 16, 2)))
-      negative = lt(const(32767), bits)
-      magnitude = sub(bits, mul(const(32768), negative))
+      if dtype == dtypes.float: output_bits = scaled
+      else:
+        half = calc(None, scaled, output=2)
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
+      self.mulacc_stage(None, base, base+64, base+128, bs_mul=base+64)
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

This rerun used the code built through the diffs above and passed all 4,096 magnitude pairs. It checks run_exp2_mul_log2 directly; it does not replace the full Tensor POW test.

The earlier focused checks also passed all 65,536 half bit patterns for EQ/NE/LT with the other operand cycling through eight special values, and all 63,488 finite half values cast to INT32. That comparison check is not every possible pair of half values.

In the earlier order, reconstructing run_math, run_exp2_mul_log2, run_cast and the then-named run_half_compare gave the same helpers as that runtime. Their focused checks, including special-value WHERE, gave `4 passed in 74.35s`. The comparison helper is now introduced as run_float_compare earlier in this draft; that historical run is not a rerun of the new order.

The saved combined run gave **7 passed, 1 failed in 557.49s**. `test_pow_full`, `test_pow`, `test_pow_neg_inf_frac_exponent` and `test_pow_zero_exponent` passed. `test_pow_const` failed at `x ** 8.0`: 617 / 2925 values exceeded the tolerance. The same test on tinygrad CPU, with the same HALF/NOOPT settings, failed at the same expression with the same 617 mismatches. This expression becomes repeated FP16 MULs, so the wider LOG2/EXP2 path does not apply.

The earlier full Tensor rerun after widening the intermediates was:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_full

test_pow_full (__main__.TestOps.test_pow_full) ... ok
Ran 1 test in 293.597s
OK
```

This historical run used the backend built from these diffs, not the later runtime. It followed a 180-second timeout and finished within a 300-second limit. Those longer limits are not used now. A fresh reconstructed run after the batching changes reached the current 30-second limit without a summary (exit 124). The inputs and tolerance remain unchanged; the new run is unresolved, not a fresh pass.

The fresh reconstructed test_pow_const run still fails at x**8.0:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_const

Mismatched elements: 617 / 2925 (21.1%)
Max relative difference among violations: 0.002876
Ran 1 test in 6.591s
FAILED (errors=1)
```

POW is therefore incomplete regardless of the passing helper and special-value tests. We leave shared core unchanged and do not relax the tolerance.

The fresh reconstructed code passes these shorter special-value checks:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_pow_neg_inf_frac_exponent TestOps.test_pow_zero_exponent TestOps.test_pow_zero_tensor

Ran 3 tests in 0.593s
OK
```

This is not a claim of every POW edge case. For example, `Tensor([1.0]) ** Tensor([nan])` returns NaN on tinygrad CPU too. A Python scalar NaN exponent instead fails in the common `simplify_pow` rewrite before reaching either backend.

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

| Group             | Implementations to verify              |
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
| **Variant tests verified** | **13 / 30** |

The rows above are an implementation inventory. The completed count includes CMPEQ, CMPNE, CMPLT, WHERE, AND, OR, XOR, ADD, SUB, NEG, TRUNC, SHL and SHR: their applicable variant methods passed, with the configuration matrices and boundary checks recorded below. This is test coverage, not a proof for every possible input or unsupported dtype. The other 17 ops remain unverified; a basic or helper-test pass is not enough.

FLOORDIV, FLOORMOD, THREEFRY and POW use tinygrad's existing decompositions. The earlier sections now dispatch FP32 ADD/SUB/NEG/MUL/FDIV and raw-storage WHERE; other FP32 paths still need their own implementation and tests. The later shift audit replaces public narrow SHL's saturating MUL path with convolution and accepts per-lane counts. Counts outside `0..bits-1` remain unsupported. Some CAST combinations still use the Python fallback. BITCAST only reinterprets storage; it does not calculate new values.

The full sweep uses forward-only FP16 with NOOPT=1. It does not establish backward, default-FP32 or optimized-kernel coverage.

### Variant audit: rounding family

These are separate current-runtime runs, not passes at the early half-only TRUNC checkpoint. Each used NOOPT=1 DEV=ROCKCHIP, default FP32, no FORWARD_ONLY flag, and a 30-second command limit:

| Test                             | Result | Time   |
|----------------------------------|--------|--------|
| test_trunc                       | Pass   | 0.303s |
| test_floor                       | Pass   | 1.124s |
| test_ceil                        | Pass   | 1.144s |
| test_round                       | Pass   | 4.430s |
| test_round_quantization_gradient | Pass   | 0.463s |

Now remove NOOPT=1 and run the same methods separately, still with default FP32 and the 30-second limit:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc
Ran 1 test in 3.535s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_floor
Ran 1 test in 5.525s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_ceil
Ran 1 test in 5.857s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_round
Ran 1 test in 14.264s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_round_quantization_gradient
Ran 1 test in 0.521s
OK
```

The first four methods explicitly request forward-only internally; removing the environment flag does not give them backward coverage. The fifth checks its composed gradient. The early TRUNC section records the corresponding half runs and its FP32 gate failure. These five methods now pass with and without optimization in FP32.

Repeat with DEFAULT_FLOAT=HALF and optimization enabled. The combined command printed ok for trunc, floor, ceil and round, then hit the 30-second limit during test_round_quantization_gradient. Run that last method separately:

```bash
$ DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_round_quantization_gradient
Ran 1 test in 0.444s
OK
```

These are current-runtime results. The combined HALF timeout is not a five-test pass, and these runs do not cover every dtype. The remaining NOOPT HALF configuration is recorded at the end of this audit below.

### Variant audit: ADD

The basic ADD pass does not cover a third input or broadcasting. Run each existing variant separately on the current runtime with DEV=ROCKCHIP, default FP32, optimization enabled and no FORWARD_ONLY flag:

| Test                   | Result | Time    |
|------------------------|--------|---------|
| test_tiny_add          | Pass   | 0.233s  |
| test_add               | Pass   | 9.457s  |
| test_add3              | Pass   | 7.244s  |
| test_broadcasted_add   | Pass   | 19.044s |
| test_broadcasted_add_2 | Pass   | 7.839s  |

Each command used the same 30-second limit. test_tiny_add is forward-only internally; the other methods also compare gradients where their inputs require them. These results cover this FP32 configuration, not every dtype, and are not passes at the blog's initial half-only ADD step.

With DEFAULT_FLOAT=HALF, a combined run printed ok for test_tiny_add, test_add and test_add3, then reached the 30-second limit during test_broadcasted_add. That shared budget does not show whether the broadcast method can finish on its own. Run both broadcast variants separately:

```bash
$ DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_broadcasted_add
Ran 1 test in 19.711s
OK

$ DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_broadcasted_add_2
Ran 1 test in 7.714s
OK
```

Both finish with gradients enabled.

### Variant audit: scalar division

The current-runtime FP32 test_scalar_div still hits the 30-second limit with NOOPT=1 FORWARD_ONLY=1. Before changing FDIV, inspect the calls from eight FP32 inputs [.25, .5, 1, 2, -.25, -.5, -1, -2]. Count run_float_binary calls without changing their inputs or results:

| Expression | Observed path                         | Time in run_float_binary |
|------------|---------------------------------------|--------------------------|
| x / 255    | One eight-lane MUL call                | 0.0226s                  |
| 1 / x      | One eight-lane FDIV call               | 0.0346s                  |
| x / 2      | No run_float_binary call              | —                        |
| 2 / x      | One eight-lane FDIV call               | 0.0347s                  |

So this is not only an FDIV problem. tinygrad already turns x/255 into multiplication by its reciprocal. That FP32 coefficient is not exactly representable as half, so run_float_scale falls back to the integer-significand MUL path. x/2 uses the existing exact-half coefficient shortcut. The reciprocal expressions still need FDIV. These eight-lane observations identify the paths; they are not a full scalar-division test pass. Keep the full test unresolved rather than narrowing the coefficient or reducing division precision just to make it faster.

Both paths normalize their operands before arithmetic. For a compile-time coefficient, its sign, exponent and significand are already known. Prepare those constants on the host; keep tensor normalization, multiplication/division, rounding and special-value selection on the NPU. This does not replace division with an approximate reciprocal.

```diff
   def run_float_scale(self, a:list, scale:float) -> list:
@@
-      return self.run_float_binary(Ops.MUL, a, [scale]*len(a))
+      return self.run_float_binary(Ops.MUL, a, [scale]*len(a), constant=(1, scale))
@@
-  def run_float_binary(self, op:Ops, a:list, b:list, half_inputs:bool=False) -> list:
+  def run_float_binary(self, op:Ops, a:list, b:list, half_inputs:bool=False, constant:tuple[int,float]|None=None) -> list:
@@
-      for values in (a, b):
+      for index,values in enumerate((a, b)):
+        if constant is not None and index == constant[0]:
+          # Decode only a compile-time coefficient, never tensor values.
+          bits = int.from_bytes(struct.pack("<f", constant[1]), "little")
+          exp = (bits >> 23) & 255
+          mant = (bits & 0x7fffff) + (8388608 if exp else 0)
+          exp = max(exp, 1)
+          if not half_inputs:
+            for n in (16, 8, 4, 2, 1):
+              if mant < 2**(24-n): mant, exp = mant * 2**n, exp-n
+          operands.append(tuple(const(x) for x in (mant, exp, bits >> 31, bits & 0x7fffffff)))
+          continue
         raw = [bytes(raw16(x, dtypes.float)) for x in values[start:start+8]]
```

Only enable this from a CONST UOp or a cast of one, not from tensor values that happen to match:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-                             (x.op is Ops.CAST and x.src[0].op is Ops.CONST)), None) if u.op is Ops.MUL else None
-            values[u] = self.run_float_scale(src_values[1-constant], scalar16(src_values[constant][0])) if constant is not None else \
-              self.run_float_binary(u.op, *src_values)
+                             (x.op is Ops.CAST and x.src[0].op is Ops.CONST)), None)
+            if u.op is Ops.MUL and constant is not None:
+              values[u] = self.run_float_scale(src_values[1-constant], scalar16(src_values[constant][0]))
+            else:
+              coefficient = None if constant is None else (constant, scalar16(src_values[constant][0]))
+              values[u] = self.run_float_binary(u.op, src_values[0], src_values[1], constant=coefficient)
```

The unchanged full FP32 test_scalar_div still reached the 30-second limit after this change. Constant setup now avoids the repeated NPU normalization work, but that is not enough to finish this variant within the limit. Keep it unresolved.

### Type-check the runtime

mypy reports three untyped result lists and reuse of raw as both a list of input words and output bytes. These are typing changes, not new NPU operations:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_float_trunc(self, a:list) -> list:
-    base, result = self.dev.input_mem.dma_addr, []
+  def run_float_trunc(self, a:list) -> list:
+    base = self.dev.input_mem.dma_addr
+    result:list = []
@@
-      return self.run_float_binary(Ops.MUL, a, [scale]*len(a), constant=(1, scale))
-    base, result = self.dev.input_mem.dma_addr, []
+      return self.run_float_binary(Ops.MUL, a, [scale]*len(a), constant=(1, scale))
+    base = self.dev.input_mem.dma_addr
+    result:list = []
@@
-    assert not half_inputs or op is Ops.FDIV
-    base, result = self.dev.input_mem.dma_addr, []
+    assert not half_inputs or op is Ops.FDIV
+    base = self.dev.input_mem.dma_addr
+    result:list = []
@@
-      raw = bytes(to_mv(self.dev.input_buf+bits-base, 32))
-      result.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
+      output = bytes(to_mv(self.dev.input_buf+bits-base, 32))
+      result.extend(typed_view(output[i*4:i*4+4], dtypes.float) for i in range(count))
```

### Comparison backward variants: FP32 math dependencies

Run the existing backward methods with default FP32 and optimization enabled:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_ne_backwards
NotImplementedError: ROCKCHIP NPU does not support Ops.LOG2 with dtypes.float
Ran 1 test in 2.682s
FAILED (errors=1)

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_lt_backwards
NotImplementedError: ROCKCHIP NPU does not support Ops.LOG2 with dtypes.float
Ran 1 test in 2.680s
FAILED (errors=1)
```

Both fail while realizing Tensor.randn, before comparing gradients. randn_like uses the Box–Muller expression cos(2*pi*u) * sqrt(-2*log(1-v)) with FP32 intermediates. Our half LOG2 path cannot handle that dtype. Try tinygrad's existing xlog2 decomposition first, just as we did for FP32 EXP2:

```diff
-from tinygrad.codegen.decomp.transcendental import xexp2
+from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
@@
     (UPat(Ops.EXP2, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xexp2(x)),
+    # Reuse native FP32 LOG2 without narrowing its input or intermediates.
+    (UPat(Ops.LOG2, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xlog2(x)),
```

A saved small probe covered subnormals, zero, negative inputs and nonfinite values.

Retest the original backward method:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_ne_backwards
NotImplementedError: ROCKCHIP NPU does not support Ops.SIN with dtypes.float
Ran 1 test in 3.231s
FAILED (errors=1)
```

LOG2 no longer stops this run. Cosine lowers through SIN, so the next missing dependency is FP32 SIN; this is still not a comparison-backward pass.

Try the same approach for SIN, using xsin's general range reduction rather than its small-input fast mode:

```diff
-from tinygrad.codegen.decomp.transcendental import xexp2, xlog2
+from tinygrad.codegen.decomp.transcendental import xexp2, xlog2, xsin
@@
     (UPat(Ops.LOG2, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xlog2(x)),
+    # Use the general native range reduction, not the small-input-only fast mode.
+    (UPat(Ops.SIN, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xsin(x)),
```

The backward test now reaches the next missing operation:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_ne_backwards
NotImplementedError: ROCKCHIP NPU does not support Ops.SQRT with dtypes.float
Ran 1 test in 4.987s
FAILED (errors=1)
```

That does not yet establish general SIN support. A separate 16-input SIN probe, including ±1e20 and nonfinite inputs, stops at:

```text
NotImplementedError: ROCKCHIP 64-bit shift requires one uniform count in 0..63
```

Native Payne–Hanek reduction shifts 64-bit words by counts derived from each input's exponent. Our helper only accepts one count per call. Keep the uniform path, but dispatch varying counts as separate one-lane NPU calls; Python routes the lanes, not the shifted bits:

```diff
   def run_u64_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
+    assert len(a) == len(b)
     counts = list(map(scalar16, b))
-    if not counts or not all_same(counts) or not 0 <= counts[0] < 64:
-      raise NotImplementedError("ROCKCHIP 64-bit shift requires one uniform count in 0..63")
+    if not counts or any(not 0 <= count < 64 for count in counts):
+      raise NotImplementedError("ROCKCHIP 64-bit shift requires counts in 0..63")
+    # A task has one shift configuration; route varying counts through separate NPU tasks.
+    if not all_same(counts): return [self.run_u64_shift(op, [x], [count], dtype)[0] for x,count in zip(a, counts)]
```

The same 16-input SIN probe now passes at rtol=2e-7, atol=1e-7.

The existing uniform 64-bit shift checks still pass. This removes the varying-count restriction for 64-bit shifts in 0..63; it does not change the narrow-shift paths. FP32 SQRT remains the next blocker in the backward test.

FP32 SQRT has an existing tinygrad decomposition too: get_transcendental_patterns lowers it to xpow(x, 0.5). Try that before adding a new hardware algorithm:

```diff
-from tinygrad.codegen.decomp.transcendental import xexp2, xlog2, xsin
+from tinygrad.codegen.decomp.transcendental import xexp2, xlog2, xsin, xpow
@@
     (UPat(Ops.SIN, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xsin(x)),
+    # Try tinygrad's native SQRT decomposition with FP32 intermediates.
+    (UPat(Ops.SQRT, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xpow(x, x.const_like(0.5))),
```

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_ne_backwards
Ran 1 test in 17.007s
OK
```

The backward method passes, but its random positive SQRT inputs do not test special values. A separate 15-input probe found:

| Input | Native decomposition | SQRT reference |
|-------|----------------------|----------------|
| -0    | +0                   | -0             |
| -inf  | +inf                 | NaN            |
| 10000 | 99.99999237060547    | 100            |

xpow has power semantics: it deliberately treats negative infinity differently from a finite negative base. SQRT needs NaN for any negative input, including -inf. For either zero, return the original input to retain its sign. These selections stay in UOps and run on the NPU:

```diff
 class RockchipRenderer(Renderer):
@@
-    # Try tinygrad's native SQRT decomposition with FP32 intermediates.
-    (UPat(Ops.SQRT, dtypes.float, src=(UPat.var("x", dtypes.float),)), lambda x: xpow(x, x.const_like(0.5))),
+    # Native POW needs SQRT's negative-input rule and preservation of signed zero.
+    (UPat(Ops.SQRT, dtypes.float, src=(UPat.var("x", dtypes.float),)),
+     lambda x: (x < 0).where(x.const_like(math.nan), x.ne(0).where(xpow(x, x.const_like(0.5)), x))),
```

The finite approximation is not bit-exact: the 10000 case differs by one ULP. The probe checked zero signs separately.

This is a bounded special-value probe, not a complete SQRT audit or an exhaustive run.

Now both unchanged backward methods pass with the SQRT special-value selections in place:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_lt_backwards
Ran 1 test in 18.007s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_ne_backwards
Ran 1 test in 17.989s
OK
```

The same unguarded SQRT-to-xpow rule remains in tinygrad's shared decomposition. We leave core unchanged and apply these selections only in RockchipRenderer.

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_eq TestOps.test_cmp_lt TestOps.test_cmp_le TestOps.test_cmp_gt TestOps.test_cmp_ge
Ran 5 tests in 5.232s
OK
```

These unchanged methods cover float, integer and bool inputs, broadcasting, constants and infinities. Their specials list still excludes NaN; this is not a replacement for the raw-pattern comparison checks.

The reconstructed test_cmp_ne_backwards and test_cmp_lt_backwards also pass separately in 17.865s and 16.991s. This confirms the new dependencies are present in the diffs, not just in the working runtime. These runs use default FP32 with optimization enabled.

Check the other forward configurations on the current runtime. Each row runs the same five methods: test_cmp_eq, test_cmp_lt, test_cmp_le, test_cmp_gt and test_cmp_ge, without changing their inputs:

| Configuration                  | Result   | Time   |
|--------------------------------|----------|--------|
| NOOPT=1, default FP32           | 5 passed | 3.279s |
| DEFAULT_FLOAT=HALF              | 5 passed | 5.372s |
| NOOPT=1 DEFAULT_FLOAT=HALF       | 5 passed | 3.205s |

These existing methods omit NaN. Saved raw-pattern probes checked NaNs and zero signs separately; do not infer that coverage from the five Tensor methods.

That covers NaNs and both zero signs in the float helper. It does not establish every integer dtype's comparison path or the remaining backward configurations.

Finish the backward configuration matrix, running each method separately:

| Configuration            | test_cmp_ne_backwards | test_cmp_lt_backwards |
|--------------------------|-----------------------|-----------------------|
| NOOPT=1, default FP32     | Pass, 12.369s         | Pass, 14.816s         |
| DEFAULT_FLOAT=HALF       | Pass, 17.208s         | Pass, 17.663s         |
| NOOPT=1 DEFAULT_FLOAT=HALF | Pass, 12.432s       | Pass, 13.855s         |

CMPEQ, CMPNE and CMPLT now pass the applicable existing comparison variants: the five forward Tensor methods, two backward methods and seven direct UOp methods. The additional raw-float and integer-boundary checks pass too. Count these three ops in progress; do not count the other math operations merely because randn now works.

### Variant audit: WHERE

Run all four existing Tensor methods: test_where, test_where_permute, test_where_nan_cond and test_inf_where. Keep the original inputs and comparison logic:

| Configuration                  | Result   | Time             |
|--------------------------------|----------|------------------|
| Default FP32                   | 4 passed | 2.448s + 0.907s  |
| NOOPT=1, default FP32           | 4 passed | 1.282s           |
| DEFAULT_FLOAT=HALF             | 4 passed | 2.645s           |
| NOOPT=1 DEFAULT_FLOAT=HALF      | 4 passed | 1.222s           |

The first row used two commands: test_where alone, then the other three together. Each command stayed below 30 seconds. These methods cover broadcast operands, integer outputs, permuted results, NaN comparisons and selecting away infinity. They request forward-only internally or directly inspect the output; removing FORWARD_ONLY does not add gradient coverage.

All applicable existing WHERE variants pass, with no skipped case in this selection. Count WHERE alongside the three comparisons: 4/30 verified against the variant tests so far.

### Variant audit: AND, OR and XOR

Run test_and, test_or, test_xor, test_bitwise_not and test_int_or. The AND method includes bool inputs, a ten-input floating sum feeding comparisons, and UINT64 masking; do not replace it with only the small integer examples.

| Configuration                  | Result   | Time   |
|--------------------------------|----------|--------|
| Default FP32, first four methods | 4 passed | 7.080s |
| NOOPT=1, default FP32           | 5 passed | 5.568s |
| DEFAULT_FLOAT=HALF             | 5 passed | 6.713s |
| NOOPT=1 DEFAULT_FLOAT=HALF      | 5 passed | 4.610s |

The first command did not include test_int_or. Its result remains separate from that four-method run.

All applicable existing bitwise variants pass, with no skips in this selection. These are forward bitwise checks, not gradient tests. AND, OR and XOR bring the verified variant count to 7/30.

### Variant audit: SUB and NEG

The existing methods are test_sub, test_scalar_sub, test_scalar_rsub and test_neg. They include scalar shapes and gradients.

The combined optimized FP32 command printed ok for all three SUB methods, then reached its 30-second limit during NEG. Run NEG separately:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_neg
Ran 1 test in 11.332s
OK
```

Run the remaining configurations without sharing one time budget across all the slower optimized methods:

| Configuration             | Methods                         | Result   | Time    |
|---------------------------|---------------------------------|----------|---------|
| NOOPT=1, default FP32      | All four                        | 4 passed | 1.695s  |
| NOOPT=1 DEFAULT_FLOAT=HALF | All four                        | 4 passed | 1.276s  |
| DEFAULT_FLOAT=HALF        | test_sub                        | 1 passed | 10.956s |
| DEFAULT_FLOAT=HALF        | test_scalar_sub, test_scalar_rsub | 2 passed | 11.001s |
| DEFAULT_FLOAT=HALF        | test_neg                        | 1 passed | 11.140s |

Every method completed in each configuration; the interrupted combined command is not reported as a four-test pass. SUB and NEG bring the verified variant count to 9/30.

Finish ADD's remaining configurations. The optimized runs were recorded earlier; now run the same five methods with NOOPT=1:

```bash
$ NOOPT=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tiny_add TestOps.test_add TestOps.test_add3 TestOps.test_broadcasted_add TestOps.test_broadcasted_add_2
Ran 5 tests in 6.875s
OK

$ NOOPT=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tiny_add TestOps.test_add TestOps.test_add3 TestOps.test_broadcasted_add TestOps.test_broadcasted_add_2
Ran 5 tests in 6.059s
OK
```

These complete the existing ADD variant checks across our FP32/HALF, optimized/NOOPT matrix, plus direct float/int32/bool cases. ADD brings verified progress to 10/30.

Finish the rounding-family matrix on the current runtime:

```bash
$ NOOPT=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_trunc TestOps.test_floor TestOps.test_ceil TestOps.test_round TestOps.test_round_quantization_gradient
Ran 5 tests in 6.207s
OK

$ NOOPT=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode
Ran 1 test in 20.749s
OK
```

The earlier rows cover FP32 with and without optimization and optimized HALF. All five existing rounding-family methods now pass in those configurations, and the FP32 division-rounding integration check passes too. Count TRUNC as the eleventh verified op; this does not claim standalone TRUNC backward coverage that the existing tests explicitly omit, or complete FDIV coverage.

### FP32 MAX variants

Next run maximum with the default FP32 dtype, without changing its inputs:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum
NotImplementedError: ROCKCHIP NPU does not support Ops.MAX with dtypes.float
Ran 1 test in 0.194s
FAILED (errors=1)
```

Our FP32 TRUNC path already uses the EW MIN/MAX selectors. Can we reuse MAX directly? Probe selector 0 through `mulacc_stage` with FP32 input and output first:

| a      | b      | NPU MAX |
|--------|--------|---------|
| -1     | 2      | 2       |
| 2      | -1     | 2       |
| NaN    | 1      | NaN     |
| 1      | NaN    | NaN     |
| -0     | +0     | +0      |
| +0     | -0     | +0      |
| -inf   | +inf   | +inf    |
| +inf   | -inf   | +inf    |

This handles finite maxima and propagates NaN from either operand. Mixed signed zeros produce +0 in this probe; this is not a claim that it preserves an operand's zero sign. Add MAX to the existing FP32 helper and dispatch:

```diff
   def run_float_alu(self, op:Ops, a:list, b:list|None=None) -> list:
-    assert op in (Ops.ADD, Ops.SUB, Ops.NEG)
+    assert op in (Ops.ADD, Ops.SUB, Ops.NEG, Ops.MAX)
@@
       if op is Ops.NEG:
         self.mulacc_stage(6, base, base+64, base+128)
         output = 128
+      elif op is Ops.MAX:
+        self.mulacc_stage(0, base, base+64, base+128)
+        output = 128
@@
-          elif u.op in (Ops.ADD, Ops.SUB, Ops.NEG) and u.dtype == dtypes.float:
+          elif u.op in (Ops.ADD, Ops.SUB, Ops.NEG, Ops.MAX) and u.dtype == dtypes.float:
             values[u] = self.run_float_alu(u.op, src_values[0], src_values[1] if len(src_values) > 1 else None)
```

The unchanged full test hit our 30-second limit. Run forward only to separate that from the MAX output check:

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum
Ran 1 test in 6.151s
OK
```

Three NaN results were wrong. Inspecting those pairs shows negative NaN operands: `0xff83036f`, `0xffbf630b` and `0xffbaa7ce`. In each pair the NPU returned the other, finite operand. Our first probe only used a canonical positive NaN, so it missed this. Direct FP32 MAX is not enough yet; we need to preserve NaN propagation for both signs before counting MAX. Progress stays 11/30.

Try MIN on the same NaNs. It keeps negative NaNs but drops positive ones; ABS clears either NaN's sign without losing its payload. So keep MAX's normal result and use MIN to catch the missing case:

```
candidate = MAX(a, b)
magnitude_bits = bits(ABS(MIN(a, b)))
nan_mask = magnitude_bits > 0x7f800000
result_bits = candidate_bits * (1 - nan_mask) + 0x7fc00000 * nan_mask
```

The last two lines use INT32 NPU operations on the same storage, not numeric FP32-to-INT32 conversion. Positive NaNs already survive MAX. Negative NaNs survive MIN, and their unsigned magnitude exceeds the infinity encoding after ABS. Multiplying the two raw words by complementary 0/1 masks avoids computing an overflowing difference between them.

The register probe returned NaN for all three failed pairs, while retaining +inf, -inf, a negative finite result, -0 and the smallest positive subnormal. Add that correction:

```diff
   def run_float_alu(self, op:Ops, a:list, b:list|None=None) -> list:
@@
       elif op is Ops.MAX:
         self.mulacc_stage(0, base, base+64, base+128)
-        output = 128
+        # MIN retains negative NaNs. ABS exposes their magnitude for an integer NaN check.
+        self.mulacc_stage(1, base, base+64, base+256)
+        self.mulacc_stage(5, base+256, base+64, base+320)
+        for offset,value in ((192, 0x7f800000), (448, 0x7fc00000), (704, 1)):
+          to_mv(self.dev.input_buf+offset, 32)[:] = struct.pack("<i", value)*8
+        self.mulacc_stage(1, base+192, base+320, base+384, precision=4, output=4, binary=True)
+        self.mulacc_stage(4, base+704, base+384, base+512, precision=4, output=4)
+        # Select raw words with disjoint 0/1 products, avoiding an overflowing difference.
+        self.mulacc_stage(0, base+128, base+512, base+576, precision=4, output=4, mul=True)
+        self.mulacc_stage(0, base+448, base+384, base+640, precision=4, output=4, mul=True)
+        self.mulacc_stage(2, base+576, base+640, base+768, precision=4, output=4)
+        output = 768
```

Now replay the diffs and rerun maximum and minimum with gradients enabled at this checkpoint:

```bash
$ NOOPT=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum TestOps.test_minimum
Ran 2 tests in 14.138s
OK
```

The reconstructed backend passed both. A separate bounded run printed `test_relu_maximum_exact ... ok`, then reached the 30-second limit in the reduction `test_max`. `test_max_nan` was skipped by its existing `this test is broken #862` decorator, not passed.

Isolate the reduction's forward pass at the same checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_max
Ran 1 test in 5.906s
OK
```

So its forward reduction works. Running `NOOPT=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_max` alone still hit the 30-second limit with gradients enabled. MAX remains uncounted while reduction and optimized variant checks are incomplete.

Why is backward so much slower? A stack sample at 15 seconds was inside `run_float_alu` during the first case's gradient. Inspecting that kernel before executing it showed:

```
global_size = (135, 1, 1), local_size = (1, 1, 1)
RANGE bound = CAST(CONST(135), int)
END has two sources: STORE and RANGE
```

It repeats the 135-element reduction for 135 workgroups. The fixed-loop batching introduced at the first masked_select timeout also applies here: the bound is constant, REG buffers are per thread, and END is unconditional. These measurements were taken before that step was moved earlier; no additional loop-batching diff is needed here.

The bounded tests now finish with gradients enabled:

```bash
$ NOOPT=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_max
Ran 1 test in 17.100s
OK

$ NOOPT=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_min
Ran 1 test in 9.515s
OK
```

The optimized FP32 `test_maximum` still hit 30 seconds. Inspect its first kernels: `global_size=(325,1,1)`, `local_size=(3,1,1)`, with GROUP but no local memory or barriers. Our local_size=1 restriction prevents batching those independent three-lane workgroups. GROUP is already a no-op in the interpreter; admitting it does not add a synchronization operation.

Keep the no-local-memory, private-buffer and loop checks, but allow these workgroups to share a batch:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-                 Ops.BUFFER, Ops.RANGE, Ops.END,
+                 Ops.BUFFER, Ops.RANGE, Ops.END, Ops.GROUP,
@@
-    batch = 8 if uniform_shifts and uniform_loops and private_buffers and simple_ends and local_size == (1,1,1) and \
+    batch = 8 if uniform_shifts and uniform_loops and private_buffers and simple_ends and \
       all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
```

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum
Ran 1 test in 14.365s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_minimum
Ran 1 test in 13.766s
OK
```

Both optimized FP32 elementwise tests now finish with their existing gradient checks. No test inputs or tolerances changed.

At the reconstructed HALF checkpoint, a combined maximum/minimum command printed `test_maximum ... ok`, then hit its shared 30-second limit during minimum. Run the unfinished method alone:

```bash
$ DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_minimum
Ran 1 test in 13.717s
OK
```

Both optimized HALF methods completed, but the interrupted combined command is not reported as a two-test pass.

```bash
$ NOOPT=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum TestOps.test_minimum
Ran 2 tests in 13.253s
OK
```

The elementwise maximum/minimum checks now cover FP32 and HALF, with and without optimization. Reduction, cumulative and index-returning variants still need their own completed checks before MAX changes the 11/30 count.

The reconstructed optimized FP32 reduction also finishes, with gradients enabled:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_max
Ran 1 test in 15.495s
OK
```

Continue with the reduction and cumulative variants at this reconstructed checkpoint:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_min
Ran 1 test in 9.635s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_cummax TestOps.test_small_cummin
test_small_cummax (__main__.TestOps.test_small_cummax) ... ok
test_small_cummin (__main__.TestOps.test_small_cummin) ... ok
Ran 2 tests in 23.921s
OK
```

The second command printed a passing unittest summary, but the outer 30-second timeout returned 124 during shutdown. Both test bodies completed; this is not a clean command exit. Keep the distinction and check the methods separately before counting this variant configuration.

The index-returning variants finish individually:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_argmax
Ran 1 test in 8.534s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_argmin
Ran 1 test in 7.599s
OK
```

These include repeated maxima/minima, signed INT32 boundaries and bool inputs. They do not replace the larger cumulative tests or the HALF/NOOPT checks.

The standalone small cumulative-max command exits cleanly:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_cummax
Ran 1 test in 18.609s
OK

$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_cummin
Ran 1 test in 3.554s
OK
```

### Bounded integer indexing

The bounded ADD/MUL path was introduced at the first masked_select timeout above. The earlier cumulative-max profile also found integer indexing expensive: ADD took 13.34s, MUL 4.05s and FP32 MAX 1.35s during an incomplete run. Keep that as historical profiling, not a new benchmark.

The full cumulative test still times out after this change, so this is not its complete fix.

Profile the larger cumulative test again. It still reaches 30 seconds, but the cost has moved: the last sample records 1.15s in bounded ADD and 0.43s in bounded MUL, versus 13.58s in general integer CMPLT. These samples cover different amounts of work, so they are not a whole-test speedup ratio.

The bounded CMPLT extension now appears at the first masked_select timeout above. The 49-pair native comparison probe and its static operand-bound gate are described there; do not add the same helper changes again here.

The unchanged `test_simple_cummax` still hits 30 seconds after the bounded CMPLT extension. Keep it incomplete. The existing comparison checks still pass:

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_cmp_lt TestOps.test_cmp_eq
Ran 2 tests in 1.242s
OK
```

This verifies the comparison regression, not the cumulative test. Progress stays 11/30.

### Repeated prefix operands

The cumulative kernel has local memory, barriers and 16 local lanes. The previous no-local-memory batching rule cannot apply. Its RANGE recomputes prefix reductions for different outputs, so check whether expensive operations receive exactly repeated inputs before adding another arithmetic path.

A read-only duplicate probe still executed every NPU call. One sample saw 2,089 UINT32 WHERE calls with only 66 distinct operand vectors, and 3,129 FP32 MAX calls with 217. Compare complete input bytes here, not float equality: -0 and +0, and different NaN payloads, must remain distinct.

Extend the per-kernel cache introduced for integer ADD/WHERE to FP32 MAX and UINT32 WHERE. These helpers also return storage backed by immutable bytes, not scratch-buffer views. Keep the same 4096-entry limit and reset between calls:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-            if (u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int)) else None
+            if (u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int), (Ops.MAX, dtypes.float), (Ops.WHERE, dtypes.uint)) else None
```

This is reuse of an observed NPU result, not Python MAX or WHERE evaluation. The full `test_simple_cummax` still timed out at 30 seconds after the change, so it remains incomplete.

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_where TestOps.test_maximum
Ran 2 tests in 2.344s
OK
```

These checks validate reuse and regressions, not completion of the larger cumulative test. It remains a timeout and MAX remains outside the 11/30 completed count.

The next duplicate probe found 2,184 bounded INT32 MUL calls with one distinct operand vector, and 4,366 CMPLT calls with 198. These are already NPU operations with statically bounded inputs. Reuse their exact results too; leave ADD out because its indexing operands change much more often.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-            if (u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int), (Ops.MAX, dtypes.float), (Ops.WHERE, dtypes.uint)) else None
+            if ((u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int), (Ops.MAX, dtypes.float), (Ops.WHERE, dtypes.uint)) or
+                (u in bounded_int32 and u.op in (Ops.MUL, Ops.CMPLT))) else None
```

This changes which existing NPU results we reuse, not how MUL or CMPLT is calculated. The cache still resets for every kernel call.

The larger cumulative test still does not finish within our limit on the working backend:

```bash
$ timeout --signal=TERM --kill-after=2s 30s env DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_simple_cummax
test_simple_cummax (__main__.TestOps.test_simple_cummax) ...
```

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_where TestOps.test_maximum TestOps.test_cmp_lt TestOps.test_cmp_eq
test_where (__main__.TestOps.test_where) ... ok
test_maximum (__main__.TestOps.test_maximum) ... ok
test_cmp_lt (__main__.TestOps.test_cmp_lt) ... ok
test_cmp_eq (__main__.TestOps.test_cmp_eq) ... ok

Ran 4 tests in 3.453s
OK
```

So this reuse survives the regression checks, but does not solve the cumulative runtime limit. MAX is still not counted as complete.

### Keep LOCAL storage separate when batching

The cumulative kernel uses LOCAL storage and barriers. Why did we exclude these? Our allocation repeats one buffer across every lane. That is correct for one workgroup, but batching several groups would make them overwrite each other's scratch.

Allocate one LOCAL buffer per group, shared only by that group's local lanes. REG stays per lane and global buffers stay shared. Fixed loops still advance together; conditional backedges remain excluded. IF already has a per-lane mask, and a barrier waits for every lane because this interpreter executes one UOp across all lanes before the next.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-                 Ops.BUFFER, Ops.RANGE, Ops.END, Ops.GROUP,
+                 Ops.BUFFER, Ops.RANGE, Ops.END, Ops.GROUP, Ops.BARRIER, Ops.IF, Ops.ENDIF,
@@
-    private_buffers = all(u.op is not Ops.BUFFER or u.addrspace is AddrSpace.REG for u in self.uops)
+    private_buffers = all(u.op is not Ops.BUFFER or u.addrspace in (AddrSpace.REG, AddrSpace.LOCAL) for u in self.uops)
@@
-      all(u.op in batch_ops and u.addrspace is not AddrSpace.LOCAL for u in self.uops) else 1
+      all(u.op in batch_ops for u in self.uops) else 1
@@
           if u.addrspace == AddrSpace.REG:
             # REGs are per thread
             values[u] = [memoryview(bytearray(u.max_numel()*u.dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
+          elif u.op is Ops.BUFFER and u.addrspace is AddrSpace.LOCAL:
+            # LOCAL storage is shared within each workgroup, never across batched groups.
+            local = [memoryview(bytearray(u.max_numel()*u.dtype.itemsize)).cast(storage_fmt) for _ in group]
+            values[u] = [buf for buf in local for _ in warp]
```

Check the storage scope directly: two lanes write their values, cross a barrier, then read the other lane's value. Use 17 groups so the last batch is partial. Distinct values in each group catch accidental sharing.

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_cummax
Ran 1 test in 18.106s
OK
```

The larger test_simple_cummax still exits 124 at 30 seconds. Separate LOCAL storage makes batching possible; it does not remove the repeated prefix arithmetic or prove the whole MAX family complete.

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum TestOps.test_minimum TestOps.test_where
Ran 3 tests in 3.192s
OK
```

### Use the wider stages for bounded arithmetic

The bounded helper now widens its tasks at the first masked_select timeout. No second width change is needed here. The following regression result belongs to the earlier audit order:

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum TestOps.test_cmp_lt TestOps.test_add
Ran 3 tests in 1.911s
OK
```

That audit still timed out on test_simple_cummax at 30 seconds. Bounded integer operations and integer WHERE now use wider tasks; FP32 MAX still processes eight lanes at a time. This is not a new completed op.

### Widen the FP32 stages too

Bounded integer arithmetic now uses wider tasks, but FP32 MAX still loops over eight lanes. The earlier FP32 ADD probe used the same layout successfully. Keep the arithmetic sequence, including the negative-NaN correction for MAX and signed-zero correction for ADD/SUB; change the storage and submission width.

Each scratch slot now holds 128 words, or 512 bytes. Slot numbers replace the old byte offsets: old offset 64 becomes slot 1, 128 becomes slot 2, and so on. The local stage helper translates those slots into DMA addresses and passes the current atom count. Constants must cover the whole task too.

```diff
   def run_float_alu(self, op:Ops, a:list, b:list|None=None) -> list:
     assert op in (Ops.ADD, Ops.SUB, Ops.NEG, Ops.MAX)
     assert (op is Ops.NEG and b is None) or (b is not None and len(a) == len(b))
     base = self.dev.input_mem.dma_addr
-    to_mv(self.dev.input_buf+192, 32)[:] = struct.pack("<i", -2147483647)*8
-    to_mv(self.dev.input_buf+448, 32)[:] = struct.pack("<i", -2147483648)*8
     result:list = []
-    for start in range(0, len(a), 8):
-      count = min(8, len(a)-start)
-      for offset,values in ((0, a), (64, b)):
-        raw = b"".join(raw16(x, dtypes.float) for x in values[start:start+8]) if values is not None else b""
-        to_mv(self.dev.input_buf+offset, 32)[:] = raw+bytes(32-len(raw))
+    for start in range(0, len(a), 128):
+      count = min(128, len(a)-start)
+      atoms = (count+7)//8
+      # Each numbered slot owns 512 bytes, enough for 128 FP32/INT32 lanes.
+      def stage(algo:int|None, lhs:int, rhs:int, out:int, **kw):
+        self.mulacc_stage(algo, base+lhs*512, base+rhs*512, base+out*512, atoms=atoms, **kw)
+      for slot,values in ((0, a), (1, b)):
+        raw = b"".join(raw16(x, dtypes.float) for x in values[start:start+count]) if values is not None else b""
+        to_mv(self.dev.input_buf+slot*512, atoms*32)[:] = raw+bytes(atoms*32-len(raw))
       if op is Ops.NEG:
-        self.mulacc_stage(6, base, base+64, base+128)
-        output = 128
+        stage(6, 0, 1, 2)
+        output = 2
       elif op is Ops.MAX:
-        self.mulacc_stage(0, base, base+64, base+128)
+        stage(0, 0, 1, 2)
         # MIN retains negative NaNs. ABS exposes their magnitude for an integer NaN check.
-        self.mulacc_stage(1, base, base+64, base+256)
-        self.mulacc_stage(5, base+256, base+64, base+320)
-        for offset,value in ((192, 0x7f800000), (448, 0x7fc00000), (704, 1)):
-          to_mv(self.dev.input_buf+offset, 32)[:] = struct.pack("<i", value)*8
-        self.mulacc_stage(1, base+192, base+320, base+384, precision=4, output=4, binary=True)
-        self.mulacc_stage(4, base+704, base+384, base+512, precision=4, output=4)
+        stage(1, 0, 1, 4)
+        stage(5, 4, 1, 5)
+        for slot,value in ((3, 0x7f800000), (7, 0x7fc00000), (11, 1)):
+          to_mv(self.dev.input_buf+slot*512, atoms*32)[:] = struct.pack("<i", value)*(atoms*8)
+        stage(1, 3, 5, 6, precision=4, output=4, binary=True)
+        stage(4, 11, 6, 8, precision=4, output=4)
         # Select raw words with disjoint 0/1 products, avoiding an overflowing difference.
-        self.mulacc_stage(0, base+128, base+512, base+576, precision=4, output=4, mul=True)
-        self.mulacc_stage(0, base+448, base+384, base+640, precision=4, output=4, mul=True)
-        self.mulacc_stage(2, base+576, base+640, base+768, precision=4, output=4)
-        output = 768
+        stage(0, 2, 8, 9, precision=4, output=4, mul=True)
+        stage(0, 7, 6, 10, precision=4, output=4, mul=True)
+        stage(2, 9, 10, 12, precision=4, output=4)
+        output = 12
       else:
-        rhs = base+64
+        for slot,value in ((3, -2147483647), (7, -2147483648)):
+          to_mv(self.dev.input_buf+slot*512, atoms*32)[:] = struct.pack("<i", value)*(atoms*8)
+        rhs = 1
         if op is Ops.SUB:
-          self.mulacc_stage(6, rhs, base, base+256)
-          rhs = base+256
-        self.mulacc_stage(2, base, rhs, base+128)
+          stage(6, rhs, 0, 4)
+          rhs = 4
+        stage(2, 0, rhs, 2)
         # ADD loses -0 + -0. As signed INT32 bits, only -0 equals INT32_MIN.
-        self.mulacc_stage(0, base, rhs, base+320, precision=4, output=4)
-        self.mulacc_stage(1, base+320, base+192, base+384, precision=4, output=4, binary=True)
-        self.mulacc_stage(0, base+448, base+384, base+512, precision=4, output=4, mul=True)
-        self.mulacc_stage(2, base+128, base+512, base+576, precision=4, output=4)
-        output = 576
-      raw = bytes(to_mv(self.dev.input_buf+output, 32))
+        stage(0, 0, rhs, 5, precision=4, output=4)
+        stage(1, 5, 3, 6, precision=4, output=4, binary=True)
+        stage(0, 7, 6, 8, precision=4, output=4, mul=True)
+        stage(2, 2, 8, 9, precision=4, output=4)
+        output = 9
+      raw = bytes(to_mv(self.dev.input_buf+output*512, atoms*32))
       result.extend(typed_view(raw[i*4:i*4+4], dtypes.float) for i in range(count))
     return result
```

The historical test_simple_cummax run reached the 30-second limit at this point. The wider integer WHERE setup now appears earlier in the tutorial; that path is no longer eight-lane-only here.

### Check the wider integer WHERE path

Integer WHERE's wider scratch slots and task layout were introduced at the first bool-WHERE timeout. The raw FP32 WHERE path now reuses that helper; check its larger inputs and tails here.

A Tensor test may still call a helper with only eight lanes. Wider register support alone does not prove that the interpreter fills those lanes.

The cumulative MAX test still times out at 30 seconds, even after these wider arithmetic paths. Do not count it as passed or assume that submission size was the only bottleneck.

```bash
$ FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_where TestOps.test_maximum TestOps.test_add TestOps.test_sub TestOps.test_neg
Ran 5 tests in 3.629s
OK
```

### What still costs time in cumulative MAX?

After widening, a bounded profile still reaches the limit in the first 512-element case. A separate 12-second Python profile counted about 1.24 million raw16 calls and 493,720 typed_view calls; packing and interpreter work matter too. This diagnostic intentionally interrupted the test and is not a correctness failure or a pass.

tinygrad's `_cumalu` in `tinygrad/mixin/op.py` pads, pools and reduces overlapping prefixes. We have not changed that core algorithm. Leave cumulative MAX incomplete while auditing the other implemented families; do not turn a timeout into a pass.

### Shift variants: per-lane counts

Now run the existing shift tests at this reconstructed checkpoint, with optimization enabled. test_lshift reaches its broadcast count `[0, 2, 4]` and fails:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_lshift TestOps.test_lshift_signed
NotImplementedError: ROCKCHIP 32-bit shift requires one uniform shift count in 0..31
```

The signed method passed in the same audit. The audit also requested test_lshift_int16, but the reconstructed TestOps did not contain that worktree-only method: `AttributeError: type object 'TestOps' has no attribute 'test_lshift_int16'`. That is a missing tutorial diff, not a hardware result; address it separately.

Why reject different counts? A convolution task has one configuration, but run_u32_shift already submits one word at a time. Pair each word with its own count instead of reusing counts[0]. The CPU chooses the task configuration; conv_shift still performs the shift on the NPU.

```diff
   def run_u32_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
-    assert op in (Ops.SHL, Ops.SHR)
+    assert op in (Ops.SHL, Ops.SHR) and len(a) == len(b)
-    b = list(map(scalar16, b))
-    if not b or not all_same(b) or not 0 <= b[0] <= 31:
-      raise NotImplementedError("ROCKCHIP 32-bit shift requires one uniform shift count in 0..31")
+    counts = [scalar16(x) for x in b]
+    if not counts or any(not 0 <= count <= 31 for count in counts):
+      raise NotImplementedError("ROCKCHIP 32-bit shift requires counts in 0..31")
-    return [typed_view(self.conv_shift(op, bytes(raw16(x, dtype)), int(b[0]), signed=dtype == dtypes.int), dtype) for x in a]
+    return [typed_view(self.conv_shift(op, bytes(raw16(x, dtype)), int(count), signed=dtype == dtypes.int), dtype)
+            for x,count in zip(a, counts)]
```

All four existing 32-bit shift methods pass on the working backend after this change:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_lshift TestOps.test_lshift_signed TestOps.test_rshift TestOps.test_rshift_signed
Ran 4 tests in 0.609s
OK
```

Restore the missing diff for the small INT16 method already present in the worktree. These inputs do not overflow; this checks the original INT16 MUL path, not general wrapping shifts:

```diff
 class TestOps(unittest.TestCase):
@@
+  def test_lshift_int16(self):
+    data = [[0, 1, 2], [1 << 8, 1 << 10, 1 << 12]]
+    tor = torch.tensor(data, dtype=torch.int16)
+    ten = Tensor(data, dtype=dtypes.int16)
+    helper_test_op([], lambda: tor << 0, lambda: ten << 0, forward_only=True)
+    helper_test_op([], lambda: tor << 2, lambda: ten << 2, forward_only=True)
+
   def test_rshift(self):
```

All 436 hunks replay. The reconstructed checkpoint now runs the same five methods:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_lshift TestOps.test_lshift_signed TestOps.test_lshift_int16 TestOps.test_rshift TestOps.test_rshift_signed
Ran 5 tests in 0.642s
OK
```

But the INT16 test uses small values. Check the known saturation limit before calling SHL complete:

| INT16 input | Shift | Torch result | ROCKCHIP result |
|------------:|------:|-------------:|----------------:|
| 32767       | 1     | -2           | 32767           |
| 16384       | 1     | -32768       | 32767           |
| -32768      | 1     | 0            | -32768          |
| -20000      | 1     | 25536        | -32768          |

The current INT16 MUL saturates; a shift must retain the low 16 bits instead. The 32-bit convolution path already moves raw bits without this saturation. Reusing that path and taking the low two output bytes is a candidate for the next fix. Narrow shift helpers also still require uniform counts. Progress stays 11/30.

### Narrow shifts without saturation

For SHL, append zero bytes to the narrow input, shift that UINT32 word, then keep the original number of low bytes. No sign extension is needed: upper bits cannot move into the retained low bits during a left shift.

SHR needs a different layout. Put the input in the high bytes and shift by `32 - bits + count`, as the earlier INT16 SHR wrapper did. Signed SHR then fills from the original sign bit. This also works for INT8/UINT8.

| Operation | Narrow input placement | 32-bit count      | Read back       |
|-----------|------------------------|-------------------|-----------------|
| SHL       | Low bytes, zero padding | count             | Low input-width bytes |
| SHR       | High bytes, zero padding | 32 - bits + count | Low input-width bytes |

A direct NPU probe matched both directions for INT8, UINT8, INT16 and UINT16, including minimum/maximum values and different counts per lane. Extend the existing wrapper instead of adding another shift implementation:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_u16_shr(self, a:list, b:list, dtype:DType) -> list:
+  def run_narrow_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
+    assert op in (Ops.SHL, Ops.SHR) and len(a) == len(b) and dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16)
     counts = list(map(scalar16, b))
-    if not counts or not all_same(counts) or not 0 <= counts[0] < 16:
-      raise NotImplementedError("ROCKCHIP 16-bit SHR requires one uniform count in 0..15")
-    wide = dtypes.int if dtype == dtypes.int16 else dtypes.uint
-    words = [typed_view(bytes(2)+bytes(raw16(x, dtype)), wide) for x in a]
-    shifted = self.run_u32_shift(Ops.SHR, words, [16+counts[0]]*len(a), wide)
-    return [typed_view(bytes(x)[:2], dtype) for x in shifted]
+    if not counts or any(not 0 <= count < dtype.bitsize for count in counts):
+      raise NotImplementedError(f"ROCKCHIP {dtype.bitsize}-bit shifts require counts in 0..{dtype.bitsize-1}")
+    wide = dtypes.int if op is Ops.SHR and dtype in dtypes.sints else dtypes.uint
+    pad = bytes(4-dtype.itemsize)
+    # SHR puts the sign bit at bit 31; SHL needs only the original low bytes.
+    words = [typed_view(pad+bytes(raw16(x, dtype)) if op is Ops.SHR else bytes(raw16(x, dtype))+pad, wide) for x in a]
+    amounts = [32-dtype.bitsize+count if op is Ops.SHR else count for count in counts]
+    shifted = self.run_u32_shift(op, words, amounts, wide)
+    return [typed_view(bytes(x)[:dtype.itemsize], dtype) for x in shifted]
@@
-          elif u.op is Ops.SHR and u.dtype in (dtypes.int16, dtypes.uint16):
-            values[u] = self.run_u16_shr(src_values[0], src_values[1], u.dtype)
+          elif u.op in (Ops.SHL, Ops.SHR) and u.dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16):
+            values[u] = self.run_narrow_shift(u.op, src_values[0], src_values[1], u.dtype)
```

The public narrow-shift dispatch no longer uses saturating INT16 MUL. Counts outside `0..bits-1` still raise; that limit has not been removed here.

The reconstructed five-method shift suite now passes in all four configurations. Each command had its own 30-second limit; no test inputs or assertions were removed:

| Configuration       | Five shift methods | Negative division/shift regression |
|---------------------|-------------------:|-----------------------------------:|
| Default, optimized  | 0.639s             | 0.134s                             |
| Default, NOOPT=1    | 0.644s             | 0.134s                             |
| HALF, optimized     | 0.659s             | 0.134s                             |
| HALF, NOOPT=1       | 0.533s             | 0.134s                             |

The methods are test_lshift, test_lshift_signed, test_lshift_int16, test_rshift and test_rshift_signed. The regression is test_idiv_shift_rewrite_negative.

Check the direct UOp tests too. The first run reports:

```text
test_shl_int32 ... skipped 'only ptx and cstyle use bitshifts'
test_shr_int32 ... skipped 'only ptx and cstyle use bitshifts'
OK (skipped=2)
```

Together with the existing Tensor shift methods and division/shift regression, the saved variant audit brought the count to **13/30**. Out-of-range counts remain an explicit limitation; these runs do not claim support for them.

### Variant audit: BITCAST

The default FP32 `TestOps.test_bitcast` passes at this reconstructed checkpoint: one test in 0.141s. That is only one same-size view, not complete dtype coverage. Keep BITCAST unverified rather than count it from this one result.

### Variant audit: MUL

Run the existing tiny, tensor, scalar and NaN/inf cases at the reconstructed checkpoint, without FORWARD_ONLY so the gradient checks also run:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tiny_mul TestOps.test_mul TestOps.test_scalar_mul TestOps.test_mul_naninf

test_tiny_mul (__main__.TestOps.test_tiny_mul) ... ok
test_mul (__main__.TestOps.test_mul) ...
# The enclosing 30-second timeout exits with status 124 here.
```

The last two methods were not reached. Run them separately, then compare HALF with and without optimization:

| Configuration             | Methods                                      | Result                 |
|---------------------------|----------------------------------------------|------------------------|
| Default FP32, optimized   | scalar_mul, mul_naninf                       | 2 passed in 1.869s     |
| HALF, optimized           | tiny_mul, mul, scalar_mul, mul_naninf         | 4 passed in 5.534s     |
| HALF, NOOPT=1             | tiny_mul, mul, scalar_mul, mul_naninf         | 4 passed in 2.244s     |

Why does the large FP32 case still take so long? Scalar coefficients use `run_float_scale`, but two tensor inputs use `run_float_binary`. Its exact FP32 calculation still processes eight lanes per batch. The wider private integer stages introduced above suggest a batching experiment; these timings alone do not prove it will fix the timeout. MUL stays unverified until the remaining variants finish and pass.

The integer stages already accept up to sixteen eight-lane atoms. Try that layout for MUL: 128 lanes need 512 bytes per scratch slot, and the last partial batch needs `ceil(count/8)` atoms. Keep FDIV at eight lanes for now. Constants and output reads must cover the same width; changing only the task width would read stale scratch bytes.

```diff
   def run_float_binary(self, op:Ops, a:list, b:list, half_inputs:bool=False, constant:tuple[int,float]|None=None) -> list:
@@
     base = self.dev.input_mem.dma_addr
     result:list = []
-    for start in range(0, len(a), 8):
-      count, slot, constants = min(8, len(a)-start), 0, {}
+    batch = 128 if op is Ops.MUL else 8
+    stride = max(64, batch*4)
+    for start in range(0, len(a), batch):
+      count, slot, constants = min(batch, len(a)-start), 0, {}
+      atoms = (count+7)//8
       def alloc(raw:bytes|None=None) -> int:
         nonlocal slot
-        addr = base+64*slot
+        addr = base+stride*slot
         slot += 1
-        assert slot*64 <= self.dev.input_mem.size
-        if raw is not None: to_mv(self.dev.input_buf+addr-base, 32)[:] = raw+bytes(32-len(raw))
+        assert slot*stride <= self.dev.input_mem.size
+        if raw is not None: to_mv(self.dev.input_buf+addr-base, atoms*32)[:] = raw+bytes(atoms*32-len(raw))
         return addr
       def const(x:int) -> int:
-        if x not in constants: constants[x] = alloc(struct.pack("<i", x)*8)
+        if x not in constants: constants[x] = alloc(struct.pack("<i", x)*(atoms*8))
         return constants[x]
       def calc(algo:int|None, x:int, y:int, **kw) -> int:
         out = alloc()
-        self.mulacc_stage(algo, x, y, out, precision=4, output=4, **kw)
+        self.mulacc_stage(algo, x, y, out, precision=4, output=4, atoms=atoms, **kw)
@@
-        raw = [bytes(raw16(x, dtypes.float)) for x in values[start:start+8]]
+        raw = [bytes(raw16(x, dtypes.float)) for x in values[start:start+count]]
@@
-      output = bytes(to_mv(self.dev.input_buf+bits-base, 32))
+      output = bytes(to_mv(self.dev.input_buf+bits-base, atoms*32))
```

The four default-FP32 Tensor methods now finish on the runtime in 11.137s, and on the reconstructed tutorial in 11.506s, both with exit status 0. All 443 hunks replay.

But `NOOPT=1` still reaches the 30-second limit during `test_mul`, after `test_tiny_mul` passes. The new helper can handle 128 lanes, but the interpreter still batches only eight workgroups. With scalar workgroups it cannot fill that wider task. That dispatch limit is the next thing to investigate; the optimized pass does not complete the MUL audit.

For kernels containing FP32 MUL, feed enough independent workgroups to fill 128 lanes. Keep all the existing safety checks and the separate LOCAL buffers; this changes the batch limit, not which kernels may be batched.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-    batch = 8 if uniform_shifts and uniform_loops and private_buffers and simple_ends and \
+    # Feed the wider FP32 MUL helper without merging a workgroup's local storage with another's.
+    groups_per_batch = max(8, 128//len(warp)) if any(u.op is Ops.MUL and u.dtype == dtypes.float for u in self.uops) else 8
+    batch = groups_per_batch if uniform_shifts and uniform_loops and private_buffers and simple_ends and \
       all(u.op in batch_ops for u in self.uops) else 1
```

The four Tensor MUL methods now finish with default FP32 and NOOPT=1: 3.596s on the runtime, then 3.070s at the reconstructed checkpoint. The reconstructed optimized run also finishes, in 11.378s. Both commands exit normally; no timeout is hidden behind an OK line.

Do not count MUL yet: product reductions and cumulative products also use it. A bounded reconstructed run of `test_small_cumprod`, `test_cumprod_zero_axis`, `test_prod_dtype_arg` and `test_prod` prints OK for the first three, then reaches the shared 30-second command limit during `test_prod`. That does not give `test_prod` its own full 30 seconds, so run it separately before diagnosing its speed. Larger cumulative-product and scatter-product variants remain pending.

Run `test_prod` alone before changing anything else:

```bash
$ DEBUG=2 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_prod

Ran 1 test in 18.815s
OK
```

This reconstructed-checkpoint command exits normally within 30 seconds. Both forward and backward checks pass. DEBUG shows the larger gradient kernels taking seconds, but no failed comparison. `mixin/gradient.py` explains why a product test is more than MUL: its backward expression counts zeros, selects a safe divisor and evaluates `ret/safe_x`. We have not changed that shared code.

The existing `test_scatter_mul` and `test_scatter_reduce_prod_zeros` also pass on this reconstructed backend, together in 7.384s with default FP32. Those tests check infinity/NaN scatter multiplication and zero-initialized product reduction respectively. These passes clear the cases from the current audit, not from the historical sweep table below.

Next `test_simple_cumprod` reaches the 30-second limit when run alone. A separate FORWARD_ONLY=1 run also exits 124 without finishing. So backward work is not the only problem; neither run reports a numerical mismatch before stopping.

Inspect `mixin/op.py` before inventing another register trick. `_cumalu` pads with the operation's identity, builds overlapping windows, then reduces each window. At length 512 it still uses that path; `_split_cumalu` only splits lengths greater than 512. That describes much more work than a simple elementwise MUL. It suggests profiling the repeated reductions next, not changing the shared core or claiming the timeout proves wrong arithmetic. MUL remains outside the completed count.

A 20-second cProfile diagnostic of the reconstructed forward-only run records 68,637 `mulacc_stage` calls. `run_float_binary` accounts for 9.830s of the sampled backend time; submission and register construction dominate underneath it. The diagnostic deliberately raises SystemExit(124), which unittest reports as an error. That is an interrupted profile, not a numerical failure or a completed test.

Can repeated operand batches reuse a result? Extend the existing per-kernel raw-byte cache to FP32 MUL. Keep MUL entries tied to their UOp, so constant-coefficient and general multiplication dispatches cannot share an entry. No new arithmetic runs on the host; a hit returns an earlier NPU result.

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          cache_key = (u.op, u.dtype, tuple(tuple(bytes(raw16(x, dt)) for x in xs) for dt,xs in zip(src_dtypes, src_values))) \
-            if ((u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int), (Ops.MAX, dtypes.float), (Ops.WHERE, dtypes.uint)) or
+          cache_key = (u if u.op is Ops.MUL else u.op, u.dtype,
+                       tuple(tuple(bytes(raw16(x, dt)) for x in xs) for dt,xs in zip(src_dtypes, src_values))) \
+            if ((u.op, u.dtype) in ((Ops.ADD, dtypes.int), (Ops.WHERE, dtypes.int), (Ops.MUL, dtypes.float),
+                                  (Ops.MAX, dtypes.float), (Ops.WHERE, dtypes.uint)) or
                 (u in bounded_int32 and u.op in (Ops.MUL, Ops.CMPLT))) else None
```

The unchanged forward-only `test_simple_cumprod` still times out at 30 seconds. Keep that limitation open rather than count MUL as complete.

### Variant audit: FDIV batching

MUL's cumulative-product limit is still open. Before changing the reduction algorithm, check division too: product gradients use it, and `run_float_binary` still limits FDIV to eight lanes.

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div

test_div (__main__.TestOps.test_div) ...
# The reconstructed command reaches the 30-second limit, exit 124.
```

The FDIV loop uses the same private INT32 stages as MUL, including the exact shifted-integer operations. Try the 128-lane layout without changing long division, rounding or special-value selection. Half division also uses this helper after widening its inputs.

```diff
   def run_float_binary(self, op:Ops, a:list, b:list, half_inputs:bool=False, constant:tuple[int,float]|None=None) -> list:
@@
-    batch = 128 if op is Ops.MUL else 8
-    stride = max(64, batch*4)
+    batch, stride = 128, 512
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-    # Feed the wider FP32 MUL helper without merging a workgroup's local storage with another's.
-    groups_per_batch = max(8, 128//len(warp)) if any(u.op is Ops.MUL and u.dtype == dtypes.float for u in self.uops) else 8
+    # Feed wider floating MUL/FDIV helpers while keeping each workgroup's local storage separate.
+    wide_binary = any((u.op is Ops.MUL and u.dtype == dtypes.float) or
+                      (u.op is Ops.FDIV and u.dtype in (dtypes.half, dtypes.float)) for u in self.uops)
+    groups_per_batch = max(8, 128//len(warp)) if wide_binary else 8
```

The unchanged default-FP32 `test_div` now passes in 10.448s on the runtime and 8.923s on the reconstructed tutorial, both with normal command exit. All 451 hunks replay; scoped lint passes. Mypy still reports the same 47 reference-backend errors.

With NOOPT=1, a combined reconstructed command prints OK for `test_div` and `test_scalar_div`, then reaches the shared 30-second limit during `test_div_naninf`. As with the product audit, test the unfinished method alone; the combined timeout is not evidence of a numerical error.

The separate reconstructed NOOPT=1 `test_div_naninf` passes in 8.330s and exits normally. Rounding-mode, integer-input and the rest of the dtype/optimization matrix still need their own results before counting FDIV.

Continue with the existing variants, one command at a time. On the reconstructed checkpoint, default-FP32 `test_div_rounding_mode` reaches its own 30-second limit without a mismatch. It includes integer and float numerators/denominators, ordinary division, truncation, floor rounding and an invalid-mode check; the timeout leaves its coverage incomplete.

The shorter `test_div_int` passes in 1.148s. It includes true division, floor/truncating division and the UINT64 maximum divided by one. With DEFAULT_FLOAT=HALF, the ordinary `test_div` also passes, including gradients, in 7.452s.

The reconstructed HALF `test_div` with NOOPT=1 passes in 6.770s. The optimized HALF scalar and NaN/inf methods pass together in 12.615s. These commands include their normal gradient checks; none use FORWARD_ONLY=1.

Default-FP32 `test_div_rounding_mode` passes unchanged with NOOPT=1 in 21.640s, with normal exit. Its optimized run timed out, so the difference needs profiling rather than a new quotient formula. FDIV remains uncounted while that configuration and the remaining variants are unresolved. All 451 tutorial hunks still replay.

Why did the 1500 branch handle this case? Its `renderer/rockchip.py::_int32_divmod` converts the integer operands to FP16, runs FDIV and TRUNC, then converts the quotient back to INT32. For CMOD it computes `a - quotient*b`. The helper calls this the “supported small INT32 division domain”. Our test uses small numerators, including ±111, and small denominators, including ±55; that shortcut is relevant here, but it is not full-width INT32 division. Copying it unconditionally would lose integers that FP16 cannot represent exactly.

The optimized checkpoint's bounded profile points to integer division too: 20 seven-lane CDIV calls take 5.654s and ten seven-lane CMOD calls take 2.786s before the probe stops. This is an interrupted profile, not a passing test. Our restoring division already computes both quotient and remainder, so next check whether the graph asks for both with identical inputs. If it does, reusing those NPU results could save work without narrowing the integer range.

The next 20-second probe records the raw operands and clears its records at each kernel entry. It sees 20 CDIV calls and ten CMOD calls; all ten CMOD calls have a matching quotient calculation in that same kernel. The probe is deliberately interrupted, not a test pass.

Keep both results from the existing division loop. Key them by dtype and exact input bytes, and clear the cache at each kernel call. Scratch addresses cannot be cached because later tasks overwrite them; copy the result words instead. Limit the cache to 128 input batches.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
-  def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
+  def run_integer(self, op:Ops, inputs:list[list], dtype:DType, division_cache:dict|None=None) -> list:
@@
     for start in range(0, len(inputs[0]), batch):
       count = min(batch, len(inputs[0])-start)
+      key = (dtype, tuple(tuple(bytes(raw16(x, dtype)) for x in xs[start:start+count]) for xs in inputs)) \
+        if division_cache is not None and op in (Ops.CDIV, Ops.CMOD) else None
+      if division_cache is not None and key is not None and key in division_cache:
+        result.extend(division_cache[key][op is Ops.CMOD])
+        continue
@@
           out = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else remainder
+          if division_cache is not None and key is not None and len(division_cache) < 128:
+            # Retain both exact NPU results before scratch is reused; no host division.
+            pair = []
+            for limbs in (quotient, remainder):
+              raw = [bytes(to_mv(self.dev.input_buf+x-base, atoms*32)) for x in limbs]
+              pair.append([typed_view(b"".join(x[i*4:i*4+1] for x in raw), dtype) for i in range(count)])
+            division_cache[key] = pair
@@
     alu_cache:dict[tuple, list] = {}
+    division_cache:dict = {}
@@
-            values[u] = self.run_integer(u.op, src_values, integer_dtype)
+            values[u] = self.run_integer(u.op, src_values, integer_dtype, division_cache=division_cache)
```

Now repeat the original optimized rounding-mode test:

```bash
$ DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_div_rounding_mode
test_div_rounding_mode (__main__.TestOps.test_div_rounding_mode) ... ok
Ran 1 test in 23.401s

OK
# The enclosing checkpoint command still reaches its 30-second wall-clock limit: exit 124.
```

The test body passes, but the full reconstructed command does not finish within the limit. Do not count this as a completed bounded run. All 459 diffs replay and scoped lint passes; mypy still reports the existing 47 errors in ops_rockchip_ref.py. Progress remains 13/30.

The ordinary runtime command, without the checkpoint reconstruction wrapper, finishes normally: `test_div_rounding_mode` passes in 24.428s under the same 30-second limit. Keep that separate from the checkpoint command's timeout above.

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
| Timeouts | Deferred. Keep the existing failure record; skip these cases during the accuracy pass. | `_tile` and grouped EW submission, for a later performance pass |

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
| Timeouts              | Deferred, not passed. Do not rerun these cases during the accuracy pass. |

The 1500 helpers are candidates for these checks, not evidence that our failures are already solved. Keep tinygrad's native decomposition first; use a different formula or LUT only after tracing an actual missing primitive or accuracy failure. The eight skips also need their decorators checked separately: skipped is neither passed nor a hardware failure.

The earlier sections have targeted checks for boolean/FP32 WHERE, FP32 comparisons, division sign handling, scratch reuse and FP32 ADD/SUB/NEG/MUL/FDIV. That does not clear their whole failure groups: masked_select still needs a completed run, and the accuracy cases need their original tests rerun. Until another complete sweep finishes, **209 / 433 is the saved baseline, not the current pass count**.

## isclose: finish the run before diagnosing accuracy

In the earlier investigation order, the reconstructed backend reached the 30-second limit on test_isclose. TRACE showed one lane per operation: comparisons, WHERE and mask operations were excluded from the batching allowlist. The preceding sections now introduce comparison/WHERE/MAX/AND batching earlier. OR is still excluded here; an expression containing it still drops back to one workgroup at a time.

Extend only that allowlist. The existing per-group LOCAL storage and restricted-loop checks stay unchanged:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
     batch_ops = {Ops.PARAM, Ops.CONST, Ops.SPECIAL, Ops.INDEX, Ops.LOAD, Ops.STORE, Ops.CAST, Ops.BITCAST,
                  Ops.BUFFER, Ops.RANGE, Ops.END, Ops.GROUP, Ops.BARRIER, Ops.IF, Ops.ENDIF,
                  Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD,
                  Ops.EXP2, Ops.LOG2, Ops.SIN,
-                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE, Ops.MAX, Ops.AND,
+                 Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.NEG, Ops.TRUNC, Ops.WHERE, Ops.MAX, Ops.AND, Ops.OR,
                  *GroupOp.Comparison, Ops.SHL, Ops.SHR, Ops.MULACC, Ops.CUSTOM, Ops.SINK, Ops.NOOP, Ops.AFTER}
```

Now the unchanged test completes within the limit. TRACE is omitted here; the failing expression is x.isclose(x + 1e-6):

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_isclose

Mismatched elements: 14 / 360 (3.89%)
Ran 1 test in 7.677s
FAILED (errors=1)
```

Before changing NPU arithmetic, compare tinygrad CPU under the same settings:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=CPU python test/backend/test_ops.py TestOps.test_isclose

Mismatched elements: 14 / 360 (3.89%)
Ran 1 test in 0.447s
FAILED (errors=1)
```

Both fail at the same expression with the same mismatch count. That does not yet identify which intermediate differs from Torch, but it rules out treating this result alone as proof that our NPU comparison needs changing. Keep the test failed; next inspect the HALF tolerance arithmetic against Torch, without loosening the assertion.

### Find the rounding boundary

Use a small FP16 input, a = 0.09765625, and b = a + 1e-6. That increment is too small to change the stored FP16 value. Does the lazy expression preserve that rounding?

| Expression                 | tinygrad CPU | ROCKCHIP     | Torch eager |
| -------------------------- | ------------ | ------------ | ----------- |
| a - b, with b still lazy   | about -1e-6  | about -1e-6  | 0           |
| a.isclose(b), b still lazy | False        | False        | True        |
| stored FP16 b              | 0.09765625   | 0.09765625   | 0.09765625  |
| a - b, after b.realize()    | 0            | 0            | 0           |
| isclose after b.realize()   | True         | True         | True        |

The same small probe gave these results for 0.08978271484375 and 0.05780029296875 too. The shared isclose formula in tinygrad/mixin/elementwise.py compares abs(a-b) with atol + rtol*abs(b). Its lazy a-(a+epsilon) path loses the intermediate HALF rounding boundary; the shared symbolic rules combine repeated terms. NOOPT=1 does not disable those general rewrites.

For the first value, the wide tolerance is about 9.865625e-7 and the evaluated HALF tolerance is 9.5367431640625e-7. Both are below the retained increment, but a stored subtraction is zero. So changing a comparison register or merely widening the tolerance does not address the missing rounding boundary.

realize() was only a diagnostic. Do not add it to test_isclose, change its inputs or relax its assertion: the original test still fails. A shared rewrite/rounding fix needs separate validation across backends; this probe is not such a fix.

Check the existing special-value cases separately. This test crosses infinity, negative infinity, NaN and zero, with equal_nan both false and true:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_isclose_edge_cases

Ran 1 test in 2.361s
OK
```

The current-runtime command exited normally within 30 seconds. Special-value selection passes this check; the finite lazy-expression failure above remains open.

## Sigmoid: inspect intermediate rounding

Check a recorded accuracy failure, leaving known timeouts alone. The unchanged test at the reconstructed variant checkpoint gives:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sigmoid

Mismatched elements: 25 / 2925 (0.855%)
[2, 22]: 0.2061767578125 (ACTUAL), 0.2059326171875 (DESIRED)
Ran 1 test in 3.173s
FAILED (errors=1)

$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=CPU python test/backend/test_ops.py TestOps.test_sigmoid

Mismatched elements: 13 / 2925 (0.444%)
Ran 1 test in 0.269s
FAILED (errors=1)
```

Both fail, but not on the same elements. Start with the first Rockchip mismatch. Tensor.sigmoid uses `1 / (1 + exp2(x * (-1/log(2))))`. A host reference calculation, rounding each operation to half, gives:

| Step                 | FP16 value      |
| -------------------- | --------------: |
| x                    | -1.349609375    |
| -1/log(2)            | -1.4423828125   |
| MUL                  | 1.9462890625    |
| EXP2                 | 3.853515625     |
| ADD 1                | 4.8515625       |
| RECIPROCAL           | 0.2061767578125 |
| Sigmoid, round once  | 0.2059326171875 |

This is not an NPU trace. It reproduces the first wrong result even with correctly rounded individual operations. Changing EXP2 to compensate is not justified: the composed half rounding already explains this lane.

The 1500 branch has the same test inputs. Its `_dpu_exp2` also uses half arithmetic, not a direct sigmoid implementation. Copying that helper does not establish a fix.

Inspect the UOps before changing anything. The diagnostic stops at program entry, before NPU arithmetic:

```text
8  CONST weakfloat -1.4423828125
9  CAST half
12 MUL half       (input, coefficient)
13 EXP2 half      (product)
14 ADD half       (1, exponential)
15 FDIV half      (1, denominator)
16 STORE
```

The coefficient is already rounded before the renderer sees it. We should not silently replace -1.4423828125 with -1/log(2): that would change the supplied expression. First ask whether keeping this coefficient but widening the intermediate operations is enough.

Recalculate the same 2925 stored test inputs on the host, using the original tolerance:

| Candidate                             | Mismatches |
| ------------------------------------- | ---------: |
| Half rounding after every operation   |         25 |
| Wide intermediates, same coefficient  |          0 |

The wide calculation uses double precision and rounds the final result to half. It is a candidate calculation, not an FP32 or NPU verification. The half calculation reproduces all 25 mismatches in count, so widening is worth testing before changing EXP2's approximation.

Try the candidate in the renderer, preserving the supplied coefficient. Recognize the complete expression, not every FDIV or EXP2:

```diff
 class RockchipRenderer(Renderer):
+  @staticmethod
+  def _pm_widen_sigmoid(u:UOp) -> UOp|None:
+    def constant(x:UOp):
+      while x.op is Ops.CAST: x = x.src[0]
+      return x.arg if x.op is Ops.CONST else None
+    if constant(u.src[0]) != 1 or u.src[1].op is not Ops.ADD: return None
+    one, exponential = u.src[1].src
+    if one.op is Ops.EXP2: one, exponential = exponential, one
+    if constant(one) != 1 or exponential.op is not Ops.EXP2: return None
+    product = exponential.src[0]
+    if product.op is not Ops.MUL or product.dtype != dtypes.half: return None
+    x, coefficient = product.src
+    if constant(x) is not None: x, coefficient = coefficient, x
+    if constant(coefficient) is None: return None
+    # Preserve the supplied coefficient; widen the arithmetic, not the tensor input.
+    exponent = x.cast(dtypes.float).alu(Ops.MUL, coefficient.cast(dtypes.float))
+    denominator = xexp2(exponent).alu(Ops.ADD, UOp.const(1, dtypes.float))
+    return UOp.const(1, dtypes.float).alu(Ops.FDIV, denominator).cast(dtypes.half)
+
   @staticmethod
   def _pm_lower_trunc(x:UOp) -> UOp:
@@
   comparison_matcher = PatternMatcher([
+    # Keep sigmoid's intermediate arithmetic wide until the final half result.
+    (UPat(Ops.FDIV, dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_sigmoid(u)),
```

The first runtime trial failed before comparing results:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sigmoid

NotImplementedError: ROCKCHIP NPU does not support Ops.CMPLT with dtypes.bool
Ran 1 test in 0.294s
FAILED (errors=1)
```

We introduced native xexp2 in render(), after normal dtype rewriting. Its new comparisons did not go through that earlier processing. Move this rule to extra_matcher, alongside the existing native FP32 math decompositions:

```diff
 class RockchipRenderer(Renderer):
@@
   extra_matcher = PatternMatcher([
+    # Keep sigmoid's intermediate arithmetic wide until the final half result.
+    (UPat(Ops.FDIV, dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_sigmoid(u)),
@@
   comparison_matcher = PatternMatcher([
-    # Keep sigmoid's intermediate arithmetic wide until the final half result.
-    (UPat(Ops.FDIV, dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_sigmoid(u)),
```

Retest the unchanged method:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sigmoid

Ran 1 test in 19.186s
OK
```

Replaying these diffs and running the same unchanged test at this checkpoint also passed: 1 test in 19.266s, with normal exit inside the 30-second limit. No test input, tolerance or tinygrad core code changed. This pass covers forward-only execution.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sigmoid_extreme

# 30-second command limit, exit 124. No test summary.
```

This historical attempt should not have been selected: test_sigmoid_extreme explicitly calls gradient(), which FORWARD_ONLY does not disable. Exclude it from this forward-only task; do not restart its timed-out run.

The alternative formula has its own short test, including explicit gradients:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sigmoid_alt_extreme

AssertionError: nan != 0.0 within 7 places (nan difference)
Ran 1 test in 0.497s
FAILED (failures=1)
```

It uses exp(x)/(1+exp(x)), not the expression matched above. FORWARD_ONLY does not remove its explicit gradient calls. This was an out-of-scope test selection, not an outstanding forward-only requirement. Do not investigate its gradient failure in this task.

## Tanh: check the composed expression

Continue with a forward-only accuracy case. test_tanh uses helper_test_op and has no explicit gradient call; FORWARD_ONLY=1 applies here.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tanh

Mismatched elements: 589 / 2925 (20.1%)
[0, 0]: 0.1923828125 (ACTUAL), 0.19287109375 (DESIRED)
Ran 1 test in 2.627s
FAILED (errors=1)
```

This is the reconstructed checkpoint after the sigmoid change. Tensor.tanh is `2*sigmoid(2*x)-1`; passing sigmoid does not establish accuracy after the final subtraction. The 1500 branch uses the same test shape. Next inspect the lowered expression before extending the matcher: we need to know whether the sigmoid pattern survived and where the half rounding occurs.

The program-entry diagnostic stops before arithmetic and shows:

```text
16 MUL half   (x, -2.884765625)
17 EXP2 half
18 ADD half   (1, exponential)
19 FDIV half  (2, denominator)
20 ADD half   (quotient, -1)
21 STORE
```

Our sigmoid matcher requires numerator 1, so it did not match this expression. Widening only the quotient would still round before subtracting 1. Near zero that subtraction can magnify the relative error.

Match the complete `2/(1+EXP2(x*c))-1` form, retaining c from the graph. Use FP32 for the final subtraction as well, then cast once to half. Keep the original numerator-1 sigmoid rule:

```diff
 class RockchipRenderer(Renderer):
@@
   def _pm_widen_sigmoid(u:UOp) -> UOp|None:
@@
-    if constant(u.src[0]) != 1 or u.src[1].op is not Ops.ADD: return None
+    bias = None
+    if u.op is Ops.ADD:
+      division, bias = u.src
+      if bias.op is Ops.FDIV: division, bias = bias, division
+      if division.op is not Ops.FDIV or constant(bias) != -1: return None
+      u = division
+    numerator = 2 if bias is not None else 1
+    if constant(u.src[0]) != numerator or u.src[1].op is not Ops.ADD: return None
@@
-    return UOp.const(1, dtypes.float).alu(Ops.FDIV, denominator).cast(dtypes.half)
+    result = UOp.const(numerator, dtypes.float).alu(Ops.FDIV, denominator)
+    if bias is not None: result = result.alu(Ops.ADD, bias.cast(dtypes.float))
+    return result.cast(dtypes.half)
@@
   extra_matcher = PatternMatcher([
-    # Keep sigmoid's intermediate arithmetic wide until the final half result.
-    (UPat(Ops.FDIV, dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_sigmoid(u)),
+    # Keep sigmoid and tanh's final subtraction wide until the half result.
+    (UPat((Ops.FDIV, Ops.ADD), dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_sigmoid(u)),
```

The unchanged runtime test now passes:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tanh

Ran 1 test in 18.718s
OK
```

Replaying the tutorial diffs and testing that reconstructed backend also passes: test_tanh in 18.333s and test_sigmoid in 19.034s. Each ran separately with FORWARD_ONLY=1 and a 30-second limit.

Next check the existing extreme-input variant. It uses helper_test_op, with no explicit gradient calls:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tanh_extreme

# 30-second command limit, exit 124. No test summary.
```

The basic tanh pass does not cover this variant. Keep it unresolved and skip further timeout runs during the accuracy pass.

## Constant POW: inspect the repeated squares

The sigmoid/tanh change does not cover powers. Run the unchanged test_pow_const at this checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_const

Mismatched elements: 617 / 2925 (21.1%)
[0, 5]: 0.01345062255859375 (ACTUAL), 0.01343536376953125 (DESIRED)
Ran 1 test in 1.084s
FAILED (errors=1)
```

The traceback points to x**8.0. test_pow_const uses helper_test_op, so FORWARD_ONLY applies. Do not substitute test_pow_const_direct: that method explicitly calls gradient.

Check simplify_pow in tinygrad/uop/symbolic.py. For an integer exponent it recursively squares the half-power. For 8, that gives:

```text
x → MUL(x, x) → MUL(square, square) → MUL(fourth, fourth)
```

This is derived from the source rule, not a new TRACE capture. Each intermediate is half, so rounding happens three times. Adding POW to ops_map would not recover the original expression after this rewrite. Our LOG2/EXP2 matcher does not apply either.

The 1500 branch has the same x**8 test. Its precise-product-sum helpers address sums of products; that alone does not establish a fix for repeated squares. Next inspect this MUL chain and compare wider intermediates before adding a backend rule. Keep the tolerance and shared core unchanged; no fix or pass is claimed here yet.

Try widening the complete repeated-square chain, then cast its result once. DEBUG=5 confirms the failing kernel contains c10=c9*c9, c11=c10*c10 and c12=c11*c11. This candidate changes intermediate rounding; it is not a new hardware POW instruction.

```diff
 class RockchipRenderer(Renderer):
+  @staticmethod
+  def _pm_widen_squares(u:UOp) -> UOp|None:
+    source, depth = u, 0
+    while source.op is Ops.MUL and source.dtype == dtypes.half and source.src[0] is source.src[1]:
+      source, depth = source.src[0], depth+1
+    if depth < 2: return None
+    value = source.cast(dtypes.float)
+    for _ in range(depth): value = value.alu(Ops.MUL, value)
+    return value.cast(dtypes.half)
+
+  square_matcher = PatternMatcher([
+    # Repeated squaring rounds once to half, rather than after every product.
+    (UPat(Ops.MUL, dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_squares(u)),
+  ])
+
@@
   def render(self, uops:list[UOp]) -> str:
     # Keep the original UOp order; remove only the temporary sink after lowering.
-    sink = graph_rewrite(UOp.sink(*uops), self.comparison_matcher)
+    sink = graph_rewrite(UOp.sink(*uops), self.square_matcher, bottom_up=True)
+    sink = graph_rewrite(sink, self.comparison_matcher)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_pow_const

# 30-second command limit, exit 124. No test summary.
```

This candidate did not finish the unchanged test. We have no passing accuracy result, so restore the previous path and defer this attempt. Do not restart it with a longer limit:

```diff
 class RockchipRenderer(Renderer):
-  @staticmethod
-  def _pm_widen_squares(u:UOp) -> UOp|None:
-    source, depth = u, 0
-    while source.op is Ops.MUL and source.dtype == dtypes.half and source.src[0] is source.src[1]:
-      source, depth = source.src[0], depth+1
-    if depth < 2: return None
-    value = source.cast(dtypes.float)
-    for _ in range(depth): value = value.alu(Ops.MUL, value)
-    return value.cast(dtypes.half)
-
-  square_matcher = PatternMatcher([
-    # Repeated squaring rounds once to half, rather than after every product.
-    (UPat(Ops.MUL, dtypes.half, name="u"), lambda u: RockchipRenderer._pm_widen_squares(u)),
-  ])
-
@@
   def render(self, uops:list[UOp]) -> str:
     # Keep the original UOp order; remove only the temporary sink after lowering.
-    sink = graph_rewrite(UOp.sink(*uops), self.square_matcher, bottom_up=True)
-    sink = graph_rewrite(sink, self.comparison_matcher)
+    sink = graph_rewrite(UOp.sink(*uops), self.comparison_matcher)
```

## Atanh: inspect the ratio before LOG2

The saved sweep failed test_atanh. Rerun it with the backend built so far; this method has no explicit gradient calls:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_atanh

Mismatched elements: 287 / 2925 (9.81%)
[0, 3]: 0.1817626953125 (ACTUAL), 0.1815185546875 (DESIRED)
[0, 4]: -0.31591796875 (ACTUAL), -0.3154296875 (DESIRED)
Ran 1 test in 3.078s
FAILED (errors=1)
```

DEBUG=5 shows the expression before kernel lowering:

```text
c19 = ((UOp.const(1.0)+c10)*(UOp.const(1.0)+c10*UOp.const(-1.0)).reciprocal()).log2()*UOp.const(0.34657359027997264)
```

This is log2((1+x)/(1-x))*ln(2)/2. The same test in the 1500 branch uses the same shapes and input ranges; its renderer has no dedicated atanh rule. Do not assume a branch pass supplies a direct implementation to copy.

Near zero, both 1+x and 1-x are close to 1. Rounding these and their ratio to half can lose a large fraction of the small logarithm. That is a candidate cause, not yet an isolated measurement. Next trace the lowered arithmetic and check the ratio before changing LOG2's polynomial or adding a wider expression matcher. The original tolerance remains unchanged.

The first failing lane has x=0.1795654296875. A diagnostic wrapper around run_fdiv left every result unchanged and recorded numerator 1.1796875, denominator 0.8203125 and quotient 1.4384765625. A host rounding calculation using those same half operands reproduces 0.1817626953125, versus reference 0.1815185546875. This locates lost precision before LOG2; it is not a host implementation of atanh.

At extra_matcher, an observational print showed LOG2(FDIV(ADD(1,x), SUB(1,x))). Try widening that ratio before tinygrad's native xlog2:

```diff
 class RockchipRenderer(Renderer):
@@
   extra_matcher = PatternMatcher([
+    # Keep atanh's 1+x, 1-x and ratio wide before the native LOG2 decomposition.
+    (UPat(Ops.LOG2, dtypes.half, src=(UPat(Ops.FDIV, src=(
+      UPat(Ops.ADD, src=[UPat.cvar("one"), UPat.var("x", dtypes.half)]),
+      UPat(Ops.SUB, src=(UPat.cvar("one"), UPat.var("x", dtypes.half))))),)),
+     lambda one,x: xlog2(one.cast(dtypes.float).alu(Ops.ADD, x.cast(dtypes.float)).alu(Ops.FDIV,
+       one.cast(dtypes.float).alu(Ops.SUB, x.cast(dtypes.float)))).cast(dtypes.half) if one.arg == 1 else None),
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_atanh

# 30-second command limit, exit 124. No test summary.
```

No passing accuracy result was obtained. Restore the previous matcher; this wider-decomposition attempt remains deferred under the timeout rule:

```diff
 class RockchipRenderer(Renderer):
@@
   extra_matcher = PatternMatcher([
-    # Keep atanh's 1+x, 1-x and ratio wide before the native LOG2 decomposition.
-    (UPat(Ops.LOG2, dtypes.half, src=(UPat(Ops.FDIV, src=(
-      UPat(Ops.ADD, src=[UPat.cvar("one"), UPat.var("x", dtypes.half)]),
-      UPat(Ops.SUB, src=(UPat.cvar("one"), UPat.var("x", dtypes.half))))),)),
-     lambda one,x: xlog2(one.cast(dtypes.float).alu(Ops.ADD, x.cast(dtypes.float)).alu(Ops.FDIV,
-       one.cast(dtypes.float).alu(Ops.SUB, x.cast(dtypes.float)))).cast(dtypes.half) if one.arg == 1 else None),
```

## Log: scale before rounding to half

Move to another short accuracy failure:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log

Mismatched elements: 8 / 2925 (0.274%)
[1, 8]: -0.86962890625 (ACTUAL), -0.86865234375 (DESIRED)
Ran 1 test in 2.750s
FAILED (errors=1)
```

log(x) is LOG2(x) times a constant. We already have run_math(LOG2, dtype=dtypes.float) from the wider POW work. Try retaining that result through the multiplication instead of rounding it to half first. Preserve the supplied coefficient; do not replace it with a more convenient constant. All three stages use the existing NPU helpers:

```text
half x → FP32 LOG2 result → FP32 multiply by the constant → half
```

First try a late matcher for LOG2 times CONST:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
         elif u.op is Ops.CUSTOM:
+          if len(u.arg) == 3 and u.arg[0] == "LOG2_SCALE" and u.dtype == dtypes.half and src_dtypes == [dtypes.half]:
+            logarithm = self.run_math(Ops.LOG2, src_values[0], dtype=dtypes.float)
+            values[u] = self.run_cast(self.run_float_scale(logarithm, u.arg[1]), dtypes.float, dtypes.half)
+            i += 1
+            continue
@@
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Scale LOG2's FP32 result before its final half rounding; preserve the supplied constant.
+    (UPat(Ops.MUL, dtypes.half, src=[UPat(Ops.LOG2, src=(UPat.var("x", dtypes.half),)), UPat.cvar("scale")]),
+     lambda x,scale: UOp(Ops.CUSTOM, src=(x,), arg=("LOG2_SCALE", float(scale.arg), dtypes.half))),
```

The same test still fails with 8 mismatches in 2.795s. Printing the MUL at render entry explains why: its second input is CAST(CONST(0.693359375), half), not a bare CONST. Match that CAST:

```diff
   comparison_matcher = PatternMatcher([
@@
-    (UPat(Ops.MUL, dtypes.half, src=[UPat(Ops.LOG2, src=(UPat.var("x", dtypes.half),)), UPat.cvar("scale")]),
+    (UPat(Ops.MUL, dtypes.half, src=[UPat(Ops.LOG2, src=(UPat.var("x", dtypes.half),)),
+      UPat(Ops.CAST, dtypes.half, src=(UPat.cvar("scale"),))]),
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log

AssertionError: CUSTOM/CUSTOMI arg must be (str, DType), got ('LOG2_SCALE', 0.693359375, dtypes.half)
Ran 1 test in 0.151s
FAILED (failures=1)
```

Now the matcher runs, but CUSTOM only accepts (name, dtype). Put the coefficient in a constant source UOp instead. Check that it really is a constant before reading it for register setup:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          if len(u.arg) == 3 and u.arg[0] == "LOG2_SCALE" and u.dtype == dtypes.half and src_dtypes == [dtypes.half]:
+          if u.arg == ("LOG2_SCALE", dtypes.half) and src_dtypes == [dtypes.half]*2:
+            assert u.src[1].op is Ops.CAST and u.src[1].src[0].op is Ops.CONST
             logarithm = self.run_math(Ops.LOG2, src_values[0], dtype=dtypes.float)
-            values[u] = self.run_cast(self.run_float_scale(logarithm, u.arg[1]), dtypes.float, dtypes.half)
+            values[u] = self.run_cast(self.run_float_scale(logarithm, scalar16(src_values[1][0])), dtypes.float, dtypes.half)
@@
   comparison_matcher = PatternMatcher([
@@
-     lambda x,scale: UOp(Ops.CUSTOM, src=(x,), arg=("LOG2_SCALE", float(scale.arg), dtypes.half))),
+     lambda x,scale: UOp(Ops.CUSTOM, src=(x, scale.cast(dtypes.half)), arg=("LOG2_SCALE", dtypes.half))),
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log

Ran 1 test in 6.285s
OK
```

This is the runtime result. It includes the existing tensor, special-value and scalar cases; it does not count every LOG2 variant as complete.

Check the existing POW matcher too. The child MUL can now become LOG2_SCALE before its parent EXP2 is matched. Keep that combination on the existing wider POW path, rather than rounding the logarithm to half before EXP2:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # A surrounding EXP2 still uses the wider POW path, not a half-rounded scaled logarithm.
+    (UPat(Ops.EXP2, dtypes.half, src=(UPat(Ops.CUSTOM, arg=("LOG2_SCALE", dtypes.half),
+      src=(UPat.var("a", dtypes.half), UPat.var("b", dtypes.half))),)),
+     lambda a,b: UOp(Ops.CUSTOM, src=(a,b), arg=("EXP2_MUL_LOG2", dtypes.half))),
```

The runtime regression run of test_log10, test_mul, test_pow_neg_inf_frac_exponent, test_pow_zero_exponent and test_pow_zero_tensor passed: 5 tests in 7.685s. These short POW cases are not full POW coverage; the known timeouts stay deferred.

Replay the diffs up to here and run the reconstructed checkpoint, not the later runtime:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_log TestOps.test_log10 TestOps.test_mul TestOps.test_pow_neg_inf_frac_exponent TestOps.test_pow_zero_exponent TestOps.test_pow_zero_tensor

Ran 6 tests in 15.393s
OK
```

## GELU: inspect the composed expression

The LOG fix passes, but that does not settle the other composed functions. Run the unchanged GELU test at this checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_gelu

Mismatched elements: 224 / 2925 (7.66%)
[0, 34]: -0.0521240234375 (ACTUAL), -0.05218505859375 (DESIRED)
Ran 1 test in 3.625s
FAILED (errors=1)
```

The test tries approximate="tanh" before approximate="none". An observational wrapper printing that argument confirms the first case fails; the erf-based case has not run. It leaves the results unchanged and reproduces the same 224 mismatches in 3.522s.

The 1500 branch has the same two test cases and no dedicated GELU handler. The shared formula is:

```text
0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x³)))
```

Inspecting the EXP2 input at render entry gives this shortened tree. All arithmetic shown is half; x is the input LOAD:

```text
MUL(
  MULACC(CAST(0.044708251953125), MUL(MUL(x, x), x), x),
  CAST(-2.302734375))
```

That diagnostic still fails with 224 mismatches in 4.134s. So the standalone TANH pass did not make this whole expression wide: the polynomial and exponent input still round to half. Next inspect the surrounding division and final multiplication before deciding which intermediates to widen. This trace identifies a candidate rounding problem, not a verified fix; do not replace the supplied coefficients or claim the erf case passed.

The surrounding STORE expression is `x / (1 + EXP2(product))`. The observer reproduces the same failure in 3.915s. Our sigmoid helper requires a constant numerator, so it misses this weighted form. First try widening the exponential and division, without changing the polynomial or coefficients:

```diff
 class RockchipRenderer(Renderer):
@@
   def _pm_widen_sigmoid(u:UOp) -> UOp|None:
@@
-    if constant(u.src[0]) != numerator or u.src[1].op is not Ops.ADD: return None
+    if (bias is not None and constant(u.src[0]) != numerator) or u.src[1].op is not Ops.ADD: return None
@@
-    result = UOp.const(numerator, dtypes.float).alu(Ops.FDIV, denominator)
+    result = u.src[0].cast(dtypes.float).alu(Ops.FDIV, denominator)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_gelu

Mismatched elements: 133 / 2925 (4.55%)
[0, 34]: -0.052093505859375 (ACTUAL), -0.05218505859375 (DESIRED)
Ran 1 test in 21.802s
FAILED (errors=1)
```

Fewer mismatches, but not a pass. The polynomial still rounds before this wider exponential. Try retaining FP32 through its ADD/MUL/MULACC nodes too; stop at the existing inputs and preserve the supplied constants. Expand MULACC into FP32 MUL and ADD, since the native FP32 path does not advertise fused MULACC:

```diff
   def _pm_widen_sigmoid(u:UOp) -> UOp|None:
@@
     # Preserve the supplied coefficient; widen the arithmetic, not the tensor input.
-    exponent = x.cast(dtypes.float).alu(Ops.MUL, coefficient.cast(dtypes.float))
+    def widen(v:UOp) -> UOp:
+      if v.dtype == dtypes.half and v.op in (Ops.ADD, Ops.MUL, Ops.MULACC):
+        operands = tuple(widen(s) for s in v.src)
+        if v.op is Ops.MULACC: return operands[0].alu(Ops.MUL, operands[1]).alu(Ops.ADD, operands[2])
+        return operands[0].alu(v.op, *operands[1:])
+      return v.cast(dtypes.float)
+    exponent = widen(x).alu(Ops.MUL, coefficient.cast(dtypes.float))
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_gelu

Mismatched elements: 12 / 2925 (0.41%)
[3, 38]: -0.05474853515625 (ACTUAL), -0.0548095703125 (DESIRED)
Ran 1 test in 20.677s
FAILED (errors=1)
```

The remaining error may come from the constants, which were already rounded before the matcher. A host diagnostic using the same seeded HALF inputs and double-precision arithmetic gives 12 mismatches with the observed coefficients, including the same first five indices. Using the original formula's 0.044715 and `-2*sqrt(2/pi)/log(2)` gives zero. This is a diagnostic calculation, not an NPU test or a fallback.

So widening the intermediates helped, but cannot recover constants already rounded to half. Keep GELU failed. Do not substitute different coefficients just because they pass: a later fix must preserve the original expression before that rounding, without changing tinygrad core. The erf-based case still has not run.

Replaying this checkpoint reproduces the same 12 mismatches in 21.985s. The existing test_sigmoid regression still passes in 19.793s. All 456 tutorial hunks replay and runtime lint passes. Mypy still reports the 47 existing errors in ops_rockchip_ref.py, not this runtime.

Can we move this matcher earlier? The current renderer exposes extra_matcher and render(), but extra_matcher runs after final symbolic rewriting, float-operand casts and decomposition. Symbolic rewriting canonicalizes CAST(CONST) through const_like, which rebuilds the constant at its stated dtype. There is no earlier renderer matcher hook here. Moving the same rule between these two hooks cannot recover the original GELU coefficients. Under our no-core-change constraint, leave this remaining precision failure explicit rather than guessing coefficients from their rounded values.

Check TANH too. The reconstructed checkpoint fails with 249 / 2925 mismatches in 19.054s. The broader rule now accepts the child `2 / denominator`, rounding it before the parent `-1` can match. Allow a nonconstant numerator for the weighted case, but leave constant 2 for the existing TANH rule:

```diff
   def _pm_widen_sigmoid(u:UOp) -> UOp|None:
@@
-    if (bias is not None and constant(u.src[0]) != numerator) or u.src[1].op is not Ops.ADD: return None
+    if u.src[1].op is not Ops.ADD: return None
+    # Leave constant 2 / denominator for the enclosing TANH subtraction to match.
+    if (bias is not None or constant(u.src[0]) is not None) and constant(u.src[0]) != numerator: return None
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_tanh

Ran 1 test in 19.617s
OK
```

This is the reconstructed checkpoint after the guard fix; the working runtime also passed in 21.022s. All 457 hunks replay and runtime lint passes. The GELU coefficient issue remains separate from this repaired regression.

## ASINH: bounded rerun

The 1500 branch's test_asinh is not equivalent to ours: its negative-large-input case allows atol=1e-2 and rtol=2e-2, and it lacks our very-large-negative and explicit [-1, 0, 1] cases. Keep our current inputs and tolerances.

The shared formula is `sign(x)*log(abs(x) + sqrt(x*x + 1))`. Before proposing another matcher, run the unchanged test with the reconstructed backend:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_asinh

# 30-second command limit, exit 124. No test summary.
```

No new accuracy result. Defer this case under the timeout rule; the older numerical failure is not cleared by this run.

## ERF: inspect the polynomial constants

The old sweep stopped at unsupported FP32 MUL. With that support now present, run the existing test at this checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_erf

Mismatched elements: 560 / 2925 (19.1%)
[0, 0]: 0.21875 (ACTUAL), 0.2176513671875 (DESIRED)
Ran 1 test in 23.275s
FAILED (errors=1)
```

This fails the first tensor case. The extreme, [-1, 0, 1] and scalar cases are not reached. The 1500 branch uses the same first case but lacks the explicit [-1, 0, 1] case; its runtime and renderer have no dedicated ERF handler.

The native expression is `s*(1 - t*P(t)*exp(-x*x))`, where `s=sign(x)` (using +1 at zero) and `t=1/(1+0.3275911*abs(x))`. Inspecting the render input shows these half-rounded polynomial constants:

```text
P coefficients: 1.0615234375, -1.453125, 1.421875, -0.284423828125, 0.2548828125
t coefficient: 0.32763671875
```

The unchanged diagnostic run still gives 560 mismatches in 26.087s. Before adding a widening rule, check x=0: t=1, so the result would be `1-sum(P coefficients) = -0.000732421875` even with exact arithmetic after rounding the coefficients. ERF(0) should be zero.

A host diagnostic with the seeded HALF inputs, these coefficients and double-precision arithmetic gives 571 mismatches. The original formula's coefficients give zero mismatches on the same inputs. This is not an NPU implementation or proof of all-input accuracy; it shows that widening alone is not enough. As with GELU, the original constants are already lost before our renderer hooks. Keep this failure explicit rather than replacing constants by guessing which formula the rounded graph came from.

## Recheck short FP32 variants

Not every old failure still needs code. The earlier FP32 gate failures predate our FP32 arithmetic support. Recheck short existing variants at this reconstructed checkpoint, without DEFAULT_FLOAT=HALF:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_add3 TestOps.test_scalar_sub TestOps.test_scalar_rsub TestOps.test_mul_naninf

test_add3 ... ok
test_scalar_sub ... ok
test_scalar_rsub ... ok
test_mul_naninf ... ok
Ran 4 tests in 1.784s
OK
```

These pass without another gate or matcher change. This covers only these four methods, not the whole saved FP32 failure group or every arithmetic variant.

Then check broadcasting and scalar multiplication at the same checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_broadcasted_add TestOps.test_broadcasted_add_2 TestOps.test_scalar_mul

test_broadcasted_add ... ok
test_broadcasted_add_2 ... ok
test_scalar_mul ... ok
Ran 3 tests in 1.781s
OK
```

The mixed-dtype maximum/minimum methods also pass at this FP32 checkpoint. They include integer limits, booleans and mixed inputs, not only the first floating tensor case:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_maximum TestOps.test_minimum

Ran 2 tests in 1.376s
OK
```

Check small matrix products too. These tests previously stopped at the FP32 arithmetic gates; the old 9x9 record had no useful traceback:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_gemm TestOps.test_9_gemm TestOps.test_small_gemm_range TestOps.test_small_gemm_eye

test_small_gemm ... ok
test_9_gemm ... ok
test_small_gemm_range ... ok
test_small_gemm_eye ... ok
Ran 4 tests in 1.344s
OK
```

These are reconstructed-checkpoint forward results with the original inputs. They do not cover larger GEMMs or backward tests.

Check padding and small reductions next, with DEBUG=2 to see the dispatched kernels:

```bash
$ DEBUG=2 NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_gemm_padded TestOps.test_sum_tiny TestOps.test_sum_relu TestOps.test_const_reduce

Ran 4 tests in 2.697s
OK
```

The log shows ROCKCHIP kernels r_16_16_16, r_2_4_2, r_60 and r_9. Constant reduction also dispatches E_9 to fill its input. These are completed bounded commands, not results inferred from removing the old FP32 gates. Larger reductions and known timeout cases remain unverified.

## ELU: subtract before the half CAST

The current test keeps the 1500 branch's ordinary, alpha=0.1 and scalar cases, and adds a large-positive case. Run it unchanged:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_elu

Mismatched elements: 87 / 2925 (2.97%)
[0, 6]: -0.220703125 (ACTUAL), -0.220947265625 (DESIRED)
Ran 1 test in 16.194s
FAILED (errors=1)
```

ELU's negative branch uses `alpha*(exp(x)-1)`. The shared exp implementation already computes in FP32, but casts to half before returning. Subtracting 1 after that rounding loses precision near zero. Unlike the GELU coefficient problem, the wider value still exists in the graph. Try delaying that CAST until after subtraction:

```text
FP32 result → CAST(half) → SUB 1
FP32 result → SUB 1 → CAST(half)
```

Match both SUB(1) and ADD(-1), using the existing FP32 NPU subtraction:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Keep cancellation against 1 wide when the input was already computed in FP32.
+    (UPat(Ops.SUB, dtypes.half, src=(UPat(Ops.CAST, dtypes.half, src=(UPat.var("x", dtypes.float),)),
+      UPat(Ops.CAST, dtypes.half, src=(UPat.const(1.),)))), lambda x: x.alu(Ops.SUB, x.const_like(1)).cast(dtypes.half)),
+    (UPat(Ops.ADD, dtypes.half, src=[UPat(Ops.CAST, dtypes.half, src=(UPat.var("x", dtypes.float),)),
+      UPat(Ops.CAST, dtypes.half, src=(UPat.const(-1.),))]), lambda x: x.alu(Ops.SUB, x.const_like(1)).cast(dtypes.half)),
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_elu

# 30-second command limit, exit 124. No test summary.
```

No completed accuracy result. Do not claim the first case passed merely because the failure was not printed. Restore the previous matcher and defer this candidate under the timeout rule:

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
-    # Keep cancellation against 1 wide when the input was already computed in FP32.
-    (UPat(Ops.SUB, dtypes.half, src=(UPat(Ops.CAST, dtypes.half, src=(UPat.var("x", dtypes.float),)),
-      UPat(Ops.CAST, dtypes.half, src=(UPat.const(1.),)))), lambda x: x.alu(Ops.SUB, x.const_like(1)).cast(dtypes.half)),
-    (UPat(Ops.ADD, dtypes.half, src=[UPat(Ops.CAST, dtypes.half, src=(UPat.var("x", dtypes.float),)),
-      UPat(Ops.CAST, dtypes.half, src=(UPat.const(-1.),))]), lambda x: x.alu(Ops.SUB, x.const_like(1)).cast(dtypes.half)),
```

## Hardswish: keep the clamped product wide

First check the short ReLU variants with default FP32 inputs. At this reconstructed checkpoint, test_relu, test_relu_exact, test_relu_maximum_exact, test_leaky_relu and test_relu6 pass: 5 tests in 4.568s, with NOOPT=1 and FORWARD_ONLY=1.

HALF Hardswish still fails:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_hardswish

Mismatched elements: 34 / 2925 (1.16%)
[4, 41]: 0.7470703125 (ACTUAL), 0.748046875 (DESIRED)
Ran 1 test in 1.754s
FAILED (errors=1)
```

The 1500 branch has the same ordinary and scalar cases; ours also checks [-3, 3]. The shared formula is `x * clamp(x+3, 0, 6) * (1/6)`. Printing the render input shows this shortened tree, with half arithmetic:

```text
lower = MAX(0, ADD(x, 3))
clamp = WHERE(CMPLT(lower, 6), lower, 6)
result = MUL(MUL(x, clamp), 0.1666259765625)
```

That observation reproduces the same 34 mismatches in 1.636s. Try FP32 intermediates through the shift, clamp and both products, casting only the final result to half. Keep the supplied constants, including the rounded scale; do not replace it with an exact 1/6.

Match this structure rather than every MUL. Check that both bounds, the bias and scale are constants and that the shifted input is the same x used by the product:

```diff
 class RockchipRenderer(Renderer):
+  @staticmethod
+  def _pm_widen_clamped_product(x:UOp, clamp:UOp, scale:UOp) -> UOp|None:
+    condition, lower, upper = clamp.src
+    if condition.op is not Ops.CMPLT or condition.src != (lower, upper) or lower.op is not Ops.MAX: return None
+    floor, shifted = lower.src
+    if floor.op is Ops.ADD: floor, shifted = shifted, floor
+    if shifted.op is not Ops.ADD or x not in shifted.src: return None
+    bias = shifted.src[1] if shifted.src[0] is x else shifted.src[0]
+    def is_constant(v:UOp) -> bool:
+      while v.op is Ops.CAST: v = v.src[0]
+      return v.op is Ops.CONST
+    if not all(is_constant(v) for v in (floor, upper, bias, scale)): return None
+    wide = x.cast(dtypes.float)
+    bounded = wide.alu(Ops.ADD, bias.cast(dtypes.float)).maximum(floor.cast(dtypes.float))
+    cap = upper.cast(dtypes.float)
+    bounded = bounded.lt(cap).where(bounded, cap)
+    return wide.alu(Ops.MUL, bounded).alu(Ops.MUL, scale.cast(dtypes.float)).cast(dtypes.half)
+
@@
   extra_matcher = PatternMatcher([
+    # Keep a constant-bounded product wide through its final scale.
+    (UPat(Ops.MUL, dtypes.half, src=[UPat(Ops.MUL, src=[UPat.var("x", dtypes.half), UPat(Ops.WHERE, name="clamp")]),
+      UPat.var("scale")]), lambda x,clamp,scale: RockchipRenderer._pm_widen_clamped_product(x, clamp, scale)),
```

The first run stops before submission: `AttributeError: 'UOp' object has no attribute 'lt'`, in 0.149s. Use the UOp ALU API:

```diff
   def _pm_widen_clamped_product(x:UOp, clamp:UOp, scale:UOp) -> UOp|None:
@@
-    bounded = bounded.lt(cap).where(bounded, cap)
+    bounded = bounded.alu(Ops.CMPLT, cap).where(bounded, cap)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_hardswish

Ran 1 test in 2.091s
OK
```

This runtime result includes the tensor, scalar and [-3, 3] cases. The arithmetic and final conversion use the existing NPU helpers; no new register mode or CPU tensor calculation was added.

Replay the diffs and check nearby operations at the reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_hardswish TestOps.test_relu6 TestOps.test_mul TestOps.test_maximum

Ran 4 tests in 4.966s
OK
```

All 462 hunks replay and runtime lint passes. This fixes Hardswish's recorded HALF failure, not every activation or every dtype/optimization configuration.

## Hardsigmoid: the boundary case still matters

The reconstructed HALF test_hardsigmoid passes in 2.768s. Now run its extreme variant rather than counting the ordinary pass as complete:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_hardsigmoid_extreme

Mismatched elements: 2 / 6 (33.3%)
[1]: 0.0001220703125 (ACTUAL), 0.0 (DESIRED)
[2]: 0.0167236328125 (ACTUAL), 0.0166015625 (DESIRED)
Ran 1 test in 3.725s
FAILED (errors=1)
```

This reaches the final [-3.1, -3, -2.9, 2.9, 3, 3.1] case. The 1500 branch lacks this boundary case and the explicit huge inputs, so its shorter test cannot establish that these pass.

The shared formula is clamp(alpha*x+0.5, 0, 1). With the half-rounded alpha, the fused calculation at x=-3 is:

```text
alpha = 0.1666259765625
alpha*(-3) + 0.5 = 0.0001220703125
```

That agrees with the observed NPU result. Check tinygrad CPU under the same settings before changing MULACC:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=CPU python test/backend/test_ops.py TestOps.test_hardsigmoid_extreme

Mismatched elements: 2 / 6 (33.3%)
Ran 1 test in 0.230s
FAILED (errors=1)
```

CPU gives the same two wrong values. The shared decomposition fuses MUL+ADD when MULACC is advertised. Splitting it here merely to change the rounding would conflict with the fused operation we implemented earlier; replacing alpha would change the supplied constant. Leave this HALF boundary failure open, without changing core or the test tolerance.

Default FP32 is a separate result. Both methods pass at the reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_hardsigmoid TestOps.test_hardsigmoid_extreme

Ran 2 tests in 5.128s
OK
```

So this is not complete across dtypes: FP32 passes, while HALF still fails the boundary variant.

## Softsign: recheck both dtypes

The native formula is `x/(1+abs(x))`; the 1500 branch has the same ordinary, scalar and [-1, 0, 1] cases. No new matcher is needed at this reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_softsign TestOps.test_softsign_exact

Ran 2 tests in 3.827s
OK

$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_softsign TestOps.test_softsign_exact

Ran 2 tests in 4.109s
OK
```

Both bounded forward commands finish normally. This covers these two methods with HALF and default FP32, not other activation formulas.

## SiLU: check the weighted sigmoid path

SiLU calls swish, whose formula is x*sigmoid(x). The 1500 branch has the same tensor and scalar cases. Our weighted-sigmoid matcher was introduced while investigating GELU; verify this other caller instead of assuming it benefits too:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_silu

Ran 1 test in 21.339s
OK
```

This is a reconstructed-checkpoint pass with the current matcher, not a before/after baseline for SiLU.

Run the separately named swish method as well, in its own bounded command:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_swish

Ran 1 test in 19.898s
OK
```

Both HALF methods pass. Default FP32 and the other sigmoid-derived functions still need their own results; these do not clear the deferred GELU/ELU failures.

## Quick GELU: inspect the combined coefficient

Quick GELU uses `x*sigmoid(1.702*x)`, not SiLU's `x*sigmoid(x)`. The 1500 branch has the same ordinary and scalar tests. At our reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_quick_gelu

Mismatched elements: 92 / 2925 (3.15%)
[0, 24]: -0.105712890625 (ACTUAL), -0.1055908203125 (DESIRED)
Ran 1 test in 21.229s
FAILED (errors=1)
```

Print the input when the weighted-sigmoid matcher fires. It already contains a combined constant:

```text
FDIV(x, ADD(1, EXP2(MUL(x, -2.455078125))))
```

The observer leaves the result unchanged: 92 mismatches in 20.501s. A double-precision host calculation of this exact expression, rounded to half at the end, also gives 92 mismatches with the same first five indices. This diagnostic does not run inside the backend.

The original two scales and intermediate rounding are no longer separate in this graph. Widening cannot reconstruct them from -2.455078125. Keep the failure, rather than replacing the supplied coefficient with one chosen to match Torch. This is another reason the SiLU pass cannot count Quick GELU as complete.

Default FP32 passes at the same reconstructed checkpoint:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_quick_gelu

Ran 1 test in 20.020s
OK
```

The HALF failure and untested extreme variant remain open. No runtime change was made for this diagnostic.

## SUM: check the requested output dtype

The default-FP32 reduction now passes at the reconstructed checkpoint, including its axes, keepdim, scalar and invalid-axis checks:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sum

Ran 1 test in 0.904s
OK
```

Its dtype variant gets further than the old FP32 gate failure:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_sum_dtype_arg

NotImplementedError: ROCKCHIP NPU does not support Ops.ADD with dtypes.double
Ran 1 test in 0.330s
FAILED (errors=1)
```

The traceback is inside the explicitly requested float64 sum. The test enters that case only when renderer.supported_dtypes() includes double. Our inherited list includes it, but our arithmetic does not implement it.

The 1500 renderer advertises only HALF and INT16, so the same test skips its conditional double case there. That is not an FP64 implementation to port. Keep this uncovered dtype visible: removing double from the list would make the method pass without verifying FP64, while casting the accumulator to FP32 would lose the requested precision. General SUM support is therefore still incomplete across dtypes.

The ordinary HALF test_sum also passes at this checkpoint in 1.050s with DEFAULT_FLOAT=HALF. Neither ordinary pass clears the explicit double failure above.

## Product and empty reductions: forward-only recheck

The earlier product result included backward work. Recheck the original method forward-only at this reconstructed checkpoint, alongside its invalid-dtype argument check and empty SUM shapes:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_prod TestOps.test_prod_dtype_arg TestOps.test_sum_with_zeros_shape

Ran 3 tests in 1.172s
OK
```

test_prod_dtype_arg only checks rejection of an invalid dtype argument; it is not evidence of FP64 product support. The known cumulative-product timeout remains deferred.

Then check mean's axis handling and the small variance edge cases. These methods use helper_test_op and have no direct backward calls:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_mean TestOps.test_mean_axis TestOps.test_mean_zero_axis TestOps.test_var_one_in_axis

Ran 4 tests in 1.841s
OK
```

The variance method checks one-element axes and several correction values, including invalid degrees of freedom. Larger variance/std tests are not covered by this command.

The matching small STD method adds SQRT, so run it separately:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_std_one_in_axis

Ran 1 test in 3.380s
OK
```

This reconstructed-checkpoint result covers that method only, not the timed-out general SQRT test.

## Cumulative sum and indexing: forward-only recheck

Next check the small cumulative sum and empty axes, then bool indices. Run the unchanged methods at this reconstructed checkpoint, with default FP32:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_small_cumsum TestOps.test_cumsum_zero_axis TestOps.test_gather_bool_index

Ran 3 tests in 0.319s
OK
```

The bool-index method includes converting nonzero floating values to bool, then to integer indices. Now check the full gather method, including negative axes and infinity values:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_gather

Ran 1 test in 1.171s
OK
```

Scatter writes through indices instead. Its ADD variant also checks infinity and NaN:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scatter TestOps.test_scatter_add

Ran 2 tests in 6.948s
OK
```

No new code was needed for these six methods. These are targeted passes, not coverage of every cumulative or scatter-reduction variant. Each command used the 30-second limit; known timeout cases remain deferred.

## Scatter exceptions: check which side failed

The saved sweep reported a missing exception. Rerun that method first:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scatter_reduce_errors

  with self.assertRaises(expected) as torch_cm:
AssertionError: RuntimeError not raised
Ran 1 test in 0.011s
FAILED (failures=1)
```

This stops in Torch, before the tinygrad call. The dtype-mismatch case casts x to half and leaves src unchanged. But prepare_test_op already creates both inputs with dtypes.default_float, so under DEFAULT_FLOAT=HALF both are half. There is no dtype mismatch to reject.

The 1500 branch has the same test. Our shared scatter validation already rejects unequal input dtypes; adding a Rockchip register mode cannot fix this reference-side failure. Try the unchanged test with default FP32, so x.half() and src really have different dtypes:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_scatter_reduce_errors TestOps.test_scatter_no_reduce_tensor_src

Ran 2 tests in 0.013s
OK
```

These check argument rejection, not NPU arithmetic. Keep the HALF configuration failure separate from missing backend support; no test or core change was made.

## Reference dtype cases: default FP32 recheck

Three other saved failures also involve the default dtype: bitcast views three floating values as INT32, stack compares the scalar 3.14 at a tight tolerance, and normalize_int compares against an explicit Torch float() conversion. Check the original methods with default FP32 before proposing another numerical fix:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_bitcast TestOps.test_stack TestOps.test_normalize_int

Ran 3 tests in 3.418s
OK
```

These passes are from the reconstructed tutorial checkpoint. They do not erase the earlier HALF results or establish every BITCAST/normalization variant. No implementation diff was needed.

## Deferred timeout: acos


The next bounded accuracy rerun was test_atan. It uses helper_test_op, so FORWARD_ONLY applies:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_atan

# 30-second command limit, exit 124. No test summary.
```

The shared expression is asin(x/sqrt(1+x*x)); there is no separate ATAN UOp to enable. This run gives no new accuracy result. Skip further test_atan runs during this pass, just like the known acos timeout below.


Skip test_acos during this accuracy pass. The observations below are from the earlier investigation, not a request to restart its timed-out run.

The saved sweep had no useful traceback for test_acos. With the reconstructed backend, the unchanged command now reaches the 30-second limit without an assertion result:

```bash
$ TRACE=1 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_acos

37 Ops.SQRT dtypes.half ... [dtypes.half]
# Stopped at the 30-second command limit (exit 124); no test summary.
```

TRACE shows one lane at a time because SQRT is not in the batching allowlist. This is a timeout, not evidence that acos is correct or incorrect.

The shared formula in tinygrad/mixin/elementwise.py is acos(x) = pi/2 - asin(x). Its asin uses a polynomial P and sqrt(1-abs(x)). Check a small set of stored HALF inputs before proposing new registers:

| Stored input    | tinygrad CPU   | ROCKCHIP       | Torch          |
| --------------- | -------------- | -------------- | -------------- |
| -0.9990234375   | 3.095703125    | 3.09765625     | 3.09765625     |
| -0.990234375    | 3              | 3              | 3.001953125    |
| 0.990234375     | 0.139770507813 | 0.1396484375   | 0.139892578125 |
| 0.9990234375    | 0.044189453125 | 0.0439453125   | 0.044189453125 |

Unlike the isclose example, these outputs do not all agree between tinygrad CPU and ROCKCHIP. Near +1, subtracting two values near pi/2 can expose intermediate rounding.

Can we reuse the same polynomial without that final cancellation? Let a = abs(x) and q = sqrt(1-a)*P(a). Substituting the existing asin formula gives:

```text
acos(x) = WHERE(x < 0, pi - q, q)
```

A separate Tensor probe ran that expression on the NPU, without changing the runtime matcher or the existing test. At 0.990234375 it gave 0.1397705078125; at 0.9990234375 it gave 0.044189453125. The probe also checked ±1, ±0.9, ±0.5, zero and an out-of-domain input 2, which remained NaN.

This is a candidate, not an implemented fix or a full test_acos pass. Next check its rounding across the input range and identify a precise lowering pattern. Do not replace the existing test with these few points or widen its tolerance.
