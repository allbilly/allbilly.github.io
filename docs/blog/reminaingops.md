

Next implement Ops.SHR. UINT32 needs zeros at the top; INT32 needs copies of its sign bit.

For `x >> n`, let `r=n%8`. Each output byte gets `floor(u/2^r)` from the current source byte, plus the low r bits of the next higher byte shifted left by 8-r. We can reuse the SHL correction with left shift 8-r, then change which columns the final convolution selects.

For `0xABCD1234 >> 4`:

| Stage             | Op / formula                   | Dtype       | Byte 0 | Byte 1 | Byte 2 | Byte 3 |
|-------------------|--------------------------------|-------------|-------:|-------:|-------:|-------:|
| Original input    | `0xABCD1234`                   | raw bytes   | `0x34` | `0x12` | `0xCD` | `0xAB` |
| CNA signed read   | s                              | INT8        | 52     | 18     | -51    | -85    |
| Task 1            | `q = floor((s+8)/16)`          | INT8        | 3      | 1      | -3     | -5     |
| CNA unsigned read | u                              | UINT8       | 52     | 18     | 205    | 171    |
| CNA input CVT     | `u-128`                        | INT8        | -76    | -110   | 77     | 43     |
| Task 2            | `carry = floor(u/16)`          | INT8        | 3      | 1      | 12     | 10     |
| Final conv        | `16*s-256*q` per source byte   | accumulator | 64     | 32     | -48    | -80    |
| Final conv, UINT  | next byte's `16*s-256*q`       | accumulator | 32     | -48    | -80    | 0      |
| Final conv, UINT  | add current carry              | INT8        | 35     | -47    | -68    | 10     |
| UINT32 result     | `0x0ABCD123`                   | raw bytes   | `0x23` | `0xD1` | `0xBC` | `0x0A` |
| INT32 result      | sign-fill gives `0xFABCD123`   | raw bytes   | `0x23` | `0xD1` | `0xBC` | `0xFA` |

0. Keep the original four bytes at scratch offset 0, lowest byte first. Reading them as INT8 or UINT8 changes their interpretation, not their bits.

1. First task: CNA reads signed INT8. Convolution multiplies by 2, then output CVT adds 1 and shifts right by 5 with rounding. This prepares the correction q:

   ```text
   q = round_nearest_even((2*s + 1)/32) = floor((s + 8)/16)
   ```

   Write duplicate q lanes at scratch offsets 16, 20 and 24. The final convolution can then use two -128 weights instead of one -256 weight, which does not fit INT8.

2. Second task: CNA reads the original bytes as UINT8. Its input converter subtracts 128 so they fit signed INT8. Convolution multiplies by 2, then output CVT adds 241 and shifts right by 5:

   ```text
   carry = round_nearest_even((2*(u - 128) + 241)/32)
         = round_nearest_even((2*u - 15)/32)
         = floor(u/16)
   ```

   Store these four carry lanes at scratch offset 48. The bias makes the rounded CVT shift give an exact integer right shift.

3. For INT32, one extra task computes `sign = floor(signed_high_byte/128)`: -1 for negative inputs, 0 otherwise. Convolution weight 2, output offset -127 and shift 8 do this on the NPU. Store copies at offset 80. UINT32 skips this task.

4. The final convolution reads the original signed bytes, q and carry directly from scratch. It computes `16*s - 128*q - 128*q` for the next higher byte and adds the current byte's carry, all in one task:

   ```text
   out[0] =  3 + (16*18  - 256*1)    =  35 = 0x23
   out[1] =  1 + (16*-51 - 256*-3)   = -47 = 0xD1
   out[2] = 12 + (16*-85 - 256*-5)   = -68 = 0xBC
   out[3] = 10                       =  10 = 0x0A  (UINT32)
   out[3] = 10 + 16*sign             =  -6 = 0xFA  (INT32)
   ```

5. Read the four output bytes as UINT32 or INT32. They give `0x0ABCD123` or `0xFABCD123`; no CPU byte assembly is needed.

So this example uses three tasks for UINT32, four for INT32. The output CVT shift register does the right shifts; convolution does the correction and neighboring-byte routing. For shifts divisible by 8, skip q and carry: convolution routes whole bytes with zero or sign fill.

No Python sign check or lane routing is needed. The sign copies live at scratch offset 80, so let the shared task helper read a 96-channel tile:

```diff
-  def conv_shl_task(self, weights:list[list[int]], out_addr:int, offset:int=0, shift:int=0, unsigned:bool=False, features:bool=False) -> None:
-    k = 64 if features else 32
+  def conv_shl_task(self, weights:list[list[int]], out_addr:int, offset:int=0, shift:int=0,
+                    unsigned:bool=False, features:bool=False, channels:int=64) -> None:
+    k = channels if features else 32
```

Keep the original word at scratch offset 0. Tasks writing q, carry and sign leave it unchanged:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def conv_shr_word(self, raw:bytes, amount:int, signed:bool=False) -> bytes:
+    output_addr = self.dev.output_mem.dma_addr
+    to_mv(self.dev.input_buf, 128)[:] = raw+bytes(128-len(raw))
+    residual, displacement = amount%8, amount//8
+    if residual == 0 and not signed:
+      self.conv_shl_task([[int(j==i+displacement) for j in range(4)] for i in range(4)], output_addr)
+    else:
+      if residual:
+        # The next higher byte supplies its low r bits, shifted left by 8-r.
+        self.conv_shl_task([[2*int(j==i%4) for j in range(4)] for i in range(12)],
+                          self.dev.input_mem.dma_addr+16, offset=-1 if residual == 1 else 1, shift=residual+1)
+        # The current byte supplies floor(unsigned_byte / 2^r).
+        self.conv_shl_task([[2*int(j==i) for j in range(4)] for i in range(4)],
+                          self.dev.input_mem.dma_addr+48, offset=256-(2**residual-1), shift=residual+1, unsigned=True)
+      if signed:
+        # floor(signed_high_byte / 128) gives -1 or 0, without a CPU sign check.
+        self.conv_shl_task([[0, 0, 0, 2] for _ in range(4)], self.dev.input_mem.dma_addr+80, offset=-127, shift=8)
+      channels = 96 if signed else 64
+      weights = [[0]*channels for _ in range(4)]
+      for i in range(4):
+        j = i+displacement
+        if j >= 4:
+          if signed: weights[i][80] = 1
+        elif residual == 0:
+          weights[i][j] = 1
+        else:
+          weights[i][48+j] = 1
+          if j < 3:
+            if residual == 1:
+              weights[i][j+1] = -128
+              weights[i][17+j], weights[i][21+j], weights[i][25+j] = 127, 127, 2
+            else:
+              weights[i][j+1] = 2**(8-residual)
+              weights[i][17+j] = weights[i][21+j] = -128
+          elif signed:
+            # The missing higher byte is the sign byte; split +128 into two weights.
+            if residual == 1: weights[i][80] = weights[i][81] = 64
+            else: weights[i][80] = 2**(8-residual)
+      self.conv_shl_task(weights, output_addr, features=True, channels=channels)
+    return bytes(to_mv(self.dev.output_buf, 4))
```


As with SHL, the wrapper only copies the original and final word's storage:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_u32_shr(self, a:list, b:list, dtype:DType) -> list:
+    if not b or not all_same(b) or not 0 <= b[0] <= 31:
+      raise NotImplementedError("ROCKCHIP UINT32 SHR requires one uniform shift count in 0..31")
+    fmt = "<I" if dtype == dtypes.uint else "<i"
+    result:list = []
+    for x in a:
+      raw = self.conv_shr_word(struct.pack(fmt, x), int(b[0]), signed=dtype == dtypes.int)
+      result.append(struct.unpack(fmt, raw)[0])
+    return result
```


Advertise SHR and route 32-bit inputs before the ordinary dtype gate:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
-           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0}
+           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           if u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_shl(src_values[0], src_values[1], u.dtype)
+          elif u.op is Ops.SHR and u.dtype in (dtypes.int, dtypes.uint):
+            values[u] = self.run_u32_shr(src_values[0], src_values[1], u.dtype)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py TestOps.test_rshift TestOps.test_rshift_signed

test_rshift (__main__.TestOps.test_rshift) ... ok
test_rshift_signed (__main__.TestOps.test_rshift_signed) ... ok

Ran 2 tests in 0.350s

OK
```

The backend also passed 8,192 signed/unsigned cases across all counts 0..31. This still processes one word at a time with a uniform count per call; the test's per-element counts are separate calls under NOOPT=1.

The runtime's existing INT16 path can use the converter directly: `round((2*x-(2^n-1))/2^(n+1)) = floor(x/2^n)`. It no longer requires the discarded bits to be zero. All 65,536 INT16 values passed for every count 0..15. That direct INT16 path is separate from the 32-bit convolution implementation above.

Progress so far

| Group                | Working now                       | Remaining                      |
|----------------------|-----------------------------------|--------------------------------|
| `GroupOp.Unary`       | `NEG`, `RECIPROCAL`                | `EXP2`, `LOG2`                 |
|                      |                                   | `SIN`, `SQRT`, `TRUNC`          |
| `GroupOp.Binary`      | `ADD`, `MUL`, `SUB`                | `AND`, `CDIV`, `CMOD`          |
|                      | `FDIV`, `MAX`                      | `FLOORDIV`, `FLOORMOD`, `POW`  |
|                      | `CMPEQ`, `CMPNE`, `CMPLT`          | `THREEFRY`, `XOR`              |
|                      | `OR` (bool), `SHL`, `SHR`          |                                |
| `GroupOp.Ternary`     | `WHERE`                           | `MULACC`                       |
| `Elementwise` extras | `CAST` (bool → FP16, mask → bool) | `BITCAST`                      |
| **Total**            | **15 / 30**                       | **15 / 30**                    |

This counts the paths covered in the blog, not full support for every dtype and edge case.

1. FP16 arithmetic runs through EW. Bool OR uses CAST → MAX → comparison; comparisons and WHERE use the lowering rules above. NaN comparisons are still unsupported.
2. UINT32/INT32 SHL and SHR now use convolution and output CVT. The NPU routes the carries and assembles the output bytes; Python copies the initial/final storage and submits tasks.
3. The 32-bit shift helpers handle counts 0..31, one word at a time, with a uniform count per call. General device-resident per-lane counts in one vector task are not implemented. INT16 SHR handles counts 0..15 exactly; the simple INT16 SHL multiply still saturates on overflow.

Both original right-shift tests passed. The mixed run of `test_rshift`, `test_rshift_signed`, `test_lshift`, `test_lshift_signed`, `test_add`, `test_maximum`, `test_sub`, `test_neg`, `test_mul`, `test_div` and `test_where` passed all 11 tests in 9.347s with `NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.

The direct SHR checks also passed 8,192 INT32/UINT32 cases and all 1,048,576 INT16 value/count combinations. These are correctness checks, not a performance claim.

Next add Ops.AND and Ops.XOR for bool inputs. We already have bool → FP16 and mask → bool CAST on the NPU, so no new registers here.

```text
AND: a, b → CAST(half) → MUL → CAST(bool)
XOR: a, b → CAST(half) → CMPNE → existing comparison formula → CAST(bool)
```

| a | b | FP16 a * b | AND | FP16 a != b / XOR |
|---|---|-----------:|-----|-------------------|
| False | False | 0 | False | False |
| False | True  | 0 | False | True  |
| True  | False | 0 | False | True  |
| True  | True  | 1 | True  | False |

1. CAST gives exact FP16 0/1 inputs. MUL is 1 only when both inputs are True, so its output is already a normalized mask.
2. XOR is True when the inputs differ. Reuse our CMPNE matcher rather than adding another hardware mode.
3. Put both rules in comparison_matcher, after the general rewrites. AND's final CAST stays a CAST; XOR's new CMPNE gets lowered by the existing rule.

```diff
 class RockchipRenderer(Renderer):
@@
   comparison_matcher = PatternMatcher([
+    # Bool AND multiplies FP16 0/1 masks; keep the final bool CAST after general rewrites.
+    (UPat(Ops.AND, dtypes.bool, name="u"),
+     lambda u: u.src[0].cast(dtypes.half).alu(Ops.MUL, u.src[1].cast(dtypes.half)).cast(dtypes.bool)),
+    # Bool XOR is inequality of the FP16 0/1 inputs; reuse comparison lowering.
+    (UPat(Ops.XOR, dtypes.bool, name="u"),
+     lambda u: u.src[0].cast(dtypes.half).ne(u.src[1].cast(dtypes.half))),
```

These rules only match dtypes.bool. We do not add AND/XOR register selectors to ops_map or relax the integer ALU gate.

The direct checks passed all four bool pairs, lengths 4, 7, 8, 9, 16, 17 and 65, broadcasting and `(a & b) ^ b`: 17 cases. NPU submissions and both input/output CAST paths were observed for the length tests.

Now test_minimum gets past its bool XOR case as well:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_minimum TestOps.test_maximum TestOps.test_where TestOps.test_add TestOps.test_mul \
    TestOps.test_rshift TestOps.test_rshift_signed TestOps.test_lshift TestOps.test_lshift_signed

test_minimum (__main__.TestOps.test_minimum) ... ok
test_maximum (__main__.TestOps.test_maximum) ... ok
test_where (__main__.TestOps.test_where) ... ok
test_add (__main__.TestOps.test_add) ... ok
test_mul (__main__.TestOps.test_mul) ... ok
test_rshift (__main__.TestOps.test_rshift) ... ok
test_rshift_signed (__main__.TestOps.test_rshift_signed) ... ok
test_lshift (__main__.TestOps.test_lshift) ... ok
test_lshift_signed (__main__.TestOps.test_lshift_signed) ... ok

Ran 9 tests in 7.014s

OK
```

The original test_and and test_xor also contain INT32 inputs. Those still fail at `tor & 0x1337` and `tor ^ 0x1337` with `NotImplementedError` for Ops.AND / Ops.XOR with dtypes.int. We have not implemented integer bitwise operations in this step, and did not change the tests.

That adds bool AND and XOR to Working now: **17 / 30**, with **13 / 30** remaining. Like bool OR, this does not count as general integer support.

Now extend AND/XOR to INT32 and UINT32 with convolution. We cannot use the bool formula on whole integers, and we do not need 32 one-bit planes.

Split each byte into four base-4 digits. The three low digits are 0..3; the highest digit is signed, -2..1:

```text
q0 = signed_byte
q1 = floor(q0/4), q2 = floor(q0/16), q3 = floor(q0/64)
d0 = q0 - 4*q1
d1 = q1 - 4*q2
d2 = q2 - 4*q3
d3 = q3

signed_byte = d0 + 4*d1 + 16*d2 + 64*d3
```

For each digit pair, AND/XOR only needs this 4×4 constant table:

| a | AND b=0 | b=1 | b=2 | b=3 | XOR b=0 | b=1 | b=2 | b=3 |
|---|--------:|----:|----:|----:|--------:|----:|----:|----:|
| 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 3 |
| 1 | 0 | 1 | 0 | 1 | 1 | 0 | 3 | 2 |
| 2 | 0 | 0 | 2 | 2 | 2 | 3 | 0 | 1 |
| 3 | 0 | 1 | 2 | 3 | 3 | 2 | 1 | 0 |

Convolution does not directly execute a bitwise instruction. We use it to evaluate this table on the NPU:

```text
index = 4*a + b                              # 0..15
step[t] = RELUX1(index - t + 1)              # 1 when index >= t, otherwise 0
result = table[0] + sum((table[t]-table[t-1]) * step[t], t=1..15)
```

For example, a=2 and b=3 gives index=11. The first eleven steps are 1, so the sum telescopes to table[11]: AND=2 or XOR=1. These are threshold lanes for a table lookup, not individual input bits.

1. Upload the two words' eight original bytes once. Three convolution/CVT tasks compute q1, q2 and q3 with the exact floor conversion from SHR.
2. One convolution subtracts adjacent quotients to get the 32 base-4 digits.
3. Eight convolution + integer RELUX1 tasks create the threshold lanes, two digit pairs per task.
4. One convolution evaluates the table using constant difference weights. For the signed top digits, add 10 to their pair index, reorder the constant table and return a signed -2..1 digit.
5. One convolution assembles each output byte using weights 1, 4, 16 and 64. Its four raw output bytes are already the INT32/UINT32 result.

That is 14 convolution tasks per word pair. Python prepares constant weights and copies initial/final storage; there are no host reads or arithmetic on intermediate tensor values. It is a correctness implementation, not a speed claim.

Let the shared convolution helper select its input address and enable integer RELUX1:

```diff
   def conv_shl_task(self, weights:list[list[int]], out_addr:int, offset:int=0, shift:int=0,
-                    unsigned:bool=False, features:bool=False, channels:int=64) -> None:
+                    unsigned:bool=False, features:bool=False, channels:int=64, input_addr:int|None=None, relux:bool=False) -> None:
@@
+    if input_addr is not None: self.npu_regs.append(E(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR, input_addr))
+    if relux:
+      self.npu_regs += [
+        E(rk.DPU, rk.REG_DPU_BN_CFG, (1 << rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT) |
+          (1 << rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT) | (1 << rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT)),
+        E(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 1), # Integer convolution mode: clamp to integer 1.
+      ]
     self.submit(cna=True)
```

The new methods keep the same raw-storage wrapper as the runtime:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def conv_bitwise_word(self, op:Ops, raw:bytes) -> bytes:
+    assert op in (Ops.AND, Ops.XOR) and len(raw) == 8
+    if self.output_offset + 32 > self.dev.output_mem.size: raise RuntimeError("ROCKCHIP intermediate buffer exhausted")
+    scratch = bytearray(640)
+    scratch[:8], scratch[160], scratch[512] = raw, 1, 1
+    to_mv(self.dev.input_buf, len(scratch))[:] = scratch
+    base = self.dev.input_mem.dma_addr
+    # Three signed quotients split all eight bytes into base-4 digits, not bit planes.
+    for digit in range(1, 4):
+      self.conv_shl_task([[2*int(i==j) for j in range(32)] for i in range(8)], base+32*digit,
+        offset=-(4**digit-1), shift=2*digit+1, features=True, channels=32)
+    weights = [[0]*128 for _ in range(32)]
+    for byte in range(8):
+      for digit in range(4):
+        weights[byte*4+digit][digit*32+byte] = 1
+        if digit < 3: weights[byte*4+digit][(digit+1)*32+byte] = -4
+    self.conv_shl_task(weights, base+128, features=True, channels=128)
+    # Pair index = 4*a+b. ReLU-X(index-t+1) is 1 exactly when index >= t.
+    for pair in range(8):
+      weights = [[0]*64 for _ in range(32)]
+      for part in range(2):
+        digit = pair*2+part
+        for t in range(1, 16):
+          row = weights[part*16+t-1]
+          row[digit], row[16+digit], row[32] = 4, 1, 1-t+(10 if digit%4 == 3 else 0)
+      self.conv_shl_task(weights, base+256+pair*32, features=True, channels=64, input_addr=base+128, relux=True)
+    lookup = ((0,0,0,0,0,1,0,1,0,0,2,2,0,1,2,3) if op is Ops.AND else
+              (0,1,2,3,1,0,3,2,2,3,0,1,3,2,1,0))
+    weights = [[0]*288 for _ in range(16)]
+    for digit in range(16):
+      table = list(lookup)
+      if digit%4 == 3:
+        # Top digits are -2..1. Shift their pair index by 10 and keep the result signed.
+        table = [lookup[4*((i//4+2)%4)+(i%4+2)%4] for i in range(16)]
+        table = [x if x < 2 else x-4 for x in table]
+      weights[digit][256] = table[0]
+      for t in range(1, 16): weights[digit][digit*16+t-1] = table[t]-table[t-1]
+    self.conv_shl_task(weights, base+544, features=True, channels=288, input_addr=base+256)
+    weights = [[0]*32 for _ in range(4)]
+    for byte in range(4):
+      for digit in range(4): weights[byte][byte*4+digit] = 4**digit
+    self.conv_shl_task(weights, self.dev.output_mem.dma_addr+self.output_offset, features=True, channels=32, input_addr=base+544)
+    return bytes(to_mv(self.dev.output_buf+self.output_offset, 4))
```

Keep the returned bytes in a typed memoryview, so the next task or STORE can copy them without a numeric conversion. No wrapper class is needed:

```diff
+def typed_view(raw:bytes|memoryview, dtype:DType) -> memoryview: return memoryview(raw).cast("B").cast(storage_fmt_for_dtype(dtype))
+
+def scalar16(value, dtype:DType|None=None):
+  if not isinstance(value, memoryview): return value
+  return from_storage_scalar(value[0], dtype) if dtype is not None else value[0]
+
+def raw16(value, dtype:DType) -> bytes|memoryview:
+  if isinstance(value, memoryview): return value.cast("B")
+  return struct.pack("<" + storage_fmt_for_dtype(dtype), to_storage_scalar(value, dtype))
+
 def _load(m, i, dtype: DType):
@@
 def _store(m, i, v, dtype: DType):
   if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
+  if isinstance(v, memoryview):
+    assert v.nbytes == dtype.itemsize
+    m.cast("B")[i*m.itemsize:i*m.itemsize+dtype.itemsize] = v.cast("B")
+    return
```

The existing Python CAST fallback still needs a number, so decode there:

```diff
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-            values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
+            values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(scalar16(x, src_dtypes[0]))) for x in src_values[0]]
```

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_u32_bitwise(self, op:Ops, a:list, b:list, dtype:DType) -> list:
+    assert len(a) == len(b)
+    def raw(x): return bytes(raw16(x, dtype))
+    return [typed_view(self.conv_bitwise_word(op, raw(x)+raw(y)), dtype) for x,y in zip(a,b)]
```

Advertise the composite operations, then dispatch 32-bit inputs before the ordinary ALU gate. Bool inputs still use the matchers above:

```diff
-CMP = 9  # Internal multi-stage comparison marker, not an EW algorithm.
+CMP = 9  # Internal lowering/dispatch marker, not an EW algorithm.
@@
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
-           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0}
+           Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0, Ops.AND: CMP, Ops.XOR: CMP}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.SHR and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_shr(src_values[0], src_values[1], u.dtype)
+          elif u.op in (Ops.AND, Ops.XOR) and u.dtype in (dtypes.int, dtypes.uint):
+            values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
```

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_and TestOps.test_xor TestOps.test_minimum TestOps.test_maximum TestOps.test_where \
    TestOps.test_add TestOps.test_mul TestOps.test_rshift TestOps.test_rshift_signed \
    TestOps.test_lshift TestOps.test_lshift_signed

test_and (__main__.TestOps.test_and) ... ok
test_xor (__main__.TestOps.test_xor) ... ok
test_minimum (__main__.TestOps.test_minimum) ... ok
test_maximum (__main__.TestOps.test_maximum) ... ok
test_where (__main__.TestOps.test_where) ... ok
test_add (__main__.TestOps.test_add) ... ok
test_mul (__main__.TestOps.test_mul) ... ok
test_rshift (__main__.TestOps.test_rshift) ... ok
test_rshift_signed (__main__.TestOps.test_rshift_signed) ... ok
test_lshift (__main__.TestOps.test_lshift) ... ok
test_lshift_signed (__main__.TestOps.test_lshift_signed) ... ok

Ran 11 tests in 26.069s

OK
```

The original integer test_and and test_xor now pass without changing their inputs. AND/XOR coverage is bool plus INT32/UINT32; other integer widths are not enabled.

The convolution path also passed all 65,536 byte pairs for AND and all 65,536 for XOR, checking the four raw output bytes directly. This includes bytes 0x80..0xff and the signed top-digit handling.

Eight Tensor checks also passed for INT32/UINT32 AND/XOR, including broadcasting, random full-width values and high-bit boundary cases. NPU submissions were observed in every check.

Next Ops.TRUNC. The CVT shift registers helped with integer division, but truncating FP16 means removing the fractional part, not shifting the whole value. The TRM lists EW ALU selector 7 as Floor and 8 as Ceil; probing confirmed both work on FP16.

```text
TRUNC(x) = MAX(FLOOR(x), MIN(CEIL(x), 0))
MIN(y, 0) = NEG(MAX(NEG(y), 0))
```

| Stage | -2.9 | -0.9 | 0.9 | 2.9 |
|-------|-----:|-----:|----:|----:|
| FLOOR(x) | -3 | -1 | 0 | 2 |
| CEIL(x) | -2 | -0 | 1 | 3 |
| MIN(CEIL(x), 0) | -2 | -0 | 0 | 0 |
| MAX(FLOOR(x), previous) | -2 | -0 | 0 | 2 |

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
   def build_registers(self, op:Ops, int16_mode:bool=False, custom:str|None=None, byte_output:bool=False, output_shift:int=0,
                       input_addr:int|None=None, weight_addr:int|None=None, output_addr:int|None=None) -> None:
@@
     exp_shift = custom == "fp16_exponent_shift_minus(16)"
-    assert custom is None or (op is Ops.CUSTOM and exp_shift and not int16_mode)
+    rounding = {"FLOOR": 7, "CEIL": 8}.get(custom or "")
+    unary = op is Ops.NEG or rounding is not None
+    assert custom is None or (op is Ops.CUSTOM and (exp_shift or rounding is not None) and not int16_mode)
+    # MIN with binary mode returns a < b; FLOOR/CEIL use the ordinary FP16 ALU path.
     binary = int16_mode and op is Ops.CMPLT
@@
-        ((1 if binary else self.ops_map[op]) << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) | # MIN in binary mode returns a < b.
+        ((rounding if rounding is not None else 1 if binary else self.ops_map[op]) << rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT) |
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

Ran 1 test in 1.355s

OK
```

All 65,536 FP16 encodings passed the direct NPU check against numpy.trunc. Non-NaN outputs matched bit for bit, including negative zero, subnormals and both infinities. NaN inputs remained NaN; their payload bits are not preserved. This is FP16 TRUNC only, with six NPU stages per atom, not a fused single-task implementation.

Progress: **18 / 30** paths covered, **12 / 30** remaining. TRUNC joins NEG and RECIPROCAL; AND/XOR now cover bool and 32-bit integers.

Rerunning test_trunc together with the eleven AND/XOR regression tests above gave `Ran 12 tests in 25.069s`, `OK`, with the same NOOPT/FP16/forward-only settings.

Next extend Ops.OR to INT32/UINT32. The convolution pipeline is unchanged; just select another two-bit table:

| a | b=0 | b=1 | b=2 | b=3 |
|---|----:|----:|----:|----:|
| 0 | 0 | 1 | 2 | 3 |
| 1 | 1 | 1 | 3 | 3 |
| 2 | 2 | 3 | 2 | 3 |
| 3 | 3 | 3 | 3 | 3 |

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
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python test/backend/test_ops.py \
    TestOps.test_or TestOps.test_and TestOps.test_xor TestOps.test_trunc TestOps.test_minimum TestOps.test_maximum

test_or (__main__.TestOps.test_or) ... ok
test_and (__main__.TestOps.test_and) ... ok
test_xor (__main__.TestOps.test_xor) ... ok
test_trunc (__main__.TestOps.test_trunc) ... ok
test_minimum (__main__.TestOps.test_minimum) ... ok
test_maximum (__main__.TestOps.test_maximum) ... ok

Ran 6 tests in 20.737s

OK
```

OR was already counted for bool, so progress stays **18 / 30**. Its coverage now includes INT32/UINT32, with the same 14 convolution tasks per word pair and no host arithmetic on intermediate values.

Four additional INT32/UINT32 Tensor checks passed (144 lanes), including broadcasting, random full-width values, sign-bit boundaries and alternating-bit patterns. Every check submitted NPU work.

Next investigate Ops.MULACC, `a*b+c`. We can run BS MUL with b from BRDMA, then EW ADD with c from ERDMA in one task. The product stays FP32 until the output converter writes FP16.

| Inputs | Separate FP16 MUL then ADD | One BS → EW task | Correctly rounded FP16 MULACC |
|--------|---------------------------:|-----------------:|-----------------------------:|
| `1.0009765625 * 1.0009765625 - 1.001953125` | 0 | `2^-20` | `2^-20` |
| `65504 * 2 - 65504` | inf | 65504 | 65504 |
| `11.8125 * -2688 - 0.00023484230041503906` | -31744 | -31744 | -31760 |
| `(-0) * 1 + (-0)` | -0 | +0 | -0 |

The first two cases show why keeping the product in FP32 helps. But the third exposes another rounding step: the exact product is -31752, halfway between two FP16 values. The small negative addend should move it towards -31760. FP32 addition loses that small amount, so the final FP16 conversion rounds the halfway value to -31744 instead.

The converter controls we tried did not fix this. Bypassing EW operand conversion misreads FP16 addends; changing its offset to negative zero or changing CVT_TYPE also did not preserve the negative-zero case.

The probe is saved in `~/npu/ops_reg/probe_fp16_mulacc.py`. A 4,096-triple random check had no numeric mismatches, but targeted halfway cases did. Passing ordinary random tests is not enough to claim correctly rounded MULACC.

MULACC is not added to ops_map yet. The native one-task path is FP32-accumulating multiply-add, not exact FP16 FMA; accepting that limitation or implementing a correction needs a separate decision. Progress stays **18 / 30**.

Lets try to correct it on the NPU. First keep both the product and sum in FP32 storage, using precision 5. Multiplying two FP16 inputs fits exactly in FP32; the rounding error comes from adding c.

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

| Case | Native one-task result | Corrected result |
|------|-----------------------:|-----------------:|
| `11.8125 * -2688 - 0.00023484230041503906` | -31744 | -31760 |
| `1.0009765625 * 1.0009765625 - 1.001953125` | `2^-20` | `2^-20` |
| `65504 * 2 - 65504` | 65504 | 65504 |
| `(-0) * 1 + (-0)` | +0 | -0 |

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
@@
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
+      E(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, 1 << rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT),
+      E(rk.DPU, rk.REG_DPU_BS_OW_CFG,
+        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT) |
+        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT) |
+        ((precision == 2 and output == 5) << rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT) | (1 << rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT)),
+      E(rk.DPU, rk.REG_DPU_SURFACE_ADD, 1 << rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT),
+      E(rk.DPU, rk.REG_DPU_EW_CFG, ew),
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

| Step | NPU operation |
|------|---------------|
| Product | FP16 BS MUL → exact FP32 p |
| Sum | FP32 ADD(p, c) → s |
| Error | FP32 TwoSum → e |
| Low bit | INT32 `q = floor(s_bits/2)`, `low = s_bits - 2*q` |
| Correction | If low is 0 and e is finite/nonzero, move s_bits one unit towards e |
| Zero sign | Restore -0 when both p and c are -0 |
| Output | FP32 → FP16 conversion |

For the low bit, `round_away((x - int(x > 0))/2)` gives `floor(x/2)`. This avoids overflowing `2*x`. For negative FP32 values, moving the bits up makes the value more negative, so the sign comparison selects +1 or -1.

The finite/nonzero error check disables the correction for infinities and NaNs. We return the hardware NaN, without promising its payload bits. The zero-sign check uses `bits < INT_MIN+1`, which matches only INT_MIN, the stored FP32 negative zero.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
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
+      product = calc(-1, base, precision=2, bs_mul=base+64)
+      addend = calc(-1, base+128, precision=2)
+      total = calc(2, product, addend)
+      # TwoSum: exact a*b+c = total + error for finite FP16 inputs.
+      v = calc(4, total, product)
+      error = calc(2, calc(4, product, calc(4, total, v)), calc(4, addend, v))
+      # Reinterpret FP32 storage as INT32. Get parity without overflowing 2*total_bits.
+      half = calc(4, total, ilt(zero, total), 4, 4, shift=1)
+      even = isub(one, isub(total, imul(half, two)))
+      abs_error = calc(5, error)  # EW algorithm 5 = ABS.
+      valid = imul(ilt(zero, abs_error), ilt(abs_error, infinity))
+      opposite = calc(5, isub(ilt(error, zero), ilt(total, zero)), precision=4, output=4)
+      direction = isub(one, imul(opposite, two))
+      odd = iadd(total, imul(imul(even, valid), direction))
+      # FP32 ADD turns -0 + -0 into +0; restore that sign from the original operands.
+      both_neg_zero = imul(ilt(product, min_plus_one), ilt(addend, min_plus_one))
+      signed = iadd(odd, imul(min_int, both_neg_zero))
+      out = calc(-1, signed, output=2) - base
+      # Four FP32 lanes per surface: half output retains the 16-byte surface stride.
+      raw = bytes(to_mv(self.dev.input_buf+out, 8)) + bytes(to_mv(self.dev.input_buf+out+16, 8))
+      result.extend(typed_view(raw[i*2:i*2+2], dtypes.half) for i in range(count))
+    return result
```

Now advertise MULACC and dispatch its three inputs. tinygrad also fuses integer MUL+ADD when MULACC is advertised, so decompose those non-FP16 cases again in the late matcher. This does not add general INT32 or FP32 MULACC support.

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
            Ops.CMPEQ: CMP, Ops.CMPNE: CMP, Ops.CMPLT: CMP, Ops.WHERE: CMP, Ops.SHL: 0, Ops.SHR: 0,
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP}
@@
 class RockchipProgram(Program['RockchipDevice']):
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op in (Ops.AND, Ops.XOR, Ops.OR) and u.dtype in (dtypes.int, dtypes.uint):
             values[u] = self.run_u32_bitwise(u.op, src_values[0], src_values[1], u.dtype)
+          elif u.op is Ops.MULACC and u.dtype == dtypes.half:
+            values[u] = self.run_mulacc(*src_values)
@@
 class RockchipRenderer(Renderer):
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


Next CDIV and CMOD. We already have TRUNC, but FDIV → TRUNC is not enough:

```text
4094 / 3 = 1364.666...
FP16 FDIV rounds it to 1365
TRUNC(1365) = 1365, but CDIV should be 1364
```

Both inputs fit FP16 exactly. This is quotient rounding, not just an input CAST problem. The direct NPU probe returned 1365 too.

First finish BITCAST. It keeps the same bytes and changes their dtype. In this interpreter the host copies storage; it does not calculate a converted value or submit a fake identity operation.

```diff
 def _load(m, i, dtype: DType):
@@
   if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
+  if m.itemsize == dtype.itemsize:
+    return typed_view(bytes(m.cast("B")[i*m.itemsize:(i+1)*m.itemsize]), dtype)
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
         elif u.op is Ops.BITCAST:
-          values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
+          assert src_dtypes[0].itemsize == u.dtype.itemsize, "bitcast itemsize mismatch"
+          values[u] = [typed_view(raw16(x, src_dtypes[0]), u.dtype) for x in src_values[0]]
```

Remove the unused bitcast import from tinygrad.dtype. Pack arithmetic inputs with raw16 rather than converting the storage reference back to a Python float. Read the output as storage references too:

```diff
   def run_npu(self, op:Ops, a:list, b:list|None=None, custom:str|None=None, dtype:DType=dtypes.half, carry:bool=False) -> list:
@@
-      packed = struct.pack("<8h" if dtype == dtypes.int16 or (op is Ops.CAST and not byte_output) else "<8e", *(lanes + [0] * (8-len(lanes))))
+      input_dtype = dtypes.int16 if dtype == dtypes.int16 or (op is Ops.CAST and not byte_output) else dtypes.half
+      packed = b"".join(raw16(x, input_dtype) for x in lanes) + bytes(2*(8-len(lanes)))
@@
-        to_mv(self.dev.weight_buf, 16)[:] = struct.pack("<8h" if dtype == dtypes.int16 else "<8e", *(rhs + [0] * (8-len(rhs))))
+        to_mv(self.dev.weight_buf, 16)[:] = b"".join(raw16(x, input_dtype) for x in rhs) + bytes(2*(8-len(rhs)))
@@
-      result.extend(struct.unpack("<" + fmt, to_mv(self.dev.output_buf, 16))[:len(lanes)])
+      out = bytes(to_mv(self.dev.output_buf, 16))
+      result.extend(typed_view(out[i:i+dtype.itemsize], dtype) for i in range(0, len(lanes)*dtype.itemsize, dtype.itemsize))
```

The DMA-address path in the current implementation does the same raw copy when gathering noncontiguous lanes; contiguous intermediate lanes can keep their DMA address. The checks which need a Python number use scalar16. This is not a fully device-resident interpreter.

The bit-pattern checks passed all 65,536 FP16 encodings, all 65,536 BF16 encodings, and selected FP32/FP64 zeros, infinities and NaN payloads. The existing test_bitcast passed with its default FP32 inputs.

For integer arithmetic, use the same base-256 idea as our shifts. Each byte becomes one zero-padded INT32 lane:

| Stage | a | b | c | d |
|-------|---|---|---|---|
| UINT32 storage | byte 0 | byte 1 | byte 2 | byte 3 |
| Scratch INT32 | 0..255 | 0..255 | 0..255 | 0..255 |
| Arithmetic | low limb | next limb | next limb | high limb |
| Result storage | low byte | low byte | low byte | low byte |

1. Copy the original bytes into the scratch lanes. Python copies bytes and inserts zero padding; it does not split values with arithmetic.
2. For SUB, compute `t = a - b - borrow`. If t is negative, add 256 and pass borrow=1 to the next limb.
3. ADD can reuse SUB with the two's-complement negative of b.
4. MUL uses byte products. Each product is at most 65025; even an eight-byte word's accumulated products fit INT32.
5. Extract carry with our converter formula: `floor(t/256) = round((2*t - 255)/512)`. Keep `t - 256*carry` as the result byte.

These are private INT32 tasks through mulacc_stage, not FP16 casts. First add the scratch and limb helpers:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
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
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def subtract(a:list[int], b:list[int]) -> tuple[list[int], int]:
+        out, borrow = [], zero
+        for x,y in zip(a, b):
+          value = sub(sub(x, y), borrow)
+          borrow = lt(value, zero)
+          out.append(add(value, mul(radix, borrow)))
+        return out, borrow
+      def choose(mask:int, a:list[int], b:list[int]) -> list[int]: return [select(mask, x, y) for x,y in zip(a, b)]
+      def negate(a:list[int]) -> list[int]: return subtract([zero]*len(a), a)[0]
+      def double(a:list[int], carry:int) -> list[int]:
+        out = []
+        for x in a:
+          value = add(add(x, x), carry)
+          carry = sub(one, lt(value, radix))
+          out.append(sub(value, mul(radix, carry)))
+        return out
+      def nonzero(a:list[int]) -> int:
+        total = zero
+        for x in a: total = add(total, x)
+        return lt(zero, total)
```

Next load the limbs and implement exact wrapping ADD/SUB/MUL, integer comparisons and WHERE. Integer WHERE selects raw bytes, so no integer is converted to FP16:

```diff
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
+      if op is Ops.WHERE: out = choose(a[0], operands[1], operands[2])
+      elif op is Ops.NEG: out = negate(a)
+      elif op is Ops.ADD: out = subtract(a, negate(b))[0]
+      elif op is Ops.SUB: out = subtract(a, b)[0]
+      elif op is Ops.MUL:
+        out, carry = [], zero
+        for i in range(width):
+          total = carry
+          for j in range(i+1): total = add(total, mul(a[j], b[i-j]))
+          # No tie: round((2*total-255)/512) is floor(total/256).
+          carry = calc(4, add(total, total), const(255), shift=9)
+          out.append(sub(total, mul(radix, carry)))
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

An extra remainder byte keeps the carry, so doubling a large unsigned remainder cannot overflow the word. Afterwards restore q's sign from both inputs and r's sign from a. CDIV by zero returns 0 and CMOD returns a, matching tinygrad's helper definitions.

For FLOORDIV, subtract 1 from q when the remainder is nonzero and the operand signs differ. For FLOORMOD, add b to r in that case:

```diff
   def run_integer(self, op:Ops, inputs:list[list], dtype:DType) -> list:
@@
+        else:
+          assert op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD)
+          numerator, divisor = choose(sign_a, negate(a), a), choose(sign_b, negate(b), b)
+          quotient, remainder = [zero]*width, [zero]*(width+1)
+          # Restoring division: feed one numerator bit into the remainder, subtract if it fits.
+          # The extra byte retains the carry; there is no array of bit planes or host arithmetic.
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
+          valid = nonzero(divisor)
+          quotient = choose(valid, choose(different_sign, negate(quotient), quotient), [zero]*width)
+          remainder = choose(sign_a, negate(remainder[:width]), remainder[:width])
+          if op in (Ops.FLOORDIV, Ops.FLOORMOD):
+            correction = mul(mul(different_sign, nonzero(remainder)), valid)
+            quotient = subtract(quotient, [correction]+[zero]*(width-1))[0]
+            remainder = subtract(remainder, negate(choose(correction, b, [zero]*width)))[0]
+          out = quotient if op in (Ops.CDIV, Ops.FLOORDIV) else remainder
+      # Rejoin the low bytes of the NPU-produced limbs. This is a storage copy, not numeric evaluation.
+      out_dtype = dtypes.bool if op in GroupOp.Comparison else dtype
+      raw_out = [bytes(to_mv(self.dev.input_buf+x-base, 32)) for x in out]
+      result.extend(typed_view(b"".join(x[i*4:i*4+1] for x in raw_out), out_dtype) for i in range(count))
+    return result
```

Remove the earlier lossy INT32 → FP16 → INT32 matchers for ADD/MUL and comparisons. Keep the float/bool comparison matchers and restrict the arithmetic WHERE matcher to half. Route integer operations before the FP16 gate:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
-          if u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
+          integer_dtype = src_dtypes[1] if u.op is Ops.WHERE else src_dtypes[0]
+          if integer_dtype in dtypes.ints and u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.NEG, Ops.MAX, Ops.WHERE,
+                                                      Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, *GroupOp.Comparison):
+            values[u] = self.run_integer(u.op, src_values, integer_dtype)
+          elif u.op is Ops.SHL and u.dtype in (dtypes.int, dtypes.uint):
```

tinygrad already lowers FLOORDIV/FLOORMOD using CDIV/CMOD plus sign correction in codegen/decomp/op.py. We do not need to change that file. The direct helper also handles both, so we can check the formula independently.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/backend/test_ops.py::TestOps::test_div_int \
    test/backend/test_ops.py::TestOps::test_mod \
    test/backend/test_ops.py::TestOps::test_fmod

3 passed in 27.07s
```

The focused checks passed 88 subtests across signed and unsigned 8/16/32/64-bit arithmetic, comparisons and division. This includes full-width random inputs, wrapping results, negative remainders and zero divisors. It is not exhaustive over every pair. INT32 division took about 0.25s for eight lanes; correctness first, not a speed claim.

Next THREEFRY. tinygrad already supplies the rounds in codegen/decomp/op.py. Now we have wrapping UINT32 ADD, convolution shifts and XOR, so leave THREEFRY out of code_for_op and let tinygrad decompose it.

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
@@
+  def run_u64_shift(self, op:Ops, a:list, b:list, dtype:DType) -> list:
+    counts = list(map(scalar16, b))
+    if not counts or not all_same(counts) or not 0 <= counts[0] < 64:
+      raise NotImplementedError("ROCKCHIP 64-bit shift requires one uniform count in 0..63")
+    amount = counts[0]
+    raw = [bytes(raw16(x, dtype)) for x in a]
+    low, high = ([typed_view(x[i:i+4], dtypes.uint) for x in raw] for i in (0, 4))
+    zero = [typed_view(bytes(4), dtypes.uint)]*len(a)
+    def left(x:list, count:int) -> list: return self.run_u32_shl(x, [count]*len(x), dtypes.uint) if count else x
+    def right(x:list, count:int, signed:bool=False) -> list:
+      return self.run_u32_shr(x, [count]*len(x), dtypes.int if signed else dtypes.uint) if count else x
+    if op is Ops.SHL:
+      lo = left(low, amount) if amount < 32 else zero
+      hi = self.run_u32_bitwise(Ops.OR, left(high, amount), right(low, 32-amount), dtypes.uint) if 0 < amount < 32 else \
+        high if amount == 0 else left(low, amount-32)
+    else:
+      signed = dtype in dtypes.sints
+      hi = right(high, min(amount, 31), signed) if signed or amount < 32 else zero
+      lo = self.run_u32_bitwise(Ops.OR, right(low, amount), left(high, 32-amount), dtypes.uint) if 0 < amount < 32 else \
+        low if amount == 0 else right(high, amount-32, signed)
+    return [typed_view(bytes(x)+bytes(y), dtype) for x,y in zip(lo, hi)]
```

AND/OR/XOR can process each 32-bit chunk independently:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
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

The 32-bit shift wrappers now accept and return raw storage too:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_u32_shl(self, a:list, b:list, dtype:DType) -> list:
+    counts = [scalar16(x) for x in b]
+    if not counts or not all_same(counts) or not 0 <= counts[0] <= 31:
+      raise NotImplementedError("ROCKCHIP UINT32 SHL requires one uniform shift count in 0..31")
+    result:list = []
+    for x in a:
+      raw = bytes(raw16(x, dtype))
+      result.append(typed_view(self.conv_shl_word(raw, int(counts[0])), dtype))
+    return result
```

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_u32_shr(self, a:list, b:list, dtype:DType) -> list:
+    counts = [scalar16(x) for x in b]
+    if not counts or not all_same(counts) or not 0 <= counts[0] <= 31:
+      raise NotImplementedError("ROCKCHIP UINT32 SHR requires one uniform shift count in 0..31")
+    result:list = []
+    for x in a:
+      raw = bytes(raw16(x, dtype))
+      result.append(typed_view(self.conv_shr_word(raw, int(counts[0]), signed=dtype == dtypes.int), dtype))
+    return result
```

These replace the earlier unpack-to-Python-integer wrappers. The convolution register sequences are unchanged.

The direct THREEFRY check matched the CPU backend on five counter/key pairs, including full-width values. The JAX vector already recorded in test_randomness.py also matched all 20 words. That module could not collect here because hypothesis is missing; the same reference vector is checked in test/device/test_rockchip_integer.py without adding a dependency.

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py::TestRockchipInteger::test_threefry_jax_reference

1 passed in 10.12s
```

The 18 earlier arithmetic/comparison/shift regressions passed again in 35.57s.

The storage paths use typed memoryviews, not a wrapper class. BITCAST changes the view format; raw16 copies its bytes. For mapped output views, input_address can recover the DMA address from the host mapping, so this does not require unpacking intermediate values.

Rerun after the storage change:

```bash
$ NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest -n0 -q \
    test/device/test_rockchip_integer.py \
    test/backend/test_ops.py::TestOps::test_maximum \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc

11 passed, 124 subtests passed in 45.95s
```

This includes the NPU's raw 0x7c01 NaN going through exponent shift to 0x3c01, plus MULACC's rounding and signed-zero cases. No NaN payload is unpacked/repacked to implement BITCAST.

Progress: **25 / 30** paths covered. Remaining: SQRT, EXP2, LOG2, POW and SIN. This is not every dtype or edge case, and the full test_ops.py sweep is still pending.

### SQRT

The 1500 branch uses 14 FP16 Newton steps. Let's check the rounding before copying it.

A bit-based starting estimate followed by three FP16 Newton steps still gave 8,030 wrong answers across all 31,743 positive finite FP16 values. Each was one ULP away. Repeating the half-precision arithmetic does not guarantee the correctly rounded root.

Instead we can search the positive FP16 encodings. Their integer order is also their numeric order. BS can multiply two FP16 inputs and write the exact product as FP32, so we can compare a candidate's square with the original input without rounding the product to half.

| Stage | Operation |
|-------|-----------|
| Input | Convert FP16 x to FP32, retaining the original half bits |
| Search | Find the largest FP16 y with y² ≤ x |
| Neighbour | z is the next FP16 value after y; g = z - y |
| Midpoint | m² = y² + y*g + (g/2)² |
| Round | x < m² selects y; x > m² selects z; a tie selects the even encoding |
| Special values | Keep ±0 and +inf; negative nonzero inputs return NaN |

1. Search encodings from 0 to 0x5c00, which represents 256. sqrt(65504) is smaller than 256. Fifteen steps cover the interval.
2. Each search step runs an exact BS square, an INT32 comparison of the positive FP32 encodings, and INT32 selection. No Python comparison chooses a lane's answer.
3. The midpoint itself may not fit in FP16. Expand its square instead: y² + y*g + (g/2)². For positive finite half inputs, these products and their sum fit in FP32 exactly.
4. The final NPU selection also handles signed zero, infinities and NaNs. Python only copies bytes between packed-half and padded-word layouts.

Add the helper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
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
+      def const(value:int) -> int:
+        if value not in constants: constants[value] = alloc(struct.pack("<i", value)*8)
+        return constants[value]
+      def read(addr:int, size:int=32) -> bytes: return bytes(to_mv(self.dev.input_buf+addr-base, size))
+      def calc(algo:int, x:int, y:int|None=None, precision:int=4, output:int=4, **kw) -> int:
+        out = alloc()
+        self.mulacc_stage(algo, x, const(0) if y is None else y, out, precision, output, **kw)
+        return out
+      def add(x:int, y:int) -> int: return calc(2, x, y)
+      def sub(x:int, y:int) -> int: return calc(4, x, y)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, binary=True)
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def half_bits(bits:int) -> int:
+        raw = read(bits)
+        return alloc(b"".join(raw[i:i+2] for i in range(0, 32, 4)))
+      def half_output(x:int) -> int:
+        out = calc(-1, x, precision=5, output=2)
+        return alloc(read(out, 8)+read(out+16, 8))
+      def square(x:int) -> int: return calc(-1, x, precision=2, output=5, bs_mul=x)
+      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8]) + bytes(2*(8-count))
+      source = alloc(raw)
+      source_bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = calc(-1, source, precision=2, output=5)
+      lo, hi = const(0), const(0x5c00)  # sqrt(max finite half) < 256.
+      for _ in range(15):
+        mid = calc(2, lo, hi, shift=1)  # Positive ties round up: ceil((lo+hi)/2).
+        above = lt(x, square(half_bits(mid)))
+        lo, hi = select(above, lo, mid), select(above, sub(mid, const(1)), hi)
+      lower, upper = half_bits(lo), half_bits(add(lo, const(1)))
+      lower32 = calc(-1, lower, precision=2, output=5)
+      upper32 = calc(-1, upper, precision=2, output=5)
+      gap = half_output(calc(4, upper32, lower32, precision=5, output=5))
+      # midpoint² = lower² + lower*gap + (gap/2)²; each product is exact FP32.
+      half_scale = alloc(struct.pack("<e", 0.5)*8)
+      half_gap = half_output(calc(-1, gap, precision=2, output=5, bs_mul=half_scale))
+      cross = calc(-1, lower, precision=2, output=5, bs_mul=gap)
+      midpoint2 = calc(2, calc(2, square(lower), cross, precision=5, output=5), square(half_gap), precision=5, output=5)
+      below, above = lt(x, midpoint2), lt(midpoint2, x)
+      tie = sub(sub(const(1), below), above)
+      parity = sub(lo, mul(calc(4, lo, const(1), shift=1), const(2)))
+      rounded = add(lo, add(above, mul(tie, parity)))
+      # Preserve signed zeros/+inf; negative nonzero inputs produce NaN. NaN payloads may pass through.
+      negative = lt(const(32767), source_bits)
+      magnitude = sub(source_bits, mul(const(32768), negative))
+      special = select(negative, const(0x7e00), source_bits)
+      special = select(lt(const(0), magnitude), special, source_bits)
+      valid = mul(sub(const(1), negative), mul(lt(const(0), magnitude), lt(magnitude, const(0x7c00))))
+      out = read(select(valid, rounded, special))
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

3 passed in 224.37s (0:03:44)
```

The ADD, MUL, maximum, MULACC and raw-NaN regressions also passed: 6 passed in 6.46s.

Progress: **26 / 30** paths covered. EXP2, LOG2, POW and SIN remain. The full test_ops.py sweep is still pending.

### EXP2, LOG2 and SIN

We can reuse the private FP32 stages from MULACC. BS accepts an FP32 main input and an FP16 multiplier, then writes FP32. The weights need four half lanes at offset 0 and four at offset 16; packing all eight together only worked for the first four lanes.

#### EXP2

```text
x = n + r
n = floor(x + 0.5)
2^x = 2^n * 2^r
```

1. Keep r in [-0.5, 0.5]. For a half input this reduced fraction fits in half exactly.
2. Evaluate the degree-eight exponential polynomial in FP32. Each BS MUL takes r as its half operand; EW ADD adds the next FP32 coefficient.
3. Add n*2^23 to the positive FP32 result's bits. This scales by 2^n without rounding to half early.
4. Convert to half once. Inputs at or below -25 round to zero; -24.5 still gives the smallest subnormal. Inputs at or above 16 overflow to infinity.

The first complete NPU sweep found one wrong rounding: input 0x11c5 gave 0x3c00 instead of 0x3c01. We add one output bit for that input using NPU integer comparisons. LLVM's [exp2f16](https://raw.githubusercontent.com/llvm/llvm-project/main/libc/src/__support/math/exp2f16.h) also lists this rounding exception. This is a constant correction, not a Python calculation on the input.

#### LOG2

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

#### SIN

The 1500 branch clamps its input to ±10000. We dont need that clamp here.

```text
q = floor(x * 2/pi + 0.5)
r = x - q * pi/2
q mod 4 selects sin(r), cos(r), -sin(r), or -cos(r)
```

1. Split q into a multiple of 256 and its remainder. Both pieces fit exactly in half.
2. Split pi/2 into three constants and subtract their products in FP32. Splitting q keeps the first two products exact.
3. Evaluate sine and cosine polynomials on the reduced half angle. Keep the remaining FP32 angle error e, then correct with sin(r+e) ≈ sin(r)+e*cos(r) and cos(r+e) ≈ cos(r)-e*sin(r).
4. Select the quadrant and sign on the NPU. Keep sin(-0) = -0; infinities and NaNs produce NaN.

The first full sweep had six one-ULP errors: three magnitudes and their negatives. The final integer selection corrects those three boundaries. Two of them, 0x51f5 and 0x5cb0, also appear in LLVM's [sinf16](https://raw.githubusercontent.com/llvm/llvm-project/main/libc/src/__support/math/sinf16.h) exception table. Our third is 0x32b3.

Add the shared helper. Its local helpers allocate scratch slots and emit tasks; they do not calculate lane results in Python:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_math(self, op:Ops, a:list) -> list:
+    assert op in (Ops.EXP2, Ops.LOG2, Ops.SIN)
+    base = self.dev.input_mem.dma_addr
+    coefficients = (tuple(math.log(2)**k/math.factorial(k) for k in range(9)) if op is Ops.EXP2 else
+                    tuple((-1)**k/((k+1)*math.log(2)) for k in range(16)))
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
+      def const(value:int|float, fmt:str="i") -> int:
+        key = (value, fmt)
+        if key not in constants: constants[key] = alloc(struct.pack("<"+fmt, value)*8)
+        return constants[key]
+      def read(addr:int, size:int=32) -> bytes: return bytes(to_mv(self.dev.input_buf+addr-base, size))
+      def calc(algo:int, x:int, y:int|None=None, precision:int=5, output:int=5, **kw) -> int:
+        out = alloc()
+        self.mulacc_stage(algo, x, const(0) if y is None else y, out, precision, output, **kw)
+        return out
+      def add(x:int, y:int) -> int: return calc(2, x, y, 4, 4)
+      def sub(x:int, y:int) -> int: return calc(4, x, y, 4, 4)
+      def mul(x:int, y:int) -> int: return calc(0, x, y, 4, 4, mul=True)
+      def lt(x:int, y:int) -> int: return calc(1, x, y, 4, 4, binary=True)
+      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
+      def exponent_scale(x:int) -> int: return mul(mul(mul(x, const(128)), const(256)), const(256))
+      def half_weights(value:int) -> int:
+        half = calc(-1, value, output=2)
+        return alloc(read(half, 8)+bytes(8)+read(half+16, 8))
+      def word_weights(value:int) -> int:
+        raw = read(value)
+        return alloc(b"".join(raw[i:i+2] for i in range(0, 16, 4))+bytes(8)+
+                     b"".join(raw[i:i+2] for i in range(16, 32, 4)))
+      def polynomial(weights:int, cs:tuple[float, ...], squared:bool=False) -> int:
+        value = const(cs[-1], "f")
+        for c in reversed(cs[:-1]):
+          value = calc(-1, value, bs_mul=weights)
+          if squared: value = calc(-1, value, bs_mul=weights)
+          value = calc(2, value, const(c, "f"))
+        return value
+      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8])+bytes(2*(8-count))
+      source = alloc(raw)
+      bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = calc(-1, source, precision=2)
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
+        weights = half_weights(reduced)
+        residual = calc(4, reduced, calc(-1, const(1., "f"), bs_mul=weights))
+        sine = calc(-1, polynomial(weights, (1., -1/6, 1/120, -1/5040, 1/362880, -1/39916800), True), bs_mul=weights)
+        cosine = polynomial(weights, (1., -1/2, 1/24, -1/720, 1/40320, -1/3628800), True)
+        # Keep the full FP32 reduction residual instead of rounding the angle to half.
+        sin_r = calc(2, sine, calc(-1, residual, bs_mul=half_weights(cosine)))
+        cos_r = calc(4, cosine, calc(-1, residual, bs_mul=half_weights(sine)))
+        odd = sub(q, mul(calc(4, mul(q, const(2)), const(1), 4, 4, shift=2), const(2)))
+        quadrant = sub(q, mul(calc(4, mul(q, const(2)), const(3), 4, 4, shift=3), const(4)))
+        sin_weight = word_weights(mul(const(0x3c00), sub(const(1), odd)))
+        cos_weight = word_weights(mul(const(0x3c00), odd))
+        scaled = calc(2, calc(-1, sin_r, bs_mul=sin_weight), calc(-1, cos_r, bs_mul=cos_weight))
+        sign_weight = word_weights(add(const(0x3c00), mul(const(32768), lt(const(1), quadrant))))
+        scaled = calc(-1, scaled, bs_mul=sign_weight)
+      elif op is Ops.EXP2:
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
+      if op is not Ops.SIN:
+        # The EXP2/LOG2 reduced fraction is exactly representable in half.
+        weights = half_weights(fraction)
+        poly = polynomial(weights, coefficients)
+        if op is Ops.EXP2:
+          scaled = add(poly, exponent_scale(exponent))  # Multiply by 2^n in the FP32 bit representation.
+        else:
+          scaled = calc(2, calc(-1, poly, bs_mul=weights), calc(-1, exponent, precision=4))
+      half = calc(-1, scaled, output=2)
+      packed = read(half, 8)+read(half+16, 8)
+      output_bits = alloc(b"".join(packed[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      negative = lt(const(32767), bits)
+      magnitude = sub(bits, mul(const(32768), negative))
+      if op is Ops.SIN:
+        # Three rounding boundaries remain after FP32 reduction; the correction is symmetric in the sign.
+        for code, correction in ((0x32b3, 1), (0x51f5, -1), (0x5cb0, -1)):
+          equal = sub(sub(const(1), lt(magnitude, const(code))), lt(const(code), magnitude))
+          output_bits = add(output_bits, mul(const(correction), equal))
+        output_bits = select(lt(const(0), magnitude), output_bits, bits)  # Keep sin(-0) = -0.
+        output_bits = select(lt(const(0x7bff), magnitude), const(0x7e00), output_bits)
+      elif op is Ops.EXP2:
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
+      output_bits = select(lt(const(0x7c00), magnitude), const(0x7e00), output_bits)
+      out = read(output_bits)
+      result.extend(typed_view(out[i*4:i*4+2], dtypes.half) for i in range(count))
+    return result
```

Advertise and dispatch the three ops:

```diff
 ops_map = {Ops.ADD: 2, Ops.MUL: 0, Ops.SUB: 4, Ops.NEG: 0, Ops.FDIV: 3, Ops.MAX: 0, Ops.RECIPROCAL: 3,
@@
-           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP, Ops.SQRT: CMP}
+           Ops.AND: CMP, Ops.XOR: CMP, Ops.OR: CMP, Ops.TRUNC: CMP, Ops.MULACC: CMP, Ops.CDIV: CMP, Ops.CMOD: CMP,
+           Ops.SQRT: CMP, Ops.EXP2: CMP, Ops.LOG2: CMP, Ops.SIN: CMP}
@@
   def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
@@
           elif u.op is Ops.SQRT and u.dtype == dtypes.half:
             values[u] = self.run_sqrt(src_values[0])
+          elif u.op in (Ops.EXP2, Ops.LOG2, Ops.SIN) and u.dtype == dtypes.half:
+            values[u] = self.run_math(u.op, src_values[0])
```

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

POW is already expanded by tinygrad's symbolic rewrite into LOG2, MUL, EXP2 and WHERE. A small probe gave NaN even for 2^3, while the reference gives 8. The old arithmetic WHERE lets an unselected NaN branch through because 0*NaN is NaN. A NaN exponent also reached the Python float-to-integer CAST and raised ValueError. These are the next issues to fix; adding POW to ops_map alone would not fix this rewritten graph.

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
-    # FP16 arithmetic selection; integers use exact raw-byte selection in run_integer.
+    # FP16 raw-bit selection reuses the exact integer WHERE path.
```

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

Update run_math to accept the private wider input/output modes:

```diff
@@ -1,5 +1,7 @@
+  def run_math(self, op:Ops, a:list, input_dtype:DType=dtypes.half, dtype:DType=dtypes.half) -> list:
-  def run_math(self, op:Ops, a:list) -> list:
     assert op in (Ops.EXP2, Ops.LOG2, Ops.SIN)
+    assert input_dtype == dtypes.half or (op is Ops.EXP2 and input_dtype == dtypes.float)
+    assert dtype == dtypes.half or (op is Ops.LOG2 and dtype == dtypes.float)
     base = self.dev.input_mem.dma_addr
     coefficients = (tuple(math.log(2)**k/math.factorial(k) for k in range(9)) if op is Ops.EXP2 else
                     tuple((-1)**k/((k+1)*math.log(2)) for k in range(16)))
@@ -27,7 +29,15 @@
       def sub(x:int, y:int) -> int: return calc(4, x, y, 4, 4)
       def mul(x:int, y:int) -> int: return calc(0, x, y, 4, 4, mul=True)
       def lt(x:int, y:int) -> int: return calc(1, x, y, 4, 4, binary=True)
+      def select(mask:int, yes:int, no:int) -> int:
+        if dtype != dtypes.float: return add(no, mul(sub(yes, no), mask))
+        # Select full FP32 bit patterns without overflowing a signed INT32 difference.
+        sy, sn = lt(yes, const(0)), lt(no, const(0))
+        hy, hn = mul(const(1073741824), sy), mul(const(1073741824), sn)
+        my, mn = add(add(yes, hy), hy), add(add(no, hn), hn)
+        magnitude = add(mn, mul(sub(my, mn), mask))
+        sign = add(sn, mul(sub(sy, sn), mask))
+        return add(magnitude, mul(const(-2147483648), sign))
-      def select(mask:int, yes:int, no:int) -> int: return add(no, mul(sub(yes, no), mask))
       def exponent_scale(x:int) -> int: return mul(mul(mul(x, const(128)), const(256)), const(256))
       def half_weights(value:int) -> int:
         half = calc(-1, value, output=2)
@@ -43,10 +53,10 @@
           if squared: value = calc(-1, value, bs_mul=weights)
           value = calc(2, value, const(c, "f"))
         return value
+      raw = b"".join(raw16(x, input_dtype) for x in a[start:start+8])+bytes(input_dtype.itemsize*(8-count))
-      raw = b"".join(raw16(x, dtypes.half) for x in a[start:start+8])+bytes(2*(8-count))
       source = alloc(raw)
+      bits = source if input_dtype == dtypes.float else alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
+      x = source if input_dtype == dtypes.float else calc(-1, source, precision=2)
-      bits = alloc(b"".join(raw[i:i+2]+bytes(2) for i in range(0, 16, 2)))
-      x = calc(-1, source, precision=2)
       if op is Ops.SIN:
         multiple = calc(7, calc(2, calc(-1, const(2/math.pi, "f"), bs_mul=half_weights(x)), const(0.5, "f")))
         q = calc(-1, multiple, output=4)
@@ -90,14 +100,25 @@
         weights = half_weights(fraction)
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
-      half = calc(-1, scaled, output=2)
-      packed = read(half, 8)+read(half+16, 8)
-      output_bits = alloc(b"".join(packed[i:i+2]+bytes(2) for i in range(0, 16, 2)))
-      negative = lt(const(32767), bits)
-      magnitude = sub(bits, mul(const(32768), negative))
       if op is Ops.SIN:
         # Three rounding boundaries remain after FP32 reduction; the correction is symmetric in the sign.
         for code, correction in ((0x32b3, 1), (0x51f5, -1), (0x5cb0, -1)):
@@ -107,16 +128,17 @@
         output_bits = select(lt(const(0x7bff), magnitude), const(0x7e00), output_bits)
       elif op is Ops.EXP2:
         # FP32 Horner lands on a half midpoint for this input; the exact exponential rounds upward.
+        code = 0x3a38a000 if input_dtype == dtypes.float else 0x11c5
+        exceptional = sub(sub(const(1), lt(bits, const(code))), lt(const(code), bits))
-        exceptional = sub(sub(const(1), lt(bits, const(0x11c5))), lt(const(0x11c5), bits))
         output_bits = add(output_bits, exceptional)
+        overflow = mul(sub(const(1), negative), lt(const(0x417fffff if input_dtype == dtypes.float else 0x4bff), magnitude))
+        underflow = mul(negative, lt(const(0x41c7ffff if input_dtype == dtypes.float else 0x4e3f), magnitude))
-        overflow = mul(sub(const(1), negative), lt(const(0x4bff), magnitude))
-        underflow = mul(negative, lt(const(0x4e3f), magnitude))
         output_bits = select(overflow, const(0x7c00), select(underflow, const(0), output_bits))
       else:
+        output_bits = select(lt(const(0x7bff), magnitude), const(infinity), output_bits)
+        output_bits = select(negative, const(nan), output_bits)
+        output_bits = select(lt(const(0), magnitude), output_bits, const(-8388608 if dtype == dtypes.float else 0xfc00))
+      output_bits = select(lt(const(0x7f800000 if input_dtype == dtypes.float else 0x7c00), magnitude), const(nan), output_bits)
-        output_bits = select(lt(const(0x7bff), magnitude), const(0x7c00), output_bits)
-        output_bits = select(negative, const(0x7e00), output_bits)
-        output_bits = select(lt(const(0), magnitude), output_bits, const(0xfc00))
-      output_bits = select(lt(const(0x7c00), magnitude), const(0x7e00), output_bits)
       out = read(output_bits)
+      result.extend(typed_view(out[i*4:i*4+dtype.itemsize], dtype) for i in range(count))
-      result.extend(typed_view(out[i*4:i*4+2], dtypes.half) for i in range(count))
     return result
```

While testing full-width selection, we found that subtracting INT32_MIN does not work as ordinary INT32 subtraction on this path. Removing a sign bit instead uses two additions of 2^30. This avoids negating INT32_MIN inside EW SUB.

Add the composed helper:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
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

### Move the integer CAST onto the NPU

POW checks whether its exponent is an integer. That CAST still used Python and raised on NaN.

The FP32→INT32 converter rounds to nearest. Toggling CVT_TYPE and CVT_ROUND did not change it. For finite values, remove the fraction first:

```text
trunc(x) = max(floor(x), min(ceil(x), 0))
         → FP32-to-INT32 converter
```

FLOOR, CEIL, MIN and MAX run in FP32 here. Converting the resulting integer-valued float no longer changes the answer. Nonfinite/out-of-range conversion still follows the hardware converter's saturation behavior; this is not a claim that every backend defines those CASTs identically.

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
+  def run_cast(self, a:list, src_dtype:DType, dtype:DType) -> list:
+    assert src_dtype in (dtypes.half, dtypes.float, dtypes.int) and dtype in (dtypes.half, dtypes.float, dtypes.int)
+    result:list = []
+    base = self.dev.input_mem.dma_addr
+    for start in range(0, len(a), 8):
+      count = min(8, len(a)-start)
+      to_mv(self.dev.input_buf, 8*src_dtype.itemsize)[:] = b"".join(raw16(x, src_dtype) for x in a[start:start+8]) + \
+        bytes(src_dtype.itemsize*(8-count))
+      to_mv(self.dev.input_buf+64, 32)[:] = bytes(32)
+      precision = {dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[src_dtype]
+      self.mulacc_stage(-1, base, base+64, base+128, precision=precision)
+      value = base+128
+      if dtype == dtypes.int:
+        # The converter rounds to nearest. Remove the fraction before converting to get truncation toward zero.
+        self.mulacc_stage(7, value, base+64, base+192)
+        self.mulacc_stage(8, value, base+64, base+256)
+        self.mulacc_stage(1, base+256, base+64, base+320)
+        self.mulacc_stage(0, base+192, base+320, base+384)
+        value = base+384
+      self.mulacc_stage(-1, value, base+64, base+448, output={dtypes.half: 2, dtypes.float: 5, dtypes.int: 4}[dtype])
+      raw = (bytes(to_mv(self.dev.input_buf+448, 8))+bytes(to_mv(self.dev.input_buf+464, 8)) if dtype == dtypes.half else
+             bytes(to_mv(self.dev.input_buf+448, 32)))
+      result.extend(typed_view(raw[i*dtype.itemsize:(i+1)*dtype.itemsize], dtype) for i in range(count))
+    return result
@@
         elif u.op is Ops.CAST:
@@
                                      dtype=u.dtype)
+          elif src_dtypes[0] in (dtypes.half, dtypes.float, dtypes.int) and u.dtype in (dtypes.half, dtypes.float, dtypes.int):
+            values[u] = self.run_cast(src_values[0], src_dtypes[0], u.dtype)
```

### Compare the FP16 encodings

The old floating-point comparison formula rejects NaN inputs. We now have exact integer comparison, so use the half encodings directly:

| Input property | NPU check |
|----------------|-----------|
| Sign | bits > 32767 |
| Magnitude | bits - sign*32768 |
| NaN | magnitude > 0x7c00 |
| Both zero | magnitude_a + magnitude_b = 0 |
| Equal | Same bits or both zero, and neither input is NaN |
| Less | Different signs select the negative input; same negative signs reverse the bit comparison |

Equality of integer encodings is 1 - (a < b) - (b < a), using native integer comparisons. NaN makes CMPEQ and CMPLT false; CMPNE is the inverse of equality. This also makes +0 and -0 compare equal.

The final 0/1 mask goes through the NPU half-to-byte converter, so the result is still a bool byte:

```diff
 class RockchipProgram(Program['RockchipDevice']):
@@
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
+      lhs, rhs = (alloc(b"".join(bytes(raw16(x, dtypes.half))+bytes(2) for x in xs[start:start+8])+bytes(4*(8-count)))
+                  for xs in (a, b))
+      sa, sb = lt(const(32767), lhs), lt(const(32767), rhs)
+      ma, mb = sub(lhs, mul(const(32768), sa)), sub(rhs, mul(const(32768), sb))
+      valid = mul(sub(one, lt(const(0x7c00), ma)), sub(one, lt(const(0x7c00), mb)))
+      both_zero = sub(one, lt(zero, calc(2, ma, mb)))
+      ab, ba = lt(lhs, rhs), lt(rhs, lhs)
+      if op is Ops.CMPLT:
+        less = select(calc(5, sub(sa, sb)), sa, select(sa, ba, ab))
+        mask = mul(mul(valid, sub(one, both_zero)), less)
+      else:
+        equal = sub(one, mul(calc(2, ab, ba), sub(one, both_zero)))
+        mask = mul(valid, equal)
+        if op is Ops.CMPNE: mask = sub(one, mask)
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

The combined run gave **7 passed, 1 failed in 557.49s**. `test_pow_full`, `test_pow`, `test_pow_neg_inf_frac_exponent` and `test_pow_zero_exponent` passed. `test_pow_const` failed at `x ** 8.0`: 617 / 2925 values exceeded the tolerance. The same test on tinygrad CPU, with the same HALF/NOOPT settings, failed at the same expression with the same 617 mismatches. This expression becomes repeated FP16 MULs, so the wider LOG2/EXP2 path does not apply.

This is not a claim of every POW edge case. For example, `Tensor([1.0]) ** Tensor([nan])` returns NaN on tinygrad CPU too. A Python scalar NaN exponent instead fails in the common `simplify_pow` rewrite before reaching either backend.

Now run the whole file, serially. It collects **433 tests**. The longer timeout allows the slower NPU decompositions to finish; it does not change the comparisons or skip tests.

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest -n0 -vv --tb=short test/backend/test_ops.py
```

`test_all_large` hit the 600-second per-test timeout; its log contains the timeout dump, and the process exited with signal 11 while dumping it. We continued the unattempted cases in separate serial pytest processes, using the same settings and unchanged source. Each process kept the 600-second timeout, with a 660-second external limit. Timeouts are not passes or skips.

The complete sweep gave **209 passed, 195 failed, 21 timed out, 8 skipped — 433 methods total**. This is the combined result, not one pytest summary. All collected methods were accounted for, and the runtime and test-file hashes stayed unchanged. The eight skips came from existing test decorators.

Count collected test methods, not the green parent line alone: pytest can print `PASSED` for a method whose subtests failed, then exit with code 1. The summary checks the exit code and `SUBFAILED` entries too. Both asymmetric-padding convolution methods failed this way on unsupported FP32 ADD; they are not passes.

Progress so far

| Group | Paths implemented |
|-------|-------------------|
| `GroupOp.Unary` | `NEG`, `RECIPROCAL`, `TRUNC` |
| | `SQRT`, `EXP2`, `LOG2`, `SIN` |
| `GroupOp.Binary` | `ADD`, `MUL`, `SUB`, `FDIV`, `MAX` |
| | `CMPEQ`, `CMPNE`, `CMPLT` |
| | `AND`, `OR`, `XOR`, `SHL`, `SHR` |
| | `CDIV`, `CMOD`, `FLOORDIV`, `FLOORMOD` |
| | `THREEFRY`, `POW` |
| `GroupOp.Ternary` | `WHERE`, `MULACC` |
| Extras | `CAST`, `BITCAST` |
| **Total** | **30 / 30 paths** |

FLOORDIV, FLOORMOD, THREEFRY and POW use tinygrad's existing decompositions. This count is not 30 fully supported ops for every dtype. General FP32 arithmetic is still gated; the FP32 stages above are private helpers. INT16 SHL still saturates on overflow, and vector shift calls require a uniform count. Some CAST combinations still use the Python fallback. BITCAST only reinterprets storage; it does not calculate new values.

The full sweep uses forward-only FP16 with NOOPT=1. It does not establish backward, default-FP32 or optimized-kernel coverage.

The first failed cases were also checked on CPU with the same settings. `test_9_gemm`, `test_acos` and `test_all` passed there; `test_acosh` failed there too (**3 passed, 1 failed in 4.19s**). A CPU failure does not excuse an NPU mismatch, but helps separate common HALF accuracy issues from backend-specific ones.

The next CPU baseline passed `test_asin` but failed `test_asinh`, `test_atan` and `test_atanh` (**1 passed, 3 failed in 5.32s**). Rockchip failed `test_asin` with 892 / 2925 values outside tolerance, so that one is not explained by the CPU baseline.

`test_any`, `test_simple_cummin` and `test_slice_fancy_indexing_tuple_indices` exposed another limit:

```text
RuntimeError: ROCKCHIP intermediate buffer exhausted
```

Each `run_npu` result keeps a page until the workgroup finishes. A long reduction can use up those pages before the next workgroup resets the offset. The guard stops instead of overwriting live results; this needs scratch-buffer reuse, not another ALU op or a CPU fallback.

Not every failed method reaches the backend. With DEFAULT_FLOAT=HALF, `test_bitcast` asks Torch to view a `(3, 3)` half tensor as INT32. Torch rejects the odd last dimension before tinygrad runs. It stays in the failed count, but is a reference-side failure for this configuration, not an NPU BITCAST result mismatch.

Likewise, a pass is not proof that every conversion ran on the NPU. `test_cast` passed, but bool → FP32 still reaches the existing Python CAST fallback. The dedicated half/FP32/INT32 conversion helper and normalized-mask conversions are separate NPU paths.

The five forward comparison methods (`test_cmp_eq`, `test_cmp_gt`, `test_cmp_ge`, `test_cmp_lt`, `test_cmp_le`) passed. Their shared special-value loop includes infinities but leaves NaN commented out; the separate raw-bit checks above cover NaNs. `test_cmp_lt_backwards` and `test_cmp_ne_backwards` failed on FP32 ADD. FORWARD_ONLY only disables gradients in the helper, not methods that explicitly call `backward()`.

`test_copysign` passed, but `test_copysign_exact` returned +1 where Torch expected -1. The same exact test passed on CPU in 3.09s. `copysign` uses `(b < 0) | (1/b < 0)` to detect the sign, including negative zero. The generic late rewrite replaces RECIPROCAL with FDIV when FDIV is advertised, so the direct RECIPROCAL guard does not protect this lowered path. Special-value FDIV behavior remains a limit; the ordinary copysign pass does not establish it.

`test_cummax` and `test_cummin` passed. `test_cumprod` failed on its `(20, 30)` case with 23 / 600 mismatches; CPU failed at the same case with the same mismatch count and maximum relative error (0.001699). `test_cumsum` instead stopped at the FP32 ADD gate.

`test_div` and `test_div_int` passed, but `test_div_naninf` returned positive infinities where some outputs should be negative. Ordinary division passing does not establish signed-zero/infinity behavior.

`test_gelu` failed with 233 / 2925 mismatches. CPU with the same HALF settings also failed, but with 150 / 2925 mismatches. So there is a shared precision issue, but it does not explain all of Rockchip's error.

`test_hardsigmoid` passed. Its extreme-value case failed on both Rockchip and CPU with the same two mismatches out of six, including 0.0001220703125 instead of 0 at the boundary.

`test_lshift`, `test_lshift_int16`, `test_lshift_signed`, `test_rshift` and `test_rshift_signed` passed in this sweep. `test_masked_fill` passed too, but `test_masked_select` reached unsupported `Ops.WHERE` with `dtypes.bool`. The implemented WHERE paths do not cover every output dtype.

`test_mul_naninf` passed. `test_mulacc_with_zero_strides` failed on dtype, not values: tinygrad returned FP32 and Torch FP16, both containing 3. The CPU baseline had the same failure under these HALF settings; this is separate from the direct MULACC checks above.

`test_pad_replicate_mode` reached its zero-height negative-padding case, where Torch raised `Calculated output H: 0 W: 5 must be >= 1`. This stays in the failed count, but that case stopped in the reference before tinygrad ran.

`test_pow` passed in the full sweep. `test_pow_const` reproduced the 617 / 2925 HALF mismatches above; `test_pow_const_direct` instead stopped at unsupported FP32 FDIV. These are different limits, not a failure of the same path.

`test_pow_full` also passed, in 316.92s. `test_pow_int` has an existing unconditional skip; it is not counted as working.

`test_quick_gelu` failed with 159 / 2925 mismatches, versus 140 / 2925 on CPU with the same settings. Like GELU, the shared HALF failure does not explain all of the NPU error.

Two scatter-reduce cases stopped in Torch: `test_scatter_reduce_errors` expected a RuntimeError that Torch did not raise, and `test_scatter_reduce_prod_zeros` hit a source/destination dtype mismatch. Both remain failed methods, not NPU arithmetic mismatches.

`test_stack` reached its final scalar assertion: HALF stores 3.14 as 3.140625, but that assertion compares against NumPy's 3.14 with rtol=1e-7. It remains a failure under these HALF settings; `test_stack_max` and `test_stack_slice` passed.

`test_tanh` failed with 611 / 2925 mismatches on Rockchip, versus 131 / 2925 on CPU with the same settings. The CPU baseline also fails, but the larger NPU error still needs investigation.

Finally rerun the focused hardware checks together, including direct MULACC:

```bash
$ TEST_TIMEOUT=600 NOOPT=1 FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP \
    python -m pytest -n0 -v --tb=short test/device/test_rockchip_integer.py \
    test/backend/test_uops.py::TestFloatUOps::test_mulacc

18 passed, 124 subtests passed in 434.68s
```

These cover every half encoding for SQRT/EXP2/LOG2/SIN, raw comparison and CAST checks, integer boundaries, wide shifts, THREEFRY and special-value WHERE. They do not replace the full sweep: its result remains **209 / 433 passed**.

Targeted lint passed for `ops_rockchip.py` and the focused test. Whole-tree mypy still reports 47 errors in `ops_rockchip_ref.py`; whole-tree ruff reports 142 errors in the reference/generated files. Those checks are not green.
