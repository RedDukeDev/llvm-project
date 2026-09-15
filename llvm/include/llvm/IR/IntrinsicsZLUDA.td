// ZLUDA-specific generic intrinsics. Should be removed by lowering passes before reaching codegen.

// Input intrinsics, this is what ZLUDA emits, all of them use NV hardware
// layout as described in PTX documentation. None of them are lowered to
// AMD hardware instructions

// The return type should be llvm_v4f32_ty to match the PTX definition,
// but if the return type is llvm_v4i32_ty it's easier to perform
// D matrix reconstitution
def int_zluda_mma_m16n8k16_f32_bf16_bf16_f32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_mma_m16n8k16_f32_f16_f16_f32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_mma_m16n8k32_s32_s8_s8_s32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// FP8 (e4m3), gfx12 native path only. Same PTX shape as the s8 form above --
// m16n8k32, 8-bit elements -- the difference is the accumulator: f32 packed
// as i32, not s32, and the AMD-side lowering picks the fp8_fp8 WMMA rather
// than the iu8 one. Everything else about the operand layout is identical, so
// this is deliberately declared with the exact same register shape as the s8
// intrinsic above rather than invented afresh.
def int_zluda_mma_m16n8k32_f32_fp8_fp8_f32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v2i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// Successful matrix layout change. We convert inputs and split outputs

// C matrix and D matrix conversion is shared between all fp16,bf16, I8
def int_zluda_cmatrix_concatenate_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_dmatrix_split_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v8i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_dmatrix_trunc_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty], [llvm_v8f32_ty],
  [IntrNoMem, IntrConvergent]
>;

// BF16/FP16
def int_zluda_amatrix_convert_amd_nv16x16 : DefaultAttrsIntrinsic<
  [llvm_v16i16_ty], [llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_bmatrix_concatenate_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_v16i16_ty], [llvm_v2i32_ty, llvm_v2i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// I8
def int_zluda_amatrix_split_amd16x16_nv16x32 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_bmatrix_reshape_amd16x16_nv32x8 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// FP8 on GFX12. NVIDIA and AMD do not just differ in the usual way here: GFX12
// halves the 8-bit WMMA operand width relative to GFX11 (v2i32, not v4i32) and
// uses a different lane distribution for K, confirmed from the AMD Matrix
// Instruction Calculator (github.com/ROCm/amd_matrix_instruction_calculator)
// against the AMD RDNA4 ISA reference and NVIDIA's own PTX ISA guide, section
// 9.7.14.5.10 (PTX ISA 8.8). Even the accumulator layout differs from GFX11's,
// so none of the existing GFX11 conversions above are reusable -- this is a
// full parallel set, not a width change to the existing ones.
def int_zluda_amatrix_convert_fp8_amd16x16_nv16x32 : DefaultAttrsIntrinsic<
  [llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_bmatrix_convert_fp8_amd16x16_nv32x8 : DefaultAttrsIntrinsic<
  [llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_cmatrix_concatenate_fp8_amd16x16_nv16x8 : DefaultAttrsIntrinsic<
  [llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty],
  [IntrNoMem, IntrConvergent]
>;
def int_zluda_dmatrix_split_fp8_nv16x8_amd16x16 : DefaultAttrsIntrinsic<
  [llvm_v4i32_ty, llvm_v4i32_ty], [llvm_v8i32_ty],
  [IntrNoMem, IntrConvergent]
>;

// TODO: custom classes to reduce code duplication in ZLUDA intrinsics
