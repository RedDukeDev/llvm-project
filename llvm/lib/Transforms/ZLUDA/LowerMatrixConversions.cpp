#include "llvm/Transforms/ZLUDA/LowerMatrixConversions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include <array>
#include <cstdlib>
#include <optional>
#include <utility>

using namespace llvm;

std::pair<Value *, uint32_t>
getLogicalCoordinatesForAMatrixAMDPhysicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Lane,
                                                      uint32_t vGPR) {
  // A i: (lane % 16)
  Value *Row = Builder.CreateAnd(Lane, 15, "row");

  // A k: 2 * GPR_num + floor(GPR_bits / 16)
  uint32_t Column = 2 * vGPR;

  return {Row, Column};
}

std::pair<uint32_t, Value *>
getLogicalCoordinatesForBMatrixAMDPhysicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Lane,
                                                      uint32_t vGPR) {
  auto [Row, Column] = getLogicalCoordinatesForAMatrixAMDPhysicalCoordinates(
      Builder, Lane, vGPR);

  return {Column, Row};
}

std::pair<Value *, Value *>
getLogicalCoordinatesForCMatrixAMDPhysicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Lane,
                                                      uint32_t vGPR) {
  // C or D i: 2 * GPR_num + floor(lane / 16)
  Value *Row = Builder.CreateAdd(Builder.getInt32(2 * vGPR),
                                 Builder.CreateLShr(Lane, 4), "row");

  // C or D j: (lane % 16)
  Value *Column = Builder.CreateAnd(Lane, 15, "column");

  return {Row, Column};
}

std::pair<Value *, Value *>
getLogicalCoordinatesForDMatrixNVPhysicalCoordinates(IRBuilder<> &Builder,
                                                     Value *Lane,
                                                     uint32_t ElementIdx) {
  // groupID = %laneid >> 2
  Value *GroupID = Builder.CreateLShr(Lane, 2, "group.id");
  // threadID_in_group = %laneid % 4
  Value *ThreadIDInGroup = Builder.CreateAnd(Lane, 3, "thread.id.in.group");

  // row = groupID + (i / 2) * 8
  Value *Row =
      Builder.CreateAdd(GroupID, Builder.getInt32((ElementIdx / 2) * 8), "row");

  // col = (threadID_in_group * 2) + (i & 0x1)
  Value *Column = Builder.CreateAdd(Builder.CreateShl(ThreadIDInGroup, 1),
                                    Builder.getInt32(ElementIdx % 2), "column");

  return {Row, Column};
}

// TODO: refactor shared logic with
// getBMatrixPhysicalCoordinatesForLogicalCoordinates
std::pair<Value *, Value *>
getAMatrixNVPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                     Value *Row,
                                                     uint32_t Column) {
  // groupID = row % 8
  Value *GroupID = Builder.CreateAnd(Row, 7);

  uint32_t ThreadIDInGroup = (Column % 8) / 2;

  // laneid = groupID * 4 + threadID_in_group
  Value *Lane = Builder.CreateAdd(Builder.CreateShl(GroupID, 2),
                                  Builder.getInt32(ThreadIDInGroup), "nv.lane");

  // PackedIdx = (row / 8) + (col / 8) * 2
  Value *PackedIdx =
      Builder.CreateAdd(Builder.CreateLShr(Row, 3),
                        Builder.getInt32((Column >> 3) * 2), "nv.idx");

  return {Lane, PackedIdx};
}

std::tuple<Value *, uint32_t, Value *>
getBMatrixNVPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                     uint32_t Row,
                                                     Value *Column) {
  Value *IsInFirst =
      Builder.CreateICmpULT(Column, Builder.getInt32(8), "is.in.first");

  // groupID = col % 8
  Value *GroupID = Builder.CreateAnd(Column, 7);

  // threadID_in_group = (row % 8) / 2
  uint32_t ThreadIDInGroup = (Row % 8) / 2;

  // laneid = groupID * 4 + ThreadIDInGroup
  Value *Lane = Builder.CreateAdd(Builder.CreateShl(GroupID, 2),
                                  Builder.getInt32(ThreadIDInGroup), "nv.lane");

  // PackedIdx = row / 8
  uint32_t ElementIdx = Row / 8;

  return {Lane, ElementIdx, IsInFirst};
}

std::tuple<Value *, Value *, Value *>
getCMatrixNVPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                     Value *Row,
                                                     Value *Column) {
  Value *IsInFirst =
      Builder.CreateICmpULT(Column, Builder.getInt32(8), "is.in.first");

  // groupID = row % 8
  Value *GroupID = Builder.CreateAnd(Row, 7);

  // threadID_in_group = (col % 8) / 2
  Value *ThreadIDInGroup =
      Builder.CreateLShr(Builder.CreateAnd(Column, 7), 1, "thread.id.in.group");

  // laneid = groupID * 4 + ThreadIDInGroup
  Value *Lane = Builder.CreateAdd(Builder.CreateShl(GroupID, 2),
                                  ThreadIDInGroup, "nv.lane");

  // i = (row / 8) * 2 + col % 2
  Value *ElementIdx =
      Builder.CreateAdd(Builder.CreateShl(Builder.CreateLShr(Row, 3), 1),
                        Builder.CreateAnd(Column, 1));

  return {Lane, ElementIdx, IsInFirst};
}

std::pair<Value *, Value *>
getDMatrixAMDPhysicalCoordinatesForLogicalCoordinates(IRBuilder<> &Builder,
                                                      Value *Row,
                                                      Value *Column) {
  // C or D[i][j] GPR: floor(i / 2)
  Value *vGPR = Builder.CreateLShr(Row, 1, "element.idx");

  // C or D[i][j] Lane: ((16 * i) % 32) + j
  Value *Lane = Builder.CreateAdd(
      Builder.CreateAnd(Builder.CreateShl(Row, 4), 31), Column, "amd.lane");

  return {Lane, vGPR};
}

static IntrinsicInst *getMatrixConversion(Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (II) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::zluda_amatrix_convert_amd_nv16x16:
    case Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16:
    case Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8:
    case Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8:
    case Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16:
    case Intrinsic::zluda_amatrix_split_amd16x16_nv16x32:
    case Intrinsic::zluda_bmatrix_reshape_amd16x16_nv32x8:
    case Intrinsic::zluda_amatrix_convert_fp8_amd16x16_nv16x32:
    case Intrinsic::zluda_bmatrix_convert_fp8_amd16x16_nv32x8:
    case Intrinsic::zluda_cmatrix_concatenate_fp8_amd16x16_nv16x8:
    case Intrinsic::zluda_dmatrix_split_fp8_nv16x8_amd16x16:
      return II;
    }
  }
  return nullptr;
}

// Diagnostic: whether to skip the cross-lane move and hand back the lane's own
// value.
//
// On the generic targets the FP8 path has been measured at 85% fragment
// translation and 0% multiply, but "translation" is two things: widening e4m3
// to f16, and moving data between lanes to turn NVIDIA's fragment layout into
// AMD's. A static instruction count cannot tell them apart; this probe can.
//
// With the probe on, every ds_bpermute this file emits disappears along with
// the lane arithmetic feeding it, while every extract, select and insert
// around them stays exactly where it was. The image is meaningless by
// construction; only the clock is read. The difference is what the cross-lane
// traffic costs, and nothing else.
static bool probeNoCrossLane() {
  static const bool on = ::getenv("ZLUDA_PROBE_NO_CROSS_LANE") != nullptr;
  return on;
}

static Value *bpermuteLane(IRBuilder<> &Builder, Value *Lane, Value *X,
                           const Twine &Name = "") {
  if (auto *C = dyn_cast<Constant>(X)) {
    return C;
  }

  if (probeNoCrossLane()) {
    return X;
  }

  Value *BitCast = Builder.CreateBitCast(X, Builder.getInt32Ty());

  Value *Permuted = Builder.CreateIntrinsic(
      Builder.getInt32Ty(), Intrinsic::amdgcn_ds_bpermute,
      {Builder.CreateShl(Lane, 2), BitCast}, nullptr, Name);

  return Builder.CreateBitCast(Permuted, X->getType());
}

// Whether to move data between lanes with permlane16/permlanex16 instead of
// ds_bpermute.
//
// ds_bpermute is a gather through local memory, and on the generic targets it
// is 57% of the hottest kernel: 176
// of them in every group call, with 78 s_waitcnt riding along. permlane16 and
// permlanex16 are GFX10+ vector-unit permutations of a 16-lane row, taken from
// the same row or from the other one, with packed 4-bit selectors that are
// uniform across the wave and shared by both rows, as measured on the
// hardware. Every conversion in this file moves
// data along a map that depends on the lane alone, so each such map is at most
// four permlanes and a per-lane select on a constant mask, and two when the
// source depends on the column alone.
//
// On by default, measured in the real kernel -- a synthetic benchmark could not
// tell the two apart, only the kernel knows what the memory round trips were
// costing. gfx11-generic, alternated, same build for both: the hottest kernel
// 71.4 -> 60.7 ms (-15%), the whole network 406 -> 381 ms (-6.2%), the output
// image bit-identical in every run. ZLUDA_PERMLANE=0 turns it off. Not yet
// measured on the fused native tiers (gfx1100/1101), RDNA2 or RDNA4.
static bool permlaneRequested() {
  static const bool on = [] {
    const char *v = ::getenv("ZLUDA_PERMLANE");
    return v == nullptr || v[0] != '0';
  }();
  return on;
}

// The value a lane expression takes in one concrete lane, or nothing if the
// expression is not plain arithmetic over the lane number and constants.
//
// The conversions below build their source lane with IRBuilder from the lane
// number, so evaluating that tree for lanes 0..31 recovers the fixed map the
// ds_bpermute would have gathered along -- without restating the formulas a
// second time, where they could drift from the ones that are actually emitted.
static std::optional<uint32_t> evalLane(Value *V, Value *LaneID, uint32_t Lane,
                                        unsigned Depth = 0) {
  if (V == LaneID)
    return Lane;
  if (auto *C = dyn_cast<ConstantInt>(V))
    return uint32_t(C->getZExtValue());
  auto *BO = dyn_cast<BinaryOperator>(V);
  if (!BO || Depth > 32)
    return std::nullopt;
  std::optional<uint32_t> L = evalLane(BO->getOperand(0), LaneID, Lane, Depth + 1);
  std::optional<uint32_t> R = evalLane(BO->getOperand(1), LaneID, Lane, Depth + 1);
  if (!L || !R)
    return std::nullopt;
  const uint32_t A = *L, B = *R;
  switch (BO->getOpcode()) {
  case Instruction::Add:
    return A + B;
  case Instruction::Sub:
    return A - B;
  case Instruction::Mul:
    return A * B;
  case Instruction::And:
    return A & B;
  case Instruction::Or:
    return A | B;
  case Instruction::Xor:
    return A ^ B;
  case Instruction::Shl:
    if (B < 32)
      return A << B;
    return std::nullopt;
  case Instruction::LShr:
    if (B < 32)
      return A >> B;
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

class LowerMatrixConversions {
public:
  LowerMatrixConversions(Function &F) : F(F), LaneID(nullptr) {}
  bool lower();

private:
  bool lowerBB(BasicBlock &BB);
  void lowerConversion(IntrinsicInst *Conversion);

  Value *aMatrixConvert(IRBuilder<> &Builder, Value *NVFragment);
  Value *aMatrixConvertI8Half(IRBuilder<> &Builder, Value *NVLowReg,
                              Value *NVHighReg);
  Value *bMatrixConcatenate(IRBuilder<> &Builder, Value *NVFragmentFirst,
                            Value *NVFragmentSecond);
  Value *cMatrixConcatenate(IRBuilder<> &Builder, Value *NVFragmentFirst,
                            Value *NVFragmentSecond);
  Value *dMatrixSplit(IRBuilder<> &Builder, Value *AMDFragment, bool Truncate);

  // GFX12 FP8: a full parallel set, laid out for the derivation in one place.
  // See the comment on the intrinsics themselves (IntrinsicsZLUDA.td) for why
  // none of the functions above carry over.
  Value *aMatrixConvertFp8Half(IRBuilder<> &Builder, Value *NvLowReg,
                               Value *NvHighReg);
  Value *bMatrixConvertFp8Half(IRBuilder<> &Builder, Value *NvRegLeft,
                               Value *NvRegRight);
  Value *cMatrixConcatenateFp8(IRBuilder<> &Builder, Value *NVFragmentFirst,
                               Value *NVFragmentSecond);
  Value *dMatrixSplitFp8(IRBuilder<> &Builder, Value *AMDFragment);

  Value *getLaneNumber();
  // The value of X in the lane SrcLane names, for every lane: permlanes where
  // the map is fixed and asked for, ds_bpermute otherwise.
  Value *permuteLane(IRBuilder<> &Builder, Value *SrcLane, Value *X,
                     const Twine &Name = "");
  // An i1 that is bit `lane` of Mask, built once per function next to the
  // lane number so every permlane select shares it.
  Value *laneCondition(uint32_t Mask);

  Function &F;
  Value *LaneID;
  DenseMap<uint32_t, Value *> LaneConditions;
};

Value *LowerMatrixConversions::getLaneNumber() {
  if (LaneID == nullptr) {
    IRBuilder<> Builder(&*F.getEntryBlock().begin());
    LaneID =
        Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {},
                                {Builder.getInt32(-1), Builder.getInt32(0)});
  }

  return LaneID;
}

Value *LowerMatrixConversions::laneCondition(uint32_t Mask) {
  auto It = LaneConditions.find(Mask);
  if (It != LaneConditions.end())
    return It->second;
  auto *Lane = cast<Instruction>(getLaneNumber());
  IRBuilder<> Builder(Lane->getNextNode());
  Value *Bit = Builder.CreateAnd(Builder.CreateLShr(Builder.getInt32(Mask), Lane),
                                 Builder.getInt32(1));
  Value *Cond = Builder.CreateICmpNE(Bit, Builder.getInt32(0), "lane.cond");
  LaneConditions[Mask] = Cond;
  return Cond;
}

Value *LowerMatrixConversions::permuteLane(IRBuilder<> &Builder,
                                           Value *SrcLane, Value *X,
                                           const Twine &Name) {
  if (isa<Constant>(X) || probeNoCrossLane() || !permlaneRequested())
    return bpermuteLane(Builder, SrcLane, X, Name);

  // The map this gather follows, lane by lane.
  Value *Lane = getLaneNumber();
  std::array<uint32_t, 32> Src;
  for (uint32_t L = 0; L < 32; ++L) {
    std::optional<uint32_t> S = evalLane(SrcLane, Lane, L);
    if (!S || *S >= 32) {
      // Which gathers stay on ds_bpermute, and why. Every map this file builds
      // evaluates today, so this should stay silent; it is here so that a new
      // conversion whose map does not will say so instead of quietly keeping
      // its memory round trips.
      if (::getenv("ZLUDA_PERMLANE_DEBUG")) {
        errs() << "[permlane] keeping ds_bpermute in " << F.getName() << ": "
               << (S ? "source lane out of range" : "map not evaluable")
               << " (lane " << L << (S ? " -> " + std::to_string(*S) : "")
               << ")\n  expression: " << *SrcLane << "\n";
      }
      return bpermuteLane(Builder, SrcLane, X, Name);
    }
    Src[L] = *S;
  }

  // Split it by kind: a lane reads either from its own row (permlane16) or
  // from the other one (permlanex16), and each kind has one column selector
  // per destination row. Indexed [cross][row][column].
  uint32_t Sel[2][2][16] = {};
  bool Used[2][2][16] = {};
  bool KindUsed[2] = {};
  bool RowUsed[2][2] = {};
  uint32_t CrossMask = 0;
  for (uint32_t L = 0; L < 32; ++L) {
    const unsigned Row = L >> 4, Col = L & 15;
    const unsigned Cross = (Src[L] >> 4) != Row;
    Sel[Cross][Row][Col] = Src[L] & 15;
    Used[Cross][Row][Col] = true;
    KindUsed[Cross] = true;
    RowUsed[Cross][Row] = true;
    if (Cross)
      CrossMask |= 1u << L;
  }

  Type *I32 = Builder.getInt32Ty();
  Value *X32 = Builder.CreateBitCast(X, I32);

  // One permlane of a kind. Row is 0 or 1 for that row's selectors, 2 for the
  // two rows' selectors merged, which is only called when they do not clash.
  auto EmitOne = [&](unsigned Cross, unsigned Row) -> Value * {
    uint32_t Lo = 0, Hi = 0;
    for (unsigned Col = 0; Col < 16; ++Col) {
      bool In0 = Used[Cross][0][Col], In1 = Used[Cross][1][Col];
      bool Use = Row == 2 ? (In0 || In1) : Used[Cross][Row][Col];
      if (!Use)
        continue;
      uint32_t S = Row == 2 ? (In0 ? Sel[Cross][0][Col] : Sel[Cross][1][Col])
                            : Sel[Cross][Row][Col];
      if (Col < 8)
        Lo |= S << (4 * Col);
      else
        Hi |= S << (4 * (Col - 8));
    }
    return Builder.CreateIntrinsic(
        I32,
        Cross ? Intrinsic::amdgcn_permlanex16 : Intrinsic::amdgcn_permlane16,
        {PoisonValue::get(I32), X32, Builder.getInt32(Lo), Builder.getInt32(Hi),
         Builder.getFalse(), Builder.getFalse()});
  };

  auto EmitKind = [&](unsigned Cross) -> Value * {
    if (!KindUsed[Cross])
      return nullptr;
    if (!RowUsed[Cross][0])
      return EmitOne(Cross, 1);
    if (!RowUsed[Cross][1])
      return EmitOne(Cross, 0);
    bool Clash = false;
    for (unsigned Col = 0; Col < 16; ++Col)
      Clash |= Used[Cross][0][Col] && Used[Cross][1][Col] &&
               Sel[Cross][0][Col] != Sel[Cross][1][Col];
    if (!Clash)
      return EmitOne(Cross, 2);
    Value *Row0 = EmitOne(Cross, 0);
    Value *Row1 = EmitOne(Cross, 1);
    return Builder.CreateSelect(laneCondition(0xFFFF0000u), Row1, Row0);
  };

  Value *Same = EmitKind(0);
  Value *Other = EmitKind(1);
  Value *Result;
  if (!Other)
    Result = Same;
  else if (!Same)
    Result = Other;
  else
    Result = Builder.CreateSelect(laneCondition(CrossMask), Other, Same, Name);
  return Builder.CreateBitCast(Result, X->getType());
}

Value *LowerMatrixConversions::aMatrixConvert(IRBuilder<> &Builder,
                                              Value *NVFragment) {
  Value *Lane = getLaneNumber();

  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false));

  for (uint32_t vGPR = 0; vGPR < 8; ++vGPR) {
    // AMD and NVIDIA formats will have the same packed bf16x2, so only need to
    // find coordinates for the first.
    auto [Row, Column] = getLogicalCoordinatesForAMatrixAMDPhysicalCoordinates(
        Builder, Lane, vGPR);

    auto [CudaTID, NVPackedIdx] =
        getAMatrixNVPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                             Column);

    // Extract the base packed value (even index: 0 or 2)
    Value *BasePackedIdx =
        Builder.CreateAnd(NVPackedIdx, Builder.getInt32(~1), "base.packed.idx");

    // reg0 = NVFragment[basePackedIdx], reg1 = NVFragment[basePackedIdx + 1]
    Value *Reg0 = Builder.CreateExtractElement(NVFragment, BasePackedIdx);
    Value *Reg1 = Builder.CreateExtractElement(
        NVFragment, Builder.CreateAdd(BasePackedIdx, Builder.getInt32(1)));

    // a_tmp0 = bpermute_lane(cudaTID, reg0)
    Value *ATmp0 = permuteLane(Builder, CudaTID, Reg0, "a.tmp0");

    // a_tmp1 = bpermute_lane(cudaTID, reg1)
    Value *ATmp1 = permuteLane(Builder, CudaTID, Reg1, "a.tmp1");

    // a_Frag_reg = (lane < 8) ? a_tmp0 : a_tmp1
    Value *LaneMod16 = Builder.CreateAnd(Lane, 15, "lane.mod16");
    Value *LaneLT8 =
        Builder.CreateICmpULT(LaneMod16, Builder.getInt32(8), "lane.lt.8");
    Value *AFragReg = Builder.CreateSelect(LaneLT8, ATmp0, ATmp1, "a.frag.reg");

    // Extract bottom 16 bits: aFrag[2 * vGPR] = bottom16AsFp16(a_Frag_reg)
    Value *Bottom16 =
        Builder.CreateTrunc(AFragReg, Builder.getInt16Ty(), "bottom16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Bottom16, 2 * vGPR);

    // Extract top 16 bits: aFrag[2 * vGPR + 1] = top16AsFp16(a_Frag_reg)
    Value *Top16 = Builder.CreateTrunc(Builder.CreateLShr(AFragReg, 16),
                                       Builder.getInt16Ty(), "top16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Top16, 2 * vGPR + 1);
  }

  return AMDFragment;
}

Value *LowerMatrixConversions::aMatrixConvertI8Half(IRBuilder<> &Builder,
                                                    Value *NvLowReg,
                                                    Value *NvHighReg) {
  Value *Lane = getLaneNumber();
  Value *QuarterLane = Builder.CreateAnd(Lane, 7, "quarter.lane");
  Value *HalfLane = Builder.CreateAnd(Lane, 15, "half.lane");
  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false));

  Value *UsesLowSrc = Builder.CreateCmp(llvm::CmpInst::ICMP_ULT, HalfLane,
                                        Builder.getInt32(8), "uses.low.src");
  for (uint32_t vGPR = 0; vGPR < 4; ++vGPR) {
    Value *SrcThread =
        Builder.CreateAdd(Builder.CreateMul(QuarterLane, Builder.getInt32(4)),
                          Builder.getInt32(vGPR), "src.thread");
    Value *APermutedLow = permuteLane(Builder, SrcThread, NvLowReg);
    Value *APermutedHigh = permuteLane(Builder, SrcThread, NvHighReg);
    Value *APermuted =
        Builder.CreateSelect(UsesLowSrc, APermutedLow, APermutedHigh);
    AMDFragment = Builder.CreateInsertElement(AMDFragment, APermuted, vGPR);
  }
  return AMDFragment;
}

Value *LowerMatrixConversions::bMatrixConcatenate(IRBuilder<> &Builder,
                                                  Value *NVFragmentFirst,
                                                  Value *NVFragmentSecond) {
  Value *Lane = getLaneNumber();

  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false));

  for (uint32_t vGPR = 0; vGPR < 8; ++vGPR) {
    // AMD and NVIDIA formats will have the same packed bf16x2, so only need to
    // find coordinates for the first.
    auto [Row, Column] = getLogicalCoordinatesForBMatrixAMDPhysicalCoordinates(
        Builder, Lane, vGPR);

    auto [CudaTID, NVPackedIdx, IsInFirst] =
        getBMatrixNVPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                             Column);

    Value *RegFirst =
        Builder.CreateExtractElement(NVFragmentFirst, NVPackedIdx);
    Value *RegSecond =
        Builder.CreateExtractElement(NVFragmentSecond, NVPackedIdx);

    // b_Frag_reg = bpermute_lane(cudaTID, reg)
    Value *BFragRegFirst =
        permuteLane(Builder, CudaTID, RegFirst, "b.frag.reg.first");
    Value *BFragRegSecond =
        permuteLane(Builder, CudaTID, RegSecond, "b.frag.reg.second");

    // Extract bottom 16 bits
    Value *Bottom16First = Builder.CreateTrunc(
        BFragRegFirst, Builder.getInt16Ty(), "bottom16.first");
    Value *Bottom16Second = Builder.CreateTrunc(
        BFragRegSecond, Builder.getInt16Ty(), "bottom16.second");
    Value *Bottom16 = Builder.CreateSelect(IsInFirst, Bottom16First,
                                           Bottom16Second, "bottom16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Bottom16, 2 * vGPR);

    // Extract top 16 bits
    Value *Top16First =
        Builder.CreateTrunc(Builder.CreateLShr(BFragRegFirst, 16),
                            Builder.getInt16Ty(), "top16.first");
    Value *Top16Second =
        Builder.CreateTrunc(Builder.CreateLShr(BFragRegSecond, 16),
                            Builder.getInt16Ty(), "top16.second");
    Value *Top16 =
        Builder.CreateSelect(IsInFirst, Top16First, Top16Second, "top16");
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Top16, 2 * vGPR + 1);
  }

  return AMDFragment;
}

Value *LowerMatrixConversions::cMatrixConcatenate(IRBuilder<> &Builder,
                                                  Value *NVFragmentFirst,
                                                  Value *NVFragmentSecond) {
  auto *RetTy = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);

  if (Constant *C0 = dyn_cast<Constant>(NVFragmentFirst)) {
    if (Constant *C1 = dyn_cast<Constant>(NVFragmentSecond)) {
      if (C0->isZeroValue() && C1->isZeroValue()) {
        return Constant::getNullValue(RetTy);
      }
    }
  }

  Value *Lane = getLaneNumber();

  Value *AMDFragment = PoisonValue::get(RetTy);

  for (uint32_t vGPR = 0; vGPR < 8; ++vGPR) {
    auto [Row, Column] = getLogicalCoordinatesForCMatrixAMDPhysicalCoordinates(
        Builder, Lane, vGPR);

    auto [CudaTID, NVElementIdx, IsInFirst] =
        getCMatrixNVPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                             Column);

    Value *BaseIdx =
        Builder.CreateAnd(NVElementIdx, Builder.getInt32(~1), "base.idx");

    Value *Ctmp0SrcFirst =
        Builder.CreateExtractElement(NVFragmentFirst, BaseIdx);
    Value *Ctmp0SrcSecond =
        Builder.CreateExtractElement(NVFragmentSecond, BaseIdx);
    Value *Ctmp1SrcFirst = Builder.CreateExtractElement(
        NVFragmentFirst, Builder.CreateAdd(BaseIdx, Builder.getInt32(1)));
    Value *Ctmp1SrcSecond = Builder.CreateExtractElement(
        NVFragmentSecond, Builder.CreateAdd(BaseIdx, Builder.getInt32(1)));

    // ctmp0 = bpermute_lane(cudaTID, ...)
    Value *Ctmp0First =
        permuteLane(Builder, CudaTID, Ctmp0SrcFirst, "ctmp0.first");
    Value *Ctmp0Second =
        permuteLane(Builder, CudaTID, Ctmp0SrcSecond, "ctmp0.second");

    // ctmp1 = bpermute_lane(cudaTID, ...)
    Value *Ctmp1First =
        permuteLane(Builder, CudaTID, Ctmp1SrcFirst, "ctmp1.first");
    Value *Ctmp1Second =
        permuteLane(Builder, CudaTID, Ctmp1SrcSecond, "ctmp1.second");

    // cFrag[vGPR] = (lIdx & 1) ? ctmp1 : ctmp0
    Value *LaneIsOdd = Builder.CreateTrunc(Builder.CreateAnd(Lane, 1),
                                           Builder.getInt1Ty(), "lane.is.odd");
    Value *CFragValueFirst = Builder.CreateSelect(
        LaneIsOdd, Ctmp1First, Ctmp0First, "cfrag.value.first");
    Value *CFragValueSecond = Builder.CreateSelect(
        LaneIsOdd, Ctmp1Second, Ctmp0Second, "cfrag.value.second");

    Value *FinalValue = Builder.CreateSelect(IsInFirst, CFragValueFirst,
                                             CFragValueSecond, "final.value");

    AMDFragment = Builder.CreateInsertElement(AMDFragment, FinalValue, vGPR);
  }

  return AMDFragment;
}

Value *LowerMatrixConversions::dMatrixSplit(IRBuilder<> &Builder,
                                            Value *AMDFragment, bool Truncate) {
  Value *Lane = getLaneNumber();

  Value *NVFragmentFirst = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false));

  for (uint32_t cChunk = 0; cChunk < 4; ++cChunk) {
    auto [Row, ColumnFirst] =
        getLogicalCoordinatesForDMatrixNVPhysicalCoordinates(Builder, Lane,
                                                             cChunk);

    auto [R_lIdxFirst, AMDvGPRFirst] =
        getDMatrixAMDPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                              ColumnFirst);

    //  r_vGPR = (AMDvGPR / 4) * 4
    Value *BaseVGPRFirst = Builder.CreateAnd(AMDvGPRFirst, Builder.getInt32(~3),
                                             "base.vgpr.first");

    // d_tmp0 = bpermute_lane(r_lIdx, dFrag[baseVGPR])
    Value *DTmp0First =
        permuteLane(Builder, R_lIdxFirst,
                     Builder.CreateExtractElement(AMDFragment, BaseVGPRFirst),
                     "d.tmp0.first");

    // d_tmp1 = bpermute_lane(r_lIdx, dFrag[baseVGPR + 1])
    Value *DTmp1First = permuteLane(
        Builder, R_lIdxFirst,
        Builder.CreateExtractElement(
            AMDFragment, Builder.CreateAdd(BaseVGPRFirst, Builder.getInt32(1))),
        "d.tmp1.first");

    // d_tmp2 = bpermute_lane(r_lIdx, dFrag[baseVGPR + 2])
    Value *DTmp2First = permuteLane(
        Builder, R_lIdxFirst,
        Builder.CreateExtractElement(
            AMDFragment, Builder.CreateAdd(BaseVGPRFirst, Builder.getInt32(2))),
        "d.tmp2.first");

    // d_tmp3 = bpermute_lane(r_lIdx, dFrag[baseVGPR + 3])
    Value *DTmp3First = permuteLane(
        Builder, R_lIdxFirst,
        Builder.CreateExtractElement(
            AMDFragment, Builder.CreateAdd(BaseVGPRFirst, Builder.getInt32(3))),
        "d.tmp3.first");

    // if (lIdx < 8) val = d_tmp0;
    // else if (lIdx < 16) val = d_tmp1;
    // else if (lIdx < 24) val = d_tmp2;
    // else val = d_tmp3;
    Value *LaneLT8 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(8), "lane.lt.8");
    Value *LaneLT16 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(16), "lane.lt.16");
    Value *LaneLT24 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(24), "lane.lt.24");

    // Build nested selects: lane < 8 ? tmp0 : (lane < 16 ? tmp1 : (lane < 24
    // ? tmp2 : tmp3))
    Value *Select23First = Builder.CreateSelect(LaneLT24, DTmp2First,
                                                DTmp3First, "select.23.first");
    Value *Select123First = Builder.CreateSelect(
        LaneLT16, DTmp1First, Select23First, "select.123.first");
    Value *ValFirst =
        Builder.CreateSelect(LaneLT8, DTmp0First, Select123First, "val.first");

    NVFragmentFirst =
        Builder.CreateInsertElement(NVFragmentFirst, ValFirst, cChunk);
  }

  if (Truncate) {
    return NVFragmentFirst;
  }

  Value *NVFragmentSecond = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false));

  for (uint32_t cChunk = 0; cChunk < 4; ++cChunk) {
    auto [Row, ColumnFirst] =
        getLogicalCoordinatesForDMatrixNVPhysicalCoordinates(Builder, Lane,
                                                             cChunk);

    // Process the second fragment (column + 8)
    Value *ColumnSecond =
        Builder.CreateAdd(ColumnFirst, Builder.getInt32(8), "column.second");
    auto [R_lIdxSecond, AMDvGPRSecond] =
        getDMatrixAMDPhysicalCoordinatesForLogicalCoordinates(Builder, Row,
                                                              ColumnSecond);

    Value *BaseVGPRSecond = Builder.CreateAnd(
        AMDvGPRSecond, Builder.getInt32(~3), "base.vgpr.second");

    Value *DTmp0Second =
        permuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(AMDFragment, BaseVGPRSecond),
                     "d.tmp0.second");

    Value *DTmp1Second =
        permuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(
                         AMDFragment, Builder.CreateAdd(BaseVGPRSecond,
                                                        Builder.getInt32(1))),
                     "d.tmp1.second");

    Value *DTmp2Second =
        permuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(
                         AMDFragment, Builder.CreateAdd(BaseVGPRSecond,
                                                        Builder.getInt32(2))),
                     "d.tmp2.second");

    Value *DTmp3Second =
        permuteLane(Builder, R_lIdxSecond,
                     Builder.CreateExtractElement(
                         AMDFragment, Builder.CreateAdd(BaseVGPRSecond,
                                                        Builder.getInt32(3))),
                     "d.tmp3.second");

    Value *LaneLT8 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(8), "lane.lt.8");
    Value *LaneLT16 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(16), "lane.lt.16");
    Value *LaneLT24 =
        Builder.CreateICmpULT(Lane, Builder.getInt32(24), "lane.lt.24");

    Value *Select23Second = Builder.CreateSelect(
        LaneLT24, DTmp2Second, DTmp3Second, "select.23.second");
    Value *Select123Second = Builder.CreateSelect(
        LaneLT16, DTmp1Second, Select23Second, "select.123.second");
    Value *ValSecond = Builder.CreateSelect(LaneLT8, DTmp0Second,
                                            Select123Second, "val.second");

    NVFragmentSecond =
        Builder.CreateInsertElement(NVFragmentSecond, ValSecond, cChunk);
  }

  Type *StructTy =
      StructType::get(NVFragmentFirst->getType(), NVFragmentSecond->getType());
  Value *Result = PoisonValue::get(StructTy);
  Result = Builder.CreateInsertValue(Result, NVFragmentFirst, 0);
  Result = Builder.CreateInsertValue(Result, NVFragmentSecond, 1);

  return Result;
}

// GFX12 FP8 matrix conversions.
//
// Derived, not guessed: from NVIDIA's PTX ISA guide (8.8), section
// 9.7.14.5.10 "Matrix Fragments for mma.m16n8k32", .e4m3 row/col:
//
//   groupID = %laneid >> 2;  threadID_in_group = %laneid % 4;
//   A: row = groupID for 0<=i<4 || 8<=i<12, else groupID+8
//      col = threadID_in_group*4 + (i&3) [+16 for i>=8]
//   B: row (=K) = threadID_in_group*4 + (i&3) [+16 for i>=4]; col = groupID
//   C/D: row = groupID for i<2, groupID+8 for i>=2
//        col = threadID_in_group*2 + (i&1)
//
// and from the AMD Matrix Instruction Calculator
// (github.com/ROCm/amd_matrix_instruction_calculator), RDNA4,
// v_wmma_f32_16x16x16_fp8_fp8, wave32, no modifiers:
//
//   A[i][k] GPR: (k/4)%2, byte (k%4);  Lane: 16*(k/8) + i
//   B[k][j] GPR: (k/4)%2, byte (k%4);  Lane: 16*(k/8) + j
//   C/D[i][j] GPR: i%8;  Lane: 16*(i/8) + j
//
// Working through both sides -- who is a function of AMD's own (lane, GPR)
// coordinates for a fixed K-half -- gives, for every one of A, B, C and D, a
// formula built from the same two pieces: which of two NVIDIA registers to
// read (a coarse selector, exactly the role zluda_amatrix_convert_amd_nv16x16
// and the I8 conversions above already play), and a source lane computed as
// group*4 + tid where group and tid come from AMD's own row/col mod 8 and
// div/mod 2. None of it reuses the GFX11 functions above: RDNA4's operand
// width is half RDNA3's for 8-bit types, its K-lane split is a different
// scheme (two lane-halves carrying different K, not the same K broadcast
// twice), and even the accumulator's GPR/lane formula differs.
//
// Unverified on hardware: the compiler accepting the intrinsic and selecting
// real instructions for it does not by itself prove the layout right.

Value *LowerMatrixConversions::aMatrixConvertFp8Half(IRBuilder<> &Builder,
                                                     Value *NvLowReg,
                                                     Value *NvHighReg) {
  Value *Lane = getLaneNumber();
  Value *Row = Builder.CreateAnd(Lane, 15, "row");        // i = lane % 16
  Value *Half = Builder.CreateLShr(Lane, 4, "half");      // 0 for lane<16, 1 otherwise
  Value *GroupID = Builder.CreateAnd(Row, 7, "group.id"); // row % 8, either branch
  Value *UsesLow =
      Builder.CreateICmpULT(Row, Builder.getInt32(8), "uses.low");

  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 2, /*Scalable=*/false));
  for (uint32_t GPR = 0; GPR < 2; ++GPR) {
    Value *Tid = Builder.CreateAdd(
        Builder.CreateMul(Half, Builder.getInt32(2)), Builder.getInt32(GPR));
    Value *SrcThread = Builder.CreateAdd(
        Builder.CreateMul(GroupID, Builder.getInt32(4)), Tid, "src.thread");
    Value *PermLow = permuteLane(Builder, SrcThread, NvLowReg);
    Value *PermHigh = permuteLane(Builder, SrcThread, NvHighReg);
    Value *Permuted = Builder.CreateSelect(UsesLow, PermLow, PermHigh);
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Permuted, GPR);
  }
  return AMDFragment;
}

Value *LowerMatrixConversions::bMatrixConvertFp8Half(IRBuilder<> &Builder,
                                                     Value *NvRegLeft,
                                                     Value *NvRegRight) {
  Value *Lane = getLaneNumber();
  Value *Col = Builder.CreateAnd(Lane, 15, "col");        // j = lane % 16
  Value *Half = Builder.CreateLShr(Lane, 4, "half");
  Value *GroupID = Builder.CreateAnd(Col, 7, "group.id"); // col % 8, either branch
  Value *UsesLeft =
      Builder.CreateICmpULT(Col, Builder.getInt32(8), "uses.left");

  Value *AMDFragment = PoisonValue::get(
      VectorType::get(Builder.getInt32Ty(), 2, /*Scalable=*/false));
  for (uint32_t GPR = 0; GPR < 2; ++GPR) {
    Value *Tid = Builder.CreateAdd(
        Builder.CreateMul(Half, Builder.getInt32(2)), Builder.getInt32(GPR));
    Value *SrcThread = Builder.CreateAdd(
        Builder.CreateMul(GroupID, Builder.getInt32(4)), Tid, "src.thread");
    Value *PermLeft = permuteLane(Builder, SrcThread, NvRegLeft);
    Value *PermRight = permuteLane(Builder, SrcThread, NvRegRight);
    Value *Permuted = Builder.CreateSelect(UsesLeft, PermLeft, PermRight);
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Permuted, GPR);
  }
  return AMDFragment;
}

Value *LowerMatrixConversions::cMatrixConcatenateFp8(IRBuilder<> &Builder,
                                                     Value *NVFragmentFirst,
                                                     Value *NVFragmentSecond) {
  auto *RetTy = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);
  if (Constant *C0 = dyn_cast<Constant>(NVFragmentFirst)) {
    if (Constant *C1 = dyn_cast<Constant>(NVFragmentSecond)) {
      if (C0->isZeroValue() && C1->isZeroValue()) {
        return Constant::getNullValue(RetTy);
      }
    }
  }

  Value *Lane = getLaneNumber();
  Value *Col = Builder.CreateAnd(Lane, 15, "col");   // j = lane % 16
  Value *Half = Builder.CreateLShr(Lane, 4, "half"); // 0 for lane<16, 1 otherwise
  // Local column within whichever 8-wide NVIDIA fragment this lane draws
  // from: identical to Col when j<8 (First), and Col-8 when j>=8 (Second) --
  // both cases collapse to the same "& 7", which is what lets First and
  // Second share one formula below.
  Value *LocalCol = Builder.CreateAnd(Col, 7, "local.col");
  Value *UsesFirst =
      Builder.CreateICmpULT(Col, Builder.getInt32(8), "uses.first");
  Value *RegIdx = Builder.CreateAdd(
      Builder.CreateMul(Half, Builder.getInt32(2)),
      Builder.CreateAnd(LocalCol, 1), "reg.idx");
  Value *Tid = Builder.CreateLShr(LocalCol, 1, "tid");

  Value *AMDFragment = PoisonValue::get(RetTy);
  for (uint32_t GPR = 0; GPR < 8; ++GPR) {
    Value *SrcThread = Builder.CreateAdd(
        Builder.getInt32(GPR * 4), Tid, "src.thread");
    Value *RegFirst = Builder.CreateExtractElement(NVFragmentFirst, RegIdx);
    Value *RegSecond = Builder.CreateExtractElement(NVFragmentSecond, RegIdx);
    Value *PermFirst = permuteLane(Builder, SrcThread, RegFirst);
    Value *PermSecond = permuteLane(Builder, SrcThread, RegSecond);
    Value *Value_ = Builder.CreateSelect(UsesFirst, PermFirst, PermSecond);
    AMDFragment = Builder.CreateInsertElement(AMDFragment, Value_, GPR);
  }
  return AMDFragment;
}

Value *LowerMatrixConversions::dMatrixSplitFp8(IRBuilder<> &Builder,
                                              Value *AMDFragment) {
  // The inverse of cMatrixConcatenateFp8: this lane speaks for an NVIDIA
  // (row, register) pair rather than an AMD (lane, GPR) pair, so groupID/tid
  // come from this lane directly, and row/g/half are derived rather than
  // given.
  Value *Lane = getLaneNumber();
  Value *GroupID = Builder.CreateLShr(Lane, 2, "group.id"); // laneid / 4
  Value *Tid = Builder.CreateAnd(Lane, 3, "tid");           // laneid % 4

  Type *V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  Value *First = PoisonValue::get(V4I32Ty);
  Value *Second = PoisonValue::get(V4I32Ty);
  for (uint32_t r = 0; r < 4; ++r) {
    // row = groupID for r<2, groupID+8 for r>=2; g = row%8 = groupID always
    // (groupID<8 by construction); half = row/8 = (r>=2) ? 1 : 0.
    Value *Row = (r < 2) ? GroupID
                         : Builder.CreateAdd(GroupID, Builder.getInt32(8));
    Value *GPR = GroupID; // row % 8
    Value *Half = Builder.getInt32(r < 2 ? 0 : 1);
    // localCol = tid*2 + (r&1)
    Value *LocalCol = Builder.CreateAdd(
        Builder.CreateMul(Tid, Builder.getInt32(2)),
        Builder.getInt32(r & 1), "local.col");
    Value *SrcLaneFirst = Builder.CreateAdd(
        Builder.CreateMul(Half, Builder.getInt32(16)), LocalCol,
        "src.lane.first");
    Value *SrcLaneSecond = Builder.CreateAdd(
        SrcLaneFirst, Builder.getInt32(8), "src.lane.second");

    Value *RegVal = Builder.CreateExtractElement(AMDFragment, GPR);
    Value *ValFirst = permuteLane(Builder, SrcLaneFirst, RegVal);
    Value *ValSecond = permuteLane(Builder, SrcLaneSecond, RegVal);
    First = Builder.CreateInsertElement(First, ValFirst, r);
    Second = Builder.CreateInsertElement(Second, ValSecond, r);
    (void)Row; // kept for the derivation's sake; folded into GPR/Half above
  }

  Type *ReturnTy = StructType::get(V4I32Ty, V4I32Ty);
  Value *Result = PoisonValue::get(ReturnTy);
  Result = Builder.CreateInsertValue(Result, First, 0);
  Result = Builder.CreateInsertValue(Result, Second, 1);
  return Result;
}

void LowerMatrixConversions::lowerConversion(IntrinsicInst *Conversion) {
  IRBuilder<> Builder(Conversion);

  switch (Conversion->getIntrinsicID()) {
  case Intrinsic::zluda_amatrix_convert_amd_nv16x16:
    Conversion->replaceAllUsesWith(
        aMatrixConvert(Builder, Conversion->getArgOperand(0)));
    Conversion->eraseFromParent();
    break;
  case Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8: {
    Value *NVMatrixFirst = Conversion->getArgOperand(0);
    Value *NVMatrixSecond = Conversion->getArgOperand(1);
    Conversion->replaceAllUsesWith(
        bMatrixConcatenate(Builder, NVMatrixFirst, NVMatrixSecond));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8: {
    Value *NVMatrixFirst = Conversion->getArgOperand(0);
    Value *NVMatrixSecond = Conversion->getArgOperand(1);
    Conversion->replaceAllUsesWith(
        cMatrixConcatenate(Builder, NVMatrixFirst, NVMatrixSecond));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16: {
    Value *AMDMatrix = Conversion->getArgOperand(0);
    auto AMDMatrixBitCast = Builder.CreateBitCast(
        AMDMatrix,
        VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false));
    Conversion->replaceAllUsesWith(
        dMatrixSplit(Builder, AMDMatrixBitCast, /*Truncate=*/true));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16: {
    Value *AMDMatrix = Conversion->getArgOperand(0);
    Conversion->replaceAllUsesWith(
        dMatrixSplit(Builder, AMDMatrix, /*Truncate=*/false));
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_amatrix_split_amd16x16_nv16x32: {
    Value *NVMatrix = Conversion->getArgOperand(0);
    auto *V0 = Builder.CreateExtractElement(NVMatrix, uint64_t(0));
    auto *V1 = Builder.CreateExtractElement(NVMatrix, uint64_t(1));
    auto *V2 = Builder.CreateExtractElement(NVMatrix, uint64_t(2));
    auto *V3 = Builder.CreateExtractElement(NVMatrix, uint64_t(3));
    auto LowHalf = aMatrixConvertI8Half(Builder, V0, V1);
    auto HighHalf = aMatrixConvertI8Half(Builder, V2, V3);
    Type *ReturnTy = StructType::get(LowHalf->getType(), HighHalf->getType());
    Value *Result = PoisonValue::get(ReturnTy);
    Result = Builder.CreateInsertValue(Result, LowHalf, 0);
    Result = Builder.CreateInsertValue(Result, HighHalf, 1);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_bmatrix_reshape_amd16x16_nv32x8: {
    Value *LeftNVMatrix = Conversion->getArgOperand(0);
    Value *RightNVMatrix = Conversion->getArgOperand(1);
    Value *UpperLeftMatrix =
        Builder.CreateExtractElement(LeftNVMatrix, uint64_t(0));
    Value *UpperRightMatrix =
        Builder.CreateExtractElement(RightNVMatrix, uint64_t(0));
    Value *LowerLeftMatrix =
        Builder.CreateExtractElement(LeftNVMatrix, uint64_t(1));
    Value *LowerRightMatrix =
        Builder.CreateExtractElement(RightNVMatrix, uint64_t(1));
    Value *B0 =
        aMatrixConvertI8Half(Builder, UpperLeftMatrix, UpperRightMatrix);
    Value *B1 =
        aMatrixConvertI8Half(Builder, LowerLeftMatrix, LowerRightMatrix);
    Type *ReturnTy = StructType::get(B0->getType(), B1->getType());
    Value *Result = PoisonValue::get(ReturnTy);
    Result = Builder.CreateInsertValue(Result, B0, 0);
    Result = Builder.CreateInsertValue(Result, B1, 1);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_amatrix_convert_fp8_amd16x16_nv16x32: {
    Value *NVMatrix = Conversion->getArgOperand(0);
    auto *V0 = Builder.CreateExtractElement(NVMatrix, uint64_t(0));
    auto *V1 = Builder.CreateExtractElement(NVMatrix, uint64_t(1));
    auto *V2 = Builder.CreateExtractElement(NVMatrix, uint64_t(2));
    auto *V3 = Builder.CreateExtractElement(NVMatrix, uint64_t(3));
    auto LowHalf = aMatrixConvertFp8Half(Builder, V0, V1);
    auto HighHalf = aMatrixConvertFp8Half(Builder, V2, V3);
    Type *ReturnTy = StructType::get(LowHalf->getType(), HighHalf->getType());
    Value *Result = PoisonValue::get(ReturnTy);
    Result = Builder.CreateInsertValue(Result, LowHalf, 0);
    Result = Builder.CreateInsertValue(Result, HighHalf, 1);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_bmatrix_convert_fp8_amd16x16_nv32x8: {
    Value *LeftNVMatrix = Conversion->getArgOperand(0);
    Value *RightNVMatrix = Conversion->getArgOperand(1);
    Value *UpperLeftMatrix =
        Builder.CreateExtractElement(LeftNVMatrix, uint64_t(0));
    Value *UpperRightMatrix =
        Builder.CreateExtractElement(RightNVMatrix, uint64_t(0));
    Value *LowerLeftMatrix =
        Builder.CreateExtractElement(LeftNVMatrix, uint64_t(1));
    Value *LowerRightMatrix =
        Builder.CreateExtractElement(RightNVMatrix, uint64_t(1));
    Value *B0 =
        bMatrixConvertFp8Half(Builder, UpperLeftMatrix, UpperRightMatrix);
    Value *B1 =
        bMatrixConvertFp8Half(Builder, LowerLeftMatrix, LowerRightMatrix);
    Type *ReturnTy = StructType::get(B0->getType(), B1->getType());
    Value *Result = PoisonValue::get(ReturnTy);
    Result = Builder.CreateInsertValue(Result, B0, 0);
    Result = Builder.CreateInsertValue(Result, B1, 1);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_cmatrix_concatenate_fp8_amd16x16_nv16x8: {
    Value *NVFragmentFirst = Conversion->getArgOperand(0);
    Value *NVFragmentSecond = Conversion->getArgOperand(1);
    Value *Result =
        cMatrixConcatenateFp8(Builder, NVFragmentFirst, NVFragmentSecond);
    Conversion->replaceAllUsesWith(Result);
    Conversion->eraseFromParent();
    break;
  }
  case Intrinsic::zluda_dmatrix_split_fp8_nv16x8_amd16x16: {
    Value *AMDMatrix = Conversion->getArgOperand(0);
    Conversion->replaceAllUsesWith(dMatrixSplitFp8(Builder, AMDMatrix));
    Conversion->eraseFromParent();
    break;
  }
  default:
    llvm_unreachable("Unexpected intrinsic");
  }
}

bool LowerMatrixConversions::lowerBB(BasicBlock &BB) {
  bool Modified = false;

  for (Instruction &I : make_early_inc_range(BB)) {
    auto Conversion = getMatrixConversion(I);

    if (Conversion) {
      lowerConversion(Conversion);
      Modified = true;
    }
  }

  return Modified;
}

bool LowerMatrixConversions::lower() {
  bool Modified = false;

  for (BasicBlock &BB : F) {
    Modified |= lowerBB(BB);
  }

  return Modified;
}

PreservedAnalyses LowerMatrixConversionsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  LowerMatrixConversions LMC(F);
  if (LMC.lower()) {
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  }

  return PreservedAnalyses::all();
}
