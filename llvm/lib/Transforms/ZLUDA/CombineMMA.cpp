#include "llvm/Transforms/ZLUDA/CombineMMA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

using namespace llvm;

// Whether an instruction could change the rounding mode or the exception
// state, which is the only thing a constrained floating point intrinsic cares
// about being moved across. Constrained intrinsics themselves only read that
// state, so they do not count; inline assembly and anything else that is not a
// plain call are refused rather than reasoned about.
static bool changesFPEnvironment(const Instruction &I) {
  const auto *Call = dyn_cast<CallBase>(&I);
  if (!Call) {
    return false;
  }
  if (isa<ConstrainedFPIntrinsic>(Call)) {
    return false;
  }
  if (Call->isInlineAsm()) {
    return true;
  }
  const Function *Callee = Call->getCalledFunction();
  if (!Callee) {
    return true;
  }
  switch (Callee->getIntrinsicID()) {
  case Intrinsic::set_rounding:
  case Intrinsic::set_fpenv:
  case Intrinsic::reset_fpenv:
    return true;
  case Intrinsic::not_intrinsic:
    return true; // an ordinary call could do anything
  default:
    return false;
  }
}

// Whether a store in one address space could write what a load in another
// reads. Only the plainly disjoint cases are claimed: shared memory, private
// memory and constant memory cannot be reached through a global pointer or
// through each other. The flat space can be any of them, so it aliases
// everything, and anything unrecognised is treated as flat.
static bool mayAlias(unsigned StoreSpace, unsigned LoadSpace) {
  auto Distinct = [](unsigned Space) {
    switch (Space) {
    case 1: // global
    case 3: // shared
    case 4: // constant
    case 5: // private
      return true;
    default: // flat, and anything this does not recognise
      return false;
    }
  };
  if (!Distinct(StoreSpace) || !Distinct(LoadSpace)) {
    return true;
  }
  return StoreSpace == LoadSpace;
}

// Why tryToReorderOperands refused, counted for ZLUDA_MMA_STATS: a dependency
// on the first multiply, or something in between that touches memory, and the
// first instruction that stood in the way.
namespace {
unsigned RefusedByDependency = 0; // the second multiply needs the first
unsigned RefusedByMemory = 0;     // something in between touches memory
std::string FirstBlocker;         // and what it was, the first time

static void noteBlocker(const Instruction &I) {
  if (!FirstBlocker.empty())
    return;
  if (const auto *Call = dyn_cast<CallBase>(&I)) {
    if (const Function *Callee = Call->getCalledFunction()) {
      FirstBlocker = Callee->getName().str();
      return;
    }
    FirstBlocker = "indirect call";
    return;
  }
  FirstBlocker = I.getOpcodeName();
}
} // namespace

// Moves the instructions that FromBefore depends on to before ToBefore. Does
// nothing other than return false if FromBefore has a dependency on ToBefore,
// and true otherwise. Based on LoadStoreVectorizer's reorder.
static bool tryToReorderOperands(IntrinsicInst *FromBefore,
                                 IntrinsicInst *ToBefore) {
  assert(FromBefore->getParent() == ToBefore->getParent());

  SmallPtrSet<Instruction *, 16> InstructionsToMove;
  SmallVector<Instruction *, 16> Worklist;

  Worklist.emplace_back(FromBefore);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    for (Value *Operand : I->operands()) {
      auto *Dependency = dyn_cast<Instruction>(Operand);
      if (!Dependency || Dependency->getOpcode() == Instruction::PHI) {
        continue;
      }

      // Ignore instructions outside of the current basic block
      if (Dependency->getParent() != FromBefore->getParent()) {
        continue;
      }

      if (Dependency == ToBefore) {
        RefusedByDependency++;
        return false;
      }

      assert(Dependency != FromBefore &&
             "Unexpected cycle while re-ordering instructions");

      if (!Dependency->comesBefore(ToBefore)) {
        // A constrained floating point intrinsic is marked as touching memory,
        // and touches none: what it accesses is the floating point environment,
        // the rounding mode and the exception flags. Moving one earlier inside
        // a block is harmless as long as nothing in between changes that
        // environment.
        //
        // ZLUDA compiles the PTX helpers in constrained mode, so a constrained
        // fpext widening the accumulator sits on the operand path of every
        // multiply; refusing it refused all 444 candidate pairs in the DLSS
        // kernels.
        if (auto *Constrained = dyn_cast<ConstrainedFPIntrinsic>(Dependency)) {
          bool EnvironmentChanged = false;
          for (auto BBI = ToBefore->getIterator(), E = Constrained->getIterator();
               BBI != E; ++BBI) {
            if (changesFPEnvironment(*BBI)) {
              EnvironmentChanged = true;
              break;
            }
          }
          if (EnvironmentChanged) {
            noteBlocker(*Dependency);
            RefusedByMemory++;
            return false;
          }
        } else if (auto *Load = dyn_cast<LoadInst>(Dependency)) {
          // A plain load may be lifted above ToBefore as long as nothing in
          // between could write what it reads. The load feeding the second
          // multiply's B operand usually sits between the two multiplies, which
          // is the ordinary shape of a loop whose loads are scheduled ahead of
          // the arithmetic that consumes them.
          //
          // The load is already executed unconditionally at this point in the
          // block, so lifting it introduces no fault that was not there; only
          // ordering against writes matters, and that is what is checked.
          if (!Load->isSimple()) {
            return false;
          }
          // A write in between only matters if it could touch what the load
          // reads. A loop that stages data through shared memory always has a
          // store between two multiplies, and a store to shared memory cannot
          // touch a global load: the address spaces are disjoint by
          // construction. Anything else that writes -- a call, an atomic, a
          // store whose space is not plainly disjoint -- is still refused.
          const unsigned LoadSpace = Load->getPointerAddressSpace();
          bool Written = false;
          for (auto BBI = ToBefore->getIterator(), E = Load->getIterator();
               BBI != E; ++BBI) {
            if (!BBI->mayWriteToMemory()) {
              continue;
            }
            auto *Store = dyn_cast<StoreInst>(&*BBI);
            if (Store && Store->isSimple() &&
                !mayAlias(Store->getPointerAddressSpace(), LoadSpace)) {
              continue;
            }
            noteBlocker(*BBI);
            Written = true;
            break;
          }
          if (Written) {
            RefusedByMemory++;
            return false;
          }
        } else if (Dependency->mayReadOrWriteMemory()) {
          // Anything else that touches memory is still refused.
          noteBlocker(*Dependency);
          RefusedByMemory++;
          return false;
        }
        InstructionsToMove.insert(Dependency);
        Worklist.emplace_back(Dependency);
      }
    }
  }

  // We only need to move the instructions in between ToBefore and FromBefore
  for (auto BBI = ToBefore->getIterator(), E = FromBefore->getIterator();
       BBI != E; ++BBI) {
    auto I = &*BBI;
    if (InstructionsToMove.contains(I)) {
      I->moveBefore(ToBefore);
    }
  }

  return true;
}

static IntrinsicInst *getBF16ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA && MMA->getIntrinsicID() ==
                 Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    return MMA;
  }
  return nullptr;
}

// The f16 form goes down the same road as bf16: the A, B, C and D conversions are
// shared, only the AMD instruction differs, and it wants <16 x half> where the bf16
// one wants <16 x i16>.
static IntrinsicInst *getF16ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA && MMA->getIntrinsicID() ==
                 Intrinsic::zluda_mma_m16n8k16_f32_f16_f16_f32) {
    return MMA;
  }
  return nullptr;
}

static IntrinsicInst *getS8ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA &&
      MMA->getIntrinsicID() == Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    return MMA;
  }
  return nullptr;
}

// Diagnostic: whether to skip the real 16x16x16 tensor multiply and replace
// it with a trivial reduction, keeping every conversion, concatenation and
// split intrinsic around it untouched.
//
// Fusion halves the WMMA count and roughly halves the kernel time, which is
// consistent with the WMMA being the cost -- but the fused kernel's
// disassembly is dominated by ds_bpermute_b32 (9295 of them in the hottest
// kernel), which is the layout glue this pass emits to turn two of NVIDIA's
// 8-wide fragments into AMD's 16-wide one, not the multiply itself. This
// answers which one actually owns the fused kernel's time: with the probe on,
// every ds_bpermute this pass emits is still there and still executes: only
// the tensor op inside it is gone.
static bool probeNoRealWMMA() {
  static const bool on = ::getenv("ZLUDA_PROBE_NO_REAL_WMMA") != nullptr;
  return on;
}

// Why multiplies were or were not fused, counted per function and printed when
// ZLUDA_MMA_STATS is set.
namespace {
struct CombineStats {
  unsigned Seen = 0;         // mma intrinsics found
  unsigned Blocks = 0;       // blocks holding at least one
  unsigned Alone = 0;        // no other mma in the block shares its A
  unsigned Paired = 0;       // a candidate partner was found
  unsigned Fused = 0;        // and the fusion went through
  unsigned ReorderFailed = 0; // it did not, because of a dependency

  void report(const Function &F) const {
    if (!Seen || !::getenv("ZLUDA_MMA_STATS"))
      return;
    errs() << "[zluda-combine-mma] " << F.getName() << ": " << Seen
           << " mma in " << Blocks << " blocks, " << Paired
           << " with a partner in the same block (" << Fused << " fused, "
           << ReorderFailed << " refused), " << Alone
           << " alone in their block; refusals: " << RefusedByDependency
           << " dependency, " << RefusedByMemory << " memory; "
           << "first blocker: " << FirstBlocker << "\n";
  }
};
} // namespace

class MMACombiner {
public:
  bool combine(Function &F);

private:
  bool combineBB(BasicBlock &BB);
  bool combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs);
  bool combineMMA(IntrinsicInst *First, IntrinsicInst *Second);

  llvm::Value *EmitAmdMmaI8(llvm::IRBuilder<> &Builder, llvm::Value *FirstA,
                            llvm::Value *FirstB, llvm::Value *SecondB,
                            llvm::Value *FirstC, llvm::Value *SecondC);

  void lowerMMA(IntrinsicInst *MMA);

  Value *combineC(IRBuilder<> &Builder, Value *FirstC, Value *SecondC);
  Value *convertC(IRBuilder<> &Builder, Value *C);

  SmallVector<Instruction *> MaybeRemove;

public:
  CombineStats Stats;
};

// If FirstC and SecondC are the result of a split, return the value before it
// was split. Otherwise concatenate the matrices.
//
// This fold is reachable only from combineMMA, i.e. only while two multiplies
// are being fused, and that is not an oversight worth "fixing" with a
// standalone peephole over the block: one was written and measured, and it
// found zero round trips to fold, on the fused path and on the generic one
// alike. On the fused path there is nothing left for it -- this fold already
// took them. On the generic path the two multiplies that would cancel sit in
// two separate invocations of the same noinline wrapper, so the split of one
// and the concatenate of the next are never in the same function, let alone
// the same block.
//
// Which is worth knowing, because the waste is real and measured: on the
// generic caches the accumulator is concatenated into AMD's layout, multiplied,
// split back, and immediately concatenated again by the next multiply in the
// chain. Reaching it means moving the conversion out of the wrapper -- keeping
// the accumulator in AMD layout across the chain and converting once at each
// end -- not pattern matching. On the generic target that path is most of the
// kernel's time.
Value *MMACombiner::combineC(IRBuilder<> &Builder, Value *FirstC,
                             Value *SecondC) {
  auto *FirstExtract = dyn_cast<ExtractValueInst>(FirstC);
  auto *SecondExtract = dyn_cast<ExtractValueInst>(SecondC);
  if (FirstExtract != nullptr && SecondExtract != nullptr) {
    auto *FirstAggregate = FirstExtract->getAggregateOperand();
    auto FirstIndices = FirstExtract->getIndices();
    auto *SecondAggregate = SecondExtract->getAggregateOperand();
    auto SecondIndices = SecondExtract->getIndices();
    if (FirstAggregate == SecondAggregate &&
        FirstIndices == ArrayRef<unsigned>{0} &&
        SecondIndices == ArrayRef<unsigned>{1}) {
      if (auto *II = dyn_cast<IntrinsicInst>(FirstAggregate)) {
        if (II->getIntrinsicID() ==
            Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16) {
          MaybeRemove.emplace_back(FirstExtract);
          MaybeRemove.emplace_back(SecondExtract);
          MaybeRemove.emplace_back(II);
          return II->getArgOperand(0);
        }
      }
    }
  }

  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);

  return Builder.CreateIntrinsic(
      V8I32Ty, Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8,
      {FirstC, SecondC});
}

// If C is the result of a truncate, return the value before it was truncated.
// Otherwise zero-extend the matrix.
Value *MMACombiner::convertC(IRBuilder<> &Builder, Value *C) {
  if (auto *II = dyn_cast<IntrinsicInst>(C)) {
    if (II->getIntrinsicID() ==
        Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16) {
      MaybeRemove.emplace_back(II);
      return II->getArgOperand(0);
    }
  }

  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);
  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);

  auto NullC = Constant::getNullValue(C->getType());
  auto IntResult = Builder.CreateIntrinsic(
      V8I32Ty, Intrinsic::zluda_cmatrix_concatenate_amd16x16_nv16x8, {C, NullC});
  return Builder.CreateBitCast(IntResult, V8F32Ty);
}

// Whether the fused instruction can be built where the second multiply sits
// rather than where the first does.
//
// The one thing that has to hold is that nothing between the two reads the
// first multiply's result, since that result is about to be defined further
// down. That check does double duty: a use in between is also precisely how
// the second multiply would come to depend on the first, and fusing those two
// would be wrong rather than merely awkward.
static bool canBuildAtSecond(IntrinsicInst *First, IntrinsicInst *Second) {
  for (User *U : First->users()) {
    auto *I = dyn_cast<Instruction>(U);
    // Anything outside this block is not obviously still dominated once the
    // definition moves down, so it is refused rather than reasoned about.
    if (!I || I->getParent() != Second->getParent())
      return false;
    if (!Second->comesBefore(I))
      return false;
  }
  return true;
}

// Combine two NVIDIA-style 16x8 MMA instructions into one AMD-style 16x16 MMA
// instruction.
bool MMACombiner::combineMMA(IntrinsicInst *First, IntrinsicInst *Second) {
  assert(First->getIntrinsicID() == Second->getIntrinsicID());
  Value *FirstA = First->getArgOperand(0);
  Value *FirstB = First->getArgOperand(1);
  Value *FirstC = First->getArgOperand(2);

  Value *SecondA = Second->getArgOperand(0);
  Value *SecondB = Second->getArgOperand(1);
  Value *SecondC = Second->getArgOperand(2);

  if (FirstA != SecondA) {
    return false;
  }

  // Two places the fused instruction can go, and the cheap one is tried first.
  //
  // At the second multiply nothing has to move, so a load sitting between the
  // two -- the ordinary shape of a software-pipelined loop, and what refused
  // every pair in the DLSS kernels -- stops mattering. At the first multiply it
  // does have to move, which is the older path and still the one to take when
  // something in between reads the first result.
  Instruction *InsertAt = Second;
  if (!canBuildAtSecond(First, Second)) {
    if (!tryToReorderOperands(Second, First)) {
      return false;
    }
    InsertAt = First;
  }

  IRBuilder<> Builder(InsertAt);

  llvm::Value *Split;
  bool IsF16 = First->getIntrinsicID() ==
               Intrinsic::zluda_mma_m16n8k16_f32_f16_f16_f32;
  if (IsF16 || First->getIntrinsicID() ==
                   Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
    auto V4I32x2Ty = StructType::get(Builder.getContext(), {V4I32Ty, V4I32Ty});
    auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);
    auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);
    auto V16I16Ty =
        VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false);

    auto ShuffledA = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_amatrix_convert_amd_nv16x16, {FirstA});
    auto CombinedB = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8,
        {FirstB, SecondB});
    auto *FirstCBitCast = Builder.CreateBitCast(FirstC, V4I32Ty);
    auto *SecondCBitCast = Builder.CreateBitCast(SecondC, V4I32Ty);
    auto CombinedC = combineC(Builder, FirstCBitCast, SecondCBitCast);
    auto CombinedCBitCast = Builder.CreateBitCast(CombinedC, V8F32Ty);

    llvm::Value *AOperand = ShuffledA;
    llvm::Value *BOperand = CombinedB;
    if (IsF16) {
      auto V16F16Ty =
          VectorType::get(Builder.getHalfTy(), 16, /*Scalable=*/false);
      AOperand = Builder.CreateBitCast(ShuffledA, V16F16Ty);
      BOperand = Builder.CreateBitCast(CombinedB, V16F16Ty);
    }
    llvm::Value *Result;
    if (probeNoRealWMMA()) {
      // Same shape as the real op (<8 x float>), computed cheaply from the
      // same operands so nothing upstream is dead-code-eliminated: every
      // conversion and permute that feeds AOperand/BOperand still has to run.
      auto V8F16Ty = VectorType::get(Builder.getHalfTy(), 8, /*Scalable=*/false);
      auto ALo = Builder.CreateShuffleVector(AOperand, ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
      auto BLo = Builder.CreateShuffleVector(BOperand, ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
      auto Sum = Builder.CreateFAdd(Builder.CreateBitCast(ALo, V8F16Ty),
                                    Builder.CreateBitCast(BLo, V8F16Ty));
      auto SumF32 = Builder.CreateFPExt(Sum, VectorType::get(Builder.getFloatTy(), 8, false));
      Result = Builder.CreateFAdd(SumF32, CombinedCBitCast);
    } else {
      Result = Builder.CreateIntrinsic(
          V8F32Ty,
          IsF16 ? Intrinsic::amdgcn_wmma_f32_16x16x16_f16
                : Intrinsic::amdgcn_wmma_f32_16x16x16_bf16,
          {AOperand, BOperand, CombinedCBitCast});
    }
    auto *ResultBitCast = Builder.CreateBitCast(Result, V8I32Ty);
    Split = Builder.CreateIntrinsic(
        V4I32x2Ty, Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16,
        {ResultBitCast});
  } else if (First->getIntrinsicID() ==
             Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    Split = EmitAmdMmaI8(Builder, FirstA, FirstB, SecondB, FirstC, SecondC);
  } else {
    llvm_unreachable("Unsupported MMA intrinsic");
  }

  auto NewFirst = Builder.CreateExtractValue(Split, {0});
  auto NewSecond = Builder.CreateExtractValue(Split, {1});

  First->replaceAllUsesWith(NewFirst);
  Second->replaceAllUsesWith(NewSecond);

  First->eraseFromParent();
  Second->eraseFromParent();

  return true;
}

llvm::Value *MMACombiner::EmitAmdMmaI8(llvm::IRBuilder<> &Builder,
                                       llvm::Value *A, llvm::Value *FirstB,
                                       llvm::Value *SecondB,
                                       llvm::Value *FirstC,
                                       llvm::Value *SecondC) {
  auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  auto V4I32x2Ty = StructType::get(Builder.getContext(), {V4I32Ty, V4I32Ty});
  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);

  auto SplitA = Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_amatrix_split_amd16x16_nv16x32, {A});
  auto ReshapedB = Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_bmatrix_reshape_amd16x16_nv32x8,
      {FirstB, SecondB});

  auto CombinedC = combineC(Builder, FirstC, SecondC);
  auto A0 = Builder.CreateExtractValue(SplitA, {0});
  auto A1 = Builder.CreateExtractValue(SplitA, {1});
  auto B0 = Builder.CreateExtractValue(ReshapedB, {0});
  auto B1 = Builder.CreateExtractValue(ReshapedB, {1});

  auto True = Builder.getTrue();
  auto False = Builder.getFalse();

  auto TempD =
      Builder.CreateIntrinsic(V8I32Ty, Intrinsic::amdgcn_wmma_i32_16x16x16_iu8,
                              {True, A0, True, B0, CombinedC, False});
  auto D =
      Builder.CreateIntrinsic(V8I32Ty, Intrinsic::amdgcn_wmma_i32_16x16x16_iu8,
                              {True, A1, True, B1, TempD, False});
  return Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_dmatrix_split_nv16x8_amd16x16, {D});
}

// Lower an NVIDIA-style 16x8 MMA instruction to an AMD-style 16x16 MMA
// instruction. The unused part of the matrix is filled with zeroes.
void MMACombiner::lowerMMA(IntrinsicInst *MMA) {

  Value *A = MMA->getArgOperand(0);
  Value *B = MMA->getArgOperand(1);
  Value *C = MMA->getArgOperand(2);

  IRBuilder<> Builder(MMA);

  llvm::Value *Result;

  llvm::Intrinsic::ID IID = MMA->getIntrinsicID();
  auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  bool IsF16 = IID == Intrinsic::zluda_mma_m16n8k16_f32_f16_f16_f32;
  if (IsF16 || IID == Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32) {
    auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);
    auto V16I16Ty =
        VectorType::get(Builder.getInt16Ty(), 16, /*Scalable=*/false);

    auto ShuffledA = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_amatrix_convert_amd_nv16x16, {A});
    auto NullB = Constant::getNullValue(B->getType());
    auto ShuffledB = Builder.CreateIntrinsic(
        V16I16Ty, Intrinsic::zluda_bmatrix_concatenate_amd16x16_nv16x8, {B, NullB});
    auto ShuffledC = convertC(Builder, C);

    llvm::Value *AOperand = ShuffledA;
    llvm::Value *BOperand = ShuffledB;
    if (IsF16) {
      auto V16F16Ty =
          VectorType::get(Builder.getHalfTy(), 16, /*Scalable=*/false);
      AOperand = Builder.CreateBitCast(ShuffledA, V16F16Ty);
      BOperand = Builder.CreateBitCast(ShuffledB, V16F16Ty);
    }
    auto *Output = Builder.CreateIntrinsic(
        V8F32Ty,
        IsF16 ? Intrinsic::amdgcn_wmma_f32_16x16x16_f16
              : Intrinsic::amdgcn_wmma_f32_16x16x16_bf16,
        {AOperand, BOperand, ShuffledC});
    Result = Builder.CreateIntrinsic(
        V4I32Ty, Intrinsic::zluda_dmatrix_trunc_nv16x8_amd16x16, {Output});
  } else if (IID == Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32) {
    auto BPadding = Constant::getNullValue(B->getType());
    auto CPadding = Constant::getNullValue(C->getType());
    llvm::Value *DoubleResult =
        EmitAmdMmaI8(Builder, A, B, BPadding, C, CPadding);
    Result = Builder.CreateExtractValue(DoubleResult, 0);
  } else {
    llvm_unreachable("Unsupported MMA intrinsic");
  }
  MMA->replaceAllUsesWith(Result);
  MMA->eraseFromParent();
}

bool MMACombiner::combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs) {
  bool Modified = !MMAs.empty();

  llvm::DenseMap<std::pair<llvm::Intrinsic::ID, llvm::Value *>, IntrinsicInst *>
      UncombinedMMAs;

  for (IntrinsicInst *MMA : MMAs) {
    std::pair<llvm::Intrinsic::ID, llvm::Value *> Key{MMA->getIntrinsicID(),
                                                      MMA->getArgOperand(0)};
    IntrinsicInst *CompatibleMMA = UncombinedMMAs.lookup(Key);
    if (CompatibleMMA) {
      Stats.Paired++;
      if (combineMMA(CompatibleMMA, MMA)) {
        Stats.Fused++;
        UncombinedMMAs.erase(Key);
      } else {
        Stats.ReorderFailed++;
        // If we failed that's likely because the MMA #2 depends on MMA #1.
        // In that case we lower MMA #1 and keep MMA #2 for future combinations.
        lowerMMA(CompatibleMMA);
        UncombinedMMAs.insert_or_assign(Key, MMA);
      }
    } else {
      UncombinedMMAs.insert_or_assign(Key, MMA);
    }
  }

  for (auto pair : UncombinedMMAs) {
    Stats.Alone++;
    lowerMMA(pair.second);
  }

  return Modified;
}

bool MMACombiner::combineBB(BasicBlock &BB) {
  // For now, we simply combine adjacent m16n8k16 MMAs if possible. This may be
  // good enough in most cases. Any MMAs that cannot be combined are lowered
  // individually.
  bool Modified = false;

  SmallVector<IntrinsicInst *> MMAs;

  for (Instruction &I : BB) {
    auto *BF16MMA = getBF16ZludaMMA(I);
    if (BF16MMA) {
      MMAs.push_back(BF16MMA);
      continue;
    }
    auto *F16MMA = getF16ZludaMMA(I);
    if (F16MMA) {
      MMAs.push_back(F16MMA);
      continue;
    }
    auto *S8MMA = getS8ZludaMMA(I);
    if (S8MMA) {
      MMAs.push_back(S8MMA);
      continue;
    }
  }

  Stats.Seen += MMAs.size();
  if (!MMAs.empty())
    Stats.Blocks++;

  Modified |= combineMMAs(MMAs);

  return Modified;
}

bool MMACombiner::combine(Function &F) {
  bool Modified = false;

  for (BasicBlock &BB : F) {
    Modified |= combineBB(BB);
  }

  for (Instruction *I : MaybeRemove) {
    if (I->user_empty()) {
      I->eraseFromParent();
      Modified = true;
    }
  }

  return Modified;
}

PreservedAnalyses CombineMMAPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  MMACombiner Combiner;
  bool Changed = Combiner.combine(F);
  Combiner.Stats.report(F);
  if (Changed) {
    return PreservedAnalyses::allInSet<CFGAnalyses>();
  }

  return PreservedAnalyses::all();
}
