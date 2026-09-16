#include "llvm/Transforms/ZLUDA/CombineMMA.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
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
//
// Takes plain instructions rather than intrinsics: pairMMAWrappers asks the
// same question about two calls that combineMMA asks about two multiplies.
//
// SkipOperand exempts one operand of FromBefore from the walk. A chained group
// needs exactly that: the accumulator of its later multiplies is the result of
// its earlier ones, which is about to be satisfied inside the group's own body
// and so does not have to be computable out here. Without the exemption the
// answer is always no, since that operand is one of the calls being replaced.
static bool tryToReorderOperands(Instruction *FromBefore, Instruction *ToBefore,
                                 int SkipOperand = -1) {
  assert(FromBefore->getParent() == ToBefore->getParent());

  SmallPtrSet<Instruction *, 16> InstructionsToMove;
  SmallVector<Instruction *, 16> Worklist;

  Worklist.emplace_back(FromBefore);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    for (unsigned OpIdx = 0, OpEnd = I->getNumOperands(); OpIdx < OpEnd; ++OpIdx) {
      if (I == FromBefore && (int)OpIdx == SkipOperand) {
        continue;
      }
      Value *Operand = I->getOperand(OpIdx);
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

// Same PTX shape as s8 (m16n8k32, 8-bit elements), native on gfx12 only --
// gfx11 never reaches this intrinsic, since the HIP source only emits it when
// __oclc_ISA_version >= 12000.
static IntrinsicInst *getFp8ZludaMMA(Instruction &I) {
  auto *MMA = dyn_cast<IntrinsicInst>(&I);
  if (MMA && MMA->getIntrinsicID() ==
                 Intrinsic::zluda_mma_m16n8k32_f32_fp8_fp8_f32) {
    return MMA;
  }
  return nullptr;
}

// The mma intrinsic is wrapped in a noinline function so that ZLUDA's own
// passes cannot fold it away. That wrapper also hides it from this pass: with
// the wrapper standing, every basic block holds calls and no intrinsics, and
// there is never a pair to fuse.
//
// Opening the wrapper at the source, by marking the call always_inline, does
// make every multiply visible -- and costs more than the fusion returns. The
// wrapper is a scheduling barrier as well as a hiding place: with it gone the
// scheduler interleaves the multiplies and the live ranges grow. Measured on
// cc_vit_1d_qkv_chained_fp8, which has no fusable pairs at all and so pays the
// cost and collects none of the gain: 585 spilled registers became 2336, 1200
// bytes of scratch became 4180, and the kernel went from 14.0 ms to 55.0 ms.
// Across the whole network that was +85 ms against the -67 ms the fusion won
// elsewhere.
//
// So the wrapper is opened here instead, and only where two calls in one block
// share operand A -- which is to say only where this pass is about to fuse
// them, and the barrier is paid for by a multiply that halves.
static bool isMMAWrapper(const Function *F) {
  if (!F || F->isDeclaration() || F->size() != 1) {
    return false;
  }
  for (const Instruction &I : F->front()) {
    const auto *Inner = dyn_cast<IntrinsicInst>(&I);
    if (!Inner) {
      continue;
    }
    switch (Inner->getIntrinsicID()) {
    case Intrinsic::zluda_mma_m16n8k16_f32_f16_f16_f32:
    case Intrinsic::zluda_mma_m16n8k16_f32_bf16_bf16_f32:
    case Intrinsic::zluda_mma_m16n8k32_s32_s8_s8_s32:
    case Intrinsic::zluda_mma_m16n8k32_f32_fp8_fp8_f32:
      return true;
    default:
      break;
    }
  }
  return false;
}

// How many wrappers were opened, reported with the rest of the statistics.
static unsigned Opened = 0;

// Group calls built by pairMMAWrappers, reported with the rest.
static unsigned PairsBuilt = 0;

// Whether opening the wrapper is asked for. Set from the host by
// zludaSetMMAOpen (>= 0 wins); otherwise the ZLUDA_MMA_OPEN environment
// variable. The host setter exists because the selective build has to turn
// fusion on and off between two compilations of the same module, and Rust's
// std::env::set_var does not reach this getenv on Windows -- the C runtime
// keeps its own copy of the environment.
static int MMAOpenOverride = -1;
static bool mmaOpenRequested() {
  if (MMAOpenOverride >= 0) {
    return MMAOpenOverride != 0;
  }
  return ::getenv("ZLUDA_MMA_OPEN") != nullptr;
}

extern "C" void zludaSetMMAOpen(int on) { MMAOpenOverride = on; }

// The wrappers carry noinline only: the C++ marks them [[clang::optnone]], but
// that attribute does not survive into the linked bitcode. The fragment
// conversion inside a wrapper is therefore already fully optimised; making it
// cheaper means emitting less of it (see LowerMatrixConversions), not letting
// the optimiser at it.

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

static bool openMMAWrappers(BasicBlock &BB) {
  // Off unless asked for. Opening the wrapper is what lets this pass fuse at
  // all, and on its own it costs more than it returns: the swin family gains
  // 20% and the vit family loses three times that. The selective build turns
  // it on only where it pays (see optimize_and_emit / selective_fuse_threshold
  // in llvm_zluda), by building each module both ways and keeping the fused
  // one only when its object did not balloon; the fusion itself is sound --
  // 264 pairs of 264 in the hottest kernel.
  if (!mmaOpenRequested()) {
    return false;
  }

  // Group the wrapped multiplies by callee and by the A operand they are
  // passed, which is the same key the fusion itself uses.
  MapVector<std::pair<Function *, Value *>, SmallVector<CallInst *, 4>> Groups;
  for (Instruction &I : BB) {
    auto *Call = dyn_cast<CallInst>(&I);
    if (!Call || Call->arg_size() < 1) {
      continue;
    }
    Function *Callee = Call->getCalledFunction();
    if (!isMMAWrapper(Callee)) {
      continue;
    }
    Groups[{Callee, Call->getArgOperand(0)}].push_back(Call);
  }

  bool Modified = false;
  for (auto &Group : Groups) {
    // A group of one is left alone: opening it would pay the barrier and
    // return nothing, which is exactly the trade that made the network slower.
    if (Group.second.size() < 2) {
      continue;
    }
    for (CallInst *Call : Group.second) {
      InlineFunctionInfo IFI;
      // The wrapper is one block, so this splices into the caller's block
      // rather than splitting it -- which matters, because the fusion only
      // pairs multiplies that end up in the same block.
      if (InlineFunction(*Call, IFI).isSuccess()) {
        Opened++;
        Modified = true;
      }
    }
  }
  return Modified;
}

// Fusing without inlining: multiplies that share an A operand become one call
// to a function that converts A once and multiplies several times.
//
// On the generic targets the cross-lane relayout of the fragments is 57% of the
// hot kernel, measured by ablation, and much of it is spent more than once:
// multiplies in that kernel share their A operand, and each call converts A for
// itself because the calls sit behind separate invocations of the same noinline
// wrapper, where nothing can see the duplication.
//
// Opening the wrappers (openMMAWrappers, above) does expose it, and the
// existing fusion then removes it -- but it inlines 512 copies into the
// caller, and the register pressure that follows is what makes fusion a loss
// on a generic target, which gets 64 VGPRs where a native gfx1100 build gets
// 96. This keeps the fusion and drops the inlining: the group function is
// noinline, so the pressure lives in its short body rather than across the
// caller's loop.
//
// On by default: on the generic target, at width two, the hot kernel goes
// 79.9 -> 70.5 ms and the whole network 422 -> 399 ms, with the image identical
// bit for bit and the object 0.5% larger. ZLUDA_MMA_PAIR=0 turns it off.
//
// It costs nothing where it does not apply: pairMMAWrappers hands the calls
// back to fusion whenever fusion is driving the build, which is every native
// tier.
static bool mmaPairRequested() {
  static const bool on = [] {
    const char *Value = ::getenv("ZLUDA_MMA_PAIR");
    return !Value || ::strcmp(Value, "0") != 0;
  }();
  return on;
}

// Whether to group a chain into one body rather than group strictly by A.
//
// The accumulator is the expensive matrix, measured: on the generic target,
// dropping the cross-lane traffic of the accumulator on the way in is 14.5% of
// the hot kernel and of the result on the way out 22.1%, while A and B together
// do not measure at all. Every multiply pays both today.
//
// A chain does not have to: where one pair's two results are the next pair's
// two accumulators, the split that leaves AMD layout and the concatenate that
// re-enters it cancel, and combineC already knows how to cancel them -- it just
// never sees both, because the two pairs sit in different group functions. In
// the hot kernel that is 32 of the 128 pairs, all of them inside one basic
// block.
//
// The cost is one A conversion: a chained group carries two distinct A operands
// where an A-grouped one carries a single shared one. A is the matrix that does
// not measure, so the trade is worth making.
//
// On by default, measured on gfx11-generic. In the IR the fold is visible as
// exactly one round trip gone per chained body: 704 permlanes become 640, which
// is the 32 of a concatenate plus the 32 of a split, with ds_bpermute and the
// WMMA count untouched. On the clock, the hot kernel goes 84.7 -> 75.8 ms with
// no run of one variant overlapping the other, and the whole network 379 ->
// 368.5 ms (-2.8%), where 14 of 16 chained runs beat every plain run. The
// output image is bit-identical in every run, as an exact cancellation
// requires, and the translated network comes out 247 KB smaller.
//
// It reaches only the generic targets: on a native tier the selective fusion
// opens the wrappers and pairMMAWrappers stands down, so there is nothing here
// to group.
static bool mmaChainRequested() {
  static const bool on = [] {
    const char *Value = ::getenv("ZLUDA_MMA_CHAIN");
    return !Value || ::strcmp(Value, "0") != 0;
  }();
  return on;
}

// Chained groups built, reported with the rest.
static unsigned ChainGroups = 0;

// How many multiplies one group function takes.
//
// The hot kernel's 512 calls fall into only 80 groups sharing an A operand.
// Taken two at a time, A is converted 256 times instead of 512; taken six at a
// time, 80 times, because inside one function body the conversion intrinsic is
// pure and identical and the optimiser folds the copies -- which it cannot do
// across a call boundary. Against that, N accumulators and N B operands are
// live at once inside the body, so register pressure grows with N.
static unsigned mmaPairWidth() {
  static const unsigned N = [] {
    // Four, measured on the hot kernel: 79.8 ms with no grouping, 70.4 at two,
    // 68.5 at four, and the smallest object (278 KB against 292 at two).
    const char *Value = ::getenv("ZLUDA_MMA_PAIR_N");
    if (!Value) {
      return 4u;
    }
    const int Parsed = ::atoi(Value);
    return Parsed >= 2 ? (unsigned)Parsed : 2u;
  }();
  return N;
}

// One multiply inside a group function: which of the group's A operands it
// takes, and where its accumulator comes from.
struct GroupSlot {
  unsigned AIndex = 0; // index into the group's A parameters
  int CFromSlot = -1;  // -1: C is a parameter; otherwise that slot's result
};

// Builds {D1..DN} f(A1..Am, then B and, where it is not satisfied internally,
// C for each slot), with every multiply inlined into it so the fusion that runs
// on it afterwards sees them as intrinsics.
//
// A slot whose accumulator comes from another slot is what makes a chain worth
// grouping: inside one body the producer's split and the consumer's
// concatenate meet, and combineC cancels the pair. Across two group calls they
// never meet, which is why that round trip survives today.
static Function *buildGroupFunction(Function *Wrapper, ArrayRef<GroupSlot> Slots,
                                    unsigned NumA, const Twine &Suffix) {
  LLVMContext &Ctx = Wrapper->getContext();
  Type *ATy = Wrapper->getArg(0)->getType();
  Type *BTy = Wrapper->getArg(1)->getType();
  Type *CTy = Wrapper->getArg(2)->getType();
  Type *RetTy = Wrapper->getReturnType();
  const unsigned Width = (unsigned)Slots.size();

  SmallVector<Type *> Returns(Width, RetTy);
  StructType *GroupTy = StructType::get(Ctx, Returns);

  SmallVector<Type *> Params(NumA, ATy);
  for (const GroupSlot &Slot : Slots) {
    Params.push_back(BTy);
    if (Slot.CFromSlot < 0) {
      Params.push_back(CTy);
    }
  }

  FunctionType *FT = FunctionType::get(GroupTy, Params, /*isVarArg=*/false);
  Function *F = Function::Create(FT, GlobalValue::InternalLinkage,
                                 Wrapper->getName() + ".group" + Suffix,
                                 Wrapper->getParent());
  // Convergent because what it wraps is: the multiply and the lane exchanges
  // around it are wave-wide operations. noinline is the whole point.
  F->addFnAttr(Attribute::Convergent);
  F->addFnAttr(Attribute::NoInline);
  F->setCallingConv(Wrapper->getCallingConv());

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> Builder(BB);

  SmallVector<CallInst *> Calls;
  unsigned NextParam = NumA;
  for (const GroupSlot &Slot : Slots) {
    Value *B = F->getArg(NextParam++);
    // A slot can only take its accumulator from an earlier one, which the
    // grouping guarantees by building producers before consumers.
    Value *C = Slot.CFromSlot < 0 ? (Value *)F->getArg(NextParam++)
                                  : (Value *)Calls[Slot.CFromSlot];
    CallInst *Call =
        Builder.CreateCall(Wrapper, {F->getArg(Slot.AIndex), B, C});
    Call->setCallingConv(Wrapper->getCallingConv());
    Calls.push_back(Call);
  }

  Value *Result = PoisonValue::get(GroupTy);
  for (unsigned I = 0; I < Width; ++I) {
    Result = Builder.CreateInsertValue(Result, Calls[I], {I});
  }
  Builder.CreateRet(Result);

  // Inline them all, so what is left in this body is Width mma intrinsics in
  // one block -- the shape the fusion already knows how to combine, and the
  // shape in which the duplicate A conversions become visible to the
  // optimiser.
  for (CallInst *Call : Calls) {
    InlineFunctionInfo IFI;
    if (!InlineFunction(*Call, IFI).isSuccess()) {
      F->eraseFromParent();
      return nullptr;
    }
  }
  return F;
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
    if ((!Seen && !Opened) || !::getenv("ZLUDA_MMA_STATS"))
      return;
    errs() << "[zluda-combine-mma] " << F.getName() << ": " << Seen
           << " mma in " << Blocks << " blocks, " << Paired
           << " with a partner in the same block (" << Fused << " fused, "
           << ReorderFailed << " refused), " << Alone
           << " alone in their block; refusals: " << RefusedByDependency
           << " dependency, " << RefusedByMemory << " memory; "
           << "first blocker: " << FirstBlocker << "; " << Opened
           << " wrappers opened; " << PairsBuilt << " group calls built ("
           << ChainGroups << " of them chained)\n";
  }
};
} // namespace

class MMACombiner {
public:
  bool combine(Function &F);

private:
  bool combineBB(BasicBlock &BB);
  bool pairMMAWrappers(BasicBlock &BB);
  bool combineMMAs(SmallVectorImpl<IntrinsicInst *> &MMAs);
  bool combineMMA(IntrinsicInst *First, IntrinsicInst *Second);

  llvm::Value *EmitAmdMmaI8(llvm::IRBuilder<> &Builder, llvm::Value *FirstA,
                            llvm::Value *FirstB, llvm::Value *SecondB,
                            llvm::Value *FirstC, llvm::Value *SecondC);
  llvm::Value *EmitAmdMmaFp8(llvm::IRBuilder<> &Builder, llvm::Value *FirstA,
                             llvm::Value *FirstB, llvm::Value *SecondB,
                             llvm::Value *FirstC, llvm::Value *SecondC);

  void lowerMMA(IntrinsicInst *MMA);

  Value *combineC(IRBuilder<> &Builder, Value *FirstC, Value *SecondC);
  Value *combineCFp8(IRBuilder<> &Builder, Value *FirstC, Value *SecondC);
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

// The FP8/GFX12 analogue of combineC: same "reuse a prior split rather than
// concatenate and immediately re-split" shortcut, aimed at the GFX12-specific
// split/concatenate pair instead of the GFX11 one.
Value *MMACombiner::combineCFp8(IRBuilder<> &Builder, Value *FirstC,
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
            Intrinsic::zluda_dmatrix_split_fp8_nv16x8_amd16x16) {
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
      V8I32Ty, Intrinsic::zluda_cmatrix_concatenate_fp8_amd16x16_nv16x8,
      {FirstC, SecondC});
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
  } else if (First->getIntrinsicID() ==
             Intrinsic::zluda_mma_m16n8k32_f32_fp8_fp8_f32) {
    Split = EmitAmdMmaFp8(Builder, FirstA, FirstB, SecondB, FirstC, SecondC);
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

// Same structure as EmitAmdMmaI8 -- gfx12's fp8_fp8 WMMA shares the s8
// (iu8) WMMA's exact operand profile, [v8f32/v8i32, v2i32, v2i32,
// v8f32/v8i32]: A and B are 2 x i32 per lane either way, and AMD's own
// documentation states the operand layout is unified across the 8-bit-element
// WMMA family (int8, int4, and by the same construction fp8). The one
// difference is the accumulator: f32 rather than s32, and the WMMA has no
// sign-of-A/sign-of-B/clamp operands the way iu8 does -- fp8 is not signed or
// unsigned, it names its own format, so the call takes exactly A, B, C.
//
// The K=32 -> two K=16 AMD ops split (SplitA/ReshapedB, unchanged from the s8
// case) and the N=8 -> N=16 fusion (FirstB/SecondB, real when fused, zero
// when not) are both reused verbatim: they move bytes between lanes without
// regard to what the bytes mean, which is exactly why the s8 infrastructure
// carries over rather than needing its own.
llvm::Value *MMACombiner::EmitAmdMmaFp8(llvm::IRBuilder<> &Builder,
                                        llvm::Value *A, llvm::Value *FirstB,
                                        llvm::Value *SecondB,
                                        llvm::Value *FirstC,
                                        llvm::Value *SecondC) {
  auto V2I32Ty = VectorType::get(Builder.getInt32Ty(), 2, /*Scalable=*/false);
  auto V2I32x2Ty = StructType::get(Builder.getContext(), {V2I32Ty, V2I32Ty});
  auto V4I32Ty = VectorType::get(Builder.getInt32Ty(), 4, /*Scalable=*/false);
  auto V4I32x2Ty = StructType::get(Builder.getContext(), {V4I32Ty, V4I32Ty});
  auto V8I32Ty = VectorType::get(Builder.getInt32Ty(), 8, /*Scalable=*/false);
  auto V8F32Ty = VectorType::get(Builder.getFloatTy(), 8, /*Scalable=*/false);

  // GFX12's own conversions: see the derivation comment above
  // aMatrixConvertFp8Half in LowerMatrixConversions.cpp. Not the GFX11 ones
  // above -- the operand width and lane layout genuinely differ between the
  // two generations for every one of A, B and the accumulator.
  auto ConvertedA = Builder.CreateIntrinsic(
      V2I32x2Ty, Intrinsic::zluda_amatrix_convert_fp8_amd16x16_nv16x32, {A});
  auto ConvertedB = Builder.CreateIntrinsic(
      V2I32x2Ty, Intrinsic::zluda_bmatrix_convert_fp8_amd16x16_nv32x8,
      {FirstB, SecondB});

  auto CombinedC = combineCFp8(Builder, FirstC, SecondC);
  auto A0 = Builder.CreateExtractValue(ConvertedA, {0});
  auto A1 = Builder.CreateExtractValue(ConvertedA, {1});
  auto B0 = Builder.CreateExtractValue(ConvertedB, {0});
  auto B1 = Builder.CreateExtractValue(ConvertedB, {1});
  auto CombinedCFloat = Builder.CreateBitCast(CombinedC, V8F32Ty);

  auto TempD = Builder.CreateIntrinsic(
      V8F32Ty, Intrinsic::amdgcn_wmma_f32_16x16x16_fp8_fp8,
      {A0, B0, CombinedCFloat});
  auto D = Builder.CreateIntrinsic(
      V8F32Ty, Intrinsic::amdgcn_wmma_f32_16x16x16_fp8_fp8, {A1, B1, TempD});
  auto DInt = Builder.CreateBitCast(D, V8I32Ty);
  return Builder.CreateIntrinsic(
      V4I32x2Ty, Intrinsic::zluda_dmatrix_split_fp8_nv16x8_amd16x16, {DInt});
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
  } else if (IID == Intrinsic::zluda_mma_m16n8k32_f32_fp8_fp8_f32) {
    auto BPadding = Constant::getNullValue(B->getType());
    auto CPadding = Constant::getNullValue(C->getType());
    llvm::Value *DoubleResult =
        EmitAmdMmaFp8(Builder, A, B, BPadding, C, CPadding);
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

// Replaces groups of wrapper calls that share A with one call to a group
// function, and fuses inside that function. See mmaPairRequested for why.
bool MMACombiner::pairMMAWrappers(BasicBlock &BB) {
  if (!mmaPairRequested()) {
    return false;
  }

  if (mmaOpenRequested() || MMAOpenOverride >= 0) {
    return false;
  }

  MapVector<std::pair<Function *, Value *>, SmallVector<CallInst *, 4>> Groups;
  // Which wrapper call takes another one's result as its accumulator. Built
  // over the same scan, since a chain is visible here and nowhere later: once
  // the calls are inside group functions the relation is gone.
  DenseMap<CallInst *, CallInst *> ConsumerOf;
  for (Instruction &I : BB) {
    auto *Call = dyn_cast<CallInst>(&I);
    if (!Call || Call->arg_size() != 3) {
      continue;
    }
    Function *Callee = Call->getCalledFunction();
    if (!isMMAWrapper(Callee)) {
      continue;
    }
    Groups[{Callee, Call->getArgOperand(0)}].push_back(Call);
    if (auto *Producer = dyn_cast<CallInst>(Call->getArgOperand(2))) {
      if (Producer->getParent() == &BB && isMMAWrapper(Producer->getCalledFunction())) {
        // One consumer per producer is all this looks for: a result read twice
        // is not a chain that can be kept in AMD layout.
        if (!ConsumerOf.count(Producer)) {
          ConsumerOf[Producer] = Call;
        } else {
          ConsumerOf[Producer] = nullptr;
        }
      }
    }
  }

  // One group function per wrapper and shape, not one per group.
  std::map<std::string, Function *> PairFunctions;
  bool Modified = false;
  // Calls already spoken for by a chained group, which the plain grouping below
  // must then leave alone.
  SmallPtrSet<CallInst *, 32> Taken;

  if (mmaChainRequested()) {
    // A pair whose two results are the next pair's two accumulators. Those four
    // calls in one body are what lets combineC cancel the split and the
    // concatenate between them.
    for (auto &Group : Groups) {
      SmallVectorImpl<CallInst *> &Calls = Group.second;
      Function *Wrapper = Group.first.first;
      for (size_t i = 0; i + 1 < Calls.size(); i += 2) {
        CallInst *X1 = Calls[i], *X2 = Calls[i + 1];
        if (Taken.count(X1) || Taken.count(X2)) {
          continue;
        }
        CallInst *Y1 = ConsumerOf.lookup(X1), *Y2 = ConsumerOf.lookup(X2);
        if (!Y1 || !Y2 || Y1 == Y2 || Taken.count(Y1) || Taken.count(Y2)) {
          continue;
        }
        // The two consumers have to be a pair themselves, or nothing cancels:
        // the fold wants both halves of one split.
        if (Y1->getArgOperand(0) != Y2->getArgOperand(0) ||
            Y1->getCalledFunction() != Wrapper || Y2->getCalledFunction() != Wrapper) {
          continue;
        }
        CallInst *First = X1;
        // The accumulator of a consumer is exempt (operand 2): the group's own
        // body will supply it. Everything else still has to be computable where
        // the single call replacing all four is about to go.
        bool Reordered =
            First->comesBefore(X2) && tryToReorderOperands(X2, First) &&
            First->comesBefore(Y1) && tryToReorderOperands(Y1, First, 2) &&
            First->comesBefore(Y2) && tryToReorderOperands(Y2, First, 2);
        if (!Reordered) {
          continue;
        }

        const GroupSlot Slots[4] = {{0, -1}, {0, -1}, {1, 0}, {1, 1}};
        // Keyed by the wrapper too: one function holds both an f16 and an fp8
        // wrapper when the kernel uses both, and a group built for one is the
        // wrong shape for the other.
        Function *&ChainF = PairFunctions[Wrapper->getName().str() + "|chain4"];
        if (!ChainF) {
          ChainF = buildGroupFunction(Wrapper, Slots, /*NumA=*/2, "chain4");
          if (!ChainF) {
            continue;
          }
          // The pass will not visit a function created while it is running.
          MMACombiner Inner;
          Inner.combine(*ChainF);
        }

        IRBuilder<> Builder(First);
        CallInst *Members[4] = {X1, X2, Y1, Y2};
        SmallVector<Value *> Args;
        Args.push_back(X1->getArgOperand(0)); // A of the producing pair
        Args.push_back(Y1->getArgOperand(0)); // A of the consuming pair
        for (unsigned J = 0; J < 4; ++J) {
          Args.push_back(Members[J]->getArgOperand(1));
          if (Slots[J].CFromSlot < 0) {
            Args.push_back(Members[J]->getArgOperand(2));
          }
        }
        CallInst *ChainCall = Builder.CreateCall(ChainF, Args);
        ChainCall->setCallingConv(ChainF->getCallingConv());

        SmallVector<Value *> Results;
        for (unsigned J = 0; J < 4; ++J) {
          Results.push_back(Builder.CreateExtractValue(ChainCall, {J}));
        }
        for (unsigned J = 0; J < 4; ++J) {
          Members[J]->replaceAllUsesWith(Results[J]);
          Taken.insert(Members[J]);
        }
        for (unsigned J = 4; J-- > 0;) {
          Members[J]->eraseFromParent();
        }
        ChainGroups++;
        PairsBuilt++;
        Modified = true;
      }
    }
    // The calls that went into a chained group are gone from the block; drop
    // them from the groups the plain path is about to walk.
    for (auto &Group : Groups) {
      SmallVectorImpl<CallInst *> &Calls = Group.second;
      Calls.erase(std::remove_if(Calls.begin(), Calls.end(),
                                 [&](CallInst *C) { return Taken.count(C) != 0; }),
                  Calls.end());
    }
  }

  for (auto &Group : Groups) {
    SmallVectorImpl<CallInst *> &Calls = Group.second;
    // Widest first, then narrower on what is left over, rather than skipping a
    // remainder that does not fill a whole group. Taking only exact multiples
    // looks harmless and is not: at width six this kernel's groups -- six point
    // four calls each on average -- leave more than half the calls untouched,
    // and the object comes out larger than doing nothing. The remainder has to
    // be swept up or the gain depends on the group sizes dividing evenly, which
    // is a property of one network and not something to rely on.
    const unsigned MaxWidth = mmaPairWidth();
    size_t i = 0;
    while (i + 2 <= Calls.size()) {
      unsigned Width = (unsigned)std::min<size_t>(MaxWidth, Calls.size() - i);
      CallInst *First = Calls[i];
      // Everything the later calls need has to be computable where the first
      // one sits, since that is where the single call replacing them all goes.
      // Asked one at a time so a refusal costs only that group, and asked in
      // order so each move sees the ones before it.
      bool Reordered = true;
      for (unsigned J = 1; J < Width && Reordered; ++J) {
        CallInst *Later = Calls[i + J];
        Reordered = First->comesBefore(Later) &&
                    tryToReorderOperands(Later, First);
      }
      if (!Reordered) {
        // Past this one only: a refusal here says nothing about a group
        // starting at the next call.
        ++i;
        continue;
      }

      Function *Wrapper = Group.first.first;
      // One function per wrapper and width, since the remainder is narrower
      // than the rest and needs a shape of its own.
      Function *&PairF =
          PairFunctions[Wrapper->getName().str() + "|width" + std::to_string(Width)];
      if (!PairF) {
        SmallVector<GroupSlot> Slots(Width, GroupSlot{0, -1});
        PairF = buildGroupFunction(Wrapper, Slots, /*NumA=*/1,
                                   "width" + Twine(Width));
        if (!PairF) {
          ++i;
          continue;
        }
        // The pass will not visit a function created while it is running, so
        // the fusion inside the new body has to be asked for here. A fresh
        // combiner: this one's own bookkeeping belongs to the function it was
        // started on.
        MMACombiner Inner;
        Inner.combine(*PairF);
      }

      IRBuilder<> Builder(First);
      SmallVector<Value *> Args;
      Args.push_back(First->getArgOperand(0)); // the shared A
      for (unsigned J = 0; J < Width; ++J) {
        Args.push_back(Calls[i + J]->getArgOperand(1));
        Args.push_back(Calls[i + J]->getArgOperand(2));
      }
      CallInst *PairCall = Builder.CreateCall(PairF, Args);
      PairCall->setCallingConv(PairF->getCallingConv());

      // Every result is defined before any of the calls it replaces, so no use
      // of any of them -- wherever it sits -- reads something not yet there.
      SmallVector<Value *> Results;
      for (unsigned J = 0; J < Width; ++J) {
        Results.push_back(Builder.CreateExtractValue(PairCall, {J}));
      }
      for (unsigned J = 0; J < Width; ++J) {
        Calls[i + J]->replaceAllUsesWith(Results[J]);
      }
      // Last to first: erasing in reverse keeps the earlier ones valid while
      // the later ones go.
      for (unsigned J = Width; J-- > 0;) {
        Calls[i + J]->eraseFromParent();
      }
      PairsBuilt++;
      Modified = true;
      i += Width;
    }
  }
  return Modified;
}

bool MMACombiner::combineBB(BasicBlock &BB) {
  // For now, we simply combine adjacent m16n8k16 MMAs if possible. This may be
  // good enough in most cases. Any MMAs that cannot be combined are lowered
  // individually.
  // The multiplies are hidden inside a noinline wrapper; the ones that have a
  // partner here are brought into the open first, and only those. Grouping
  // comes first: it consumes the calls that opening would otherwise inline,
  // and the two are alternatives, not stages.
  bool Modified = pairMMAWrappers(BB);
  Modified |= openMMAWrappers(BB);

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
    auto *Fp8MMA = getFp8ZludaMMA(I);
    if (Fp8MMA) {
      MMAs.push_back(Fp8MMA);
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
