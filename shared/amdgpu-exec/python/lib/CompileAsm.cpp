//===- CompileAsm.cpp - AMDGPU Assembly Compilation ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiles AMDGPU assembly to an ELF binary using LLVM MC, then links it
// to an HSA code object using LLD.
//
// File adapted from https://github.com/iree-org/aster
//
//===----------------------------------------------------------------------===//

#include "API.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include <string>
#include <string_view>

static void initBackend() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
    // If the `AMDGPU` LLVM target was built, initialize it.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
  });
}

// Parses an LLVM IR string into a Module. Prints to stderr and returns failure
// on error.
static llvm::FailureOr<std::unique_ptr<llvm::Module>>
parseIrModule(std::string_view irCode, llvm::LLVMContext &ctx) {
  llvm::SMDiagnostic err;
  llvm::MemoryBufferRef bufRef(llvm::StringRef(irCode.data(), irCode.size()),
                               "<llvm-ir>");
  std::unique_ptr<llvm::Module> mod = llvm::parseIR(bufRef, err, ctx);
  if (!mod) {
    llvm::errs() << "failed to parse LLVM IR: " << err.getMessage() << "\n";
    return llvm::failure();
  }
  return std::move(mod);
}

// Bitmask of device bitcode libraries to link in.
enum class DeviceLibs : uint32_t {
  None = 0,
  Ockl = 1 << 0,
  Ocml = 1 << 1,
  OpenCL = 1 << 2,
  Hip = 1 << 3,
};

static DeviceLibs operator|(DeviceLibs a, DeviceLibs b) {
  return static_cast<DeviceLibs>(static_cast<uint32_t>(a) |
                                 static_cast<uint32_t>(b));
}
static bool hasFlag(DeviceLibs libs, DeviceLibs flag) {
  return (static_cast<uint32_t>(libs) & static_cast<uint32_t>(flag)) != 0;
}

// Inspects external function declarations to determine which device libraries
// are required.
static DeviceLibs detectRequiredLibs(llvm::Module &mod) {
  DeviceLibs libs = DeviceLibs::None;
  for (llvm::Function &f : mod.functions()) {
    if (!f.hasExternalLinkage() || !f.hasName() || f.hasExactDefinition())
      continue;
    llvm::StringRef name = f.getName();
    if (name == "printf")
      libs = libs | DeviceLibs::OpenCL | DeviceLibs::Ockl | DeviceLibs::Ocml;
    else if (name.starts_with("__ockl_"))
      libs = libs | DeviceLibs::Ockl;
    else if (name.starts_with("__ocml_"))
      libs = libs | DeviceLibs::Ocml;
    else if (name == "__atomic_work_item_fence")
      libs = libs | DeviceLibs::Hip;
  }
  // ocml depends on ockl, so always pull ockl when ocml is required.
  if (hasFlag(libs, DeviceLibs::Ocml))
    libs = libs | DeviceLibs::Ockl;
  return libs;
}

// Sets the amdhsa_code_object_version module flag if not already present. This
// governs the emitted code object ABI independently of device-lib linking.
static void setCodeObjectVersion(llvm::Module &mod, AbiVersion abiVersion) {
  if (!mod.getModuleFlag("amdhsa_code_object_version"))
    mod.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                      static_cast<int>(abiVersion));
}

// Adds the OCLC control variable globals required by device libraries.
static void addControlVariables(llvm::Module &mod, llvm::StringRef chip,
                                DeviceLibs libs, const CompileOptions &opts) {
  if (libs == DeviceLibs::None)
    return;

  llvm::LLVMContext &ctx = mod.getContext();
  int abiVal = static_cast<int>(opts.abiVersion);

  auto addVar = [&](llvm::StringRef name, uint32_t value, uint32_t bits) {
    if (mod.getNamedGlobal(name))
      return;
    llvm::IntegerType *ty = llvm::IntegerType::getIntNTy(ctx, bits);
    auto *gv = new llvm::GlobalVariable(
        mod, ty, /*isConstant=*/true, llvm::GlobalValue::LinkOnceODRLinkage,
        llvm::ConstantInt::get(ty, value), name,
        /*insertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
        /*addressSpace=*/4);
    gv->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    gv->setAlignment(llvm::MaybeAlign(bits / 8));
    gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  };

  if (hasFlag(libs, DeviceLibs::Ocml)) {
    bool fast = opts.fastMath;
    addVar("__oclc_finite_only_opt", opts.finiteOnly || fast, 8);
    addVar("__oclc_daz_opt", opts.daz || fast, 8);
    addVar("__oclc_correctly_rounded_sqrt32", opts.correctSqrt && !fast, 8);
    addVar("__oclc_unsafe_math_opt", opts.unsafeMath || fast, 8);
  }

  if (hasFlag(libs, DeviceLibs::Ocml) || hasFlag(libs, DeviceLibs::Ockl)) {
    addVar("__oclc_wavefrontsize64", opts.wave64, 8);
    llvm::AMDGPU::IsaVersion isa = llvm::AMDGPU::getIsaVersion(chip);
    uint32_t isaNum = isa.Minor + 100 * isa.Stepping + 1000 * isa.Major;
    addVar("__oclc_ISA_version", isaNum, 32);
    addVar("__oclc_ABI_version", static_cast<uint32_t>(abiVal), 32);
  }
}

// Loads a single bitcode file and synchronises its data layout and triple with
// those of the parent module.
static std::unique_ptr<llvm::Module>
loadBitcodeLib(llvm::StringRef path, llvm::LLVMContext &ctx,
               const llvm::Module &parent) {
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> lib = llvm::getLazyIRFileModule(path, err, ctx);
  if (!lib) {
    llvm::errs() << "failed to load bitcode library " << path << ": "
                 << err.getMessage() << "\n";
    return nullptr;
  }
  lib->setDataLayout(parent.getDataLayout());
  lib->setTargetTriple(parent.getTargetTriple());
  return lib;
}

// Links required device bitcode libraries into mod. Only called when
// opts.rocmPath has a value.
static bool linkDeviceLibs(llvm::Module &mod, llvm::LLVMContext &ctx,
                           DeviceLibs libs, llvm::StringRef rocmPath) {
  if (libs == DeviceLibs::None)
    return true;

  llvm::SmallString<256> bitcodePath(rocmPath);
  llvm::sys::path::append(bitcodePath, "amdgcn", "bitcode");

  if (!llvm::sys::fs::is_directory(bitcodePath)) {
    llvm::errs() << "ROCm bitcode directory not found: " << bitcodePath << "\n";
    return false;
  }

  auto tryLink = [&](llvm::StringRef libName) -> bool {
    llvm::SmallString<256> libPath(bitcodePath);
    llvm::sys::path::append(libPath, libName);
    std::unique_ptr<llvm::Module> lib = loadBitcodeLib(libPath, ctx, mod);
    if (!lib)
      return false;
    llvm::Linker linker(mod);
    bool err = linker.linkInModule(
        std::move(lib), llvm::Linker::Flags::LinkOnlyNeeded,
        [](llvm::Module &m, const llvm::StringSet<> &gvs) {
          llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
            return !gv.hasName() || (gvs.count(gv.getName()) == 0);
          });
        });
    if (err) {
      llvm::errs() << "failed to link bitcode library: " << libName << "\n";
      return false;
    }
    return true;
  };

  if (hasFlag(libs, DeviceLibs::Ocml) && !tryLink("ocml.bc"))
    return false;
  if (hasFlag(libs, DeviceLibs::Ockl) && !tryLink("ockl.bc"))
    return false;
  if (hasFlag(libs, DeviceLibs::OpenCL) && !tryLink("opencl.bc"))
    return false;
  if (hasFlag(libs, DeviceLibs::Hip) && !tryLink("hip.bc"))
    return false;
  return true;
}

// Maps OptLevel to llvm::OptimizationLevel.
static llvm::OptimizationLevel toOptimizationLevel(OptLevel level) {
  switch (level) {
  case OptLevel::O0:
    return llvm::OptimizationLevel::O0;
  case OptLevel::O1:
    return llvm::OptimizationLevel::O1;
  case OptLevel::O2:
    return llvm::OptimizationLevel::O2;
  case OptLevel::O3:
    return llvm::OptimizationLevel::O3;
  }
  llvm_unreachable("unhandled OptLevel");
}

// Maps OptLevel to llvm::CodeGenOptLevel.
static llvm::CodeGenOptLevel toCodeGenOptLevel(OptLevel level) {
  switch (level) {
  case OptLevel::O0:
    return llvm::CodeGenOptLevel::None;
  case OptLevel::O1:
    return llvm::CodeGenOptLevel::Less;
  case OptLevel::O2:
    return llvm::CodeGenOptLevel::Default;
  case OptLevel::O3:
    return llvm::CodeGenOptLevel::Aggressive;
  }
  llvm_unreachable("unhandled OptLevel");
}

// Runs the new pass manager optimization pipeline on mod.
static void optimizeModule(llvm::Module &mod, llvm::TargetMachine &tm,
                           OptLevel level) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PipelineTuningOptions pto;
  pto.LoopUnrolling = true;
  pto.LoopInterleaving = true;
  pto.LoopVectorization = true;
  pto.SLPVectorization = true;

  llvm::PassBuilder pb(&tm, pto);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel ol = toOptimizationLevel(level);
  // Sync CodeGenOptLevel so the TargetMachine selects matching schedules.
  tm.setOptLevel(toCodeGenOptLevel(level));

  llvm::ModulePassManager mpm;
  mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  mpm.run(mod, mam);
}

// Compiles LLVM IR source to AMDGPU assembly text. Prints to stderr and
// returns failure on error.
llvm::FailureOr<std::string> llvmIrToAsm(std::string_view irCode,
                                         std::string_view chip,
                                         std::string_view features,
                                         std::string_view triple,
                                         CompileOptions opts) {
  initBackend();
  llvm::Triple targetTriple(llvm::Triple::normalize(triple));

  std::string lookupError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, lookupError);
  if (!target) {
    llvm::errs() << "failed to lookup target: " << lookupError << "\n";
    return llvm::failure();
  }

  // Ensure the wavefront-size target feature matches opts.wave64 so codegen
  // and the device-lib control variables agree. An explicit feature in the
  // caller-provided string takes precedence.
  std::string effectiveFeatures(features);
  if (effectiveFeatures.find("wavefrontsize") == std::string::npos) {
    if (!effectiveFeatures.empty())
      effectiveFeatures += ",";
    effectiveFeatures += opts.wave64 ? "+wavefrontsize64" : "+wavefrontsize32";
  }

  const llvm::TargetOptions tmOptions;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      targetTriple, chip, effectiveFeatures, tmOptions, llvm::Reloc::PIC_));
  if (!tm) {
    llvm::errs() << "failed to create TargetMachine\n";
    return llvm::failure();
  }

  llvm::LLVMContext ctx;
  llvm::FailureOr<std::unique_ptr<llvm::Module>> mod =
      parseIrModule(irCode, ctx);
  if (llvm::failed(mod))
    return llvm::failure();
  (*mod)->setTargetTriple(targetTriple);
  (*mod)->setDataLayout(tm->createDataLayout());

  // The code object version applies regardless of device-lib linking.
  setCodeObjectVersion(**mod, opts.abiVersion);

  if (opts.rocmPath) {
    DeviceLibs libs = detectRequiredLibs(**mod);
    addControlVariables(**mod, chip, libs, opts);
    if (!linkDeviceLibs(**mod, ctx, libs, *opts.rocmPath))
      return llvm::failure();
  }

  optimizeModule(**mod, *tm, opts.optLevel);

  llvm::SmallVector<char, 0> asmBuf;
  llvm::raw_svector_ostream os(asmBuf);
  llvm::legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, os, nullptr,
                              llvm::CodeGenFileType::AssemblyFile)) {
    llvm::errs() << "TargetMachine cannot emit assembly\n";
    return llvm::failure();
  }
  pm.run(**mod);

  return std::string(asmBuf.begin(), asmBuf.end());
}

// Assembles AMDGPU ISA source to an ELF binary using LLVM MC. Prints to
// stderr and returns failure on error.
llvm::FailureOr<llvm::SmallVector<char>> compileAsm(std::string_view asmCode,
                                                    std::string_view chip,
                                                    std::string_view features,
                                                    std::string_view triple) {
  initBackend();
  // Normalize the target triple
  llvm::Triple targetTriple(llvm::Triple::normalize(triple));

  // Lookup the target
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "failed to lookup target: " << error << "\n";
    return llvm::failure();
  }

  // Create source manager with the assembly code
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(asmCode),
                            llvm::SMLoc());

  // Create MC target options
  const llvm::MCTargetOptions mcOptions;

  // Create register info, asm info, and subtarget info
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(targetTriple));
  assert(mri && "failed to create MCRegisterInfo");
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, targetTriple, mcOptions));
  assert(mai && "failed to create MCAsmInfo");
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(targetTriple, chip, features));
  assert(sti && "failed to create MCSubtargetInfo");

  // Create MC context
  llvm::MCContext ctx(targetTriple, *mai, *mri, *sti, &srcMgr);

  // Create object file info
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(
      target->createMCObjectFileInfo(ctx, /*PIC=*/false,
                                     /*LargeCodeModel=*/false));
  assert(mofi && "failed to create MCObjectFileInfo");
  ctx.setObjectFileInfo(mofi.get());

  // Set compilation directory
  llvm::SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  // Create instruction info
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());
  assert(mcii && "failed to create MCInstrInfo");

  llvm::SmallVector<char> binaryBuf;
  llvm::raw_svector_ostream os(binaryBuf);

  // Create code emitter and assembler backend
  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  assert(ce && "failed to create MCCodeEmitter");
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  assert(mab && "failed to create MCAsmBackend");

  // Create object streamer
  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  mcStreamer.reset(target->createMCObjectStreamer(
      targetTriple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti));
  assert(mcStreamer && "failed to create MCObjectStreamer");

  // Create assembly parser
  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  assert(parser && "failed to create MCAsmParser");

  // Create target-specific assembly parser
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii));
  assert(tap && "assembler initialization error");

  // Set the target parser and run the assembly parser
  parser->setTargetParser(*tap);
  if (parser->Run(false)) {
    llvm::errs() << "assembly parsing failed\n";
    return llvm::failure();
  }

  return binaryBuf;
}

LLD_HAS_DRIVER(elf)

// Links an ELF binary to an HSA code object using LLD. Prints to stderr and
// returns failure on error.
llvm::FailureOr<llvm::SmallVector<char>>
linkBinary(llvm::ArrayRef<char> binary) {
  int tempIsaBinaryFd = -1;
  llvm::SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    llvm::errs() << "failed to create a temporary file for the ISA binary\n";
    return llvm::failure();
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);

  // Write the binary to the temporary file
  {
    llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, /*shouldClose=*/true);
    tempIsaBinaryOs.write(binary.data(), static_cast<int64_t>(binary.size()));
    tempIsaBinaryOs.flush();
  }

  // Create a temporary file for the HSA code object
  llvm::SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco",
                                         tempHsacoFilename)) {
    llvm::errs()
        << "failed to create a temporary file for the HSA code object\n";
    return llvm::failure();
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  {
    // LLD is not thread-safe; serialize invocations.
    static llvm::sys::Mutex mutex;
    const llvm::sys::ScopedLock lock(mutex);
    // Invoke lld. Expect a true return value from lld.
    if (!lld::elf::link({"ld.lld", "-shared", tempIsaBinaryFilename.c_str(),
                         "-o", tempHsacoFilename.c_str()},
                        llvm::outs(), llvm::errs(), false, false)) {
      llvm::errs() << "lld invocation error\n";
      lld::CommonLinkerContext::destroy();
      return llvm::failure();
    }
    lld::CommonLinkerContext::destroy();
  }

  auto hsacoFile =
      llvm::MemoryBuffer::getFile(tempHsacoFilename, /*IsText=*/false);
  if (!hsacoFile) {
    llvm::errs() << "failed to read the HSA code object from the temp file\n";
    return llvm::failure();
  }

  llvm::StringRef buf = (*hsacoFile)->getBuffer();
  return llvm::SmallVector<char>(buf.begin(), buf.end());
}
