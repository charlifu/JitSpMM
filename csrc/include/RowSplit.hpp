#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <argparse/argparse.hpp>
#include <asmjit/x86.h>

const uint64_t step = 256;

double codegen = 0.0;

asmjit::x86::Zmm zmms[32] = {
    asmjit::x86::zmm0, asmjit::x86::zmm1, asmjit::x86::zmm2,
    asmjit::x86::zmm3, asmjit::x86::zmm4, asmjit::x86::zmm5,
    asmjit::x86::zmm6, asmjit::x86::zmm7, asmjit::x86::zmm8,
    asmjit::x86::zmm9, asmjit::x86::zmm10, asmjit::x86::zmm11,
    asmjit::x86::zmm12, asmjit::x86::zmm13, asmjit::x86::zmm14,
    asmjit::x86::zmm15, asmjit::x86::zmm16, asmjit::x86::zmm17,
    asmjit::x86::zmm18, asmjit::x86::zmm19, asmjit::x86::zmm20,
    asmjit::x86::zmm21, asmjit::x86::zmm22, asmjit::x86::zmm23,
    asmjit::x86::zmm24, asmjit::x86::zmm25, asmjit::x86::zmm26,
    asmjit::x86::zmm27, asmjit::x86::zmm28, asmjit::x86::zmm29,
    asmjit::x86::zmm30, asmjit::x86::zmm31
};

asmjit::x86::Ymm ymms[32] = {
    asmjit::x86::ymm0, asmjit::x86::ymm1, asmjit::x86::ymm2,
    asmjit::x86::ymm3, asmjit::x86::ymm4, asmjit::x86::ymm5,
    asmjit::x86::ymm6, asmjit::x86::ymm7, asmjit::x86::ymm8,
    asmjit::x86::ymm9, asmjit::x86::ymm10, asmjit::x86::ymm11,
    asmjit::x86::ymm12, asmjit::x86::ymm13, asmjit::x86::ymm14,
    asmjit::x86::ymm15, asmjit::x86::ymm16, asmjit::x86::ymm17,
    asmjit::x86::ymm18, asmjit::x86::ymm19, asmjit::x86::ymm20,
    asmjit::x86::ymm21, asmjit::x86::ymm22, asmjit::x86::ymm23,
    asmjit::x86::ymm24, asmjit::x86::ymm25, asmjit::x86::ymm26,
    asmjit::x86::ymm27, asmjit::x86::ymm28, asmjit::x86::ymm29,
    asmjit::x86::ymm30, asmjit::x86::ymm31
};

asmjit::x86::Xmm xmms[32] = {
    asmjit::x86::xmm0, asmjit::x86::xmm1, asmjit::x86::xmm2,
    asmjit::x86::xmm3, asmjit::x86::xmm4, asmjit::x86::xmm5,
    asmjit::x86::xmm6, asmjit::x86::xmm7, asmjit::x86::xmm8,
    asmjit::x86::xmm9, asmjit::x86::xmm10, asmjit::x86::xmm11,
    asmjit::x86::xmm12, asmjit::x86::xmm13, asmjit::x86::xmm14,
    asmjit::x86::xmm15, asmjit::x86::xmm16, asmjit::x86::xmm17,
    asmjit::x86::xmm18, asmjit::x86::xmm19, asmjit::x86::xmm20,
    asmjit::x86::xmm21, asmjit::x86::xmm22, asmjit::x86::xmm23,
    asmjit::x86::xmm24, asmjit::x86::xmm25, asmjit::x86::xmm26,
    asmjit::x86::xmm27, asmjit::x86::xmm28, asmjit::x86::xmm29,
    asmjit::x86::xmm30, asmjit::x86::xmm31
};

template <typename ptype, typename dtype, typename vtype>
void spmm_csr_row_split_dynamic_aot(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const ptype *indptr, const dtype *indices,
                        const vtype *values, const vtype *feats,
                        vtype *results) {
#pragma omp parallel for schedule(dynamic, 128)
  for (int64_t row = 0; row < nrow; ++row) {
    for (int64_t f = 0; f < nfeat; ++f) {
      vtype result = 0.0;
      int64_t spos = indptr[row], epos = indptr[row + 1];
      for (int64_t idx = spos; idx < epos; ++idx) {
        int64_t col = indices[idx];
        result += values[idx] * feats[col * nfeat + f];
      }
      results[row * nfeat + f] = result;
    }
  }
}

template <typename ptype, typename dtype, typename vtype>
void spmm_csr_row_split_dynamic_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const ptype *indptr, const dtype *indices,
                        const vtype *values, const vtype *feats,
                        vtype *results) {};

void emit_nnzloop(asmjit::x86::Assembler & as, int64_t nfeat) {
  using namespace asmjit;
  Label nnzloop_start = as.newLabel();
  Label nnzloop_end = as.newLabel();

  size_t cur = 0;
  for (; cur < nfeat / 16; ++cur)
    as.vxorps(zmms[cur], zmms[cur], zmms[cur]);
  if ((nfeat % 16) >= 8) {
    as.vxorps(ymms[cur], ymms[cur], ymms[cur]);
    cur++;
  }
  if ((nfeat % 8) >= 4) {
    as.vxorps(xmms[cur], xmms[cur], xmms[cur]);
    cur++;
  }
  for (int i = 0; i < (nfeat % 4); ++i,++cur)
    as.pxor(xmms[cur], xmms[cur]);

  as.cmp(x86::r10, x86::r11);
  as.je(nnzloop_end);
  as.align(AlignMode::kCode, 16);
  as.bind(nnzloop_start);

  as.mov(x86::r12d, x86::dword_ptr(x86::r13, x86::r10, 2)); // r12 = indices[r10]
  as.imul(x86::rax, x86::r12, int64_t(nfeat)); 
  as.lea(x86::rax, x86::ptr(x86::r8, x86::rax, 2U));
  as.vbroadcastss(zmms[31], x86::dword_ptr(x86::rcx, x86::r10, 2U));
  for (cur = 0; cur < nfeat / 16; ++cur) 
    as.vfmadd231ps(zmms[cur], zmms[31], x86::zmmword_ptr(x86::rax, int32_t(64*cur)));
  if ((nfeat % 16) >= 8) 
    as.vfmadd231ps(ymms[cur++], ymms[31], x86::ymmword_ptr(x86::rax, int32_t(64*(nfeat / 16))));
  if ((nfeat % 8) >= 4) 
    as.vfmadd231ps(xmms[cur++], xmms[31], x86::xmmword_ptr(x86::rax, int32_t(32*(nfeat / 8))));
  for (int i = 0; i < (nfeat % 4); ++i, ++cur) 
    as.vfmadd231ss(xmms[cur], xmms[31], x86::dword_ptr(x86::rax, int32_t(16*(nfeat/4) + i * 4)));

  as.inc(x86::r10);
  as.cmp(x86::r10, x86::r11);
  as.jl(nnzloop_start);
  as.bind(nnzloop_end);
  as.imul(x86::rax, x86::rdi, int64_t(nfeat));
  as.lea(x86::rax, x86::ptr(x86::r9, x86::rax, 2));
  for (cur = 0; cur < nfeat / 16; ++cur) 
    as.vmovups(x86::zmmword_ptr(x86::rax, int32_t(64*cur)), zmms[cur]);
  if ((nfeat % 16) >= 8)
    as.vmovups(x86::ymmword_ptr(x86::rax, int32_t(64*(nfeat / 16))), ymms[cur++]);
  if ((nfeat % 8) >= 4)
    as.vmovups(x86::xmmword_ptr(x86::rax, int32_t(32*(nfeat / 8))), xmms[cur++]);
  for (int i = 0; i < (nfeat % 4); ++i, ++cur)
    as.movss(x86::dword_ptr(x86::rax, int32_t(16*(nfeat / 4) + i * 4)), xmms[cur]);
}

template<>
void spmm_csr_row_split_dynamic_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const uint32_t *indptr, const uint32_t *indices,
                        const float *values, const float *feats,
                        float *results) {
  double t = omp_get_wtime(); 
  using namespace asmjit;
  typedef void (*Func)();
  uint64_t * NEXT = new uint64_t(0L);
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Assembler as(&code);
  
  as.push(x86::rbx);
  as.push(x86::r12);
  as.push(x86::r13);
  as.push(x86::r14);

  as.mov(x86::r14, uint64_t(indptr));
  as.mov(x86::r13, uint64_t(indices));

  as.mov(x86::rcx, uint64_t(values));
  as.mov(x86::rbx, uint64_t(nrow));

  as.mov(x86::r8, uint64_t(feats));
  as.mov(x86::r9, uint64_t(results));

  // requst workload batch
  Label rloop_start = as.newLabel();
  Label rloop_end = as.newLabel();

  as.bind(rloop_start);
  as.mov(x86::rdi, step);
  as.lock(); as.xadd(x86::qword_ptr(uint64_t(NEXT)), x86::rdi);

  as.cmp(x86::rdi, x86::rbx);
  as.jge(rloop_end);
  as.mov(x86::rsi, x86::rdi);
  as.add(x86::rsi, step);
  as.cmp(x86::rsi, x86::rbx);
  as.cmova(x86::rsi, x86::rbx);
  
  // working on a batch of 128 rows
  Label bloop_start = as.newLabel();
  Label bloop_end = as.newLabel();
  as.cmp(x86::rdi, x86::rsi);
  as.jge(bloop_end);
  as.bind(bloop_start);

  as.mov(x86::r10d, x86::dword_ptr(x86::r14, x86::rdi, 2));
  as.mov(x86::r11d, x86::dword_ptr(x86::r14, x86::rdi, 2, 4));

  emit_nnzloop(as, nfeat);

  as.inc(x86::rdi);
  as.cmp(x86::rdi, x86::rsi);
  as.jl(bloop_start);
  as.bind(bloop_end);

  as.jmp(rloop_start);
  as.bind(rloop_end);

  as.pop(x86::r14);
  as.pop(x86::r13);
  as.pop(x86::r12);
  as.pop(x86::rbx);
  as.ret();

  Func fn;
  rt.add(&fn, &code);

  #pragma omp parallel
  {
    fn();
  }

  rt.release(fn);
  delete NEXT;
}

template<>
void spmm_csr_row_split_dynamic_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const uint64_t *indptr, const uint32_t *indices,
                        const float *values, const float *feats,
                        float *results) { 
  double t = omp_get_wtime();
  using namespace asmjit;
  typedef void (*Func)();
  uint64_t * NEXT = new uint64_t(0L);
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Assembler as(&code);
  
  as.push(x86::rbx);
  as.push(x86::r12);
  as.push(x86::r13);
  as.push(x86::r14);

  as.mov(x86::r14, uint64_t(indptr));
  as.mov(x86::r13, uint64_t(indices));
  as.mov(x86::rcx, uint64_t(values));
  as.mov(x86::rbx, uint64_t(nrow));
  as.mov(x86::r8, uint64_t(feats));
  as.mov(x86::r9, uint64_t(results));

  // requst workload batch
  Label rloop_start = as.newLabel();
  Label rloop_end = as.newLabel();

  as.bind(rloop_start);
  as.mov(x86::rdi, step);
  as.lock(); as.xadd(x86::qword_ptr(uint64_t(NEXT)), x86::rdi);

  as.cmp(x86::rdi, x86::rbx);
  as.jge(rloop_end);
  as.mov(x86::rsi, x86::rdi);
  as.add(x86::rsi, step);
  as.cmp(x86::rsi, x86::rbx);
  as.cmova(x86::rsi, x86::rbx);
  
  // working on a batch of 128 rows
  Label bloop_start = as.newLabel();
  Label bloop_end = as.newLabel();
  as.bind(bloop_start);
  as.cmp(x86::rdi, x86::rsi);
  as.jge(bloop_end);

  as.mov(x86::r10, x86::qword_ptr(x86::r14, x86::rdi, 3));
  as.mov(x86::r11, x86::qword_ptr(x86::r14, x86::rdi, 3, 8));

  emit_nnzloop(as, nfeat);

  as.inc(x86::rdi);
  as.jmp(bloop_start);
  as.bind(bloop_end);

  as.jmp(rloop_start);
  as.bind(rloop_end);

  as.pop(x86::r14);
  as.pop(x86::r13);
  as.pop(x86::r12);
  as.pop(x86::rbx);
  as.ret();

  Func fn;
  rt.add(&fn, &code);

  #pragma omp parallel 
  {
    fn();
  }

  rt.release(fn);
  delete NEXT;
}