#include <stdlib.h>
#include <argparse/argparse.hpp>
#include <asmjit/x86.h>


template <typename ptype, typename dtype, typename vtype>
void spmm_csr_scalar_aot(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const ptype *indptr, const dtype *indices,
                        const vtype *values, const vtype *feats,
                        vtype *results) {
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
void spmm_csr_scalar_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const ptype *indptr, const dtype *indices,
                        const vtype *values, const vtype *feats,
                        vtype *results);


template<>
void spmm_csr_scalar_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const uint32_t *indptr, const uint32_t *indices,
                        const float *values, const float *feats,
                        float *results) { 
  using namespace asmjit;
  typedef void (*Func)();
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Assembler as(&code);
  
  as.push(x86::r12);
  as.push(x86::r13);
  as.push(x86::r14);

  as.mov(x86::rsi, uint64_t(nrow));
  as.mov(x86::r14, uint64_t(indptr));
  as.mov(x86::r13, uint64_t(indices));
  as.mov(x86::rcx, uint64_t(values));
  as.mov(x86::r8, uint64_t(feats));
  as.mov(x86::r9, uint64_t(results));

  as.xor_(x86::rdi, x86::rdi);

  Label sta = as.newLabel();
  Label end = as.newLabel();
  as.bind(sta);
  as.cmp(x86::rdi, x86::rsi);
  as.jge(end);

  as.mov(x86::r10d, x86::dword_ptr(x86::r14, x86::rdi, 2));
  as.inc(x86::rdi);
  as.mov(x86::r11d, x86::dword_ptr(x86::r14, x86::rdi, 2));
  as.dec(x86::rdi);
  Label loop_start = as.newLabel();
  Label loop_end = as.newLabel();

  for (size_t i = 0; i < nfeat; ++i)
    as.xorps(xmms[i], xmms[i]);

  as.bind(loop_start);
  as.cmp(x86::r10, x86::r11);
  as.je(loop_end);

  as.mov(x86::r12d, x86::dword_ptr(x86::r13, x86::r10, 2));
  as.mov(x86::rax, uint64_t(nfeat));
  as.mul(x86::r12);
  as.vmovss(xmms[31], x86::dword_ptr(x86::rcx, x86::r10, 2));
  for (size_t i = 0; i < nfeat; ++i) {
    as.vfmadd231ss(xmms[i], xmms[31], x86::dword_ptr(x86::r8, x86::rax, 2, int32_t(4*i)));
  }
  as.inc(x86::r10);
  as.jmp(loop_start);
  as.bind(loop_end);
  as.mov(x86::rax, uint64_t(nfeat));
  as.mul(x86::rdi);
  for (size_t i = 0; i < nfeat; ++i) {
    as.vmovss(x86::dword_ptr(x86::r9, x86::rax, 2, int32_t(4*i)), xmms[i]);
  }
  as.inc(x86::rdi);
  as.jmp(sta);
  as.bind(end);

  as.pop(x86::r14);
  as.pop(x86::r13);
  as.pop(x86::r12);
  as.ret();
  Func fn;
  rt.add(&fn, &code);

  fn();

  rt.release(fn);
}

template<>
void spmm_csr_scalar_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const uint64_t *indptr, const uint32_t *indices,
                        const float *values, const float *feats,
                        float *results) { 
  using namespace asmjit;
  typedef void (*Func)();
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Assembler as(&code);
  
  as.push(x86::r12);
  as.push(x86::r13);
  as.push(x86::r14);
  
  as.mov(x86::rsi, uint64_t(nrow));
  as.mov(x86::r14, uint64_t(indptr));
  as.mov(x86::r13, uint64_t(indices));
  as.mov(x86::rcx, uint64_t(values));
  as.mov(x86::r8, uint64_t(feats));
  as.mov(x86::r9, uint64_t(results));

  as.xor_(x86::rdi, x86::rdi);

  Label sta = as.newLabel();
  Label end = as.newLabel();
  as.bind(sta);
  as.cmp(x86::rdi, x86::rsi);
  as.jge(end);

  as.mov(x86::r10, x86::qword_ptr(x86::r14, x86::rdi, 3));
  as.inc(x86::rdi);
  as.mov(x86::r11, x86::qword_ptr(x86::r14, x86::rdi, 3));
  as.dec(x86::rdi);
  Label loop_start = as.newLabel();
  Label loop_end = as.newLabel();

  for (size_t i = 0; i < nfeat; ++i)
    as.xorps(xmms[i], xmms[i]);

  as.bind(loop_start);
  as.cmp(x86::r10, x86::r11);
  as.je(loop_end);

  as.mov(x86::r12d, x86::dword_ptr(x86::r13, x86::r10, 2));
  as.mov(x86::rax, uint64_t(nfeat));
  as.mul(x86::r12);
  as.vmovss(xmms[31], x86::dword_ptr(x86::rcx, x86::r10, 2));
  for (size_t i = 0; i < nfeat; ++i) {
    as.vfmadd231ss(xmms[i], xmms[31], x86::dword_ptr(x86::r8, x86::rax, 2, int32_t(4*i)));
  }
  as.inc(x86::r10);
  as.jmp(loop_start);
  as.bind(loop_end);
  as.mov(x86::rax, uint64_t(nfeat));
  as.mul(x86::rdi);
  for (size_t i = 0; i < nfeat; ++i) {
    as.vmovss(x86::dword_ptr(x86::r9, x86::rax, 2, int32_t(4*i)), xmms[i]);
  }

  as.inc(x86::rdi);
  as.jmp(sta);
  as.bind(end);

  as.pop(x86::r14);
  as.pop(x86::r13);
  as.pop(x86::r12);
  as.ret();
  Func fn;
  rt.add(&fn, &code);

  fn();

  rt.release(fn);
}