#include <stdlib.h>
#include <omp.h>
#include <asmjit/x86.h>

template <typename ptype, typename dtype, typename vtype>
void spmm_csr_nnz_split_aot(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const ptype *indptr, const dtype *indices,
                        const vtype *values, const vtype *feats,
                        vtype *results) {

  int num_threads = omp_get_max_threads();
  int64_t seglen = (nnz + num_threads - 1) / num_threads;
  vector<int64_t> row_carry_out(num_threads, 0);
  vector<vector<vtype>> value_carry_out(num_threads, vector<vtype>(nfeat, 0.0));
#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int64_t spos = seglen * thread_idx;
    int64_t epos = min(spos + seglen, nnz);

    int strow, endrow;

    int64_t l = 0, r = nrow;
    while (l < r) {
      int64_t mid = l + ((r - l) >> 1) + 1;
      if (indptr[mid] <= spos)
        l = mid;
      else
        r = mid - 1;
    }
    strow = l;

    l = 0;
    r = nrow;
    while (l < r) {
      int64_t mid = l + ((r - l) >> 1) + 1;
      if (indptr[mid] <= epos)
        l = mid;
      else
        r = mid - 1;
    }
    endrow = l;

    for (int64_t row = strow, idx=spos; row < endrow; ++row) {
      for (int64_t f = 0; f < nfeat; ++f) {
        vtype running_total = 0.0;
        for (int64_t pos = idx; pos < indptr[row+1]; ++pos)
          running_total += values[pos] * feats[indices[pos] * nfeat + f];
        results[row*nfeat+f] = running_total;
      }
      idx = indptr[row+1];
    }

    for (int64_t f = 0; f < nfeat; ++f) {
      vtype running_total = 0.0;
      for (int64_t pos = indptr[endrow]; pos < epos; ++pos)
        running_total += values[pos] * feats[indices[pos] * nfeat + f];
      value_carry_out[thread_idx][f] = running_total;
    }
    row_carry_out[thread_idx] = endrow;

  }
  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out[tid] < nrow) {
      for (int64_t f = 0; f < nfeat; ++f)
        results[row_carry_out[tid] * nfeat + f] += value_carry_out[tid][f];
    }
  }
}

template <typename ptype, typename dtype, typename vtype>
void spmm_csr_nnz_split_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const ptype *indptr, const dtype *indices,
                        const vtype *values, const vtype *feats,
                        vtype *results) {};

template <>
void spmm_csr_nnz_split_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const uint32_t *indptr, const uint32_t *indices,
                        const float *values, const float *feats,
                        float *results) {
  using namespace asmjit;
  typedef void (*Func)(int64_t, int64_t);
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Assembler as(&code);
  
  as.push(x86::r12);
  as.push(x86::r13);
  as.push(x86::r14);

  as.mov(x86::r14, uint64_t(indptr));
  as.mov(x86::r13, uint64_t(indices));
  as.mov(x86::rcx, uint64_t(values));
  as.mov(x86::r8, uint64_t(feats));
  as.mov(x86::r9, uint64_t(results));

  Label sta = as.newLabel();
  Label end = as.newLabel();
  as.bind(sta);
  as.cmp(x86::rdi, x86::rsi);
  as.jge(end);

  as.mov(x86::r10d, x86::dword_ptr(x86::r14, x86::rdi, 2));
  as.mov(x86::r11d, x86::dword_ptr(x86::r14, x86::rdi, 2, 4));

  emit_nnzloop(as, nfeat);

  as.inc(x86::rdi);
  as.jmp(sta);
  as.bind(end);

  as.pop(x86::r14);
  as.pop(x86::r13);
  as.pop(x86::r12);
  as.ret();
  Func fn;
  rt.add(&fn, &code);

  int num_threads = omp_get_max_threads();
  int64_t seglen = (nnz + num_threads - 1) / num_threads;
  vector<int64_t> row_carry_out(num_threads, 0);
  vector<vector<float>> value_carry_out(num_threads, vector<float>(nfeat, 0.0));
#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int64_t spos = min(nnz, seglen * thread_idx);
    int64_t epos = min(spos + seglen, nnz);

    int strow, endrow;
 
    int64_t l = 0, r = nrow;
    while (l < r) {
      int64_t mid = l + ((r - l) >> 1) + 1;
      if (indptr[mid] <= spos)
        l = mid;
      else
        r = mid - 1;
    }
    strow = l;

    l = 0;
    r = nrow;
    while (l < r) {
      int64_t mid = l + ((r - l) >> 1) + 1;
      if (indptr[mid] <= epos)
        l = mid;
      else
        r = mid - 1;
    }
    endrow = l;

    if (strow < endrow) {
      for (int64_t f= 0; f < nfeat; ++f) {
        float running_total = 0.0;
        for (int64_t pos = spos; pos < indptr[strow+1]; ++pos)
          running_total += values[pos] * feats[indices[pos]*nfeat + f];
        results[strow*nfeat+f] = running_total;
      }
    }

    fn(strow+1, endrow);

    for (int64_t f = 0; f < nfeat; ++f) {
      float running_total = 0.0;
      for (int64_t pos = indptr[endrow]; pos < epos; ++pos)
        running_total += values[pos] * feats[indices[pos] * nfeat + f];
      value_carry_out[thread_idx][f] = running_total;
    }
    row_carry_out[thread_idx] = endrow;
  }
  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out[tid] < nrow) {
      for (int64_t f = 0; f < nfeat; ++f)
        results[row_carry_out[tid] * nfeat + f] += value_carry_out[tid][f];
    }
  }

  rt.release(fn);
}

template <>
void spmm_csr_nnz_split_jit(int64_t nrow, int64_t ncol, int64_t nnz, int64_t nfeat,
                        const uint64_t *indptr, const uint32_t *indices,
                        const float *values, const float *feats,
                        float *results) {
  using namespace asmjit;
  typedef void (*Func)(int64_t, int64_t);
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Assembler as(&code);
  
  as.push(x86::r12);
  as.push(x86::r13);
  as.push(x86::r14);

  as.mov(x86::r14, uint64_t(indptr));
  as.mov(x86::r13, uint64_t(indices));
  as.mov(x86::rcx, uint64_t(values));
  as.mov(x86::r8, uint64_t(feats));
  as.mov(x86::r9, uint64_t(results));

  Label sta = as.newLabel();
  Label end = as.newLabel();
  as.cmp(x86::rdi, x86::rsi);
  as.jge(end);
  as.bind(sta);

  as.mov(x86::r10, x86::qword_ptr(x86::r14, x86::rdi, 3));
  as.mov(x86::r11, x86::qword_ptr(x86::r14, x86::rdi, 3, 8));

  emit_nnzloop(as, nfeat);

  as.inc(x86::rdi);
  as.cmp(x86::rdi, x86::rsi);
  as.jl(sta);
  as.bind(end);

  as.pop(x86::r14);
  as.pop(x86::r13);
  as.pop(x86::r12);
  as.ret();

  Func fn;
  rt.add(&fn, &code);

  int num_threads = omp_get_max_threads();
  int64_t seglen = (nnz + num_threads - 1) / num_threads;
  vector<int64_t> row_carry_out(num_threads, 0);
  vector<vector<float>> value_carry_out(num_threads, vector<float>(nfeat, 0.0));
#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int64_t spos = seglen * thread_idx;
    int64_t epos = min(spos + seglen, nnz);

    int strow, endrow;

    int64_t l = 0, r = nrow;
    while (l < r) {
      int64_t mid = l + ((r - l) >> 1) + 1;
      if (indptr[mid] <= spos)
        l = mid;
      else
        r = mid - 1;
    }
    strow = l;

    l = 0;
    r = nrow;
    while (l < r) {
      int64_t mid = l + ((r - l) >> 1) + 1;
      if (indptr[mid] <= epos)
        l = mid;
      else
        r = mid - 1;
    }
    endrow = l;

    if (strow < endrow) {
      for (int64_t f= 0; f < nfeat; ++f) {
        float running_total = 0.0;
        for (int64_t pos = spos; pos < indptr[strow+1]; ++pos)
          running_total += values[pos] * feats[indices[pos]*nfeat + f];
        results[strow*nfeat+f] = running_total;
      }
    }

    fn(strow+1, endrow);

    for (int64_t f = 0; f < nfeat; ++f) {
      float running_total = 0.0;
      for (int64_t pos = indptr[endrow]; pos < epos; ++pos)
        running_total += values[pos] * feats[indices[pos] * nfeat + f];
      value_carry_out[thread_idx][f] = running_total;
    }
    row_carry_out[thread_idx] = endrow;

  }
  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out[tid] < nrow) {
      for (int64_t f = 0; f < nfeat; ++f)
        results[row_carry_out[tid] * nfeat + f] += value_carry_out[tid][f];
    }
  }

  rt.release(fn);
}