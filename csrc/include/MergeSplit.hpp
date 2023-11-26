#include <stdlib.h>
#include <utility>
#include <asmjit/x86.h>

using namespace std;

template <typename ptype>
pair<int64_t, int64_t> merge_path_search(int64_t diagonal, int64_t nrow,
                                         int64_t nnz, const ptype *indptr) {

  int64_t x_min = max(diagonal - nnz, 0L);
  int64_t x_max = min(diagonal, nrow);

  while (x_min < x_max) {
    int64_t mid = (x_min + x_max) >> 1;
    if (indptr[mid] <= diagonal - mid - 1)
      x_min = mid + 1;
    else
      x_max = mid;
  }
  return make_pair(min(x_min, nrow), diagonal - x_min);
}

template <typename ptype, typename dtype, typename vtype>
void spmm_csr_merge_split_aot(int64_t nrow, int64_t ncol, int64_t nnz,
                          int64_t nfeat, const ptype *indptr,
                          const dtype *indices, const vtype *values,
                          const vtype *feats, vtype *results) {

  int num_threads = omp_get_max_threads();
  int64_t num_merge_items = nrow + nnz;
  int64_t seglen = (num_merge_items + num_threads - 1) / num_threads;
  vector<int64_t> row_carry_out(num_threads, 0L);
  vector<vector<vtype>> value_carry_out(num_threads, vector<vtype>(nfeat, 0.0));
#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int64_t diagonal = min(seglen * thread_idx, num_merge_items);
    int64_t diagonal_end = min(diagonal + seglen, num_merge_items);

    auto thread_coord =
        merge_path_search<ptype>(diagonal, nrow, nnz, indptr + 1);
    auto thread_coord_end =
        merge_path_search<ptype>(diagonal_end, nrow, nnz, indptr + 1);

    auto tmp = thread_coord;
    for (; tmp.first < thread_coord_end.first; ++tmp.first) {
      for (int64_t f = 0; f < nfeat; ++f) {
        vtype running_total = 0.0;
        for (int64_t pos = tmp.second; pos < indptr[tmp.first + 1]; ++pos)
          running_total +=
              values[pos] * feats[indices[pos] * nfeat + f];
        results[tmp.first * nfeat + f] = running_total;
      }
      tmp.second = indptr[tmp.first + 1];
    }
    for (int64_t f = 0; f < nfeat; ++f) {
      vtype running_total = 0.0; 
      for (int64_t pos = tmp.second; pos < thread_coord_end.second; ++pos)
        running_total +=
            values[pos] * feats[indices[pos] * nfeat + f];
      value_carry_out[thread_idx][f] = running_total;
    }
    row_carry_out[thread_idx] = thread_coord_end.first;
  }

  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out[tid] < nrow) {
      for (int64_t f = 0; f < nfeat; ++f)
        results[row_carry_out[tid] * nfeat + f] += value_carry_out[tid][f];
    }
  }
}

template <typename ptype, typename dtype, typename vtype>
void spmm_csr_merge_split_jit(int64_t nrow, int64_t ncol, int64_t nnz,
                          int64_t nfeat, const ptype *indptr,
                          const dtype *indices, const vtype *values,
                          const vtype *feats, vtype *results) {};

template<>
void spmm_csr_merge_split_jit(int64_t nrow, int64_t ncol, int64_t nnz,
                          int64_t nfeat, const uint32_t *indptr,
                          const uint32_t *indices, const float *values,
                          const float *feats, float *results) {

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

  as.mov(x86::r10d, x86::dword_ptr(x86::r14, x86::rdi, 2));
  as.mov(x86::r11d, x86::dword_ptr(x86::r14, x86::rdi, 2, 4));

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
  int64_t num_merge_items = nrow + nnz;
  int64_t seglen = (num_merge_items + num_threads - 1) / num_threads;
  vector<int64_t> row_carry_out(num_threads, 0L);
  vector<vector<float>> value_carry_out(num_threads, vector<float>(nfeat, 0.0));
#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int64_t diagonal = min(seglen * thread_idx, num_merge_items);
    int64_t diagonal_end = min(diagonal + seglen, num_merge_items);

    auto thread_coord =
        merge_path_search<uint32_t>(diagonal, nrow, nnz, indptr + 1);
    auto thread_coord_end =
        merge_path_search<uint32_t>(diagonal_end, nrow, nnz, indptr + 1);

    if (thread_coord.first < thread_coord_end.first) {
      for (int64_t f = 0; f < nfeat; ++f) {
        float running_total = 0.0;
        for (int64_t pos = thread_coord.second; pos < indptr[thread_coord.first+1]; ++pos) 
          running_total += values[pos] * feats[indices[pos] * nfeat + f];
        results[thread_coord.first * nfeat + f] = running_total;
      }
    }

    fn(thread_coord.first+1, thread_coord_end.first);

    for (int64_t f = 0; f < nfeat; ++f) {
      float running_total = 0.0; 
      for (int64_t pos = indptr[thread_coord_end.first]; pos < thread_coord_end.second; ++pos)
        running_total +=
            values[pos] * feats[indices[pos] * nfeat + f];
      value_carry_out[thread_idx][f] = running_total;
    }
    row_carry_out[thread_idx] = thread_coord_end.first;
  }

  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out[tid] < nrow) {
      for (int64_t f = 0; f < nfeat; ++f)
        results[row_carry_out[tid] * nfeat + f] += value_carry_out[tid][f];
    }
  }

  rt.release(fn);
}

template<>
void spmm_csr_merge_split_jit(int64_t nrow, int64_t ncol, int64_t nnz,
                          int64_t nfeat, const uint64_t *indptr,
                          const uint32_t *indices, const float *values,
                          const float *feats, float *results) {

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
  int64_t num_merge_items = nrow + nnz;
  int64_t seglen = (num_merge_items + num_threads - 1) / num_threads;
  vector<int64_t> row_carry_out(num_threads, 0L);
  vector<vector<float>> value_carry_out(num_threads, vector<float>(nfeat, 0.0));
#pragma omp parallel
  {
    int thread_idx = omp_get_thread_num();
    int64_t diagonal = min(seglen * thread_idx, num_merge_items);
    int64_t diagonal_end = min(diagonal + seglen, num_merge_items);

    auto thread_coord =
        merge_path_search<uint64_t>(diagonal, nrow, nnz, indptr + 1);
    auto thread_coord_end =
        merge_path_search<uint64_t>(diagonal_end, nrow, nnz, indptr + 1);

    if (thread_coord.first < thread_coord_end.first) {
      for (int64_t f = 0; f < nfeat; ++f) {
        float running_total = 0.0;
        for (int64_t pos = thread_coord.second; pos < indptr[thread_coord.first+1]; ++pos) 
          running_total += values[pos] * feats[indices[pos] * nfeat + f];
        results[thread_coord.first * nfeat + f] = running_total;
      }
    }

    fn(thread_coord.first+1, thread_coord_end.first);

    for (int64_t f = 0; f < nfeat; ++f) {
      float running_total = 0.0; 
      for (int64_t pos = indptr[thread_coord_end.first]; pos < thread_coord_end.second; ++pos)
        running_total +=
            values[pos] * feats[indices[pos] * nfeat + f];
      value_carry_out[thread_idx][f] = running_total;
    }
    row_carry_out[thread_idx] = thread_coord_end.first;
  }

  for (int tid = 0; tid < num_threads - 1; ++tid) {
    if (row_carry_out[tid] < nrow) {
      for (int64_t f = 0; f < nfeat; ++f)
        results[row_carry_out[tid] * nfeat + f] += value_carry_out[tid][f];
    }
  }

  rt.release(fn);
}