#include <argparse/argparse.hpp>
#include <chrono>
#include <string.h>

#include "SpMat.h"
#include "RowSplit.hpp"
#include "NnzSplit.hpp"
#include "MergeSplit.hpp"
#include "Scalar.hpp"

int main(int argc, char *argv[]) {

  // add arguments
  argparse::ArgumentParser program("spmm");
  program.add_argument("-i", "--input")
      .required()
      .help("input matrix market file");
  program.add_argument("-f", "--feat-len")
      .default_value(16)
      .required()
      .scan<'d', int>();
  program.add_argument("-r", "--run")
      .default_value(10)
      .required()
      .scan<'d', int>();
  program.add_argument("-p", "--prof").default_value(false).implicit_value(true);
  program.add_argument("--impl")
      .default_value(string("row"))
      .action([](const string &value) {
        static const vector<string> choices = {"row", "nnz", "merge", "scalar"};
        if (std::find(choices.begin(), choices.end(), value) != choices.end())
          return value;
        return string("row");
      })
      .required();
  program.add_argument("-m", "--method")
      .default_value(string("jit"))
      .action([](const string &value) {
        static const vector<string> choices = {"aot", "jit"};
        if (std::find(choices.begin(), choices.end(), value) != choices.end())
          return value;
        return string("jit");
      })
      .required();

  // parse the arguments
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // get input and output file names
  auto dataset = program.get<string>("-i");
  auto ifn = dataset;
  int64_t nfeat = program.get<int>("-f");
  auto impl = program.get<string>("--impl");
  auto method  = program.get<string>("-m");
  int run = program.get<int>("-r");
  bool prof = program.get<bool>("--prof");

  cout << "Feature size: " << nfeat << endl;
  cout << "Using: " << impl << endl;
  cout << "Run: " << run << endl;
  cout << "Method: " << method << endl;

  SpMatCsr csr(ifn);
  float *feats = (float *)malloc(4 * nfeat * csr.ncol);

  for (int64_t i = 0; i < nfeat * csr.ncol; ++i) {
    feats[i] = rand() / (float)RAND_MAX;
  }

  float *results = (float *)malloc(4 * nfeat * csr.nrow);
  if (!prof) {
    DTYPE_SWITCH(csr.ptype, ptype, {
      DTYPE_SWITCH(csr.dtype, dtype, {
        double time = 0.0;
        if (impl == "row") {
          if (method == "aot") {
            for (int i = 0; i < run; ++i) {
              Timer t;
              t.tick();
              spmm_csr_row_split_dynamic_aot<ptype, dtype, float>(
                  csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                  (dtype *)csr.indices, (float *)csr.values, feats, results);
              t.tock();
              if (i >= 2)
                time += t.duration();
            }
          } else {
            for (int i = 0; i < run; ++i) {
              Timer t;
              t.tick();
              spmm_csr_row_split_dynamic_jit<ptype, dtype, float>(
                  csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                  (dtype *)csr.indices, (float *)csr.values, feats, results);
              t.tock();
              if (i >= 2)
                time += t.duration();
            }
          }
        } else if (impl == "nnz") {
          if (method == "aot") {
            for (int i = 0; i < run; ++i) {
              Timer t;
              t.tick();
              spmm_csr_nnz_split_aot<ptype, dtype, float>(
                  csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                  (dtype *)csr.indices, (float *)csr.values, feats, results);
              t.tock();
              if (i >= 2)
                time += t.duration();
            }
          } else {
            for (int i = 0; i < run; ++i) {
              Timer t;
              t.tick();
              spmm_csr_nnz_split_jit<ptype, dtype, float>(
                  csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                  (dtype *)csr.indices, (float *)csr.values, feats, results);
              t.tock();
              if (i >= 2)
                time += t.duration();
            }
          }
        } else {
          if (method == "aot") {
            for (int i = 0; i < run; ++i) {
              Timer t;
              t.tick();
              spmm_csr_merge_split_aot<ptype, dtype, float>(
                  csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                  (dtype *)csr.indices, (float *)csr.values, feats, results);
              t.tock();
              if (i >= 2)
                time += t.duration();
            }
          } else {
            for (int i = 0; i < run; ++i) {
              Timer t;
              t.tick();
              spmm_csr_merge_split_jit<ptype, dtype, float>(
                  csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                  (dtype *)csr.indices, (float *)csr.values, feats, results);
              t.tock();
              if (i >= 2)
                time += t.duration();
            }
          }
        }
        time /= (run - 2);
        printf("Total Time: %.4f ms\n", time);
      });
    });
  } else {
    DTYPE_SWITCH(csr.ptype, ptype, {
      DTYPE_SWITCH(csr.dtype, dtype, {
        if (impl == "row") {
          if (method == "aot") {
            System::profile(dataset + "_row", [&]() {
              for (int i = 0; i < run; ++i) {
                spmm_csr_row_split_dynamic_aot<ptype, dtype, float>(
                    csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                    (dtype *)csr.indices, (float *)csr.values, feats, results);
              }
            });
          } else {
            System::profile(dataset + "_row", [&]() {
              for (int i = 0; i < run; ++i) {
                spmm_csr_row_split_dynamic_jit<ptype, dtype, float>(
                    csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                    (dtype *)csr.indices, (float *)csr.values, feats, results);
              }
            });
          }
        } else if (impl == "nnz") {
          if (method == "aot") {
            System::profile([&]() {
              for (int i = 0; i < run; ++i) {
                spmm_csr_nnz_split_aot<ptype, dtype, float>(
                    csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                    (dtype *)csr.indices, (float *)csr.values, feats, results);
              }
            });
          } else {
            System::profile([&]() {
              for (int i = 0; i < run; ++i) {
                spmm_csr_nnz_split_jit<ptype, dtype, float>(
                    csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                    (dtype *)csr.indices, (float *)csr.values, feats, results);
              }
            });
          }
        } else {
          if (method == "aot") {
            System::profile([&]() {
              for (int i = 0; i < run; ++i) {
                spmm_csr_merge_split_aot<ptype, dtype, float>(
                    csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                    (dtype *)csr.indices, (float *)csr.values, feats, results);
              }
            });
          } else {
            System::profile([&]() {
              for (int i = 0; i < run; ++i) {
                spmm_csr_merge_split_jit<ptype, dtype, float>(
                    csr.nrow, csr.ncol, csr.nnz, nfeat, (ptype *)csr.indptr,
                    (dtype *)csr.indices, (float *)csr.values, feats, results);
              }
            });
          }
        }
      });
    });
  }
  free((void *)results);

  free((void *)feats); 
  return 0;
}
