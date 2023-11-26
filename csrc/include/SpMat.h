#ifndef SPMAT_H_
#define SPMAT_H_

#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <functional>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <omp.h>

using namespace std;

enum Type { uint32 = 0U, uint64 = 1U, float32 = 2U, float64 = 3U };

struct SpMatCsr {
  Type ptype, dtype, vtype;
  int64_t nrow, ncol, nnz;
  void *indptr, *indices, *values;

  SpMatCsr(const string &fn) {
    ifstream inf(fn, ios::in | ios::binary);
    inf.read((char *)&ptype, 4);
    inf.read((char *)&dtype, 4);
    inf.read((char *)&vtype, 4);

    cout << "Sparse Matrix Array Types: " << ptype << " " << dtype << " "
         << vtype << endl;

    inf.read((char *)&nrow, 8);
    inf.read((char *)&ncol, 8);
    inf.read((char *)&nnz, 8);

    cout << "Sparse Matrix Size: " << nrow << " " << ncol << " " << nnz << endl;

    if (ptype == Type::uint32) {
      indptr = malloc(4 * (nrow + 1));
      inf.read((char *)indptr, 4 * (nrow + 1));
    } else {
      indptr = malloc(8 * (nrow + 1));
      inf.read((char *)indptr, 8 * (nrow + 1));
    }

    if (dtype == Type::uint32) {
      indices = malloc(4 * nnz);
      inf.read((char *)indices, 4 * nnz);
    } else {
      indices = malloc(8 * nnz);
      inf.read((char *)indices, 8 * nnz);
    }

    if (vtype == Type::float32) {
      values = malloc(4 * nnz);
      inf.read((char *)values, 4 * nnz);
    } else {
      values = malloc(8 * nnz);
      inf.read((char *)values, 8 * nnz);
    }
  }

  ~SpMatCsr() {
    free(indptr);
    free(indices);
    free(values);
  }
};

#define DTYPE_SWITCH(val, dtype, ...) do {                    \
  if ((val) == Type::uint32) {                                \
    typedef uint32_t dtype;                                   \
    {__VA_ARGS__}                                             \
  } else if ((val) == Type::uint64) {                         \
    typedef uint64_t dtype;                                   \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    std::cout << "dtype can only be uint32 or uint64";        \
  }                                                           \
} while (0)

template <class DT = std::chrono::milliseconds,
          class ClockT = std::chrono::high_resolution_clock>
class Timer
{
    using timep_t = typename ClockT::time_point;
    timep_t _start = ClockT::now(), _end = {};

public:
    inline void tick() { 
        _end = timep_t{}; 
        _start = ClockT::now(); 
    }
    
    inline void tock() { _end = ClockT::now(); }
    
    template <class T = DT> 
    auto duration() const { 
        // assert(_end != timep_t{}); 
        return std::chrono::duration_cast<T>(_end - _start).count(); 
    }
};

struct System {
  static void profile(const std::string &name, std::function<void()> body) {
    std::string filename =
        name.find(".data") == std::string::npos ? (name + ".data") : name;

    // Launch profiler
    pid_t pid;
    std::stringstream s;
    s << getpid();
    pid = fork();
    if (pid == 0) {
      // auto fd=open("/dev/null",O_RDWR);
      // dup2(fd,1);
      // dup2(fd,2);
      // exit(execl("/usr/bin/perf","perf","record","-o",filename.c_str(),"-p",s.str().c_str(),nullptr));
      exit(execl("/usr/bin/perf", "perf", "stat", "-e", "instructions,branches,branch-misses,L1-dcache-loads", "-p", s.str().c_str(),
                 nullptr));
    }

    // Run body
    body();

    // Kill profiler
    kill(pid, SIGINT);
    waitpid(pid, nullptr, 0);
  }

  static void profile(std::function<void()> body) {
    profile("perf.data", body);
  }
};

#endif // SPMAT_H_