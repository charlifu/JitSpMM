# JitSpMM (CGO'24)

---
Software requirement
-----
CMake (3.13+)

gcc/clang/icc (c++17 support)

---
Hardware
------
x86 CPU with avx512 extension.

Use command `lscpu` to check if your cpu supports avx512 or not.

----
Build
------
```bash
git submodule update --init --recursive
mkdir -p third_party/asmjit/build/ && cd third_party/asmjit/build/
cmake -G Ninja ..
cmake --build .
cmake --install . --prefix .
cd -
mkdir -p ./build && cd build
cmake -G Ninja ..
cmake --build .
```

----
Execute
------
Run JitSpMM on the toy example:

```bash
./spmm -i ../data/1138_bus/1138_bus.csrbin
```

To run JitSpMM on other dataset, please download the mtx file from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/). And convert it to our own binary format using `mm_to_csrbin`:
```bash
./mm_to_csrbin -i [input .mtx file] -o [output .csrbin file]
```

Run `./spmm -h` to see more options.

----
Reference
-------

If you use JitSpMM in your project, please cite the following paper.

```python
@article{fu2023jitspmm,
  title={JITSPMM: Just-in-Time Instruction Generation for Accelerated Sparse Matrix-Matrix Multiplication},
  author={Fu, Qiang and Rolinger, Thomas B and Huang, H Howie},
  journal={arXiv preprint arXiv:2312.05639},
  year={2023}
}
```
