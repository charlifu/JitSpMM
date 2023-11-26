#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>

#include <argparse/argparse.hpp>

using namespace std;

enum Type {
  uint32 = 0U,
  uint64 = 1U,
  float32 = 2U,
  float64 = 3U
};

template<typename ptype,
         typename dtype,
         typename vtype>
void save_to_csrbin(int64_t nrow,
                    int64_t ncol,
                    int64_t nnz,
                    const vector<Type> & types,
                    ifstream & fin, 
                    const vector<string> & formats,
                    const string & ofn) {
  vector<map<dtype, vtype>> G(nrow);
  cout << typeid(ptype).name() << " " << typeid(dtype).name() << " " << typeid(vtype).name() << endl;
  
  string line;
  while (!fin.eof()) {
    getline(fin, line);
    uint64_t u, v;
    double val = 1.0;

    istringstream tmpstm(line);
    tmpstm >> u >> v;
    if (formats[3][0] != 'p')
      tmpstm >> val;

    G[u - 1].insert({dtype(v - 1), vtype(val)});
    if (formats[4] != "general") {
      if (formats[4] == "skew-symmetric")
        G[v - 1].insert({dtype(u - 1), vtype(-val)});
      else
        G[v - 1].insert({dtype(u - 1), vtype(val)});
      
    }
  }
  fin.close();

  vector<ptype> indptr;
  vector<dtype> indices;
  vector<vtype> values;

  for (uint64_t i = 0; i < nrow; ++i) {
    indptr.push_back((ptype)indices.size());
    for (auto &p : G[i]) {
      indices.push_back(p.first);
      values.push_back(p.second);
    }
  }

  indptr.push_back(indices.size());
  assert(indptr.size() == nrow + 1);
  assert(indices.size() == values.size());
  // assert(indices.size() == nnz);
  nnz = indices.size();

  ofstream fout(ofn, ios::out | ios::binary);
  cout << "Sparse Matrix Array Types: ";
  for (int i = 0; i < 3; ++i) {
    cout << types[i] << " ";
    fout.write((char *)&types[i], 4);
  }
  cout << endl;
  fout.write((char *)&nrow, 8);
  fout.write((char *)&ncol, 8);
  fout.write((char *)&nnz, 8);

  fout.write((char *)indptr.data(), sizeof(ptype) * (nrow + 1));
  fout.write((char *)indices.data(), sizeof(dtype) * nnz);
  fout.write((char *)values.data(), sizeof(vtype) * nnz);
  fout.close();
}

int main(int argc, char *argv[]) {

  // add arguments
  argparse::ArgumentParser program("mm_to_csrbin");
  program.add_argument("-i", "--input")
    .required()
    .help("input matrix market file");

  program.add_argument("-o", "--output")
    .required()
    .help("output csr binary file");

  // parse the arguments
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // get input and output file names
  auto ifn = program.get<string>("-i");
  auto ofn = program.get<string>("-o");

  // size of the sparse matrix
  int64_t nrow, ncol, nnz;

  ifstream fin(ifn, ios::in);

  // read the first line for the format
  string line;
  getline(fin, line);
  vector<string> formats(5);
  istringstream(line) >> formats[0] >> formats[1] >> formats[2] >> formats[3] >> formats[4];
  assert(formats[0] == "%%MatrixMarket");
  assert(formats[1] == "matrix");
  assert(formats[2] == "coordinate");

  // read until numbers
  while (line[0] == '%')
    getline(fin, line);

  // read the size of the sparse matrix
  istringstream(line) >> nrow >> ncol >> nnz;
  if (formats[4] != "general")
    nnz *= 2;

  Type dtype;
  if (nrow <= UINT32_MAX && ncol <= UINT32_MAX)
    dtype = Type::uint32;
  else
    dtype = Type::uint64;
  
  Type ptype;
  if (nnz <= UINT32_MAX)
    ptype = Type::uint32;
  else
    ptype = Type::uint64;

  if (ptype == Type::uint32 && dtype == Type::uint32)
    save_to_csrbin<uint32_t, uint32_t, float>(nrow, ncol, nnz, {ptype, dtype, Type::float32}, fin, formats, ofn);
  else if (ptype == Type::uint32 && dtype == Type::uint64)
    save_to_csrbin<uint32_t, uint64_t, float>(nrow, ncol, nnz, {ptype, dtype, Type::float32}, fin, formats, ofn);
  else if (ptype == Type::uint64 && dtype == Type::uint32)
    save_to_csrbin<uint64_t, uint32_t, float>(nrow, ncol, nnz, {ptype, dtype, Type::float32}, fin, formats, ofn);
  else
    save_to_csrbin<uint64_t, uint64_t, float>(nrow, ncol, nnz, {ptype, dtype, Type::float32}, fin, formats, ofn);
}
