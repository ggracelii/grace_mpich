# Grace MPICH with HIP + RCCL Support

This repository builds a custom version of MPICH that supports GPU-aware collectives using HIP and RCCL.

##  Setup Instructions

### 1. Clone the repository and run autogen
```bash
git clone https://github.com/ggracelii/grace_mpich.git
cd grace_mpich
./autogen.sh
```

### 2. Create and enter build directory
```bash
mkdir -p build/install
cd build
```

### 3. Configuration & make scripts
Create a configuration script titled `cf.sh`:
```bash
#!/bin/bash

HIP_PATH=/soft/compilers/rocm/rocm-6.3.2
HIP_INC=$HIP_PATH/include
HIP_LIB=$HIP_PATH/lib
LLVM_PATH=$HIP_PATH/llvm

export HIPCC=$HIP_PATH/bin/hipcc
export CC=$LLVM_PATH/bin/clang
export CXX=$LLVM_PATH/bin/clang++

export PATH=$HIP_PATH/bin:$LLVM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HIP_LIB:$LD_LIBRARY_PATH

RCCL_DIR=$HOME/rccl/build
RCCL_INC=$RCCL_DIR/include/rccl
RCCL_LIB=$RCCL_DIR/lib

export LD_LIBRARY_PATH=$RCCL_LIB:$LD_LIBRARY_PATH

export CPPFLAGS="-DENABLE_CCLCOMM -DENABLE_RCCL"
export CFLAGS="-g -O0 -I${RCCL_INC} -I${HIP_INC}"
export CXXFLAGS="-g -O0 -I${RCCL_INC} -I${HIP_INC}"
export LDFLAGS="-L${RCCL_LIB} -L${HIP_LIB}"
export LIBS="-lrccl -lamdhip64"

make clean || true

../configure \
  CFLAGS="$CFLAGS" \
  CXXFLAGS="$CXXFLAGS" \
  CPPFLAGS="$CPPFLAGS" \
  LDFLAGS="$LDFLAGS" \
  LIBS="$LIBS" \
  --prefix=$(pwd)/install \
  --with-libfabric=embedded
```
This sets up HIP and RCCL include/library paths and configures MPICH accordingly.

Create a make script titled `remake.sh`:
```bash
#!/bin/bash

make -j $(nproc) 2>&1|tee make.log
make install 2>&1|tee install.log
```
This will build MPICH and log the output to `make.log` and `install.log`.

### 4. Build and install
Run:
```bash
chmod +x cf.sh remake.sh
./cf.sh && ./remake.sh
```

### Verification
Check the `install/bin` directory and ensure you see both `mpicc` and `mpiexec`.
