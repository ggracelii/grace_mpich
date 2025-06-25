#!/bin/bash
set -e

# === Print Banner Function ===
print_block() {
  echo
  echo "========================================"
  printf " %-35s\n" "$1"
  echo "========================================"
  echo
}

# Step 0: Run autogen.sh
print_block "Skipping autogen.sh"
# ./autogen.sh

# Step 1: Create and enter build/install directory
print_block "Preparing build directory"
rm -rf build
mkdir -p build/install
cd build

# Step 2: Configure environment
print_block "Setting HIP, RCCL, and UCX environment"

HIP_PATH=/soft/compilers/rocm/rocm-6.3.2
HIP_INC=$HIP_PATH/include
HIP_LIB=$HIP_PATH/lib
export HIPCC=$HIP_PATH/bin/hipcc

LLVM_PATH=$HIP_PATH/llvm
export CC=$LLVM_PATH/bin/clang
export CXX=$LLVM_PATH/bin/clang++

export PATH=$HIP_PATH/bin:$LLVM_PATH/bin:$PATH

RCCL_BASE=$HOME/rccl/build/release
RCCL_INC=$RCCL_BASE/include/rccl
RCCL_LIB=$RCCL_BASE

UCX_PATH=$HOME/ucx/install
UCX_INC=$UCX_PATH/include
UCX_LIB=$UCX_PATH/lib

export LD_LIBRARY_PATH=$UCX_LIB:$RCCL_LIB:$HIP_LIB:$LD_LIBRARY_PATH

export GFX_ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]\+' | head -n1)
echo "Detected GPU architecture: $GFX_ARCH"
export CXXFLAGS="--offload-arch=${GFX_ARCH}"
export HIPCCFLAGS="--offload-arch=${GFX_ARCH}"

# Standard build flags
export CPPFLAGS="-DENABLE_CCLCOMM -DENABLE_RCCL -I${HIP_INC} -I${RCCL_INC} -I${UCX_INC}"
export CFLAGS="-I${HIP_INC} -I${RCCL_INC} -I${UCX_INC}"
export LDFLAGS="-L${HIP_LIB} -L${RCCL_LIB} -L${UCX_LIB} -Wl,-rpath,${UCX_LIB} -Wl,-rpath,${HIP_LIB}"
export LIBS="-lrccl -lamdhip64"


# Step 3: Configure MPICH
print_block "Running configure"
../configure \
  --prefix=$(pwd)/install \
  --with-hip=$HIP_PATH \
  --with-rccl-include=$RCCL_INC \
  --with-rccl-lib=$RCCL_LIB \
  --with-device=ch4:ucx \
  --with-ucx-include=$UCX_INC \
  --with-ucx-lib=$UCX_LIB \
  --enable-fast=all \
  --with-pm=hydra \
  CPPFLAGS="$CPPFLAGS" \
  CFLAGS="$CFLAGS" \
  CXXFLAGS="$CXXFLAGS" \
  LDFLAGS="$LDFLAGS" \
  LIBS="$LIBS"

# Step 4: Build
print_block "Running make"
make -j $(nproc) 2>&1 | tee make.log

# Step 5: Install
print_block "Running make install"
make install 2>&1 | tee install.log
