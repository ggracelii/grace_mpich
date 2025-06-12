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
# print_block "Running autogen.sh"
#./autogen.sh
print_block "Skipping autogen.sh"

# Step 1: Create and enter build/install directory
print_block "Preparing build directory"
rm -rf build
mkdir -p build/install
cd build

# Step 2: Configure environment
print_block "Setting HIP and RCCL environment"

HIP_PATH=/soft/compilers/rocm/rocm-6.3.2
HIP_INC=$HIP_PATH/include
HIP_LIB=$HIP_PATH/lib
LLVM_PATH=$HIP_PATH/llvm

export HIPCC=$HIP_PATH/bin/hipcc                  # For HIP-based builds (e.g. RCCL)
export CC=$LLVM_PATH/bin/clang                    # For host code compilation
export CXX=$LLVM_PATH/bin/clang++                 # For host C++ code

export PATH=$HIP_PATH/bin:$LLVM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HIP_LIB:$LD_LIBRARY_PATH

RCCL_DIR=$HOME/rccl/build
RCCL_INC=$RCCL_DIR/include/rccl
RCCL_LIB=$RCCL_DIR/lib

export LD_LIBRARY_PATH=$RCCL_LIB:$LD_LIBRARY_PATH

export CPPFLAGS="-DENABLE_CCLCOMM -DENABLE_RCCL"
export CFLAGS="-I${RCCL_INC} -I${HIP_INC}"
export LDFLAGS="-L${RCCL_LIB} -L${HIP_LIB}"
export LIBS="-lrccl -lamdhip64"

make clean || true

# Step 3: Configure
print_block "Running configure"
../configure \
  --prefix=$(pwd)/install \
  --with-libfabric=embedded

# Step 4: Build
print_block "Running make"
make -j $(nproc) 2>&1 | tee make.log

# Step 5: Install
print_block "Running make install"
make install 2>&1 | tee install.log
