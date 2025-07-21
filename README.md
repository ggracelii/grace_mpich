# Grace MPICH with HIP + RCCL Support

This repository builds a custom version of MPICH that supports GPU-aware collectives using HIP and RCCL.

##  Setup Instructions

### Clone the repository and run build.sh
```bash
./build.sh
```
This will run `autogen.sh`, creat build and install directories, configure MPICH to be built with RCCL support, and run `make` and `make install`.

### Verification
Check the `install/bin` directory and ensure you see both `mpicc` and `mpiexec`.
