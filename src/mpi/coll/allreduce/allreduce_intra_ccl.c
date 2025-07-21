/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

/*
 * This "algorithm" is the generic wrapper for
 * using a CCL (e.g., NCCL, RCCL etc.) to
 * complete the collective operation.
 */

/*
 * TODO: Decide how to handle MPIR_CVAR_ALLREDUCE_CCL_auto when multiple CCL backends are enabled.
 * We cannot support two separate CCL_auto cases if both NCCL and RCCL are built into MPICH.
 */

int MPIR_Allreduce_intra_ccl(const void *sendbuf, void *recvbuf, MPI_Aint count,
                             MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm_ptr, int ccl,
                             int coll_attr)
{
    int rank_ = comm_ptr->rank;
    if (rank_ == 0) { printf("*                       MPIR_Allreduce_intra_ccl called\n"); fflush(stdout); }
    switch (ccl) {
#ifdef ENABLE_NCCL
    case MPIR_CVAR_ALLREDUCE_CCL_auto:
    case MPIR_CVAR_ALLREDUCE_CCL_nccl:
        if (MPIR_NCCL_check_requirements_red_op(sendbuf, recvbuf, datatype, op)) {
            return MPIR_NCCL_Allreduce(sendbuf, recvbuf, count, datatype, op, comm_ptr,
                                       errflag);
        }
        break;
#endif

#ifdef ENABLE_RCCL
    case MPIR_CVAR_ALLREDUCE_CCL_auto:
    case MPIR_CVAR_ALLREDUCE_CCL_rccl:
        // printf("[DEBUG] MPICH Actual sendbuf: %p, recvbuf: %p\n", sendbuf, recvbuf);
        if (MPIR_RCCL_check_requirements_red_op(sendbuf, recvbuf, datatype, op)) {
            if (rank_ == 0) { printf(">> MPIR_Allreduce_intra_ccl: Using RCCL backend\n"); fflush(stdout); }
            return MPIR_RCCL_Allreduce(sendbuf, recvbuf, count, datatype, op, comm_ptr,
                                       errflag);
        } else {
            if (rank_ == 0) {printf(">> MPIR_Allreduce_intra_ccl: RCCL requirements not met, falling back\n"); fflush(stdout); }
        }
        break;
#endif
        default:
            goto fallback;
    }

fallback:
    if (rank_ == 0) { printf(">> MPIR_Allreduce_intra_ccl: Using fallback (recursive doubling)\n"); fflush(stdout); }
    return MPIR_Allreduce_intra_recursive_doubling(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
}
