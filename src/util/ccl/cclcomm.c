/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

#ifdef ENABLE_CCLCOMM

int MPIR_CCLcomm_init(MPIR_Comm * comm)
{
    int rank;
    MPI_Comm_rank(comm->handle, &rank);
    
    int mpi_errno = MPI_SUCCESS;
    MPIR_CCLcomm *cclcomm;
    cclcomm = MPL_malloc(sizeof(MPIR_CCLcomm), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!cclcomm, mpi_errno, MPI_ERR_OTHER, "**nomem");

    cclcomm->comm = comm;
    comm->cclcomm = cclcomm;
    cclcomm->subcomms_initialized = 0;
    for (int i = 0; i < N_SUBCOMMS; ++i)
        cclcomm->subcomms[i] = MPI_COMM_NULL;

    if (!comm->cclcomm->subcomms_initialized) {
        for (int i = 0; i < N_SUBCOMMS; ++i) {
            MPI_Comm_split(comm->handle, rank % N_SUBCOMMS == i ? i : MPI_UNDEFINED, rank,
                        &comm->cclcomm->subcomms[i]);
        }
        comm->cclcomm->subcomm_count = N_SUBCOMMS;
        comm->cclcomm->subcomms_initialized = 1;
    }

#ifdef ENABLE_NCCL
    cclcomm->ncclcomm = 0;      // Initialize the ncclcomm to 0
#endif /*ENABLE_NCCL */

#ifdef ENABLE_RCCL
    cclcomm->rcclcomm = 0;      // Initialize the rcclcomm to 0
#endif /*ENABLE_RCCL */

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_CCLcomm_free(MPIR_Comm * comm_ptr)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_Assert(comm_ptr->cclcomm);

    MPIR_CCLcomm *cclcomm = comm_ptr->cclcomm;
    if (cclcomm->subcomms_initialized) {
    for (int i = 0; i < cclcomm->subcomm_count; ++i) {
        if (cclcomm->subcomms[i] != MPI_COMM_NULL) {
            MPI_Comm_free(&cclcomm->subcomms[i]);
            cclcomm->subcomms[i] = MPI_COMM_NULL;
        }
    }
}

#ifdef ENABLE_NCCL
    if (comm_ptr->cclcomm->ncclcomm) {
        mpi_errno = MPIR_NCCLcomm_free(comm_ptr);
        if (mpi_errno != MPL_SUCCESS) {
            goto fn_fail;
        }
    }
#endif

#ifdef ENABLE_RCCL
    if (comm_ptr->cclcomm->rcclcomm) {
        mpi_errno = MPIR_RCCLcomm_free(comm_ptr);
        if (mpi_errno != MPL_SUCCESS) {
            goto fn_fail;
        }
    }
#endif

    MPL_free(comm_ptr->cclcomm);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif /* ENABLE_CCLCOMM */
