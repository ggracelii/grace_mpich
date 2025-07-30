/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

#ifdef ENABLE_CCLCOMM

int MPIR_CCLcomm_init(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_CCLcomm *cclcomm;

    cclcomm = MPL_malloc(sizeof(MPIR_CCLcomm), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!cclcomm, mpi_errno, MPI_ERR_OTHER, "**nomem");

    cclcomm->comm = comm;
    comm->cclcomm = cclcomm;

#ifdef ENABLE_NCCL
    cclcomm->ncclcomm = 0;
#endif

#ifdef ENABLE_RCCL
    cclcomm->rcclcomm = 0;
    mpi_errno = MPIR_RCCLcomm_init(comm, comm->rank);
    MPIR_ERR_CHECK(mpi_errno);
#endif

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

#ifdef ENABLE_NCCL
    if (cclcomm->ncclcomm) {
        mpi_errno = MPIR_NCCLcomm_free(comm_ptr);
        if (mpi_errno != MPL_SUCCESS)
            goto fn_fail;
    }
#endif

#ifdef ENABLE_RCCL
    if (cclcomm->rcclcomm) {
        mpi_errno = MPIR_RCCLcomm_free(comm_ptr);
        if (mpi_errno != MPL_SUCCESS)
            goto fn_fail;
    }
#endif

    MPL_free(cclcomm);
    comm_ptr->cclcomm = NULL;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}


#endif /* ENABLE_CCLCOMM */