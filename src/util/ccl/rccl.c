/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

#ifdef ENABLE_RCCL

#define HIP_ERR_CHECK(ret)             \
    if (unlikely((ret) != hipSuccess)) \
    goto fn_fail

#define NCCL_ERR_CHECK(ret)             \
    if (unlikely((ret) != ncclSuccess)) \
        goto fn_fail
/*
 * Static helper functions
 */

 /* og init
 static int MPIR_RCCLcomm_init(MPIR_Comm * comm_ptr, int rank)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size = comm_ptr->local_size;
    hipError_t ret;
    ncclResult_t n_ret;

    MPIR_RCCLcomm *rcclcomm;
    rcclcomm = MPL_malloc(sizeof(MPIR_RCCLcomm), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!rcclcomm, mpi_errno, MPI_ERR_OTHER, "**nomem");

    if (rank == 0) {
        ncclGetUniqueId(&(rcclcomm->id));
    }

    mpi_errno = MPIR_Bcast_impl(&(rcclcomm->id), sizeof(rcclcomm->id), MPIR_UINT8, 0, comm_ptr,
                                MPI_SUCCESS);
    MPIR_ERR_CHECK(mpi_errno);

    ret = hipStreamCreate(&(rcclcomm->stream));
    HIP_ERR_CHECK(ret);
    
    n_ret = ncclCommInitRank(&(rcclcomm->rcclcomm), comm_size, rcclcomm->id, rank);
    NCCL_ERR_CHECK(n_ret);

    comm_ptr->cclcomm->rcclcomm = rcclcomm;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
*/

/* init with 4 streams */
static int MPIR_RCCLcomm_init(MPIR_Comm * comm_ptr, int rank)
{
    int mpi_errno = MPI_SUCCESS;
    int comm_size = comm_ptr->local_size;
    hipError_t ret;
    ncclResult_t n_ret;

    MPIR_RCCLcomm *rcclcomm;
    rcclcomm = MPL_malloc(sizeof(MPIR_RCCLcomm), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!rcclcomm, mpi_errno, MPI_ERR_OTHER, "**nomem");

    if (rank == 0) {
        ncclGetUniqueId(&(rcclcomm->id));
    }

    mpi_errno = MPIR_Bcast_impl(&(rcclcomm->id), sizeof(rcclcomm->id), MPIR_UINT8, 0, comm_ptr,
                                MPI_SUCCESS);
    MPIR_ERR_CHECK(mpi_errno);

    ret = hipStreamCreate(&(rcclcomm->stream));
    HIP_ERR_CHECK(ret);

    rcclcomm->stream_count = MPIR_RCCL_MAX_STREAMS;
    rcclcomm->streams_initialized = true;
    for (int i = 0; i < rcclcomm->stream_count; ++i) {
        ret = hipStreamCreateWithFlags(&(rcclcomm->split_streams[i]), hipStreamNonBlocking);
        HIP_ERR_CHECK(ret);
    }

    n_ret = ncclCommInitRank(&(rcclcomm->rcclcomm), comm_size, rcclcomm->id, rank);
    NCCL_ERR_CHECK(n_ret);

    comm_ptr->cclcomm->rcclcomm = rcclcomm;

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int MPIR_RCCL_check_init_and_init(MPIR_Comm * comm_ptr, int rank)
{
    int mpi_errno = MPI_SUCCESS;

    if (!comm_ptr->cclcomm) {
        mpi_errno = MPIR_CCLcomm_init(comm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    }

    if (!comm_ptr->cclcomm->rcclcomm) {
        mpi_errno = MPIR_RCCLcomm_init(comm_ptr, comm_ptr->rank);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int MPIR_RCCL_red_op_is_supported(MPI_Op op)
{
    switch (op) {
        case MPI_SUM:
        case MPI_PROD:
        case MPI_MIN:
        case MPI_MAX:
            return 1;
        default:
            return 0;
    }
}

static int MPIR_RCCL_get_red_op(MPI_Op op, ncclRedOp_t * redOp)
{
    int mpi_errno = MPI_SUCCESS;

    switch (op) {
        case MPI_SUM:
            *redOp = ncclSum;
            break;
        case MPI_PROD:
            *redOp = ncclProd;
            break;
        case MPI_MIN:
            *redOp = ncclMin;
            break;
        case MPI_MAX:
            *redOp = ncclMax;
            break;
        default:
            goto fn_fail;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    mpi_errno = MPI_ERR_ARG;
    goto fn_exit;
}

static int MPIR_RCCL_datatype_is_supported(MPI_Datatype dtype)
{
    switch (MPIR_DATATYPE_GET_RAW_INTERNAL(dtype)) {
        case MPIR_INT8:
        case MPIR_UINT8:
        case MPIR_INT32:
        case MPIR_UINT32:
        case MPIR_INT64:
        case MPIR_UINT64:
        case MPIR_FLOAT16:
        case MPIR_FLOAT32:
        case MPIR_FLOAT64:
            return 1;
        default:
            return 0;
    }
}

static int MPIR_RCCL_get_datatype(MPI_Datatype dtype, ncclDataType_t * rccl_dtype)
{
    int mpi_errno = MPI_SUCCESS;

    switch (MPIR_DATATYPE_GET_RAW_INTERNAL(dtype)) {
        case MPIR_INT8:
            *rccl_dtype = ncclInt8;
            break;
        case MPIR_UINT8:
            *rccl_dtype = ncclUint8;
            break;
        case MPIR_INT32:
            *rccl_dtype = ncclInt32;
            break;
        case MPIR_UINT32:
            *rccl_dtype = ncclUint32;
            break;
        case MPIR_INT64:
            *rccl_dtype = ncclInt64;
            break;
        case MPIR_UINT64:
            *rccl_dtype = ncclUint64;
            break;
        case MPIR_FLOAT16:
            *rccl_dtype = ncclFloat16;
            break;
        case MPIR_FLOAT32:
            *rccl_dtype = ncclFloat32;
            break;
        case MPIR_FLOAT64:
            *rccl_dtype = ncclFloat64;
            break;
        default:
            goto fn_fail;
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    mpi_errno = MPI_ERR_ARG;
    goto fn_exit;
}

/*
 * External functions
 */

// int MPIR_RCCL_check_requirements_red_op(const void *sendbuf, void *recvbuf, MPI_Datatype datatype,
//                                         MPI_Op op)
// {
//     if (!MPIR_RCCL_red_op_is_supported(op) || !MPIR_RCCL_datatype_is_supported(datatype) ||
//         !MPIR_CCL_check_both_gpu_bufs(sendbuf, recvbuf)) {
//         return 0;
//     }

//     return 1;
// }


#include <stdio.h>

int MPIR_RCCL_check_requirements_red_op(const void *sendbuf, void *recvbuf, MPI_Datatype datatype,
                                        MPI_Op op)
{
    int op_supported = MPIR_RCCL_red_op_is_supported(op);
    int dtype_supported = MPIR_RCCL_datatype_is_supported(datatype);
    int buffers_on_gpu = MPIR_CCL_check_both_gpu_bufs(sendbuf, recvbuf);

    int rank = MPIR_Comm_rank(MPIR_Process.comm_world);
    if (rank == 0) {
        printf("[RCCL Check] op_supported=%d, dtype_supported=%d, buffers_on_gpu=%d\n", op_supported, dtype_supported, buffers_on_gpu);
        fflush(stdout);
        // if (!op_supported) { printf("[RCCL Check] Unsupported MPI_Op passed: %d\n", op); fflush(stdout);}
        // if (!dtype_supported) { printf("[RCCL Check] Unsupported MPI_Datatype passed: %d\n", datatype); fflush(stdout); }
        // if (!buffers_on_gpu) { printf("[RCCL Check] Buffers not on GPU (sendbuf=%p, recvbuf=%p)\n", sendbuf, recvbuf); fflush(stdout); }
    }

    if (!op_supported || !dtype_supported || !buffers_on_gpu) {
        return 0;
    }

    if (rank == 0) { printf("[RCCL Check] All requirements met. Using RCCL.\n"); fflush(stdout); }
    return 1;
}

/* og allreduce
int MPIR_RCCL_Allreduce(const void *sendbuf, void *recvbuf, MPI_Aint count, MPI_Datatype datatype,
                        MPI_Op op, MPIR_Comm * comm_ptr, int coll_attr)
{
    int mpi_errno = MPI_SUCCESS;
    hipError_t ret;
    ncclResult_t n_ret;

    ncclRedOp_t rcclOp;
    mpi_errno = MPIR_RCCL_get_red_op(op, &rcclOp);
    MPIR_ERR_CHECK(mpi_errno);

    ncclDataType_t rcclDatatype;
    mpi_errno = MPIR_RCCL_get_datatype(datatype, &rcclDatatype);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIR_RCCL_check_init_and_init(comm_ptr, comm_ptr->rank);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_RCCLcomm *rcclcomm = comm_ptr->cclcomm->rcclcomm;

    MPL_pointer_attr_t recv_attr;
    mpi_errno = MPL_gpu_query_pointer_attr(recvbuf, &recv_attr);
    MPIR_ERR_CHECK(mpi_errno);
    ret = hipSetDevice(recv_attr.device_attr.device);
    HIP_ERR_CHECK(ret);

    if (sendbuf == MPI_IN_PLACE) {
        sendbuf = recvbuf;
    }

    n_ret = ncclAllReduce(sendbuf, recvbuf, count, rcclDatatype, rcclOp, rcclcomm->rcclcomm,
                        rcclcomm->stream);
    NCCL_ERR_CHECK(n_ret);
    ret = hipStreamSynchronize(rcclcomm->stream);
    HIP_ERR_CHECK(ret);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
*/

/* splitting into 4 streams */
int MPIR_RCCL_Allreduce(const void *sendbuf, void *recvbuf, MPI_Aint count,
                        MPI_Datatype datatype, MPI_Op op,
                        MPIR_Comm *comm_ptr, int coll_attr)
{
    int mpi_errno = MPI_SUCCESS;
    hipError_t ret;
    ncclResult_t n_ret;

    ncclRedOp_t rcclOp;
    mpi_errno = MPIR_RCCL_get_red_op(op, &rcclOp);
    MPIR_ERR_CHECK(mpi_errno);

    ncclDataType_t rcclDatatype;
    mpi_errno = MPIR_RCCL_get_datatype(datatype, &rcclDatatype);
    MPIR_ERR_CHECK(mpi_errno);

    mpi_errno = MPIR_RCCL_check_init_and_init(comm_ptr, comm_ptr->rank);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_RCCLcomm *rcclcomm = comm_ptr->cclcomm->rcclcomm;

    MPL_pointer_attr_t recv_attr;
    mpi_errno = MPL_gpu_query_pointer_attr(recvbuf, &recv_attr);
    MPIR_ERR_CHECK(mpi_errno);

    ret = hipSetDevice(recv_attr.device_attr.device);
    HIP_ERR_CHECK(ret);

    if (sendbuf == MPI_IN_PLACE)
        sendbuf = recvbuf;

    int stream_count = rcclcomm->stream_count;
    hipStream_t *streams = rcclcomm->split_streams;

    MPI_Aint type_size;
    MPIR_Datatype_get_size_macro(datatype, type_size);
    MPI_Aint chunk_count = (count + stream_count - 1) / stream_count;

    for (int i = 0; i < stream_count; ++i) {
        MPI_Aint offset = i * chunk_count;
        MPI_Aint this_chunk = MPL_MIN(chunk_count, count - offset);
        if (this_chunk <= 0) continue;

        const void *send_ptr = (const char *)sendbuf + offset * type_size;
        void *recv_ptr = (char *)recvbuf + offset * type_size;

        printf("RCCL AllReduce: Processing chunk %d/%d (count=%ld)\n",
            i + 1, stream_count, (long)this_chunk);

        n_ret = ncclAllReduce(send_ptr, recv_ptr, this_chunk, rcclDatatype, rcclOp,
                            rcclcomm->rcclcomm, streams[i]);
        NCCL_ERR_CHECK(n_ret);
    }

    for (int i = 0; i < stream_count; ++i) {
        ret = hipStreamSynchronize(streams[i]);
        HIP_ERR_CHECK(ret);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* og free 
int MPIR_RCCLcomm_free(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    hipError_t ret;
    ncclResult_t n_ret;

    MPIR_Assert(comm->cclcomm->rcclcomm);
    MPIR_CCLcomm *cclcomm = comm->cclcomm;

    ret = hipStreamSynchronize(cclcomm->rcclcomm->stream);
    HIP_ERR_CHECK(ret);
    n_ret = ncclCommDestroy(cclcomm->rcclcomm->rcclcomm);
    NCCL_ERR_CHECK(n_ret);
    ret = hipStreamDestroy(cclcomm->rcclcomm->stream);
    HIP_ERR_CHECK(ret);

    MPL_free(cclcomm->rcclcomm);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
*/ 

/* 4 streams free */
int MPIR_RCCLcomm_free(MPIR_Comm * comm)
{
    int mpi_errno = MPI_SUCCESS;
    hipError_t ret;
    ncclResult_t n_ret;

    MPIR_Assert(comm->cclcomm->rcclcomm);
    MPIR_CCLcomm *cclcomm = comm->cclcomm;

    ret = hipStreamSynchronize(cclcomm->rcclcomm->stream);
    HIP_ERR_CHECK(ret);

    if (cclcomm->rcclcomm->streams_initialized) {
        for (int i = 0; i < cclcomm->rcclcomm->stream_count; ++i) {
            ret = hipStreamSynchronize(cclcomm->rcclcomm->split_streams[i]);
            HIP_ERR_CHECK(ret);
            ret = hipStreamDestroy(cclcomm->rcclcomm->split_streams[i]);
            HIP_ERR_CHECK(ret);
        }
    }

    n_ret = ncclCommDestroy(cclcomm->rcclcomm->rcclcomm);
    NCCL_ERR_CHECK(n_ret);
    ret = hipStreamDestroy(cclcomm->rcclcomm->stream);
    HIP_ERR_CHECK(ret);

    MPL_free(cclcomm->rcclcomm);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

#endif /* ENABLE_RCCL */
