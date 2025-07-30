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
int MPIR_RCCLcomm_init(MPIR_Comm *comm_ptr, int rank)
{
    printf("[Rank %d] Entered MPIR_RCCLcomm_init\n", rank);
    int mpi_errno = MPI_SUCCESS;
    hipError_t ret;
    ncclResult_t n_ret;

    MPIR_RCCLcomm *rcclcomm = MPL_malloc(sizeof(MPIR_RCCLcomm), MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!rcclcomm, mpi_errno, MPI_ERR_OTHER, "**nomem");

    int stream_count = MPIR_CVAR_N_SUBCOMMS > 0 ? MPIR_CVAR_N_SUBCOMMS : 4;
    rcclcomm->stream_count = stream_count;
    rcclcomm->split_streams = MPL_malloc(sizeof(hipStream_t) * stream_count, MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!rcclcomm->split_streams, mpi_errno, MPI_ERR_OTHER, "**nomem");

    // Generate and broadcast NCCL ID
    // if (rank == 0) {
    //     printf("[Rank %d] About to call ncclGetUniqueId\n", rank);
    //     fflush(stdout);
    //     ncclGetUniqueId(&(rcclcomm->id));
    //     printf("[Rank %d] Successfully got unique ID\n", rank);
    //     fflush(stdout);
    // }

    // Initialize multiple subcomms
    printf("[Rank %d] Initializing RCCL communicators for %d subcomms\n", rank, MPIR_CVAR_N_SUBCOMMS);
    fflush(stdout);
    rcclcomm->split_comms = MPL_malloc(sizeof(ncclComm_t) * stream_count, MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!rcclcomm->split_comms, mpi_errno, MPI_ERR_OTHER, "**nomem");
    printf("[Rank %d] Successfully allocated split_comms\n", rank);
    fflush(stdout);

    //generate 4 unique IDs for 4 subcomms
    ncclUniqueId *rccl_ids = MPL_malloc(sizeof(ncclUniqueId) * MPIR_CVAR_N_SUBCOMMS, MPL_MEM_OTHER);
    MPIR_ERR_CHKANDJUMP(!rccl_ids, mpi_errno, MPI_ERR_OTHER, "**nomem");
    if (rank == 0) {
        for (int i = 0; i < MPIR_CVAR_N_SUBCOMMS; ++i) {
            ncclGetUniqueId(&rccl_ids[i]);
        }
    }
    MPI_Bcast(rccl_ids, sizeof(ncclUniqueId) * MPIR_CVAR_N_SUBCOMMS, MPI_BYTE, 0, MPI_COMM_WORLD);

    // printf("[Rank %d] About to call Bcast\n", rank);
    // fflush(stdout);
    // mpi_errno = MPIR_Bcast_impl(&(rcclcomm->id), sizeof(rcclcomm->id), MPIR_UINT8, 0, comm_ptr, MPI_SUCCESS);
    // MPIR_ERR_CHECK(mpi_errno);
    // printf("[Rank %d] Successfully called Bcast\n", rank);
    // fflush(stdout);

    // Create the default stream
    // printf("[Rank %d] About to create default RCCL stream\n", rank);
    // fflush(stdout);
    // ret = hipStreamCreate(&(rcclcomm->stream));
    // HIP_ERR_CHECK(ret);
    // printf("[Rank %d] Successfully created default RCCL stream\n", rank);
    // fflush(stdout);

    // Create multiple streams
    printf("[Rank %d] Creating %d split streams\n", rank, stream_count);
    fflush(stdout);
    rcclcomm->streams_initialized = true;
    for (int i = 0; i < stream_count; ++i) {
        ret = hipStreamCreateWithFlags(&(rcclcomm->split_streams[i]), hipStreamNonBlocking);
        HIP_ERR_CHECK(ret);
    }
    printf("[Rank %d] Successfully created split streams\n", rank);
    fflush(stdout);

    // Create the NCCL communicator
    // int nranks = comm_ptr->local_size;
    // printf("[Rank %d] About to call ncclCommInitRank\n", rank);
    // fflush(stdout);
    // printf("[Rank %d] nranks=%d, comm_ptr->rank=%d\n",
    //    rank, nranks, comm_ptr->rank);
    // fflush(stdout);
    // n_ret = ncclCommInitRank(&(rcclcomm->rcclcomm), nranks, rcclcomm->id, comm_ptr->rank);
    // printf("[Rank %d] ncclCommInitRank returned %d (%s)\n",
    //    rank, n_ret, ncclGetErrorString(n_ret));
    // fflush(stdout);
    // NCCL_ERR_CHECK(n_ret);
    // printf("[Rank %d] Successfully initialized NCCL communicator\n", rank);
    // fflush(stdout);

    // comm_ptr->cclcomm->rcclcomm = rcclcomm;
    // printf("[Rank %d] cclcomm ptr: %p  | rcclcomm assigned: %p\n",
    //    rank, (void*)comm_ptr->cclcomm, (void*)rcclcomm);
    // fflush(stdout);
    // if (comm_ptr->cclcomm->rcclcomm == NULL) {
    //     printf("[Rank %d] FATAL: rcclcomm assignment failed!\n", rank);
    //     fflush(stdout);
    // }

    // Initialize the RCCL communicator for each subcomm
    int nranks = comm_ptr->local_size;
    printf("[Rank %d] Initializing RCCL communicators for %d subcomms\n", rank, MPIR_CVAR_N_SUBCOMMS);
    fflush(stdout);
    for (int i = 0; i < MPIR_CVAR_N_SUBCOMMS; ++i) {
        n_ret = ncclCommInitRank(&(rcclcomm->split_comms[i]), nranks, rccl_ids[i], rank);
        if (n_ret != ncclSuccess) {
            printf("!!![Rank %d] ERROR: Failed to init split_comm[%d]: %s\n",
                    rank, i, ncclGetErrorString(n_ret));
            mpi_errno = MPI_ERR_OTHER;
            goto fn_fail;
        }
        printf("[Rank %d] Successfully initialized split_comm[%d]\n", rank, i);
        fflush(stdout);
    }
    printf("[Rank %d] Successfully initialized RCCL communicators for all subcomms\n", rank);
    fflush(stdout);

    comm_ptr->cclcomm->rcclcomm = rcclcomm;

    printf("[Rank %d] Leaving MPIR_RCCLcomm_init\n", rank);
    fflush(stdout);

    MPL_free(rccl_ids);
    
fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

static int MPIR_RCCL_check_init_and_init(MPIR_Comm * comm_ptr, int rank)
{
    int mpi_errno = MPI_SUCCESS;

     if (!comm_ptr->cclcomm) {
        printf("[Rank %d] Initializing cclcomm...\n", rank);
        mpi_errno = MPIR_CCLcomm_init(comm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    }

    if (!comm_ptr->cclcomm->rcclcomm) {
        printf("[Rank %d] Initializing rcclcomm...\n", rank);
        mpi_errno = MPIR_RCCLcomm_init(comm_ptr, comm_ptr->rank);
        MPIR_ERR_CHECK(mpi_errno);
    }

    if (!comm_ptr->cclcomm->rcclcomm) {
        printf("[Rank %d] ERROR: rcclcomm is still NULL after init\n", rank);
    } else {
        printf("[Rank %d] SUCCESS: rcclcomm is initialized\n", rank);
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
    ncclResult_t n_ret;
    hipError_t ret;

    ncclRedOp_t rcclOp;
    ncclDataType_t rcclDatatype;
    MPIR_RCCLcomm *rcclcomm;

    MPIR_ERR_CHKANDJUMP(comm_ptr == NULL, mpi_errno, MPI_ERR_COMM, "**comm");

    int rank = comm_ptr->rank;
    printf("[Rank %d] Entering MPIR_RCCL_Allreduce\n", rank);

    mpi_errno = MPIR_RCCL_check_init_and_init(comm_ptr, rank);
    MPIR_ERR_CHECK(mpi_errno);

    printf("[Rank %d] In Allreduce, cclcomm ptr: %p  | rcclcomm now: %p\n",
       rank, (void*)comm_ptr->cclcomm, (void*)comm_ptr->cclcomm->rcclcomm);
       
    rcclcomm = comm_ptr->cclcomm ? comm_ptr->cclcomm->rcclcomm : NULL;
    if (!rcclcomm) {
        printf("[Rank %d] ERROR: rcclcomm is NULL!\n", rank);
        mpi_errno = MPI_ERR_OTHER;
        goto fn_fail;
    }

    printf("[Rank %d] rcclcomm initialized at %p\n", rank, (void *)rcclcomm);

    mpi_errno = MPIR_RCCL_get_red_op(op, &rcclOp);
    MPIR_ERR_CHECK(mpi_errno);
    mpi_errno = MPIR_RCCL_get_datatype(datatype, &rcclDatatype);
    MPIR_ERR_CHECK(mpi_errno);

    // Determine device and set it
    MPL_pointer_attr_t recv_attr;
    mpi_errno = MPL_gpu_query_pointer_attr(recvbuf, &recv_attr);
    MPIR_ERR_CHECK(mpi_errno);
    printf("[Rank %d] recvbuf pointer attribute device: %d\n", rank, recv_attr.device_attr.device);

    ret = hipSetDevice(recv_attr.device_attr.device);
    HIP_ERR_CHECK(ret);

    if (sendbuf == MPI_IN_PLACE) {
        printf("[Rank %d] Using MPI_IN_PLACE\n", rank);
        sendbuf = recvbuf;
    }

    int stream_count = rcclcomm->stream_count;
    hipStream_t *streams = rcclcomm->split_streams;

    if (!streams) {
        printf("[Rank %d] ERROR: split_streams is NULL!\n", rank);
        mpi_errno = MPI_ERR_OTHER;
        goto fn_fail;
    }

    printf("[Rank %d] Using %d streams\n", rank, stream_count);

    MPI_Aint type_size;
    MPIR_Datatype_get_size_macro(datatype, type_size);
    MPI_Aint chunk_count = (count + stream_count - 1) / stream_count;

    printf("[Rank %d] Total count = %ld, chunk_count = %ld, type_size = %ld\n",
           rank, (long)count, (long)chunk_count, (long)type_size);

    for (int i = 0; i < stream_count; ++i) {
        MPI_Aint offset = i * chunk_count;
        MPI_Aint this_chunk = MPL_MIN(chunk_count, count - offset);
        if (this_chunk <= 0) continue;

        const void *send_ptr = (const char *)sendbuf + offset * type_size;
        void *recv_ptr = (char *)recvbuf + offset * type_size;

        printf("[Rank %d] Chunk %d: offset=%ld, count=%ld, send_ptr=%p, recv_ptr=%p, stream=%p\n",
               rank, i, (long)offset, (long)this_chunk, send_ptr, recv_ptr, (void *)streams[i]);

        //1 comm 4 streams
        // n_ret = ncclAllReduce(send_ptr, recv_ptr, this_chunk, rcclDatatype, rcclOp,
        //                       rcclcomm->rcclcomm, streams[i]);

        // 4 comms 4 streams
        n_ret = ncclAllReduce(send_ptr, recv_ptr, this_chunk, rcclDatatype, rcclOp,
              rcclcomm->split_comms[i], streams[i]);
        NCCL_ERR_CHECK(n_ret);
    }

    for (int i = 0; i < stream_count; ++i) {
        printf("[Rank %d] Synchronizing stream %d (%p)\n", rank, i, (void *)streams[i]);
        HIP_ERR_CHECK(hipStreamSynchronize(streams[i]));
    }

    printf("[Rank %d] Finished MPIR_RCCL_Allreduce successfully.\n", rank);

fn_exit:
    return mpi_errno;
fn_fail:
    printf("[Rank %d] Exiting MPIR_RCCL_Allreduce with error.\n", rank);
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
int MPIR_RCCLcomm_free(MPIR_Comm *comm)
{
    int mpi_errno = MPI_SUCCESS;
    hipError_t ret;
    ncclResult_t n_ret;

    MPIR_Assert(comm->cclcomm && comm->cclcomm->rcclcomm);
    MPIR_RCCLcomm *rcclcomm = comm->cclcomm->rcclcomm;

    // Sync and destroy default stream
    // HIP_ERR_CHECK(hipStreamSynchronize(rcclcomm->stream));
    // HIP_ERR_CHECK(hipStreamDestroy(rcclcomm->stream));

    // Sync and destroy all split streams
    for (int i = 0; i < rcclcomm->stream_count; ++i) {
        if (rcclcomm->split_comms && rcclcomm->split_comms[i]) {
            ncclCommDestroy(rcclcomm->split_comms[i]);
        }
        if (rcclcomm->split_streams && rcclcomm->split_streams[i]) {
            hipStreamDestroy(rcclcomm->split_streams[i]);
        }
    }

    MPL_free(rcclcomm->split_comms);
    MPL_free(rcclcomm->split_streams);
    MPL_free(rcclcomm);
    comm->cclcomm->rcclcomm = NULL;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#endif /* ENABLE_RCCL */