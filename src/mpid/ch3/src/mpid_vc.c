/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"

#if defined(HAVE_LIMITS_H)
#include <limits.h>
#endif
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_ERRNO_H)
#include <errno.h>
#endif
#include <ctype.h>

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE
      category    : CH3
      type        : int
      default     : 131072
      class       : none
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        This cvar controls the message size at which CH3 switches
        from eager to rendezvous mode.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

/* What is the arrangement of VCRT and VCR and VC? 
   
   Each VC (the virtual connection itself) is referred to by a reference 
   (pointer) or VCR.  
   Each communicator has a VCRT, which is nothing more than a 
   structure containing a count (size) and an array of pointers to 
   virtual connections (as an abstraction, this could be a sparse
   array, allowing a more scalable representation on massively 
   parallel systems).

 */

/*@
  MPIDI_VCRT_Create - Create a table of VC references

  Notes:
  This routine only provides space for the VC references.  Those should
  be added by assigning to elements of the vc array within the 
  'MPIDI_VCRT' object.
  @*/
int MPIDI_VCRT_Create(int size, struct MPIDI_VCRT **vcrt_ptr)
{
    MPIDI_VCRT_t * vcrt;
    int mpi_errno = MPI_SUCCESS;
    MPIR_CHKPMEM_DECL();

    MPIR_FUNC_ENTER;

    MPIR_CHKPMEM_MALLOC(vcrt, sizeof(MPIDI_VCRT_t) + (size - 1) * sizeof(MPIDI_VC_t *), MPL_MEM_ADDRESS);
    vcrt->handle = HANDLE_SET_KIND(0, HANDLE_KIND_INVALID);
    MPIR_Object_set_ref(vcrt, 1);
    vcrt->size = size;
    *vcrt_ptr = vcrt;

 fn_exit:
    MPIR_CHKPMEM_COMMIT();
    MPIR_FUNC_EXIT;
    return mpi_errno;
 fn_fail:
    /* --BEGIN ERROR HANDLING-- */
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}

/*@
  MPIDI_VCRT_Add_ref - Add a reference to a VC reference table

  Notes:
  This is called when a communicator duplicates its group of processes.
  It is used in 'commutil.c' and in routines to create communicators from
  dynamic process operations.  It does not change the state of any of the
  virtual connections (VCs).
  @*/
int MPIDI_VCRT_Add_ref(struct MPIDI_VCRT *vcrt)
{

    MPIR_FUNC_ENTER;
    MPIR_Object_add_ref(vcrt);
    MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_REFCOUNT,TYPICAL,(MPL_DBG_FDEST, "Incr VCRT %p ref count",vcrt));
    MPIR_FUNC_EXIT;
    return MPI_SUCCESS;
}

/* FIXME: What should this do?  See proc group and vc discussion */

/*@
  MPIDI_VCRT_Release - Release a reference to a VC reference table

  Notes:
  
  @*/
int MPIDI_VCRT_Release(struct MPIDI_VCRT *vcrt)
{
    int in_use;
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_ENTER;

    MPIR_Object_release_ref(vcrt, &in_use);
    MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_REFCOUNT,TYPICAL,(MPL_DBG_FDEST, "Decr VCRT %p ref count",vcrt));
    
    /* If this VC reference table is no longer in use, we can
       decrement the reference count of each of the VCs.  If the
       count on the VCs goes to zero, then we can decrement the
       ref count on the process group and so on. 
    */
    if (!in_use) {
	int i, inuse;

	for (i = 0; i < vcrt->size; i++)
	{
	    MPIDI_VC_t * const vc = vcrt->vcr_table[i];
	    
	    MPIDI_VC_release_ref(vc, &in_use);

	    if (vc->lpid >= MPIR_Process.size && MPIR_Object_get_ref(vc) == 1) {
                /* release vc from dynamic process */
		MPIDI_VC_release_ref(vc, &in_use);
	    }

	    if (!in_use)
	    {
		/* If the VC is myself then skip the close message */
		if (vc->pg == MPIDI_Process.my_pg && 
		    vc->pg_rank == MPIDI_Process.my_pg_rank)
		{
                    MPIDI_PG_release_ref(vc->pg, &inuse);
                    if (inuse == 0)
                    {
                        MPIDI_PG_Destroy(vc->pg);
                    }
		    continue;
		}
		
		/* FIXME: the correct test is ACTIVE or REMOTE_CLOSE */
		/*if (vc->state != MPIDI_VC_STATE_INACTIVE) { */
		if (vc->state == MPIDI_VC_STATE_ACTIVE ||
		    vc->state == MPIDI_VC_STATE_REMOTE_CLOSE)
		{
		    MPIDI_CH3U_VC_SendClose( vc, i );
		}
		else
		{
                    MPIDI_PG_release_ref(vc->pg, &inuse);
                    if (inuse == 0)
                    {
                        MPIDI_PG_Destroy(vc->pg);
                    }

		    MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_OTHER,VERBOSE,(MPL_DBG_FDEST,
                            "vc=%p: not sending a close to %d, vc in state %s",
			     vc, i, MPIDI_VC_GetStateString(vc->state)));
		}

                /* NOTE: we used to * MPIDI_CH3_VC_Destroy(&(pg->vct[i])))
                   here but that is incorrect.  According to the standard, it's
                   entirely possible (likely even) that this VC might still be
                   connected.  VCs are now destroyed when the PG that "owns"
                   them is destroyed (see MPIDI_PG_Destroy). [goodell@ 2008-06-13] */
	    }
	}

	MPL_free(vcrt);
    }

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

/*@
  MPIDI_VCR_Dup - Duplicate a virtual connection reference

  Notes:
  If the VC is being used for the first time in a VC reference
  table, the reference count is set to two, not one, in order to
  distinguish between freeing a communicator with 'MPI_Comm_free' and
  'MPI_Comm_disconnect', and the reference count on the process group
  is incremented (to indicate that the process group is in use).
  While this has no effect on the process group of 'MPI_COMM_WORLD',
  it is important for process groups accessed through 'MPI_Comm_spawn'
  or 'MPI_Comm_connect/MPI_Comm_accept'.
  
  @*/
int MPIDI_VCR_Dup(MPIDI_VCR orig_vcr, MPIDI_VCR * new_vcr)
{

    MPIR_FUNC_ENTER;

    /* We are allowed to create a vc that belongs to no process group 
     as part of the initial connect/accept action, so in that case,
     ignore the pg ref count update */
    /* XXX DJG FIXME-MT should we be checking this? */
    /* we probably need a test-and-incr operation or equivalent to avoid races */
    if (MPIR_Object_get_ref(orig_vcr) == 0 && orig_vcr->pg) {
	MPIDI_VC_add_ref( orig_vcr );
	MPIDI_VC_add_ref( orig_vcr );
	MPIDI_PG_add_ref( orig_vcr->pg );
    }
    else {
	MPIDI_VC_add_ref(orig_vcr);
    }
    MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_REFCOUNT,TYPICAL,(MPL_DBG_FDEST,"Incr VCR %p ref count",orig_vcr));
    *new_vcr = orig_vcr;
    MPIR_FUNC_EXIT;
    return MPI_SUCCESS;
}

/* 
 * The following routines convert to/from the global pids, which are 
 * represented as pairs of ints (process group id, rank in that process group)
 */

/* FIXME: These routines belong in a different place */
int MPIDI_GPID_GetAllInComm( MPIR_Comm *comm_ptr, int local_size,
                             MPIDI_Gpid local_gpids[], int *singlePG )
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int *gpid = (int*)&local_gpids[0];
    int lastPGID = -1, pgid;
    MPIDI_VCR vc;

    MPIR_FUNC_ENTER;

    MPIR_Assert(comm_ptr->local_size == local_size);
    
    *singlePG = 1;
    for (i=0; i<comm_ptr->local_size; i++) {
	vc = comm_ptr->dev.vcrt->vcr_table[i];

	/* Get the process group id as an int */
	MPIDI_PG_IdToNum( vc->pg, &pgid );

	*gpid++ = pgid;
	if (lastPGID != pgid) { 
	    if (lastPGID != -1)
		*singlePG = 0;
	    lastPGID = pgid;
	}
	*gpid++ = vc->pg_rank;

        MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_OTHER,VERBOSE, (MPL_DBG_FDEST,
                         "pgid=%d vc->pg_rank=%d",
                         pgid, vc->pg_rank));
    }
    
    MPIR_FUNC_EXIT;
    return mpi_errno;
}

int MPIDI_GPID_Get( MPIR_Comm *comm_ptr, int rank, MPIDI_Gpid *in_gpid )
{
    int      pgid;
    MPIDI_VCR vc;
    int*     gpid = (int*)in_gpid;
    vc = comm_ptr->dev.vcrt->vcr_table[rank];

    /* Get the process group id as an int */
    MPIDI_PG_IdToNum( vc->pg, &pgid );
    
    gpid[0] = pgid;
    gpid[1] = vc->pg_rank;
    
    return 0;
}

/* 
 * The following is a very simple code for looping through 
 * the GPIDs.  Note that this code requires that all processes
 * have information on the process groups.
 */
int MPIDI_GPID_ToLpidArray( int size, MPIDI_Gpid in_gpid[], MPIR_Lpid lpid[] )
{
    int i, mpi_errno = MPI_SUCCESS;
    int pgid;
    MPIDI_PG_t *pg = 0;
    MPIDI_PG_iterator iter;
    int *gpid = (int*)&in_gpid[0];

    for (i=0; i<size; i++) {
        MPIDI_PG_Get_iterator(&iter);
	do {
	    MPIDI_PG_Get_next( &iter, &pg );
	    if (!pg) {
		/* --BEGIN ERROR HANDLING-- */
		/* Internal error.  This gpid is unknown on this process */
		/* A printf is NEVER valid in code that might be executed
		   by the user, even in an error case (use 
		   MPL_internal_error_printf if you need to print
		   an error message and its not appropriate to use the
		   regular error code system */
		/* printf("No matching pg foung for id = %d\n", pgid ); */
		lpid[i] = -1;
		MPIR_ERR_SET2(mpi_errno,MPI_ERR_INTERN, "**unknowngpid",
			      "**unknowngpid %d %d", gpid[0], gpid[1] );
		return mpi_errno;
		/* --END ERROR HANDLING-- */
	    }
	    MPIDI_PG_IdToNum( pg, &pgid );

	    if (pgid == gpid[0]) {
		/* found the process group.  gpid[1] is the rank in 
		   this process group */
		/* Sanity check on size */
		if (pg->size > gpid[1]) {
		    lpid[i] = pg->vct[gpid[1]].lpid;
		}
		else {
		    /* --BEGIN ERROR HANDLING-- */
		    lpid[i] = -1;
		    MPIR_ERR_SET2(mpi_errno,MPI_ERR_INTERN, "**unknowngpid",
				  "**unknowngpid %d %d", gpid[0], gpid[1] );
		    return mpi_errno;
		    /* --END ERROR HANDLING-- */
		}
		/* printf( "lpid[%d] = %d for gpid = (%d)%d\n", i, lpid[i], 
		   gpid[0], gpid[1] ); */
		break;
	    }
	} while (1);
	gpid += 2;
    }

    return mpi_errno;
}

static inline int MPIDI_LPID_GetAllInComm(MPIR_Comm *comm_ptr, int local_size,
                                          MPIR_Lpid local_lpids[])
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Assert( comm_ptr->local_size == local_size );
    for (int i=0; i<comm_ptr->local_size; i++) {
        local_lpids[i] = comm_ptr->dev.vcrt->vcr_table[i]->lpid;
    }
    return mpi_errno;
}

#ifdef HAVE_ERROR_CHECKING
#define N_STATIC_LPID32 128
/*@
  check_disjoint_lpids - Exchange address mapping for intercomm creation.
 @*/
static int check_disjoint_lpids(MPIR_Lpid lpids1[], int n1, MPIR_Lpid lpids2[], int n2)
{
    int i, mask_size, idx, bit;
    uint64_t maxlpid = 0;
    int mpi_errno = MPI_SUCCESS;
    uint32_t lpidmaskPrealloc[N_STATIC_LPID32];
    uint32_t *lpidmask;
    MPIR_CHKLMEM_DECL();

    /* Find the max lpid */
    for (i=0; i<n1; i++) {
        if (lpids1[i] > maxlpid) maxlpid = lpids1[i];
    }
    for (i=0; i<n2; i++) {
        MPIR_Assert(lpids2[i] <= INT_MAX);
        if (lpids2[i] > maxlpid) maxlpid = lpids2[i];
    }
    MPIR_Assert(maxlpid <= INT_MAX);

    mask_size = (maxlpid / 32) + 1;

    if (mask_size > N_STATIC_LPID32) {
        MPIR_CHKLMEM_MALLOC(lpidmask, mask_size*sizeof(uint32_t));
    }
    else {
        lpidmask = lpidmaskPrealloc;
    }

    /* zero the bitvector array */
    memset(lpidmask, 0x00, mask_size*sizeof(*lpidmask));

    /* Set the bits for the first array */
    for (i=0; i<n1; i++) {
        idx = lpids1[i] / 32;
        bit = lpids1[i] % 32;
        lpidmask[idx] = lpidmask[idx] | (1 << bit);
        MPIR_Assert(idx < mask_size);
    }

    /* Look for any duplicates in the second array */
    for (i=0; i<n2; i++) {
        idx = lpids2[i] / 32;
        bit = lpids2[i] % 32;
        if (lpidmask[idx] & (1 << bit)) {
            MPIR_ERR_SET1(mpi_errno,MPI_ERR_COMM,
                          "**dupprocesses", "**dupprocesses %d", lpids2[i] );
            goto fn_fail;
        }
        /* Add a check on duplicates *within* group 2 */
        lpidmask[idx] = lpidmask[idx] | (1 << bit);
        MPIR_Assert(idx < mask_size);
    }

    /* Also fall through for normal return */
 fn_fail:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
}
#endif /* HAVE_ERROR_CHECKING */

/*@
  MPID_Intercomm_exchange - Exchange remote info for intercomm creation.
 @*/
int MPID_Intercomm_exchange(MPIR_Comm *local_comm_ptr, int local_leader,
                            MPIR_Comm *peer_comm_ptr, int remote_leader, int tag,
                            int context_id, int *remote_context_id,
                            int *remote_size, MPIR_Lpid **remote_lpids, int timeout)
{
    int mpi_errno = MPI_SUCCESS;
    int singlePG;
    int local_size;
    MPIR_Lpid *local_lpids=0;
    MPIDI_Gpid *local_gpids=NULL, *remote_gpids=NULL;
    MPIR_CHKLMEM_DECL();

    if (local_comm_ptr->rank == local_leader) {

        /* First, exchange the group information.  If we were certain
           that the groups were disjoint, we could exchange possible
           context ids at the same time, saving one communication.
           But experience has shown that that is a risky assumption.
        */
        /* Exchange information with my peer.  Use sendrecv */
        local_size = local_comm_ptr->local_size;

        /* printf( "About to sendrecv in intercomm_create\n" );fflush(stdout);*/
        MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_OTHER,VERBOSE,(MPL_DBG_FDEST,"rank %d sendrecv to rank %d", peer_comm_ptr->rank,
                                       remote_leader));
        int local_ints[2] = {local_size, context_id};
        int remote_ints[2];
        mpi_errno = MPIC_Sendrecv(local_ints, 2, MPIR_INT_INTERNAL,
                                  remote_leader, tag,
                                  remote_ints, 2, MPIR_INT_INTERNAL,
                                  remote_leader, tag,
                                  peer_comm_ptr, MPI_STATUS_IGNORE, 0 );
        MPIR_ERR_CHECK(mpi_errno);

        *remote_size = remote_ints[0];
        *remote_context_id = remote_ints[1];
        MPL_DBG_MSG_FMT(MPIDI_CH3_DBG_OTHER,VERBOSE,(MPL_DBG_FDEST, "local size = %d, remote size = %d", local_size,
                                       *remote_size ));
        /* With this information, we can now send and receive the
           global process ids from the peer. */
        MPIR_CHKLMEM_MALLOC(remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid));
        *remote_lpids = MPL_malloc((*remote_size)*sizeof(MPIR_Lpid), MPL_MEM_ADDRESS);
        MPIR_CHKLMEM_MALLOC(local_gpids, local_size*sizeof(MPIDI_Gpid));
        MPIR_CHKLMEM_MALLOC(local_lpids, local_size*sizeof(MPIR_Lpid));

        mpi_errno = MPIDI_GPID_GetAllInComm( local_comm_ptr, local_size, local_gpids, &singlePG );
        MPIR_ERR_CHECK(mpi_errno);

        /* Exchange the lpid arrays */
        mpi_errno = MPIC_Sendrecv( local_gpids, local_size*sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL,
                                      remote_leader, tag,
                                      remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL,
                                      remote_leader, tag, peer_comm_ptr,
                                      MPI_STATUS_IGNORE, 0 );
        MPIR_ERR_CHECK(mpi_errno);


        /* Convert the remote gpids to the lpids */
        mpi_errno = MPIDI_GPID_ToLpidArray( *remote_size, remote_gpids, *remote_lpids );
        MPIR_ERR_CHECK(mpi_errno);

        /* Get our own lpids */
        mpi_errno = MPIDI_LPID_GetAllInComm( local_comm_ptr, local_size, local_lpids );
        MPIR_ERR_CHECK(mpi_errno);

#       ifdef HAVE_ERROR_CHECKING
        {
            MPID_BEGIN_ERROR_CHECKS;
            {
                /* Now that we have both the local and remote processes,
                   check for any overlap */
                mpi_errno = check_disjoint_lpids( local_lpids, local_size, *remote_lpids, *remote_size );
                MPIR_ERR_CHECK(mpi_errno);
            }
            MPID_END_ERROR_CHECKS;
        }
#       endif /* HAVE_ERROR_CHECKING */

        /* At this point, we're done with the local lpids; they'll
           be freed with the other local memory on exit */

    } /* End of the first phase of the leader communication */
    /* Leaders can now swap context ids and then broadcast the value
       to the local group of processes */
    int comm_info[3];
    if (local_comm_ptr->rank == local_leader) {
        /* Now, send all of our local processes the remote_lpids,
           along with the final context id */
        comm_info[0] = *remote_size;
        comm_info[1] = *remote_context_id;
        MPL_DBG_MSG(MPIDI_CH3_DBG_OTHER,VERBOSE,"About to bcast on local_comm");
        mpi_errno = MPIR_Bcast( comm_info, 2, MPIR_INT_INTERNAL, local_leader, local_comm_ptr, 0 );
        MPIR_ERR_CHECK(mpi_errno);
        mpi_errno = MPIR_Bcast( remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL, local_leader,
                                     local_comm_ptr, 0 );
        MPIR_ERR_CHECK(mpi_errno);
        MPL_DBG_MSG_D(MPIDI_CH3_DBG_OTHER,VERBOSE,"end of bcast on local_comm of size %d",
                       local_comm_ptr->local_size );
    }
    else
    {
        /* we're the other processes */
        MPL_DBG_MSG(MPIDI_CH3_DBG_OTHER,VERBOSE,"About to receive bcast on local_comm");
        mpi_errno = MPIR_Bcast( comm_info, 2, MPIR_INT_INTERNAL, local_leader, local_comm_ptr, 0 );
        MPIR_ERR_CHECK(mpi_errno);
        *remote_size = comm_info[0];
        MPIR_CHKLMEM_MALLOC(remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid));
        *remote_lpids = MPL_malloc((*remote_size)*sizeof(MPIR_Lpid), MPL_MEM_ADDRESS);
        mpi_errno = MPIR_Bcast( remote_gpids, (*remote_size)*sizeof(MPIDI_Gpid), MPIR_BYTE_INTERNAL, local_leader,
                                     local_comm_ptr, 0 );
        MPIR_ERR_CHECK(mpi_errno);

        /* Extract the context and group sign information */
        *remote_context_id = comm_info[1];
    }

    /* Finish up by giving the device the opportunity to update
       any other information among these processes.  Note that the
       new intercomm has not been set up; in fact, we haven't yet
       attempted to set up the connection tables.

       In the case of the ch3 device, this calls MPID_PG_ForwardPGInfo
       to ensure that all processes have the information about all
       process groups.  This must be done before the call
       to MPID_GPID_ToLpidArray, as that call needs to know about
       all of the process groups.
    */
    MPIDI_PG_ForwardPGInfo( peer_comm_ptr, local_comm_ptr,
                            *remote_size, (const MPIDI_Gpid*)remote_gpids, local_leader );


    /* Finally, if we are not the local leader, we need to
       convert the remote gpids to local pids.  This must be done
       after we allow the device to handle any steps that it needs to
       take to ensure that all processes contain the necessary process
       group information */
    if (local_comm_ptr->rank != local_leader) {
        mpi_errno = MPIDI_GPID_ToLpidArray( *remote_size, remote_gpids, *remote_lpids );
        MPIR_ERR_CHECK(mpi_errno);
    }

fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

/*@
  MPID_Create_intercomm_from_lpids - Create a new communicator from a given set
  of lpids.  

  Notes:
  This is used to create a communicator that is not a subset of some
  existing communicator, for example, in a 'MPI_Comm_spawn' or 
  'MPI_Comm_connect/MPI_Comm_accept'.  Thus, it is only used for intercommunicators.
 @*/
int MPID_Create_intercomm_from_lpids( MPIR_Comm *newcomm_ptr,
			    int size, const MPIR_Lpid lpids[] )
{
    int mpi_errno = MPI_SUCCESS;

    return mpi_errno;
}

/* The following is a temporary hook to ensure that all processes in 
   a communicator have a set of process groups.
 
   All arguments are input (all processes in comm must have gpids)

   First: all processes check to see if they have information on all
   of the process groups mentioned by id in the array of gpids.

   The local result is LANDed with Allreduce.
   If any process is missing process group information, then the
   root process broadcasts the process group information as a string; 
   each process then uses this information to update to local process group
   information (in the KVS cache that contains information about 
   contacting any process in the process groups).
*/
int MPIDI_PG_ForwardPGInfo( MPIR_Comm *peer_ptr, MPIR_Comm *comm_ptr,
			   int nPGids, const MPIDI_Gpid in_gpids[],
			   int root )
{
    int mpi_errno = MPI_SUCCESS;
    int i, allfound = 1, pgid, pgidWorld;
    MPIDI_PG_t *pg = 0;
    MPIDI_PG_iterator iter;
    
    const int *gpids = (const int*)&in_gpids[0];

    /* Get the pgid for CommWorld (always attached to the first process 
       group) */
    MPIDI_PG_Get_iterator(&iter);
    MPIDI_PG_Get_next( &iter, &pg );
    MPIDI_PG_IdToNum( pg, &pgidWorld );
    
    /* Extract the unique process groups */
    for (i=0; i<nPGids && allfound; i++) {
	if (gpids[0] != pgidWorld) {
	    /* Add this gpid to the list of values to check */
	    /* FIXME: For testing, we just test in place */
            MPIDI_PG_Get_iterator(&iter);
	    do {
                MPIDI_PG_Get_next( &iter, &pg );
		if (!pg) {
		    /* We don't know this pgid */
		    allfound = 0;
		    break;
		}
		MPIDI_PG_IdToNum( pg, &pgid );
	    } while (pgid != gpids[0]);
	}
	gpids += 2;
    }

    /* See if everyone is happy */
    mpi_errno = MPIR_Allreduce( MPI_IN_PLACE, &allfound, 1, MPIR_INT_INTERNAL, MPI_LAND, comm_ptr, 0 );
    MPIR_ERR_CHECK(mpi_errno);
    
    if (allfound) return MPI_SUCCESS;

    /* FIXME: We need a cleaner way to handle this case than using an ifdef.
       We could have an empty version of MPID_PG_BCast in ch3u_port.c, but
       that's a rather crude way of addressing this problem.  Better is to
       make the handling of local and remote PIDS for the dynamic process
       case part of the dynamic process "module"; devices that don't support
       dynamic processes (and hence have only COMM_WORLD) could optimize for 
       that case */
#ifndef MPIDI_CH3_HAS_NO_DYNAMIC_PROCESS
    /* We need to share the process groups.  We use routines
       from ch3u_port.c */
    MPID_PG_BCast( peer_ptr, comm_ptr, root );
#endif
 fn_exit:
    return MPI_SUCCESS;
 fn_fail:
    goto fn_exit;
}

/* ----------------------------------------------------------------------- */
/* Routines to initialize a VC */

/*
 * The lpid counter counts new processes that this process knows about.
 */
int MPIDI_lpid_counter = 0;

/* Fully initialize a VC.  This invokes the channel-specific 
   VC initialization routine MPIDI_CH3_VC_Init . */
int MPIDI_VC_Init( MPIDI_VC_t *vc, MPIDI_PG_t *pg, int rank )
{
    vc->state = MPIDI_VC_STATE_INACTIVE;
    vc->handle  = HANDLE_SET_MPI_KIND(0, MPIR_VCONN);
    MPIR_Object_set_ref(vc, 0);
    vc->pg      = pg;
    vc->pg_rank = rank;
    vc->lpid    = MPIDI_lpid_counter++;
    vc->node_id = -1;
    MPIDI_VC_Init_seqnum_send(vc);
    MPIDI_VC_Init_seqnum_recv(vc);
    vc->rndvSend_fn      = MPIDI_CH3_RndvSend;
    vc->rndvRecv_fn      = MPIDI_CH3_RecvRndv;
    vc->ready_eager_max_msg_sz = -1; /* no limit */;
    vc->eager_max_msg_sz = MPIR_CVAR_CH3_EAGER_MAX_MSG_SIZE;

    vc->sendNoncontig_fn = MPIDI_CH3_SendNoncontig_iov;
#ifdef ENABLE_COMM_OVERRIDES
    vc->comm_ops         = NULL;
#endif
    MPIDI_CH3_VC_Init(vc);
    MPIDI_DBG_PrintVCState(vc);

    return MPI_SUCCESS;
}

/* ----------------------------------------------------------------------- */
/* Routines to vend topology information. */

static int g_max_node_id = -1;
char MPIU_hostname[MAX_HOSTNAME_LEN] = "_UNKNOWN_"; /* '_' is an illegal char for a hostname so */
                                                    /* this will never match */

int MPID_Get_node_id(MPIR_Comm *comm, int rank, int *id_p)
{
    MPIR_Lpid lpid = MPIR_comm_rank_to_lpid(comm, rank);
    if (lpid >= 0 && lpid < MPIR_Process.size) {
        *id_p = MPIR_Process.node_map[lpid];
    } else {
        *id_p = -1;
    }
    return MPI_SUCCESS;
}

/* Providing a comm argument permits optimization, but this function is always
   allowed to return the max for the universe. */
int MPID_Get_max_node_id(MPIR_Comm *comm, int *max_id_p)
{
    /* easiest way to implement this is to track it at PG create/destroy time */
    *max_id_p = g_max_node_id;
    MPIR_Assert(*max_id_p >= 0);
    return MPI_SUCCESS;
}

int MPIDI_Populate_vc_node_ids(MPIDI_PG_t *pg, int our_pg_rank)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    int *out_nodemap;
    out_nodemap = MPIR_Process.node_map;
    g_max_node_id = MPIR_Process.size - 1;

    for (i = 0; i < pg->size; i++) {
        pg->vct[i].node_id = out_nodemap[i];
    }

    return mpi_errno;
}

