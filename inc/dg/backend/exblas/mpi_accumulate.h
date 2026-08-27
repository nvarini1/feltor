/**
 *  @file mpi_accumulate.h
 *  @brief Primitives for an MPI Reduction
 *
 *  @authors
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include <mpi.h>
#include <algorithm>
#include <array>
#include <vector>
#include <map>
#include "accumulate.h"

namespace dg
{
namespace exblas {

/**
 * @brief This function can be used to partition communicators for the \c
 * exblas::reduce_mpi_cpu function
 *
 * If the ranks in \c comm are aligned in rows of 128 (or any number <= 256)
 * \c comm_mod is the 1d communicator for the rows and \c comm_mod_reduce
 * is the communicator for the columns. The last row may contain fewer than 128
 * ranks and \c comm_mod may contain 1 rank fewer than the others in the last
 * columns
 *
 * @param comm the input communicator (unmodified, may not be \c MPI_COMM_NULL)
 * @param comm_mod the line communicator in \c comm, consists of all rank/mod
 * ranks
 * @param comm_mod_reduce the column communicator in \c comm consists of all
 * rank%mod ranks
 * @attention The creation of new communicators involves communication between all
 * participation processes (comm in this case).
 * In order to avoid excessive creation of new MPI communicators
 * (there is a limit to how many a program can create), the function keeps
 * a (static) record of which communicators it has been called with. If you
 * repeatedly call this function with the same \c comm only the first call will
 * actually create new communicators.
 */
inline void mpi_reduce_communicator(MPI_Comm comm,
    MPI_Comm* comm_mod, MPI_Comm* comm_mod_reduce){
    //we keep track of communicators that were created in the past
    // (static function variables are constructed the program
    // encounters them and destruced at program termination)
    static std::map<MPI_Comm, std::array<MPI_Comm, 2>> comm_mods;
    assert( comm != MPI_COMM_NULL);
    if( comm_mods.count(comm) == 1 )
    {
        *comm_mod = comm_mods[comm][0];
        *comm_mod_reduce = comm_mods[comm][1];
        return;
    }
    int rank, size;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    // For example
    // xxxxxxxxxxxxxxxx // 16 (comm_mod)
    // xxxxxxxxxxxxxxxx // 16
    // xxxxxx           //  6
    // 3333332222222222 // (comm_mod_reduce)
    int mod = 128;
    // 0-127, 128-255, ...
    MPI_Comm_split( comm, rank/mod, rank%mod, comm_mod); //collective call
    // Here we split rank%mod instead of returning MPI_COMM_NULL for != 0
    // https://github.com/open-mpi/ompi/issues/13081
    MPI_Comm_split( comm, rank%mod, rank, comm_mod_reduce);

    comm_mods[comm] = {*comm_mod, *comm_mod_reduce};
}

/*! @brief reduce a number of superaccumulators distributed among mpi processes

 * We cannot sum more than 256 accumulators before we need to normalize again, so
 * we need to split the reduction into several steps if more than 256 processes
 * are involved. This function normalizes, reduces, normalizes, reduces and
 * broadcasts the result to all participating processes.  As usual the resulting
 * superaccumulator is unnormalized.
 * @param num_superacc number of Superaccumulators each process holds
 * @param in unnormalized input superaccumulators ( must be of size
 * num_superacc*\c exblas::BIN_COUNT, allocated on the cpu) (read/write,
 * undefined on out)
 * @param out each process contains the result on output( must be of size
 * num_superacc*\c exblas::BIN_COUNT, allocated on the cpu) (write, may not
 * alias in)
 * @param comm The complete MPI communicator
 * @param comm_mod This is the line communicator of up to 128 ranks ( or any
 * other number <256)
 * @param comm_mod_reduce This is the column communicator consisting of all
 * rank 0, 1, 2, ...,127 processes in all comm_mod
 * @sa \c exblas::mpi_reduce_communicator to generate the required communicators
*/
inline void reduce_mpi_cpu(  unsigned num_superacc, int64_t* in, int64_t* out,
MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce, int* status = nullptr )
{
    for( unsigned i=0; i<num_superacc; i++)
    {
        int imin=exblas::IMIN, imax=exblas::IMAX;
        cpu::Normalize(&in[i*exblas::BIN_COUNT], imin, imax);
    }
    const unsigned N = num_superacc*exblas::BIN_COUNT;
    // When a status word is supplied, fold it into the SAME collective(s) as one
    // extra MPI_LONG reduced with MPI_SUM, instead of issuing a separate
    // MPI_Allreduce(MPI_MAX) per dot. The exblas status is 0 (ok) / nonzero
    // (NaN-Inf or overflow); a summed nonzero preserves the caller's status!=0
    // error check exactly while removing one latency-bound collective per
    // reduction -- the dominant cost of the communication-bound elliptic solves.
    // The status slot lives PAST the superaccumulators, so the Normalize passes
    // (which only touch [i*BIN_COUNT, ..) for i<num_superacc) never disturb it.
    const bool fold = (status != nullptr);
    const unsigned total = N + (fold ? 1u : 0u);
    std::vector<int64_t> aug_in, aug_out;
    int64_t *pin = in, *pout = out;
    if( fold)
    {
        aug_in.assign( in, in+N);
        aug_in.push_back( static_cast<int64_t>( *status));
        aug_out.assign( total, (int64_t)0);
        pin = aug_in.data();
        pout = aug_out.data();
    }

    int size_comm, size_comm_mod;
    MPI_Comm_size( comm, &size_comm);
    MPI_Comm_size( comm_mod, &size_comm_mod);
    if( size_comm == size_comm_mod) // size < 128
    {
        MPI_Allreduce( pin, pout, total, MPI_LONG, MPI_SUM, comm);
    }
    else
    {
        MPI_Reduce(pin, pout, total, MPI_LONG, MPI_SUM, 0, comm_mod);
        int rank;
        MPI_Comm_rank( comm_mod, &rank);
        if(rank == 0)
        {
            int size;
            MPI_Comm_size( comm_mod_reduce, &size);
            if( size > 1)
            {
                for( unsigned i=0; i<num_superacc; i++)
                {
                    int imin=exblas::IMIN, imax=exblas::IMAX;
                    cpu::Normalize(&pout[i*exblas::BIN_COUNT], imin, imax);
                    for( int k=0; k<exblas::BIN_COUNT; k++)
                        pin[i*BIN_COUNT+k] = pout[i*BIN_COUNT+k];
                }
                if( fold) pin[N] = pout[N]; // carry the partial status sum forward
                MPI_Reduce(pin, pout, total, MPI_LONG,
                    MPI_SUM, 0, comm_mod_reduce);
            }
        }
        MPI_Bcast( pout, total, MPI_LONG, 0, comm);
    }

    if( fold)
    {
        std::copy( aug_out.begin(), aug_out.begin()+N, out);
        *status = static_cast<int>( aug_out[N]);
    }
}

}//namespace exblas
} //namespace dg
