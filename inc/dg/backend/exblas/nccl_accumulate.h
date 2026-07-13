/**
 *  @file nccl_accumulate.h
 *  @brief On-device NCCL reduction of exblas superaccumulators
 *
 *  Companion to exblas/mpi_accumulate.h (reduce_mpi_cpu). Where reduce_mpi_cpu
 *  copies each superaccumulator to the host and reduces it with a blocking
 *  MPI_Allreduce, the helper here keeps the accumulators on the GPU and reduces
 *  them with a single ncclAllReduce (device->device over NVLink), copying only
 *  the final reduced result back to the host once.
 *
 *  Only compiled when DG_WITH_NCCL is defined (i.e. dg::nccl_mpi == true), which
 *  is an MPI + CUDA build where <mpi.h>, nccl.h and the exblas CUDA kernels are
 *  all already live in the translation unit.
 */
#pragma once
#ifdef DG_WITH_NCCL

#include <array>
#include <vector>
#include <mpi.h>
#include <thrust/device_vector.h>

#include "accumulate.h"          // exblas::cpu::Round, exblas::BIN_COUNT
#include "exdot_cuda.cuh"        // exblas::exdot_gpu (device weighted dot)
#include "../mpi_gather.h"       // dg::detail::getNcclComm / ncclCommSynchronize

namespace dg
{
namespace exblas
{
namespace detail
{

/*! @brief On-device NCCL reduction of three weighted local dot products.
 *
 * Computes, exactly and reproducibly, the three global weighted inner products
 *   g_k = sum_procs sum_i a_k[i] * W[i] * b_k[i]     (k = 0,1,2)
 * The three local exact superaccumulators are laid out contiguously in one
 * device buffer and reduced in a SINGLE ncclAllReduce (ncclInt64 + ncclSum,
 * which is an exact integer sum -- bit-for-bit identical to the MPI_LONG/MPI_SUM
 * fast path in reduce_mpi_cpu). Valid as long as the number of ranks cannot
 * overflow the un-normalized superaccumulators (the documented exblas bound is
 * <= 256 accumulators); the caller must guard larger communicators.
 *
 * @param comm the MPI communicator the reduction runs over (used to look up the
 *   cached NCCL communicator via dg::detail::getNcclComm)
 * @param W local weight container (must expose .data() and .size())
 * @param a0,b0,a1,b1,a2,b2 local operand containers (same length as W)
 * @param status set to 1 if any local dot flagged a non-finite value, reduced
 *   with MPI_MAX so every rank agrees (matches reduce_mpi_cpu semantics)
 * @return the three rounded double results {g0, g1, g2}
 */
template<class LC, class MC>
inline std::array<double,3> fused_wdot_nccl(
        MPI_Comm comm, const MC& W,
        const LC& a0, const LC& b0,
        const LC& a1, const LC& b1,
        const LC& a2, const LC& b2,
        int* status)
{
    constexpr int NB = exblas::BIN_COUNT;
    const unsigned size = a0.size();

    // --- three local exact superaccumulators packed in ONE device buffer ---
    // Reused across iterations (static) so there is no per-call allocation.
    // exdot_gpu writes the accumulator on the default stream and reads its
    // device-side error flag on the host before returning, which synchronizes
    // the default stream -- so d_acc is complete before the NCCL call below.
    static thrust::device_vector<int64_t> d_accV( 3*NB);
    int64_t* d_acc = thrust::raw_pointer_cast( d_accV.data());
    auto raw = [](const auto& c){ return thrust::raw_pointer_cast( c.data()); };

    int s = 0, st = 0;
    exblas::exdot_gpu( size, raw(a0), raw(W), raw(b0), d_acc + 0*NB, &st); s |= st;
    exblas::exdot_gpu( size, raw(a1), raw(W), raw(b1), d_acc + 1*NB, &st); s |= st;
    exblas::exdot_gpu( size, raw(a2), raw(W), raw(b2), d_acc + 2*NB, &st); s |= st;

    // --- single on-device reduction of all three superaccumulators ---
    ncclComm_t ncclcomm;
    cudaStream_t stream;
    dg::detail::getNcclComm( comm, &ncclcomm, &stream);
    ncclAllReduce( d_acc, d_acc, 3*NB, ncclInt64, ncclSum, ncclcomm, stream);
    // Wait for the collective to be issued, then for the stream to drain
    // (getNcclComm creates non-blocking communicators -- see mpi_gather.h).
    dg::detail::ncclCommSynchronize( ncclcomm);
    cudaStreamSynchronize( stream);

    // --- single D2H copy of the reduced accumulators, then round on host ---
    std::vector<int64_t> out( 3*NB);
    cudaError_t code = cudaMemcpy( &out[0], d_acc, 3*NB*sizeof(int64_t),
            cudaMemcpyDeviceToHost);
    if( code != cudaSuccess)
        throw dg::Error( dg::Message(_ping_)<<cudaGetErrorString(code));

    // status (non-finite detection) is tiny -- keep it on MPI, MPI_MAX so all
    // ranks agree, matching reduce_mpi_cpu's separate status Allreduce.
    *status = s;
    MPI_Allreduce( MPI_IN_PLACE, status, 1, MPI_INT, MPI_MAX, comm);

    return { exblas::cpu::Round( &out[0*NB]),
             exblas::cpu::Round( &out[1*NB]),
             exblas::cpu::Round( &out[2*NB]) };
}

} //namespace detail
} //namespace exblas
} //namespace dg

#endif // DG_WITH_NCCL
