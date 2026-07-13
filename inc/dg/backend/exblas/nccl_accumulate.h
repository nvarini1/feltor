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
 *  Only compiled when DG_WITH_NCCL and MPI_VERSION are both defined (an MPI +
 *  CUDA + NCCL build, dg::nccl_mpi == true), where <mpi.h>, nccl.h and the exblas
 *  CUDA kernels are all already live in the translation unit.
 */
#pragma once
// Guarded by BOTH DG_WITH_NCCL and MPI_VERSION. This header must NOT include
// <mpi.h> or mpi_gather.h: DG_WITH_NCCL is defined project-wide (it is set on
// the dg_dg INTERFACE target, so even the non-MPI *_b benchmark targets see it),
// so pulling an MPI header in here would flip MPI_VERSION mid-TU and break
// config.h / the dg::x:: grid aliases -- the exact hazard pcg_merged.h's include
// guard avoids. In a real feltor_mpi + NCCL build, <mpi.h>, nccl.h and
// dg::detail::getNcclComm are already live transitively when this is included.
#if defined(DG_WITH_NCCL) && defined(MPI_VERSION)

#include <array>
#include <vector>
#include <thrust/device_vector.h>

#include "accumulate.h"          // exblas::cpu::Round, exblas::BIN_COUNT
#include "exdot_cuda.cuh"        // exblas::exdot_gpu (device weighted dot)
#include "../exceptions.h"       // dg::Error / dg::Message / _ping_

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
 * The three local exact superaccumulators (plus a trailing status word) are
 * laid out contiguously in one device buffer and reduced in a SINGLE
 * ncclAllReduce (ncclInt64 + ncclSum, which is an exact integer sum --
 * bit-for-bit identical to the MPI_LONG/MPI_SUM fast path in reduce_mpi_cpu).
 * No MPI collective is issued at all: the non-finite status flag rides along in
 * the same reduction instead of a separate MPI_Allreduce. Valid as long as the
 * number of ranks cannot overflow the un-normalized superaccumulators (the
 * documented exblas bound is <= 256 accumulators); the caller must guard larger
 * communicators.
 *
 * @param comm the MPI communicator the reduction runs over (used to look up the
 *   cached NCCL communicator via dg::detail::getNcclComm)
 * @param W local weight container (must expose .data() and .size())
 * @param a0,b0,a1,b1,a2,b2 local operand containers (same length as W)
 * @param status set to 1 (on every rank) if any rank's local dot flagged a
 *   non-finite value; folded into the ncclAllReduce (ncclSum over a 0/1 flag is
 *   equivalent to the MPI_MAX reduce_mpi_cpu uses), so no separate collective
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

    // --- three local superaccumulators + one status word in ONE device buffer ---
    // Layout: [ acc0 | acc1 | acc2 | status ] = 3*NB + 1 int64 words. Reused
    // across iterations (static) so there is no per-call allocation.
    // exdot_gpu writes the accumulator on the default stream and reads its
    // device-side error flag on the host before returning, which synchronizes
    // the default stream -- so the accumulators are complete before the NCCL
    // call below.
    static thrust::device_vector<int64_t> d_accV( 3*NB + 1);
    int64_t* d_acc = thrust::raw_pointer_cast( d_accV.data());
    auto raw = [](const auto& c){ return thrust::raw_pointer_cast( c.data()); };

    int s = 0, st = 0;
    exblas::exdot_gpu( size, raw(a0), raw(W), raw(b0), d_acc + 0*NB, &st); s |= st;
    exblas::exdot_gpu( size, raw(a1), raw(W), raw(b1), d_acc + 1*NB, &st); s |= st;
    exblas::exdot_gpu( size, raw(a2), raw(W), raw(b2), d_acc + 2*NB, &st); s |= st;

    ncclComm_t ncclcomm;
    cudaStream_t stream;
    dg::detail::getNcclComm( comm, &ncclcomm, &stream);

    // Fold the local status (non-finite) flag into the reduction buffer so it
    // rides along in the SAME collective -- this removes the per-call
    // MPI_Allreduce that previously reduced status separately (a residual MPI
    // collective paired 1:1 with every NCCL call, serializing the two comm
    // layers). ncclSum over a 0/1 flag is equivalent to MPI_MAX here: the sum is
    // nonzero iff any rank flagged non-finite, and with <= 256 ranks (the exblas
    // bound the caller guards) it cannot overflow int64. Enqueued on `stream`
    // ahead of the collective so the write is ordered before the read; hs stays
    // in scope until the stream is synchronized below.
    int64_t hs = s;
    cudaMemcpyAsync( d_acc + 3*NB, &hs, sizeof(int64_t),
            cudaMemcpyHostToDevice, stream);

    // --- single on-device reduction of the accumulators AND the status word ---
    ncclAllReduce( d_acc, d_acc, 3*NB + 1, ncclInt64, ncclSum, ncclcomm, stream);
    // Wait for the collective to be issued, then for the stream to drain
    // (getNcclComm creates non-blocking communicators -- see mpi_gather.h).
    dg::detail::ncclCommSynchronize( ncclcomm);
    cudaStreamSynchronize( stream);

    // --- single D2H copy of the reduced accumulators + status, round on host ---
    std::vector<int64_t> out( 3*NB + 1);
    cudaError_t code = cudaMemcpy( &out[0], d_acc, (3*NB + 1)*sizeof(int64_t),
            cudaMemcpyDeviceToHost);
    if( code != cudaSuccess)
        throw dg::Error( dg::Message(_ping_)<<cudaGetErrorString(code));

    // globally-agreed status: nonzero iff any rank saw a non-finite value.
    *status = (out[3*NB] != 0) ? 1 : 0;

    return { exblas::cpu::Round( &out[0*NB]),
             exblas::cpu::Round( &out[1*NB]),
             exblas::cpu::Round( &out[2*NB]) };
}

} //namespace detail
} //namespace exblas
} //namespace dg

#endif // DG_WITH_NCCL && MPI_VERSION
