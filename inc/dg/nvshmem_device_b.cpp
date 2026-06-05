// Benchmark: device-initiated vs host-initiated NVSHMEM PUT (A/B)
// ===============================================================
//
// Build:  configure with -DFELTOR_WITH_NVSHMEM=ON (same as nvshmem_b), then
//             cmake --build <dir> --target dg_nvshmem_device_b
//         binary lands at <build>/inc/dg/nvshmem_device_b
// Run:    srun --gpu-bind=none ./nvshmem_device_b      (no stdin needed)
//
// Why this exists
// ---------------
// The production NVSHMEM halo path (mpi_gather.h: nvshmem_global_gather_init)
// is HOST-initiated: the host loops over chunks issuing nvshmemx_putmem_on_stream,
// then a global nvshmemx_barrier_all_on_stream. That uses NVSHMEM as a
// stream-ordered transport -- functionally close to CUDA-aware MPI.
//
// NVSHMEM's actual strength is DEVICE-initiated RMA: the CUDA kernel itself
// issues the puts. This miniapp measures increment (1) of that idea --
//   "device-issued put, host-side completion" --
// in isolation, as a clean A/B against the existing host-issued path.
//
// Both rounds move the SAME payload with the SAME completion
// (nvshmemx_barrier_all_on_stream + event bridge). The ONLY variable is how the
// puts are issued:
//
//   HOST   : for(dst) nvshmemx_putmem_on_stream(...)        // n_pes-1 host calls
//   DEVICE : put_all_to_all<<<n_pes,threads,stream>>>(...)  // ONE kernel launch,
//            each block issues nvshmemx_putmem_nbi_block(...)// puts from the GPU
//
// What this isolates: per-chunk host-issue overhead. The production dy/dz halo
// has many small chunks, so the host path pays n_chunks separate stream
// submissions; the device path collapses them into a single launch. If the
// device round is faster here, fusing the put into the gather kernel is worth it.
//
// NOT yet measured (increment 2): point-to-point signals
// (nvshmemx_putmem_signal_nbi + nvshmem_signal_wait_until) to replace the global
// barrier, and reading the symmetric buffer in-place from the outer symv kernel
// to drop the final symmetric->caller copy. See the discussion in mpi_gather.h.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>

#if defined(DG_WITH_NVSHMEM) && defined(WITH_MPI)

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "backend/mpi_init.h"  // dg::mpi_init: cudaSetDevice(rank % num_devices)
#include "backend/timer.h"

#define CUDA_CHECK( call ) do {                                            \
    cudaError_t err__ = (call);                                            \
    if( err__ != cudaSuccess) {                                            \
        std::cerr << "FATAL CUDA " << __FILE__ << ":" << __LINE__ << " "   \
                  << cudaGetErrorString(err__) << "\n";                    \
        MPI_Abort( MPI_COMM_WORLD, 2);                                     \
    } } while(0)

// Device-initiated all-to-all PUT.  One thread-block per destination PE; the
// whole block cooperatively issues the transfer of `src` into slot `my_pe` on
// the destination (so the receiver knows who each chunk came from -- identical
// layout to the host path and to the gather's m_nvshmem_recv).
//
// nvshmemx_putmem_nbi_block is a BLOCK-scoped collective: every thread in the
// block must call it. blockIdx.x is uniform across the block, so the dst==my_pe
// branch does not split the block -> the collective call is always reached by
// all threads. _nbi = non-blocking; completion is enforced afterwards by the
// stream-ordered barrier (host side), matching the host path's completion.
__global__ void put_all_to_all( double* sym_recv, const double* src,
    std::size_t count, int my_pe, int n_pes)
{
    const int dst = blockIdx.x;
    if( dst >= n_pes) return;
    double* dest = sym_recv + (std::size_t)my_pe * count; // slot my_pe on `dst`
    if( dst == my_pe)
    {
        // self -> local copy, block-strided (no RMA to self)
        for( std::size_t i = threadIdx.x; i < count; i += blockDim.x)
            dest[i] = src[i];
    }
    else
    {
        nvshmemx_putmem_nbi_block( dest, src, count * sizeof(double), dst);
    }
}

int main( int argc, char* argv[])
{
    dg::mpi_init( &argc, &argv);  // sets the CUDA device before NVSHMEM init

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    // --- init NVSHMEM exactly as nvshmem_b / mpi_matrix_b do -------------
    nvshmemx_init_attr_t attr = {};            // zero-init: no garbage fields
    MPI_Comm bootstrap = MPI_COMM_WORLD;
    attr.mpi_comm = &bootstrap;
    int status = nvshmemx_init_attr( NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if( status != 0)
    {
        std::cerr << "FATAL: nvshmemx_init_attr returned " << status
                  << " -- NVSHMEM runtime did not initialise.\n";
        MPI_Abort( MPI_COMM_WORLD, 1);
    }
    nvshmem_barrier_all();  // force all PEs past init before any RMA is posted

    const int my_pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();
    int ppn = nvshmem_team_n_pes( NVSHMEMX_TEAM_NODE);
    if( ppn <= 0) ppn = 1;

    if( my_pe != rank || n_pes != size)
        std::cerr << "WARN rank/PE mismatch: rank=" << rank << " pe=" << my_pe
                  << " size=" << size << " n_pes=" << n_pes << "\n";

    if( rank == 0)
    {
        std::cout <<
            "NVSHMEM device-initiated vs host-initiated PUT (A/B)\n"
            "====================================================\n";
        std::cout << "PEs           : " << n_pes << "\n";
        std::cout << "PEs per node  : " << ppn << "  (nodes ~ " << (n_pes+ppn-1)/ppn << ")\n";
        std::cout << "Pattern       : all-to-all PUT, every ordered PE pair\n";
        std::cout << "HOST   round  : nvshmemx_putmem_on_stream loop (production path)\n";
        std::cout << "DEVICE round  : put_all_to_all kernel + nvshmemx_putmem_nbi_block\n\n";
    }

    // --- buffers (identical to nvshmem_b) --------------------------------
    const std::size_t count = 1024;                 // 8 KiB per message
    const std::size_t bytes = count * sizeof(double);

    double* sym_recv = static_cast<double*>( nvshmem_malloc( n_pes * bytes));
    if( !sym_recv)
    {
        std::cerr << "FATAL: nvshmem_malloc returned null on PE " << my_pe << "\n";
        MPI_Abort( MPI_COMM_WORLD, 3);
    }

    double* src = nullptr;
    CUDA_CHECK( cudaMalloc( &src, bytes));
    {
        std::vector<double> h( count);
        for( std::size_t i = 0; i < count; i++)
            h[i] = 1.0e6 * (double)my_pe + (double)i;   // unique per (pe,i)
        CUDA_CHECK( cudaMemcpy( src, h.data(), bytes, cudaMemcpyHostToDevice));
    }

    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate( &stream));
    cudaEvent_t event;
    CUDA_CHECK( cudaEventCreateWithFlags( &event, cudaEventDisableTiming));

    // shared completion: identical for both rounds, so it cancels out of the
    // comparison -- only the issue mechanism differs.
    auto complete_on_stream = [&]()
    {
        nvshmemx_barrier_all_on_stream( stream);
        CUDA_CHECK( cudaEventRecord( event, stream));
        CUDA_CHECK( cudaStreamWaitEvent( cudaStreamDefault, event, 0));
        CUDA_CHECK( cudaStreamSynchronize( stream));
    };

    // HOST-initiated round: the production gather mechanism.
    auto put_round_host = [&]()
    {
        for( int dst = 0; dst < n_pes; dst++)
        {
            if( dst == my_pe)
                CUDA_CHECK( cudaMemcpyAsync( sym_recv + my_pe * count, src, bytes,
                    cudaMemcpyDeviceToDevice, stream));
            else
                nvshmemx_putmem_on_stream( sym_recv + my_pe * count, src, bytes,
                    dst, stream);
        }
        complete_on_stream();
    };

    // DEVICE-initiated round: one kernel launch issues every put from the GPU.
    const int threads = 256;
    auto put_round_device = [&]()
    {
        put_all_to_all<<< n_pes, threads, 0, stream>>>(
            sym_recv, src, count, my_pe, n_pes);
        CUDA_CHECK( cudaGetLastError());
        complete_on_stream();
    };

    // helper: verify sym_recv holds each sender's pattern, count inter-node edges
    auto verify = [&]() -> int
    {
        std::vector<double> h_recv( n_pes * count);
        CUDA_CHECK( cudaMemcpy( h_recv.data(), sym_recv, n_pes * bytes,
            cudaMemcpyDeviceToHost));
        int errors = 0;
        for( int s = 0; s < n_pes; s++)
            for( std::size_t i = 0; i < count; i++)
            {
                const double expect = 1.0e6 * (double)s + (double)i;
                if( h_recv[s * count + i] != expect) { errors++; break; }
            }
        return errors;
    };
    int inter_node_in = 0;
    for( int s = 0; s < n_pes; s++)
        if( (s / ppn) != (my_pe / ppn)) inter_node_in++;

    // --- correctness: run each round, check independently -----------------
    CUDA_CHECK( cudaMemset( sym_recv, 0, n_pes * bytes));
    put_round_host();
    const int err_host = verify();

    CUDA_CHECK( cudaMemset( sym_recv, 0, n_pes * bytes));
    put_round_device();
    const int err_device = verify();

    int tot_err_host = 0, tot_err_dev = 0, tot_inter = 0;
    MPI_Reduce( &err_host,       &tot_err_host, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce( &err_device,     &tot_err_dev,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce( &inter_node_in,  &tot_inter,    1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- timing: only meaningful once both pass --------------------------
    const int multi = 100;
    dg::Timer timer;

    nvshmem_barrier_all();
    timer.tic( MPI_COMM_WORLD);
    for( int it = 0; it < multi; it++) put_round_host();
    timer.toc( MPI_COMM_WORLD);
    const double t_host = timer.diff() / multi;

    nvshmem_barrier_all();
    timer.tic( MPI_COMM_WORLD);
    for( int it = 0; it < multi; it++) put_round_device();
    timer.toc( MPI_COMM_WORLD);
    const double t_dev = timer.diff() / multi;

    if( rank == 0)
    {
        const double net_gb = (double)(n_pes - 1) * bytes / 1e9;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Inter-node edges exercised : " << tot_inter
                  << " of " << n_pes * (n_pes - 1) << " remote edges\n\n";
        std::cout << "HOST   round (putmem_on_stream) : " << std::setw(9)
                  << t_host * 1e6 << " us   (" << net_gb / t_host << " GB/s)\n";
        std::cout << "DEVICE round (nbi_block kernel) : " << std::setw(9)
                  << t_dev  * 1e6 << " us   (" << net_gb / t_dev  << " GB/s)\n";
        if( t_dev > 0.)
            std::cout << "Speedup (host/device)           : " << std::setw(9)
                      << t_host / t_dev << " x\n";
        std::cout << "\nRESULT: "
                  << ((tot_err_host == 0 && tot_err_dev == 0) ? "PASS" : "FAIL")
                  << "   (host errors=" << tot_err_host
                  << ", device errors=" << tot_err_dev << ")\n";
    }

    cudaEventDestroy( event);
    cudaStreamDestroy( stream);
    cudaFree( src);
    nvshmem_free( sym_recv);
    nvshmem_finalize();
    MPI_Finalize();
    return (tot_err_host == 0 && tot_err_dev == 0) ? 0 : 4;
}

#else // !(DG_WITH_NVSHMEM && WITH_MPI)

int main()
{
    std::cout << "nvshmem_device_b: built without NVSHMEM+MPI "
                 "(needs -DFELTOR_WITH_NVSHMEM=ON and MPI) -- nothing to test.\n";
    return 0;
}

#endif
