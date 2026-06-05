// Benchmark / smoke-test: NVSHMEM inter-node PUT in isolation
// ===========================================================
//
// Build:  configure with -DFELTOR_WITH_NVSHMEM=ON (same as mpi_matrix_b), then
//             cmake --build <dir> --target dg_nvshmem_b
//         the binary lands next to mpi_matrix_b: <build>/inc/dg/nvshmem_b
// Run:    srun --gpu-bind=none ./nvshmem_b        (no stdin needed)
//
// Why this exists
// ---------------
// mpi_matrix_b exercises NVSHMEM only through the full production symv path
// (grid -> MPISparseBlockMat -> MPIContiguousGather). When the 16-GPU / 4-node
// run dies at the first halo exchange with IBV_WC_LOC_PROT_ERR (ibv_poll_cq
// status 4), it is impossible to tell from that log whether the fault is in the
// transport/GPUDirect-RDMA registration or somewhere in feltor's gather logic.
//
// This miniapp reproduces ONLY the NVSHMEM operations the gather performs, in
// the same order, with the same APIs (see inc/dg/backend/mpi_gather.h):
//
//     dg::mpi_init                       -> cudaSetDevice(rank % ndev)  (identical binding)
//     nvshmemx_init_attr(.._WITH_MPI..)  -> bootstrap over MPI_COMM_WORLD
//     nvshmem_barrier_all                -> force all PEs past init
//     nvshmem_malloc                     -> symmetric recv buffer
//     nvshmemx_putmem_on_stream          -> the PUT that aborts on a bad transport
//     nvshmemx_barrier_all_on_stream     -> stream-ordered completion
//     cudaEventRecord / cudaStreamWaitEvent
//
// It performs an all-to-all of small messages, so EVERY ordered PE pair -- in
// particular every inter-node link -- issues a real RDMA PUT. If a transport
// cannot register/reach peer GPU memory it fails here, with a ~10-line program
// instead of the whole solver, and the result is a clear PASS/FAIL plus a count
// of how many of the exercised edges crossed a node boundary.
//
// Sweep transports exactly as for mpi_matrix_b, e.g.
//     NVSHMEM_REMOTE_TRANSPORT=ibrc   srun ... ./nvshmem_b
//     NVSHMEM_IB_ENABLE_IBGDA=1       srun ... ./nvshmem_b
// and wrap each in `timeout 120` so a deadlocked proxy cannot eat the walltime.

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>  // std::getenv
#include <cstddef>  // std::size_t

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

int main( int argc, char* argv[])
{
    dg::mpi_init( &argc, &argv);  // sets the CUDA device before NVSHMEM init

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    // --- init NVSHMEM exactly as mpi_matrix_b does -----------------------
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
    // PEs that share a node (block distribution => node id = pe / ppn). Used
    // only to report which exercised edges are inter-node.
    int ppn = nvshmem_team_n_pes( NVSHMEMX_TEAM_NODE);
    if( ppn <= 0) ppn = 1;

    if( my_pe != rank || n_pes != size)
        // mpi_init binds MPI rank -> device; NVSHMEM PE should equal world rank.
        std::cerr << "WARN rank/PE mismatch: rank=" << rank << " pe=" << my_pe
                  << " size=" << size << " n_pes=" << n_pes << "\n";

    if( rank == 0)
    {
        std::cout <<
            "NVSHMEM inter-node PUT smoke test\n"
            "=================================\n";
        std::cout << "PEs           : " << n_pes << "\n";
        std::cout << "PEs per node  : " << ppn << "  (nodes ~ " << (n_pes+ppn-1)/ppn << ")\n";
        const char* t = std::getenv( "NVSHMEM_REMOTE_TRANSPORT");
        const char* g = std::getenv( "NVSHMEM_IB_ENABLE_IBGDA");
        std::cout << "Transport env : NVSHMEM_REMOTE_TRANSPORT="
                  << (t ? t : "(unset)")
                  << "  NVSHMEM_IB_ENABLE_IBGDA=" << (g ? g : "(unset)") << "\n";
        std::cout << "Pattern       : all-to-all PUT, every ordered PE pair\n\n";
    }

    // --- buffers ---------------------------------------------------------
    // count doubles per message; small -> this is a connectivity/latency test.
    const std::size_t count = 1024;                 // 8 KiB per message
    const std::size_t bytes = count * sizeof(double);

    // Symmetric recv buffer: one slot per (potential) sender. Sender s writes
    // into slot s on the target, so the receiver knows who each chunk is from.
    // nvshmem_malloc mirrors the gather's m_nvshmem_recv allocation.
    double* sym_recv = static_cast<double*>( nvshmem_malloc( n_pes * bytes));
    if( !sym_recv)
    {
        std::cerr << "FATAL: nvshmem_malloc returned null on PE " << my_pe << "\n";
        MPI_Abort( MPI_COMM_WORLD, 3);
    }
    CUDA_CHECK( cudaMemset( sym_recv, 0, n_pes * bytes));

    // Local (non-symmetric) source, filled with a PE-identifying pattern so the
    // receiver can verify the bytes actually came from the right peer.
    double* src = nullptr;
    CUDA_CHECK( cudaMalloc( &src, bytes));
    {
        std::vector<double> h( count);
        for( std::size_t i = 0; i < count; i++)
            h[i] = 1.0e6 * (double)my_pe + (double)i;   // unique per (pe,i)
        CUDA_CHECK( cudaMemcpy( src, h.data(), bytes, cudaMemcpyHostToDevice));
    }

    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate( &stream));           // mirror m_nvshmem_stream
    cudaEvent_t event;
    CUDA_CHECK( cudaEventCreateWithFlags( &event, cudaEventDisableTiming));

    // Helper: one all-to-all round of PUTs, completed on the stream exactly as
    // the gather completes its halo exchange.
    auto put_round = [&]()
    {
        for( int dst = 0; dst < n_pes; dst++)
        {
            if( dst == my_pe)
            {
                // self -> local D2D copy on the same stream (as in the gather)
                CUDA_CHECK( cudaMemcpyAsync( sym_recv + my_pe * count, src, bytes,
                    cudaMemcpyDeviceToDevice, stream));
            }
            else
            {
                nvshmemx_putmem_on_stream( sym_recv + my_pe * count, src, bytes,
                    dst, stream);
            }
        }
        // Stream-ordered barrier + bridge back to the default stream, mirroring
        // global_gather_wait in mpi_gather.h.
        nvshmemx_barrier_all_on_stream( stream);
        CUDA_CHECK( cudaEventRecord( event, stream));
        CUDA_CHECK( cudaStreamWaitEvent( cudaStreamDefault, event, 0));
        CUDA_CHECK( cudaStreamSynchronize( stream));
    };

    // --- correctness pass: this is where a broken transport aborts --------
    put_round();

    std::vector<double> h_recv( n_pes * count);
    CUDA_CHECK( cudaMemcpy( h_recv.data(), sym_recv, n_pes * bytes,
        cudaMemcpyDeviceToHost));

    int local_errors = 0;
    int inter_node_in = 0;   // how many incoming edges crossed a node boundary
    for( int s = 0; s < n_pes; s++)
    {
        if( (s / ppn) != (my_pe / ppn)) inter_node_in++;
        for( std::size_t i = 0; i < count; i++)
        {
            const double expect = 1.0e6 * (double)s + (double)i;
            if( h_recv[s * count + i] != expect) { local_errors++; break; }
        }
    }

    int total_errors = 0, total_inter = 0;
    MPI_Reduce( &local_errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce( &inter_node_in, &total_inter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // --- timing pass: only meaningful once correctness holds --------------
    const int multi = 100;
    dg::Timer timer;
    nvshmem_barrier_all();
    timer.tic( MPI_COMM_WORLD);
    for( int it = 0; it < multi; it++)
        put_round();
    timer.toc( MPI_COMM_WORLD);
    const double t_round = timer.diff() / multi;

    if( rank == 0)
    {
        // bytes actually PUT over the network per round, per PE: (n_pes-1) msgs
        const double net_gb = (double)(n_pes - 1) * bytes / 1e9;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Inter-node edges exercised : " << total_inter
                  << " of " << n_pes * (n_pes - 1) << " remote edges\n";
        std::cout << "All-to-all PUT round       : " << t_round * 1e6 << " us\n";
        std::cout << "Per-PE net PUT volume      : " << net_gb * 1e3 << " MB/round\n";
        if( t_round > 0.)
            std::cout << "Per-PE PUT bandwidth       : "
                      << net_gb / t_round << " GB/s\n";
        std::cout << "\nRESULT: " << (total_errors == 0 ? "PASS" : "FAIL")
                  << "  (" << total_errors << " PE(s) with mismatched data)\n";
    }

    cudaEventDestroy( event);
    cudaStreamDestroy( stream);
    cudaFree( src);
    nvshmem_free( sym_recv);
    nvshmem_finalize();
    MPI_Finalize();
    return total_errors == 0 ? 0 : 4;
}

#else // !(DG_WITH_NVSHMEM && WITH_MPI)

int main()
{
    std::cout << "nvshmem_b: built without NVSHMEM+MPI "
                 "(needs -DFELTOR_WITH_NVSHMEM=ON and MPI) -- nothing to test.\n";
    return 0;
}

#endif
