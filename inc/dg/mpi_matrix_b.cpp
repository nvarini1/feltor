// Benchmark: MPI matrix-vector multiply (symv) in MPISparseBlockMat
//
// Build:  make mpi_matrix_mpib    (inside inc/dg/)
// Run:    mpirun -n 4 ./mpi_matrix_mpib
//
// MPISparseBlockMat::symv has four sequential phases:
//
//   [global_gather_init] ... [inner symv] ... [global_gather_wait] ... [outer symv]
//    post non-blocking MPI    local rows       sync / finish halo       boundary rows
//
// The inner symv runs while comm is in flight (overlap). If it takes longer
// than the halo exchange, the communication cost is fully hidden.
//
// Backend selection is a compile-time choice resolved inside the production
// symv path (dg::blas2::symv -> MPISparseBlockMat::symv -> MPIContiguousGather):
//
//   default                  CUDA-aware MPI (Isend/Irecv on device pointers)
//   -DFELTOR_WITH_NCCL=ON     NCCL (ncclSend/ncclRecv)
//   -DFELTOR_WITH_NVSHMEM=ON  NVSHMEM (nvshmemx_putmem_on_stream)
//
// This single benchmark therefore measures all three backends; rebuild with
// the corresponding cmake flag and compare. NVSHMEM additionally requires the
// runtime to be initialised here in main() (see below).
//
// Reported metrics
//   t_total  : full symv (what production code pays)
//   t_inner  : inner compute alone (the overlap budget)
//   t_comm   : pure communication, init+wait with no inner compute between them
//   t_outer  : boundary-row compute, derived as total - max(inner, comm)
//   exposed  : communication stall = max(0, comm - inner)
//   hiding % : min(1, inner/comm) * 100

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

#ifdef DG_WITH_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif // DG_WITH_NVSHMEM

#include "backend/mpi_init.h"
#include "backend/timer.h"
#include "blas.h"
#include "topology/mpi_derivatives.h"
#include "topology/mpi_evaluation.h"
#include "topology/mpi_weights.h"

int main(int argc, char* argv[])
{
    dg::mpi_init(&argc, &argv);  // sets the CUDA device before NVSHMEM init

#ifdef DG_WITH_NVSHMEM
    // NVSHMEM must be initialised after cudaSetDevice (done by mpi_init above)
    // and before the first symv that triggers nvshmem_malloc in the gather path.
    nvshmemx_init_attr_t nvshmem_attr;
    MPI_Comm nvshmem_bootstrap = MPI_COMM_WORLD;
    nvshmem_attr.mpi_comm = &nvshmem_bootstrap;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &nvshmem_attr);
#endif // DG_WITH_NVSHMEM

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    DG_RANK0 {
        std::cout <<
            "Benchmark: MPISparseBlockMat symv phase breakdown\n"
            "=================================================\n";
        std::cout << "Backend  : ";
        if constexpr (dg::nvshmem_mpi)
            std::cout << "NVSHMEM\n";
        else if constexpr (dg::nccl_mpi)
            std::cout << "NCCL\n";
        else if constexpr (dg::cuda_aware_mpi)
            std::cout << "CUDA-aware MPI\n";
        else
            std::cout << "MPI (host-copy)\n";
        std::cout <<
            "Phases:\n"
            "  inner : inner-row compute (overlaps with comm)\n"
            "  comm  : halo exchange measured without overlap\n"
            "  outer : boundary-row compute after comm completes\n"
            "  total : end-to-end symv\n\n";
    }

    // all-periodic: every process has a halo neighbor, avoids boundary special-cases
    MPI_Comm comm = dg::mpi_cart_create({dg::PER, dg::PER, dg::PER}, std::cin,
        MPI_COMM_WORLD, true, true, std::cout);
    unsigned n, Nx, Ny, Nz;
    dg::mpi_read_grid(n, {&Nx, &Ny, &Nz}, comm, std::cin, true, std::cout);

    using value_type = double;
    dg::RealMPIGrid3d<value_type> grid( // [0,2π]³: standard DG spectral domain
        0., 2.*M_PI, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, comm);

    dg::x::DVec x = dg::construct<dg::x::DVec>(dg::evaluate(dg::zero, grid));
    dg::x::DVec y(x);

    // 3 memory ops per element: read x, read matrix entry, write y.
    // Used to convert elapsed time to an effective memory bandwidth figure.
    value_type gbytes = (value_type)grid.global().size() * sizeof(value_type) / 1e9;
    DG_RANK0 std::cout << "Global vector size: " << gbytes << " GB\n\n";

    const int multi = 100; // enough iterations to drive timer noise below ~1%
    dg::Timer t;

    auto benchmark = [&](const char* name, dg::x::DMatrix& M)
    {
        dg::blas2::symv(M, x, y); // warm-up: prime GPU kernels and MPI channels
        MPI_Barrier(comm);

        // 1. total symv: full production path (init→inner→wait→outer)
        t.tic(comm); // Barrier inside tic synchronises all ranks before the clock starts
        for(int i = 0; i < multi; i++)
            dg::blas2::symv(M, x, y);
        t.toc(comm);
        const double t_total = t.diff() / multi;

        // 2. inner rows only: the overlap budget — if >= t_comm, MPI is fully hidden
        t.tic(comm);
        for(int i = 0; i < multi; i++)
            dg::blas2::symv(M.inner_matrix(), x.data(), y.data());
        t.toc(comm);
        const double t_inner = t.diff() / multi;

        // 3. back-to-back init+wait: irreducible MPI cost, no compute overlap
        double t_comm = 0.;
        if(M.mpi_gather().isCommunicating())
        {
            // device pointers into the received halo buffer, one per foreign column
            thrust::device_vector<const value_type*> buf_ptrs(
                M.mpi_gather().buffer_size());
            M.mpi_gather().global_gather_init(x.data()); // warm-up: flush first-call handshake
            M.mpi_gather().global_gather_wait(x.data(), buf_ptrs);
            MPI_Barrier(comm);

            t.tic(comm);
            for(int i = 0; i < multi; i++)
            {
                M.mpi_gather().global_gather_init(x.data()); // post non-blocking sends/recvs
                M.mpi_gather().global_gather_wait(x.data(), buf_ptrs); // block until halos arrive
            }
            t.toc(comm);
            t_comm = t.diff() / multi;
        }

        // derived: outer = remainder after the longer of inner/comm; exposed = stall the CPU actually waits
        const double t_exposed = std::max(0., t_comm - t_inner);
        const double t_outer   = t_total - std::max(t_inner, t_comm);
        const double hiding    = M.mpi_gather().isCommunicating() // fraction of MPI latency covered, capped at 100%
                                 ? std::min(1., t_inner / t_comm) * 100.
                                 : 100.;

        DG_RANK0
        {
            std::cout << "--- " << name << " ---\n";
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  total  symv : " << std::setw(9) << t_total*1e3  << " ms"
                      << "   (" << std::setprecision(1) << 3.*gbytes/t_total << " GB/s)\n";
            std::cout << std::setprecision(4);
            std::cout << "  inner  comp : " << std::setw(9) << t_inner*1e3  << " ms"
                      << "   (" << std::setprecision(1) << 3.*gbytes/t_inner << " GB/s)\n";
            if(M.mpi_gather().isCommunicating())
            {
                const char* comm_backend = dg::nvshmem_mpi ? "NVSHM"
                                         : dg::nccl_mpi    ? "NCCL"
                                         :                   "MPI ";
                std::cout << std::setprecision(4);
                std::cout << "  comm  (" << comm_backend << ") : " << std::setw(9) << t_comm*1e3   << " ms\n";
                std::cout << "  outer  comp : " << std::setw(9) << t_outer*1e3  << " ms\n";
                std::cout << "  exposed comm: " << std::setw(9) << t_exposed*1e3<< " ms\n";
                std::cout << "  hiding      : " << std::setprecision(1)
                          << std::setw(8) << hiding << " %\n";
            }
            else
                std::cout << "  (no MPI communication in this direction)\n";
            std::cout << "\n";
        }
    };

    // dx usually has no MPI comm (x is the fast index); dy/dz each need a halo exchange
    dg::x::DMatrix Dx, Dy, Dz;
    dg::blas2::transfer(dg::create::derivative(0, grid, dg::PER, dg::centered), Dx);
    dg::blas2::transfer(dg::create::derivative(1, grid, dg::PER, dg::centered), Dy);
    dg::blas2::transfer(dg::create::derivative(2, grid, dg::PER, dg::centered), Dz);

    benchmark("dx centered", Dx);
    benchmark("dy centered", Dy);
    benchmark("dz centered", Dz);

#ifdef DG_WITH_NVSHMEM
    nvshmem_finalize();
#endif // DG_WITH_NVSHMEM
    MPI_Finalize();
    return 0;
}
