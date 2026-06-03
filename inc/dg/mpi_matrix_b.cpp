// Benchmark: MPI matrix-vector multiply (symv) in MPISparseBlockMat
//
// Build:  make mpi_matrix_mpib    (inside inc/dg/)
// Run:    mpirun -n 4 ./mpi_matrix_mpib
//
// For each derivative direction (dx, dy, dz) the benchmark times the four
// phases of MPISparseBlockMat::symv separately:
//
//   [global_gather_init] ... [inner symv] ... [global_gather_wait] ... [outer symv]
//    post (negligible)        compute          sync / finish           boundary rows
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

#include "backend/mpi_init.h"
#include "backend/timer.h"
#include "blas.h"
#include "topology/mpi_derivatives.h"
#include "topology/mpi_evaluation.h"
#include "topology/mpi_weights.h"

int main(int argc, char* argv[])
{
    dg::mpi_init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    DG_RANK0 std::cout <<
        "Benchmark: MPISparseBlockMat symv phase breakdown\n"
        "=================================================\n"
        "Phases:\n"
        "  inner : inner-row compute (overlaps with MPI)\n"
        "  comm  : MPI halo exchange measured without overlap\n"
        "  outer : boundary-row compute after MPI completes\n"
        "  total : end-to-end symv\n\n";

    MPI_Comm comm = dg::mpi_cart_create({dg::PER, dg::PER, dg::PER}, std::cin,
        MPI_COMM_WORLD, true, true, std::cout);
    unsigned n, Nx, Ny, Nz;
    dg::mpi_read_grid(n, {&Nx, &Ny, &Nz}, comm, std::cin, true, std::cout);

    using value_type = double;
    dg::RealMPIGrid3d<value_type> grid(
        0., 2.*M_PI, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, comm);

    dg::x::DVec x = dg::construct<dg::x::DVec>(dg::evaluate(dg::zero, grid));
    dg::x::DVec y(x);

    // global size in GB for bandwidth computation (3 memops: read x, read M, write y)
    value_type gbytes = (value_type)grid.global().size() * sizeof(value_type) / 1e9;
    DG_RANK0 std::cout << "Global vector size: " << gbytes << " GB\n\n";

    const int multi = 100;
    dg::Timer t;

    auto benchmark = [&](const char* name, dg::x::DMatrix& M)
    {
        // warm-up pass
        dg::blas2::symv(M, x, y);
        MPI_Barrier(comm);

        // --- 1. total symv ---
        t.tic(comm);
        for(int i = 0; i < multi; i++)
            dg::blas2::symv(M, x, y);
        t.toc(comm);
        const double t_total = t.diff() / multi;

        // --- 2. inner compute only ---
        t.tic(comm);
        for(int i = 0; i < multi; i++)
            dg::blas2::symv(M.inner_matrix(), x.data(), y.data());
        t.toc(comm);
        const double t_inner = t.diff() / multi;

        // --- 3. pure communication (init + wait, no inner compute between) ---
        double t_comm = 0.;
        if(M.mpi_gather().isCommunicating())
        {
            thrust::device_vector<const value_type*> buf_ptrs(
                M.mpi_gather().buffer_size());
            // warm-up
            M.mpi_gather().global_gather_init(x.data());
            M.mpi_gather().global_gather_wait(x.data(), buf_ptrs);
            MPI_Barrier(comm);

            t.tic(comm);
            for(int i = 0; i < multi; i++)
            {
                M.mpi_gather().global_gather_init(x.data());
                M.mpi_gather().global_gather_wait(x.data(), buf_ptrs);
            }
            t.toc(comm);
            t_comm = t.diff() / multi;
        }

        // --- derived metrics ---
        const double t_exposed = std::max(0., t_comm - t_inner);
        const double t_outer   = t_total - std::max(t_inner, t_comm);
        const double hiding    = M.mpi_gather().isCommunicating()
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
                std::cout << std::setprecision(4);
                std::cout << "  comm  (MPI) : " << std::setw(9) << t_comm*1e3   << " ms\n";
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

    dg::x::DMatrix Dx, Dy, Dz;
    dg::blas2::transfer(dg::create::derivative(0, grid, dg::PER, dg::centered), Dx);
    dg::blas2::transfer(dg::create::derivative(1, grid, dg::PER, dg::centered), Dy);
    dg::blas2::transfer(dg::create::derivative(2, grid, dg::PER, dg::centered), Dz);

    benchmark("dx centered", Dx);
    benchmark("dy centered", Dy);
    benchmark("dz centered", Dz);

    MPI_Finalize();
    return 0;
}
