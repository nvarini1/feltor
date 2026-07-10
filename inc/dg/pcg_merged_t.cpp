// Validation: PCGmerged (single-reduction CG) vs stock PCG on a 2D elliptic
// (negative Laplacian) problem. Checks that both converge to the same solution
// and reports iteration counts.
//
// Build (host, serial/OMP is enough to validate correctness):
//   g++  -std=c++17 -I../ -I../../inc pcg_merged_t.cpp -o pcg_merged_t
//   mpic++ -std=c++17 -DWITH_MPI -I../ -I../../inc pcg_merged_t.cpp -o pcg_merged_mpi
//
// The GPU path is exercised by compiling the same TU with nvcc (device=gpu).

#include <iostream>
#include <iomanip>
#include "dg/algorithm.h"
#include "dg/pcg_merged.h"

int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    MPI_Comm comm;
    // 1D-style helper picks a decomposition; adapt np as needed on the cluster
    int np[2] = {1,1};
    dg::mpi_init2d( dg::DIR, dg::DIR, comm, std::cin, np);
#else
    (void)argc; (void)argv;
#endif

    const unsigned n = 3, Nx = 64, Ny = 64;
    const double lx = 2.*M_PI, ly = 2.*M_PI;
#ifdef WITH_MPI
    dg::CartesianMPIGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::DIR, comm);
#else
    dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::DIR);
#endif
    const dg::x::DVec w2d = dg::create::weights( grid);

    // rhs and analytic solution for -Laplace u = f, u = sin x sin y
    auto sol = [](double x, double y){ return sin(x)*sin(y); };
    auto rhs = [](double x, double y){ return 2.*sin(x)*sin(y); };
    dg::x::DVec b = dg::evaluate( rhs, grid);
    const dg::x::DVec ana = dg::evaluate( sol, grid);

    dg::Elliptic2d<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> pol( grid);
    dg::blas2::symv( w2d, b, b); // apply weights to rhs (as elliptic is self-adjoint in W)

    const double eps = 1e-8;
    const unsigned max_iter = grid.size();

    // --- stock PCG ---
    dg::x::DVec x1( dg::evaluate( dg::zero, grid));
    dg::PCG<dg::x::DVec> pcg( x1, max_iter);
    unsigned it1 = pcg.solve( pol, x1, b, pol.precond(), pol.weights(), eps);

    // --- PCGmerged ---
    dg::x::DVec x2( dg::evaluate( dg::zero, grid));
    dg::PCGmerged<dg::x::DVec> pcgm( x2, max_iter);
    unsigned it2 = pcgm.solve( pol, x2, b, pol.precond(), pol.weights(), eps);

    // --- compare ---
    dg::x::DVec err1( x1), err2( x2);
    dg::blas1::axpby( 1., ana, -1., err1);
    dg::blas1::axpby( 1., ana, -1., err2);
    dg::blas1::axpby( 1., x1, -1., x2); // x2 <- x2 - x1 (solver-vs-solver diff)
    const double e1 = sqrt( dg::blas2::dot( err1, w2d, err1));
    const double e2 = sqrt( dg::blas2::dot( err2, w2d, err2));
    const double dd = sqrt( dg::blas2::dot( x2,   w2d, x2  ));

    DG_RANK0 std::cout << std::scientific << std::setprecision(3)
        << "PCG      iters=" << it1 << "  err_vs_analytic=" << e1 << "\n"
        << "PCGmerged iters=" << it2 << "  err_vs_analytic=" << e2 << "\n"
        << "||x_merged - x_pcg||_W = " << dd << "  (expect ~ solver tol)\n";

    const bool ok = (e2 < 1e-4) && (dd < 1e-4);
    DG_RANK0 std::cout << (ok ? "PASSED\n" : "FAILED\n");

#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return ok ? 0 : 1;
}
