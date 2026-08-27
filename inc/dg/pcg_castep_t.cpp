#include <iostream>
#include <iomanip>

#ifdef WITH_MPI
#include <mpi.h>
#include "backend/mpi_init.h"
#endif

#include "pcg.h"
#include "pcg_castep.h"
#include "elliptic.h"
#include "catch2/catch_all.hpp"

// Validation: PCGcastep (s-step / communication-avoiding CG, one MPI_Allreduce
// per s iterations) must converge to the same solution as the stock dg::PCG on
// an SPD elliptic problem, for a range of block sizes s.

static const double lx = 2.*M_PI;
static const double ly = 2.*M_PI;

static double fct(double x, double y){ return sin(y)*sin(x); }
static double laplace_fct( double x, double y){ return 2.*sin(y)*sin(x); }
static double initial( double, double){ return 0.; }

TEST_CASE( "PCGcastep matches PCG")
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
#endif
    const unsigned n = 4, Nx = 36, Ny = 48;
    dg::x::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::PER
#ifdef WITH_MPI
        , comm
#endif
    );
    const unsigned max_iter = n*n*Nx*Ny;
    const double eps = 1e-8;

    const dg::x::DVec w2d      = dg::create::weights( grid);
    const dg::x::DVec solution = dg::evaluate( fct, grid);
    const dg::x::DVec b        = dg::evaluate( laplace_fct, grid);

    dg::Elliptic<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec> A( grid, dg::forward);

    // reference PCG solution
    dg::x::DVec x1 = dg::evaluate( initial, grid);
    dg::PCG<dg::x::DVec> pcg( x1, max_iter);
    unsigned it1 = pcg.solve( A, x1, b, A.precond(), A.weights(), eps);

    for( unsigned s : { 1u, 2u, 3u, 4u, 8u } )
    {
        DYNAMIC_SECTION( "block size s=" << s )
        {
            dg::x::DVec x2 = dg::evaluate( initial, grid);
            dg::PCGcastep<dg::x::DVec> cacg( x2, max_iter, s);
            unsigned it2 = cacg.solve( A, x2, b, A.precond(), A.weights(), eps);

            dg::x::DVec err( solution);
            dg::blas1::axpby( 1., x2, -1., err);              // err = x2 - solution
            const double e_ana = sqrt( dg::blas2::dot( err, w2d, err));

            dg::x::DVec d( x1);
            dg::blas1::axpby( 1., x2, -1., d);               // d = x2 - x1
            const double e_slv = sqrt( dg::blas2::dot( d, w2d, d));

            INFO( "PCG iters=" << it1 << "  PCGcastep(s=" << s << ") iters=" << it2);
            INFO( "err vs analytic = " << e_ana << ", solver-vs-solver = " << e_slv);
            CHECK( e_ana < 1e-4 );        // converges to the true solution
            CHECK( e_slv < 1e-5 );        // agrees with PCG to solver tolerance
        }
    }
}
