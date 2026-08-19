#ifndef _DG_PCG_MERGED_
#define _DG_PCG_MERGED_

#include <cmath>
#include <array>
#include <vector>
#include <utility>
#include <type_traits>

#include "blas.h"
#include "functors.h"
#include "extrapolation.h"
#include "backend/typedefs.h"
#include "backend/exblas/accumulate.h"        // exblas::cpu::Round
// NOTE: do NOT include <mpi.h> / exblas/mpi_accumulate.h here. Like pcg.h, this
// header must not pull in MPI headers itself (that flips MPI_VERSION mid-TU and
// breaks config.h / the dg::x:: grid aliases). Under an MPI build the symbols
// reduce_mpi_cpu / mpi_reduce_communicator are already provided transitively by
// blas.h's MPI dispatch (backend/blas1_dispatch_mpi.h) and are used below only
// inside #ifdef MPI_VERSION.
#if defined(DG_WITH_NCCL) && defined(MPI_VERSION)
// Only active in an MPI+CUDA (NCCL) build, where <mpi.h>/nccl.h/exblas CUDA
// kernels are already live. MPI_VERSION is required in the guard because
// DG_WITH_NCCL alone is also defined for the non-MPI *_b benchmark targets --
// including this header there would flip MPI_VERSION mid-TU. blas.h (included
// above) has already established MPI_VERSION for a real MPI build by this point.
#include "backend/exblas/nccl_accumulate.h" // exblas::detail::fused_wdot_nccl
#endif

/*!@file
 * Single-reduction Preconditioned CG (Chronopoulos-Gear) solver.
 *
 * Motivation: the standard dg::PCG performs two mandatory weighted dot
 * products per iteration (for alpha and beta), each of which triggers a
 * blocking device->host copy of the exact accumulator plus, under MPI, an
 * MPI_Allreduce. On small per-GPU problems (e.g. z-only domain decomposition)
 * these synchronizations dominate the elliptic solve wall time.
 *
 * This variant reorganizes the recurrences so that both inner products land at
 * the SAME point of the iteration and are computed from a SINGLE fused
 * reduction (one host copy + one MPI collective). The residual norm used for
 * the stopping criterion is folded into the same reduction, so convergence can
 * be checked every iteration for free.
 *
 * Trade-offs vs dg::PCG:
 *   - NOT bit-reproducible against dg::PCG: every individual inner product is
 *     still computed exactly, but the s = w + beta*s recurrence replaces a
 *     fresh A*p, so the floating-point path differs.
 *   - Needs 5 work vectors (r,z,p,s,w) instead of 3.
 *   - Slightly less robust for extremely ill-conditioned systems (a known
 *     property of CG variants that avoid recomputing A*p). Fine for the
 *     well-conditioned polarization / Helmholtz solves.
 *
 * Reference: Chronopoulos & Gear, "s-step iterative methods for symmetric
 * linear systems" (1989); Ghysels & Vanroose, "Hiding global synchronization
 * latency in the preconditioned Conjugate Gradient algorithm" (2014).
 */

namespace dg{

template< class ContainerType>
class PCGmerged
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>;

    PCGmerged() = default;
    PCGmerged( const ContainerType& copyable, unsigned max_iterations):
        r(copyable), z(r), p(r), s(r), w(r), max_iter(max_iterations){}

    void set_max( unsigned new_max) {max_iter = new_max;}
    unsigned get_max() const {return max_iter;}
    const ContainerType& copyable()const{ return r;}
    void set_verbose( bool verbose){ m_verbose = verbose;}
    void set_throw_on_fail( bool t){ m_throw_on_fail = t;}

    template<class ...Params>
    void construct( Params&& ...ps){ *this = PCGmerged( std::forward<Params>( ps)...); }

    /**
     * @brief Solve \f$ Ax = b\f$ with a single-reduction preconditioned CG.
     *
     * Same interface and stopping criterion as dg::PCG::solve so it is a
     * drop-in replacement. The last argument (test_frequency) is accepted for
     * signature compatibility but ignored: the residual norm is available for
     * free in every iteration.
     *
     * @return number of iterations used
     */
    template< class MatrixType0, class ContainerType0, class ContainerType1,
              class MatrixType1, class ContainerType2 >
    unsigned solve( MatrixType0&& A, ContainerType0& x, const ContainerType1& b,
            MatrixType1&& P, const ContainerType2& W, value_type eps = 1e-12,
            value_type nrmb_correction = 1, int /*test_frequency*/ = 1);

  private:
    // g[0] = <a0,W,b0>,  g[1] = <a1,W,b1>,  g[2] = <a2,W,b2>
    // computed with a SINGLE device->host copy per operand and, under MPI, a
    // SINGLE MPI_Allreduce over all three exact accumulators.
    template<class M, class C>
    std::array<value_type,3> fused_wdot(
        const M& Wgt,
        const C& a0, const C& b0,
        const C& a1, const C& b1,
        const C& a2, const C& b2) const;

    // General K-dot fused reduction: g[k] = <a_k, W, b_k>, all K computed from
    // a SINGLE host copy and, under MPI, a SINGLE collective. Used to collapse
    // the per-solve SETUP reductions (nrmb, initial residual, gamma, delta) into
    // one collective; the hot loop keeps the specialized 3-dot fused_wdot above.
    template<class M, class C>
    std::vector<value_type> fused_wdot_n(
        const M& Wgt,
        const std::vector<std::pair<const C*, const C*> >& pairs) const;

    ContainerType r, z, p, s, w;
    unsigned max_iter = 0;
    bool m_verbose = false, m_throw_on_fail = true;
};

///@cond
template< class ContainerType>
template<class M, class C>
std::array<get_value_type<ContainerType>,3>
PCGmerged<ContainerType>::fused_wdot(
        const M& Wgt,
        const C& a0, const C& b0,
        const C& a1, const C& b1,
        const C& a2, const C& b2) const
{
    int status = 0;
    constexpr int NB = exblas::BIN_COUNT;
#ifdef MPI_VERSION
    if constexpr( dg::nccl_mpi ) // true only in a DG_WITH_NCCL build
    {
#ifdef DG_WITH_NCCL
        // On-device single reduction via NCCL. Bit-for-bit identical to the MPI
        // path below as long as a single un-normalized int64 sum cannot overflow
        // (exblas bound: <= 256 accumulators); fall through to MPI otherwise.
        int comm_size = 0;
        MPI_Comm_size( a0.communicator(), &comm_size);
        if( comm_size <= 256 )
            return dg::exblas::detail::fused_wdot_nccl(
                    a0.communicator(), Wgt.data(),
                    a0.data(), b0.data(),
                    a1.data(), b1.data(),
                    a2.data(), b2.data(), &status);
#endif
    }
    // --- local exact accumulators (each does one D2H copy of its superacc) ---
    // NOTE: assumes MPI containers expose .data() (local container) and
    // .communicator(). doDot_superacc is the weighted 3-vector local reduction.
    std::vector<int64_t> l0 = dg::blas2::detail::doDot_superacc( &status, a0.data(), Wgt.data(), b0.data());
    std::vector<int64_t> l1 = dg::blas2::detail::doDot_superacc( &status, a1.data(), Wgt.data(), b1.data());
    std::vector<int64_t> l2 = dg::blas2::detail::doDot_superacc( &status, a2.data(), Wgt.data(), b2.data());

    // --- pack the three superaccs and reduce them in ONE collective ---
    std::vector<int64_t> in( 3*NB), out( 3*NB, (int64_t)0);
    std::copy( l0.begin(), l0.end(), in.begin() + 0*NB);
    std::copy( l1.begin(), l1.end(), in.begin() + 1*NB);
    std::copy( l2.begin(), l2.end(), in.begin() + 2*NB);

    MPI_Comm comm = a0.communicator(), comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_red);
    dg::exblas::reduce_mpi_cpu( 3, in.data(), out.data(), comm, comm_mod, comm_red);

    return { exblas::cpu::Round( &out[0*NB]),
             exblas::cpu::Round( &out[1*NB]),
             exblas::cpu::Round( &out[2*NB]) };
#else
    std::vector<int64_t> l0 = dg::blas2::detail::doDot_superacc( &status, a0, Wgt, b0);
    std::vector<int64_t> l1 = dg::blas2::detail::doDot_superacc( &status, a1, Wgt, b1);
    std::vector<int64_t> l2 = dg::blas2::detail::doDot_superacc( &status, a2, Wgt, b2);
    return { exblas::cpu::Round( l0.data()),
             exblas::cpu::Round( l1.data()),
             exblas::cpu::Round( l2.data()) };
#endif
}

template< class ContainerType>
template<class M, class C>
std::vector<get_value_type<ContainerType>>
PCGmerged<ContainerType>::fused_wdot_n(
        const M& Wgt,
        const std::vector<std::pair<const C*, const C*> >& pairs) const
{
    int status = 0;
    const int K = static_cast<int>( pairs.size());
    constexpr int NB = exblas::BIN_COUNT;
#ifdef MPI_VERSION
    if constexpr( dg::nccl_mpi ) // true only in a DG_WITH_NCCL build
    {
#ifdef DG_WITH_NCCL
        int comm_size = 0;
        MPI_Comm_size( pairs[0].first->communicator(), &comm_size);
        if( comm_size <= 256 )
        {
            // Reduce all K exact accumulators on-device in one ncclAllReduce.
            // The local containers (device_vectors) are the .data() members.
            using LC = std::decay_t<decltype( pairs[0].first->data())>;
            std::vector<std::pair<const LC*, const LC*> > lpairs;
            lpairs.reserve( K);
            for( const auto& pr : pairs)
                lpairs.emplace_back( &pr.first->data(), &pr.second->data());
            return dg::exblas::detail::fused_wdot_nccl_n(
                    pairs[0].first->communicator(), Wgt.data(), lpairs, &status);
        }
#endif
    }
    // --- local exact accumulators (one D2H copy of each superacc) packed and
    //     reduced in ONE MPI collective over all K accumulators ---
    std::vector<int64_t> in( K*NB), out( K*NB, (int64_t)0);
    for( int k = 0; k < K; k++)
    {
        std::vector<int64_t> lk = dg::blas2::detail::doDot_superacc(
                &status, pairs[k].first->data(), Wgt.data(), pairs[k].second->data());
        std::copy( lk.begin(), lk.end(), in.begin() + k*NB);
    }
    MPI_Comm comm = pairs[0].first->communicator(), comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_red);
    dg::exblas::reduce_mpi_cpu( K, in.data(), out.data(), comm, comm_mod, comm_red);

    std::vector<value_type> res( K);
    for( int k = 0; k < K; k++)
        res[k] = exblas::cpu::Round( &out[k*NB]);
    return res;
#else
    std::vector<value_type> res( K);
    for( int k = 0; k < K; k++)
    {
        std::vector<int64_t> lk = dg::blas2::detail::doDot_superacc(
                &status, *pairs[k].first, Wgt, *pairs[k].second);
        res[k] = exblas::cpu::Round( lk.data());
    }
    return res;
#endif
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1,
          class Preconditioner, class ContainerType2>
unsigned PCGmerged< ContainerType>::solve(
        Matrix&& A, ContainerType0& x, const ContainerType1& b,
        Preconditioner&& P, const ContainerType2& W,
        value_type eps, value_type nrmb_correction, int)
{
#ifdef MPI_VERSION
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    // r = b - A x
    blas2::symv( std::forward<Matrix>(A), x, r);
    blas1::axpby( 1., b, -1., r);

    // z = P r ;  p = z ;  s = A p
    // These preconditioner/matrix applies are hoisted ABOVE the setup reduction
    // so that all four setup inner products are available at the SAME point and
    // can be folded into a SINGLE collective. Trade-off vs dg::PCG: the two
    // symv's are done unconditionally, even in the (rare, in a time-stepping
    // context with an extrapolated initial guess) case where the initial
    // residual already satisfies the stopping criterion. Two extra local SpMVs
    // are far cheaper than the three global reductions this saves per solve --
    // and the coarse-grid solves, where the dot products dominate (see
    // MultigridCG2d::solve), do only a handful of iterations, so collapsing the
    // setup reductions is exactly where it pays off.
    blas2::symv( std::forward<Preconditioner>(P), r, z);
    blas1::copy( z, p);
    blas2::symv( std::forward<Matrix>(A), p, s);

    // ---- single fused setup reduction ----
    //   g[0] = <b,W,b> = nrmb^2         (right-hand-side norm, for the tolerance)
    //   g[1] = <r,W,r> = |r|_W^2        (initial residual, for early return)
    //   g[2] = <r,W,z>                  (gamma)
    //   g[3] = <p,W,s>                  (delta)
    // Down from four separate blas2::dot collectives to one. When the RHS
    // container type differs from the work container type the b-norm cannot join
    // the K=3 fused list, so it falls back to its own dot (setup 4 -> 2); in the
    // common feltor case (all containers identical) all four fuse (setup 4 -> 1).
    value_type nrmb, res2, gamma, delta;
    if constexpr( std::is_same_v<std::decay_t<ContainerType1>, ContainerType> )
    {
        std::vector<std::pair<const ContainerType*, const ContainerType*> > pr = {
            { &b, &b }, { &r, &r }, { &r, &z }, { &p, &s } };
        std::vector<value_type> g = fused_wdot_n( W, pr);
        nrmb = sqrt( g[0]); res2 = g[1]; gamma = g[2]; delta = g[3];
    }
    else
    {
        nrmb = sqrt( blas2::dot( W, b));
        std::vector<std::pair<const ContainerType*, const ContainerType*> > pr = {
            { &r, &r }, { &r, &z }, { &p, &s } };
        std::vector<value_type> g = fused_wdot_n( W, pr);
        res2 = g[0]; gamma = g[1]; delta = g[2];
    }

    value_type tol = eps*(nrmb + nrmb_correction);
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    if( sqrt( res2) < tol)
        return 0;

    value_type alpha = gamma/delta;

    for( unsigned i=1; i<max_iter; i++)
    {
        blas1::axpby(  alpha, p, 1., x);       // x += alpha p
        blas1::axpby( -alpha, s, 1., r);       // r -= alpha s

        blas2::symv( std::forward<Preconditioner>(P), r, z); // z = P r
        blas2::symv( std::forward<Matrix>(A),        z, w);  // w = A z  (the only SpMV)

        // ---- single fused reduction: <r,W,z>, <w,W,z>, <r,W,r> ----
        std::array<value_type,3> g = fused_wdot( W, r, z, w, z, r, r);
        const value_type rz = g[0], wz = g[1], rr = g[2];

        if( m_verbose)
            DG_RANK0 std::cout << "# PCGmerged i="<<i<<" |r|_W="<<sqrt(rr)
                               <<" tol="<<tol<<"\n";
        if( sqrt( rr) < tol)
            return i;

        value_type beta = rz/gamma;
        alpha = rz / ( wz - beta*rz/alpha);    // no extra dot for the new alpha
        gamma = rz;

        blas1::axpby( 1., z, beta, p);         // p = z + beta p
        blas1::axpby( 1., w, beta, s);         // s = w + beta s  (recurrence, no SpMV)
    }
    if( m_throw_on_fail)
        throw dg::Fail( tol, Message(_ping_)
            <<"After "<<max_iter<<" PCGmerged iterations with rtol "<<eps
            <<" and atol "<<eps*nrmb_correction );
    return max_iter;
}
///@endcond

} //namespace dg

#endif //_DG_PCG_MERGED_
