#ifndef _DG_PCG_CASTEP_
#define _DG_PCG_CASTEP_

#include <cmath>
#include <array>
#include <vector>
#include <tuple>
#include <cstdlib>       // std::getenv, std::atoi
#include <type_traits>

#include "blas.h"
#include "functors.h"
#include "extrapolation.h"
#include "backend/typedefs.h"
#include "backend/exblas/accumulate.h"        // exblas::cpu::Round
// NOTE: like pcg.h / pcg_merged.h this header must NOT include <mpi.h> /
// exblas/mpi_accumulate.h itself (that flips MPI_VERSION mid-TU and breaks
// config.h / the dg::x:: grid aliases). Under an MPI build reduce_mpi_cpu /
// mpi_reduce_communicator are provided transitively by blas.h's MPI dispatch
// and are used below only inside #ifdef MPI_VERSION.

/*!@file
 * s-step / communication-avoiding preconditioned CG (dg::PCGcastep).
 *
 * Motivation: on a small per-GPU problem (z-only domain decomposition) the
 * elliptic solve is bound by the MPI_Allreduce inside every inner product, and
 * profiling shows the cost is a per-collective RANDOM stall tail, not the
 * per-call floor. dg::PCGmerged already collapses the two hot-loop dots into
 * ONE reduction per iteration. This solver goes below one-per-iteration:
 *
 *   - Every s iterations it builds a Krylov "matrix-powers" basis
 *     V = [p, Ahat p, ..., Ahat^s p, r, Ahat r, ..., Ahat^{s-1} r]
 *     with s local SpMVs (halo exchange only -- NO global reduction), then
 *   - computes the full Gram matrix G = V^T W V in a SINGLE global collective,
 *   - advances s CG iterations by tiny local (2s+1)-dim coordinate recurrences.
 *
 * => one MPI_Allreduce per s iterations (s-fold fewer synchronizations).
 *
 * Diagonal preconditioning is handled by the SYMMETRIC SPLIT L = sqrt(P):
 * solve Ahat xhat = L b with Ahat = L A L, then x = L xhat. Since A and the
 * diagonal P are both self-adjoint in the volume-weighted <.,.>_W inner product,
 * plain (single-basis) CA-CG applies and is mathematically identical to
 * dg::PCG. The true residual norm |r|_W that dg::PCG stops on is recovered for
 * free from a second Gram block G2 = V^T (W/P) V folded into the same collective
 * (because r_hat = L r  =>  <r_hat, W/P, r_hat> = <r, W, r>).
 *
 * Trade-offs vs dg::PCG:
 *   - NOT bit-reproducible (a different, mathematically-equivalent recurrence);
 *     every individual inner product is still computed exactly (exblas).
 *   - The monomial basis conditioning degrades with s; s in [2,5] is safe on the
 *     well-conditioned polarization / Helmholtz solves (validated numerically:
 *     iterates track dg::PCG to ~1e-13 at s=4). Default s=4.
 *   - Needs 2s+4 work vectors.
 *   - s local SpMVs are done up front per block even if convergence occurs
 *     mid-block; cheap here (compute is ~8% of solve wall) and the point is the
 *     ONE collective per block.
 *
 * References: Chronopoulos & Gear (1989); Hoemmen, "Communication-avoiding
 * Krylov subspace methods" (PhD thesis, 2010); Carson, "Communication-avoiding
 * Krylov subspace methods in finite precision" (PhD thesis, 2015).
 */

namespace dg{

template< class ContainerType>
class PCGcastep
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>;

    PCGcastep() = default;
    ///@param copyable a container of the size used in the solve
    ///@param max_iterations maximum number of CG iterations
    ///@param s block size (reductions are cut by a factor s); s in [1,8], default 4
    PCGcastep( const ContainerType& copyable, unsigned max_iterations,
               unsigned s = 4):
        m_L(copyable), m_WP(copyable), m_r(copyable), m_p(copyable),
        m_dxhat(copyable), m_tmp(copyable),
        max_iter(max_iterations)
    {
        // env override lets the block size be swept on the cluster without a
        // rebuild (e.g. DG_CASTEP_S=2 ./feltor ...). Ignored if unset/invalid.
        if( const char* e = std::getenv( "DG_CASTEP_S"))
        {
            int v = std::atoi( e);
            if( v > 0) s = (unsigned)v;
        }
        set_block_size( s);
    }

    void set_max( unsigned new_max) {max_iter = new_max;}
    unsigned get_max() const {return max_iter;}
    void set_block_size( unsigned s)
    {
        if( s < 1) s = 1;
        if( s > 8) s = 8;   // monomial basis unstable beyond ~8
        m_s = s;
        m_V.assign( 2*m_s+1, m_r);   // 2s+1 basis vectors, sized like m_r
    }
    unsigned get_block_size() const { return m_s;}
    const ContainerType& copyable()const{ return m_r;}
    void set_verbose( bool verbose){ m_verbose = verbose;}
    void set_throw_on_fail( bool t){ m_throw_on_fail = t;}

    template<class ...Params>
    void construct( Params&& ...ps){ *this = PCGcastep( std::forward<Params>( ps)...); }

    /**
     * @brief Solve \f$ Ax=b\f$ with s-step preconditioned CG.
     *
     * Drop-in replacement for dg::PCG::solve / dg::PCGmerged::solve (same
     * arguments and stopping criterion). @c test_frequency is accepted for
     * signature compatibility but ignored (the residual norm is available every
     * iteration).
     * @return number of iterations used
     */
    template< class MatrixType0, class ContainerType0, class ContainerType1,
              class MatrixType1, class ContainerType2 >
    unsigned solve( MatrixType0&& A, ContainerType0& x, const ContainerType1& b,
            MatrixType1&& P, const ContainerType2& W, value_type eps = 1e-12,
            value_type nrmb_correction = 1, int /*test_frequency*/ = 1);

  private:
    // one fused global reduction of an arbitrary list of weighted inner products
    // g[k] = <a_k, w_k, b_k>. Under MPI all K exact accumulators are packed and
    // reduced in a SINGLE MPI_Allreduce; serially each is rounded locally.
    template<class M>
    std::vector<value_type> fused_dots(
        const std::vector<std::tuple<const ContainerType*, const M*,
                                     const ContainerType*> >& triples) const;

    // small dense host helpers on (Mb x Mb) row-major matrices
    static value_type quad( const std::vector<value_type>& G, int Mb,
        const std::vector<value_type>& u, const std::vector<value_type>& v)
    {   // u^T G v
        value_type t = 0;
        for( int a=0; a<Mb; a++)
        {
            value_type ua = u[a];
            if( ua == 0) continue;
            const value_type* Ga = &G[(size_t)a*Mb];
            for( int c=0; c<Mb; c++) t += ua*Ga[c]*v[c];
        }
        return t;
    }

    ContainerType m_L, m_WP, m_r, m_p, m_dxhat, m_tmp;
    std::vector<ContainerType> m_V;
    unsigned max_iter = 0, m_s = 4;
    bool m_verbose = false, m_throw_on_fail = true;
};

///@cond
template< class ContainerType>
template<class M>
std::vector<get_value_type<ContainerType>>
PCGcastep<ContainerType>::fused_dots(
    const std::vector<std::tuple<const ContainerType*, const M*,
                                 const ContainerType*> >& triples) const
{
    int status = 0;
    const int K = static_cast<int>( triples.size());
    constexpr int NB = exblas::BIN_COUNT;
    std::vector<value_type> res( K);
#ifdef MPI_VERSION
    // pack all K local exact accumulators, reduce them in ONE collective
    std::vector<int64_t> in( (size_t)K*NB), out( (size_t)K*NB, (int64_t)0);
    for( int k=0; k<K; k++)
    {
        const auto& [a,w,b] = triples[k];
        std::vector<int64_t> lk = dg::blas2::detail::doDot_superacc(
                &status, a->data(), w->data(), b->data());
        std::copy( lk.begin(), lk.end(), in.begin() + (size_t)k*NB);
    }
    MPI_Comm comm = std::get<0>(triples[0])->communicator(), comm_mod, comm_red;
    dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_red);
    dg::exblas::reduce_mpi_cpu( K, in.data(), out.data(), comm, comm_mod, comm_red);
    for( int k=0; k<K; k++) res[k] = exblas::cpu::Round( &out[(size_t)k*NB]);
#else
    for( int k=0; k<K; k++)
    {
        const auto& [a,w,b] = triples[k];
        std::vector<int64_t> lk = dg::blas2::detail::doDot_superacc(
                &status, *a, *w, *b);
        res[k] = exblas::cpu::Round( lk.data());
    }
#endif
    if( status != 0 && m_throw_on_fail)
        throw dg::Error( Message(_ping_) << "PCGcastep: NaN/Inf in reduction");
    return res;
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1,
          class Preconditioner, class ContainerType2>
unsigned PCGcastep< ContainerType>::solve(
        Matrix&& A, ContainerType0& x, const ContainerType1& b,
        Preconditioner&& P, const ContainerType2& W,
        value_type eps, value_type nrmb_correction, int)
{
    using M = ContainerType2;
    Matrix& Aref = A;   // bind the forwarding ref once; A is reused every SpMV
    // --- symmetric split of the diagonal preconditioner:  L = sqrt(P) ---
    blas1::transform( P, m_L, dg::SQRT<value_type>());
    blas1::pointwiseDivide( W, P, m_WP);     // W/P  (recovers |r|_W below)

    // r0 = b - A x0  (m_tmp), rhat = L r0 (m_r), phat_0 = rhat_0, correction = 0
    blas2::symv( Aref, x, m_tmp);
    blas1::axpby( 1., b, -1., m_tmp);        // m_tmp = r0 (true residual)
    blas1::pointwiseDot( m_L, m_tmp, m_r);   // m_r = L .* r0 = rhat
    blas1::copy( m_r, m_p);
    blas1::scal( m_dxhat, 0.);

    // --- setup reduction (ONE collective): nrmb^2 = <b,W,b>, res0^2 = <r0,W,r0> ---
    std::vector<std::tuple<const ContainerType*, const M*, const ContainerType*> > st;
    // b/x may be foreign container types; cast is unnecessary because doDot is
    // generic, but the tuple element type is ContainerType* -- so use m_tmp (r0)
    // and a temporary for b. We reduce <r0,W,r0> here and get <b,W,b> separately
    // only if the container type matches; otherwise fall back to blas2::dot.
    value_type nrmb, res2;
    if constexpr( std::is_same_v<std::decay_t<ContainerType1>, ContainerType> )
    {
        st.push_back( std::make_tuple( &b, &W, &b));
        st.push_back( std::make_tuple( (const ContainerType*)&m_tmp, &W,
                                       (const ContainerType*)&m_tmp));
        std::vector<value_type> g = fused_dots( st);
        nrmb = sqrt( g[0]); res2 = g[1];
    }
    else
    {
        nrmb = sqrt( blas2::dot( W, b));
        st.push_back( std::make_tuple( (const ContainerType*)&m_tmp, &W,
                                       (const ContainerType*)&m_tmp));
        std::vector<value_type> g = fused_dots( st);
        res2 = g[0];
    }

    value_type tol = eps*(nrmb + nrmb_correction);
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    if( sqrt( res2) < tol)
        return 0;

    unsigned iter = 0;
    bool converged = false;

    while( iter < max_iter && !converged)
    {
        const unsigned iter_before = iter;
        const int sblk = (int)std::min<unsigned>( m_s, max_iter - iter);
        const int Mb = 2*sblk + 1;

        // ---- matrix-powers kernel: V = [p, Ahat p,...,Ahat^s p,  r, Ahat r,...] ----
        // Ahat v = L .* ( A ( L .* v) ) : one SpMV (halo comm), NO global reduction
        blas1::copy( m_p, m_V[0]);
        for( int i=0; i<sblk; i++)
        {
            blas1::pointwiseDot( m_L, m_V[i], m_tmp);   // tmp = L .* V[i]
            blas2::symv( Aref, m_tmp, m_V[i+1]);        // V[i+1] = A tmp
            blas1::pointwiseDot( m_L, m_V[i+1], m_V[i+1]); // V[i+1] = L .* V[i+1]
        }
        blas1::copy( m_r, m_V[sblk+1]);
        for( int i=0; i<sblk-1; i++)
        {
            int j = sblk+1+i;
            blas1::pointwiseDot( m_L, m_V[j], m_tmp);
            blas2::symv( Aref, m_tmp, m_V[j+1]);
            blas1::pointwiseDot( m_L, m_V[j+1], m_V[j+1]);
        }

        // ---- change-of-basis B (Mb x Mb): Ahat V[:,i] = V B[:,i] (monomial shifts) ----
        std::vector<value_type> Bm( (size_t)Mb*Mb, 0.);
        for( int i=0; i<sblk; i++)        Bm[(size_t)(i+1)*Mb + i] = 1.;    // p-block
        for( int i=0; i<sblk-1; i++)      Bm[(size_t)(sblk+1+i+1)*Mb + (sblk+1+i)] = 1.; // r-block

        // ---- Gram matrices in ONE collective: G = V^T W V, G2 = V^T (W/P) V ----
        std::vector<std::tuple<const ContainerType*, const M*, const ContainerType*> > tr;
        tr.reserve( Mb*(Mb+1));
        for( int a=0; a<Mb; a++) for( int c=a; c<Mb; c++)
            tr.push_back( std::make_tuple( &m_V[a], &W,   &m_V[c]));
        for( int a=0; a<Mb; a++) for( int c=a; c<Mb; c++)
            tr.push_back( std::make_tuple( &m_V[a], &m_WP, &m_V[c]));
        std::vector<value_type> gd = fused_dots( tr);
        std::vector<value_type> G( (size_t)Mb*Mb), G2( (size_t)Mb*Mb);
        int idx = 0;
        for( int a=0; a<Mb; a++) for( int c=a; c<Mb; c++)
        { G[(size_t)a*Mb+c]=G[(size_t)c*Mb+a]=gd[idx]; idx++; }
        for( int a=0; a<Mb; a++) for( int c=a; c<Mb; c++)
        { G2[(size_t)a*Mb+c]=G2[(size_t)c*Mb+a]=gd[idx]; idx++; }

        // ---- inner loop: local (2s+1)-dim coordinate CG recurrences (no comm) ----
        std::vector<value_type> pc(Mb,0.), rc(Mb,0.), xc(Mb,0.), Bp(Mb,0.);
        pc[0] = 1.; rc[sblk+1] = 1.;
        value_type rr_old = quad( G, Mb, rc, rc);   // <rhat,rhat>_W  (= gamma)
        for( int j=0; j<sblk; j++)
        {
            for( int r0=0; r0<Mb; r0++)
            {
                value_type t = 0; const value_type* Br = &Bm[(size_t)r0*Mb];
                for( int c=0; c<Mb; c++) t += Br[c]*pc[c];
                Bp[r0] = t;
            }
            value_type delta = quad( G, Mb, pc, Bp);   // <phat, Ahat phat>_W
            if( !(delta > 0) || !std::isfinite(rr_old) )
            {   // basis breakdown -> reconstruct progress so far and restart block
                break;
            }
            value_type alpha = rr_old/delta;
            for( int t=0; t<Mb; t++){ xc[t]+=alpha*pc[t]; rc[t]-=alpha*Bp[t]; }
            value_type rr_new  = quad( G,  Mb, rc, rc);
            value_type trueres = quad( G2, Mb, rc, rc);  // <r,W,r> (unpreconditioned)
            iter++;
            if( m_verbose)
                DG_RANK0 std::cout << "# PCGcastep i="<<iter<<" |r|_W="
                    <<sqrt(std::max((value_type)0,trueres))<<" tol="<<tol<<"\n";
            if( sqrt( std::max((value_type)0,trueres)) < tol)
            {
                converged = true;
                break;
            }
            value_type beta = rr_new/rr_old;
            for( int t=0; t<Mb; t++) pc[t] = rc[t] + beta*pc[t];
            rr_old = rr_new;
        }

        // ---- reconstruct: dxhat += V xc ; p = V pc ; r = V rc ----
        for( int t=0; t<Mb; t++)
            if( xc[t] != 0.) blas1::axpby( xc[t], m_V[t], 1., m_dxhat);
        if( !converged)
        {
            blas1::scal( m_p, 0.);
            blas1::scal( m_r, 0.);
            for( int t=0; t<Mb; t++)
            {
                if( pc[t] != 0.) blas1::axpby( pc[t], m_V[t], 1., m_p);
                if( rc[t] != 0.) blas1::axpby( rc[t], m_V[t], 1., m_r);
            }
        }
        if( iter == iter_before)   // immediate breakdown, no progress -> give up
            break;                 // (handled by throw/return below)
    }

    // x = x0 + L .* dxhat
    blas1::pointwiseDot( 1., m_L, m_dxhat, 1., x);

    if( !converged && m_throw_on_fail)
        throw dg::Fail( tol, Message(_ping_)
            <<"After "<<max_iter<<" PCGcastep iterations with rtol "<<eps
            <<" and atol "<<eps*nrmb_correction );
    return iter;
}
///@endcond

} //namespace dg

#endif //_DG_PCG_CASTEP_
