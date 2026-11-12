"""
    BernsteinTrace

Polynomial traces on boundary facets written in the Bernstein basis.

On a straight facet `F`, parametrised by the reference coordinate `s ∈ [0,1]`,
a polynomial of degree `p` is written

    φ(s) = Σ_{i=0}^{p} c_i B_i^p(s),      B_i^p(s) = C(p,i) sᶦ (1-s)^(p-i).

Because the `B_i^p` are non-negative and sum to one, the coefficients bound
the polynomial (convex-hull property):

    min_i c_i ≤ φ(s) ≤ max_i c_i      for every s ∈ [0,1].

Hence `c_i ≥ 0 ∀i` is a *certificate* that `φ ≥ 0` on the whole facet, for any
degree `p`.  This is what lets the Signorini cone be imposed pointwise for every
polynomial degree (see `ScalarSignorini`): for `p = 1` the condition is exact
(a linear function is ≥ 0 on a segment iff its two end values are), for `p ≥ 2`
it is a conforming (conservative) inner approximation of the cone.

The change of basis "values at the `p+1` Lagrange nodes of the segment →
Bernstein coefficients" is one small matrix per degree, built from the actual
node coordinates so that nothing depends on Gridap's node ordering.

This mirrors `wip/ThermoContactFEM/src2d/BernsteinTrace.jl` of GridapContactV2.jl.
"""
module BernsteinTrace

using Gridap
using Gridap.ReferenceFEs
using Gridap.CellData
using Gridap.Arrays
using FillArrays
using LinearAlgebra

export bernstein_nodes, bernstein_transform, bernstein_basis,
       facet_bernstein_coeffs

"""
    bernstein_nodes(p) -> Vector{Point{1}}

The `p+1` sample points on the reference segment: the Lagrange nodes of degree `p`.
"""
bernstein_nodes(p::Int) =
  get_node_coordinates(ReferenceFE(SEGMENT, lagrangian, Float64, p))

"""
    bernstein_basis(p, s) -> Vector{Float64}

Values `[B_0^p(s), …, B_p^p(s)]` of the Bernstein basis of degree `p` at `s ∈ [0,1]`.
"""
bernstein_basis(p::Int, s::Real) = [binomial(p, i) * s^i * (1 - s)^(p - i) for i in 0:p]

"""
    bernstein_transform(nodes) -> Matrix{Float64}

The `(p+1)×(p+1)` matrix `A⁻¹` that maps the values of a degree-`p` polynomial at
`nodes` to its Bernstein coefficients (`A[j,i] = B_i^p(s_j)`).  The ordering of
`nodes` is irrelevant: `A` is built from the coordinates themselves.
"""
function bernstein_transform(nodes::AbstractVector)
  p = length(nodes) - 1
  A = [binomial(p, i) * nodes[j][1]^i * (1 - nodes[j][1])^(p - i)
       for j in 1:p+1, i in 0:p]
  cond(A) > 1e10 && @warn "Bernstein transform ill-conditioned at degree $p" cond(A)
  inv(A)
end

"""
    facet_bernstein_coeffs(fh, p) -> Vector{Vector{Float64}}

Bernstein coefficients of degree `p` of the facet field `fh` on every cell of the
(boundary) triangulation it lives on.  `fh` must be a polynomial of degree ≤ `p`
along each facet; then the transform is exact.  The field is read in the
reference domain of the facet, which is where the Bernstein nodes live.
"""
function facet_bernstein_coeffs(fh, p::Int;
                                nodes = bernstein_nodes(p),
                                transform = bernstein_transform(nodes))
  fr  = change_domain(fh, ReferenceDomain())
  cw  = get_data(fr)
  val = lazy_map(evaluate, cw, Fill(nodes, length(cw)))
  [transform * collect(v) for v in val]
end

end # module BernsteinTrace
