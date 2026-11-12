"""
    StenbergPostprocess

Local postprocessing (2.10) of

    I. Muga, S. Rojas, P. Vega, "An adaptive superconvergent mixed finite
    element method based on local residual minimization", SIAM J. Numer. Anal.
    61(5) (2023) 2084–2105, Section 2.4.

Notation of the paper: q = -∇u is the flux, (u_h, q_h) ∈ V_h^{p-1} × H_h^p a
mixed approximation (p ≥ 1 is the BDM degree).  The postprocessed scalar
ũ_h ∈ V_h^{p+1} solves, on each element K,

    (∇ũ_h, ∇v)_K = -(q_h, ∇v)_K      ∀ v ∈ V_{⋆,K}^{p+1},     (2.10a)
    (ũ_h, 1)_K   = (u_h, 1)_K.                                 (2.10b)

This is the *modified* variant of Stenberg's postprocessing used in the paper
(cf. Lovadina–Stenberg, eqs. (2.32)–(2.33)): the right-hand side of (2.10a) is
-(q_h, ∇v)_K, not Stenberg's original (f, v)_K - ⟨q_h·n, v⟩_{∂K}.

By Proposition 3.1 of the paper, ũ_h coincides with the residual-minimization
postprocess ν_h of (3.2) (`ResidualMinimization`): testing (3.2a) with
v ∈ V_{⋆,K}^{p+1} ⊂ V_{⋆,K}^{p+2} and using (3.2b) gives (2.10a).

## Implementation

As in `ResidualMinimization`, the zero-mean test space is not built.  (2.10) is
posed on the full P_{p+1}(K) with one scalar multiplier ξ_K for (2.10b):

    [ A    m ] [ũ]   [ -(q_h, ∇w)_K ]
    [ mᵀ   0 ] [ξ] = [  (u_h, 1)_K  ]

with A = (∇ũ, ∇w)_K and m = (w, 1)_K.  Testing with w = 1 gives ξ_K = 0, so the
rows for w ∈ P_{p+1}(K) reduce to (2.10a).  Each element is solved densely from
Gridap's cell matrices; no global system is assembled.
"""
module StenbergPostprocess

using Gridap
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Arrays
using LinearAlgebra

export stenberg_postprocess

"""
    stenberg_postprocess(model, qh, uh, p; qdegree) -> (; ũh, V)

Solve the local problems (2.10) of Muga–Rojas–Vega (2023) on every cell of
`model`.

  * `qh` : the discrete flux in the paper's convention, q_h ≈ -∇u (a `CellField`);
  * `uh` : the discrete scalar variable (a `CellField`), only its cell means are used;
  * `p`  : the BDM degree of `qh` (u_h ∈ P_{p-1}).

Returns ũ_h ∈ P_{p+1} as an `FEFunction` on the discontinuous Lagrangian space
`V` (the same space as `Vν` of `residual_minimization_postprocess`).
`qdegree` (default `2(p+2)`, as in `ResidualMinimization`) integrates every term
exactly.
"""
function stenberg_postprocess(model, qh, uh, p::Int; qdegree::Int = 2 * (p + 2))
  p ≥ 1 || error("the BDM degree p must be ≥ 1")
  Ω  = Triangulation(model)
  dΩ = Measure(Ω, qdegree)

  V = TestFESpace(model, ReferenceFE(lagrangian, Float64, p + 1), conformity = :L2)
  w = get_fe_basis(V);  e = get_trial_fe_basis(V)

  cellvals(dc) = collect(get_contribution(dc, Ω))
  A = cellvals(∫(∇(w) ⋅ ∇(e))dΩ)              # (∇ũ, ∇w)_K       n × n
  m = cellvals(∫(w)dΩ)                         # (w, 1)_K         n
  r = cellvals(∫(-(qh ⋅ ∇(w)))dΩ)              # -(q_h, ∇w)_K     n
  ū = cellvals(∫(uh)dΩ)                        # (u_h, 1)_K       scalar

  ids = get_cell_dof_ids(V)
  x   = zeros(num_free_dofs(V))

  for K in 1:num_cells(Ω)
    n = length(m[K])
    M = zeros(n + 1, n + 1)
    M[1:n, 1:n] = A[K]
    M[1:n, n+1] = m[K];  M[n+1, 1:n] = m[K]

    rhs = zeros(n + 1)
    rhs[1:n] = r[K]
    rhs[n+1] = ū[K]

    x[ids[K]] = (M \ rhs)[1:n]
  end

  return (; ũh = FEFunction(V, x), V)
end

end # module StenbergPostprocess
