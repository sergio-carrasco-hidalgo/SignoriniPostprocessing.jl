"""
    ResidualMinimization

Postprocessing based on local residual minimization, Section 3.1 of

    I. Muga, S. Rojas, P. Vega, "An adaptive superconvergent mixed finite
    element method based on local residual minimization", SIAM J. Numer. Anal.
    61(5) (2023) 2084–2105.

Notation of the paper: q = -∇u is the flux, (u_h, q_h) ∈ V_h^{p-1} × H_h^p a
mixed approximation (p ≥ 1 is the BDM degree).  The postprocessed scalar
ν_h ∈ V_h^{p+1} is defined element by element through (3.1),

    ν_h|_K = argmin_{w_K ∈ V_K^{p+1}}  ½ ‖q_h + ∇w_K‖²_{*,K},     (ν_h, 1)_K = (u_h, 1)_K,

with ‖·‖_{*,K} the discrete dual norm (2.2)–(2.3) over ∇V_{⋆,K}^{p+2}.  This is
equivalent to the local mixed system (3.2): find (ε_K, ν_K) ∈ V_{⋆,K}^{p+2} × V_K^{p+1}

    (∇ε_K, ∇v)_K + (∇ν_K, ∇v)_K = -(q_h, ∇v)_K     ∀ v ∈ V_{⋆,K}^{p+2},   (3.2a)
    (∇w, ∇ε_K)_K                = 0                ∀ w ∈ V_{⋆,K}^{p+1},   (3.2b)
    (ν_K, 1)_K                  = (u_h, 1)_K.                              (3.2c)

ε_K is the Riesz representative of the local residual (Remark 3.4) and, by
Proposition 3.1, ν_h coincides with Stenberg's postprocessing (2.10).

## Implementation

The zero-mean spaces V_⋆ are not built explicitly.  Instead, (3.2) is posed on
the full spaces P_{p+2}(K) × P_{p+1}(K) with two scalar multipliers per element,
λ_K for (ε_K, 1)_K = 0 and ξ_K for (3.2c):

    [ A_εε   B_εν   m_ε   0  ] [ε]   [ -(q_h, ∇v) ]
    [ B_ενᵀ  0      0     m_ν] [ν] = [     0      ]
    [ m_εᵀ   0      0     0  ] [λ]   [     0      ]
    [ 0      m_νᵀ   0     0  ] [ξ]   [ (u_h, 1)_K ]

Testing with v = 1 and w = 1 gives λ_K = ξ_K = 0, so the rows for v ∈ P_{p+2}
and w ∈ P_{p+1} reduce to (3.2a)–(3.2b).  The system is non-singular (take
v = ε_K, w = ν_K).  It is solved densely, one element at a time, from the cell
matrices Gridap provides: no global system is assembled.
"""
module ResidualMinimization

using Gridap
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Arrays
using LinearAlgebra

export residual_minimization_postprocess

"""
    residual_minimization_postprocess(model, qh, uh, p; qdegree) -> (; νh, εh, Vν, Vε)

Solve the local problems (3.2) of Muga–Rojas–Vega (2023) on every cell of
`model`.

  * `qh` : the discrete flux in the paper's convention, q_h ≈ -∇u (a `CellField`);
  * `uh` : the discrete scalar variable (a `CellField`), only its cell means are used;
  * `p`  : the BDM degree of `qh` (u_h ∈ P_{p-1}).

Returns ν_h ∈ P_{p+1} and ε_h ∈ P_{p+2} (zero mean on every element) as
`FEFunction`s on discontinuous Lagrangian spaces, together with those spaces.
`qdegree` (default `2(p+2)`) integrates every term of (3.2) exactly.
"""
function residual_minimization_postprocess(model, qh, uh, p::Int;
                                           qdegree::Int = 2 * (p + 2))
  p ≥ 1 || error("the BDM degree p must be ≥ 1")
  Ω  = Triangulation(model)
  dΩ = Measure(Ω, qdegree)

  Vε = TestFESpace(model, ReferenceFE(lagrangian, Float64, p + 2), conformity = :L2)
  Vν = TestFESpace(model, ReferenceFE(lagrangian, Float64, p + 1), conformity = :L2)
  vε = get_fe_basis(Vε);  eε = get_trial_fe_basis(Vε)
  wν = get_fe_basis(Vν);  eν = get_trial_fe_basis(Vν)

  cellvals(dc) = collect(get_contribution(dc, Ω))
  Aεε = cellvals(∫(∇(vε) ⋅ ∇(eε))dΩ)          # (∇ε, ∇v)_K        nε × nε
  Bεν = cellvals(∫(∇(vε) ⋅ ∇(eν))dΩ)          # (∇ν, ∇v)_K        nε × nν
  mε  = cellvals(∫(vε)dΩ)                     # (v, 1)_K          nε
  mν  = cellvals(∫(wν)dΩ)                     # (w, 1)_K          nν
  rε  = cellvals(∫(-(qh ⋅ ∇(vε)))dΩ)          # -(q_h, ∇v)_K      nε
  ū   = cellvals(∫(uh)dΩ)                     # (u_h, 1)_K        scalar

  idsε = get_cell_dof_ids(Vε)
  idsν = get_cell_dof_ids(Vν)
  xε = zeros(num_free_dofs(Vε))
  xν = zeros(num_free_dofs(Vν))

  for K in 1:num_cells(Ω)
    nε = length(mε[K]);  nν = length(mν[K]);  n = nε + nν + 2
    iε = 1:nε;  iν = nε .+ (1:nν);  iλ = nε + nν + 1;  iξ = nε + nν + 2

    M = zeros(n, n)
    M[iε, iε] = Aεε[K]
    M[iε, iν] = Bεν[K];  M[iν, iε] = transpose(Bεν[K])
    M[iε, iλ] = mε[K];   M[iλ, iε] = mε[K]
    M[iν, iξ] = mν[K];   M[iξ, iν] = mν[K]

    rhs = zeros(n)
    rhs[iε] = rε[K]
    rhs[iξ] = ū[K]

    sol = M \ rhs
    xε[idsε[K]] = sol[iε]
    xν[idsν[K]] = sol[iν]
  end

  return (; νh = FEFunction(Vν, xν), εh = FEFunction(Vε, xε), Vν, Vε)
end

end # module ResidualMinimization
