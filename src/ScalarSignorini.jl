"""
    ScalarSignorini

Scalar Signorini problem in dual-mixed form, discretised as a mixed variational
inequality on a convex cone, **without a Lagrange multiplier**.

Continuous problem (Wang, Hu & Zhou, JCAM 2024, eq. (3)):

    σ - ∇u = 0,   div σ + f = 0          in Ω,
    u = u_D                               on Γ_D,
    u ≥ 0,  σ·n ≥ 0,  u (σ·n) = 0         on Γ_S.

`solve_scalar_signorini` also accepts, as optional data, a reaction term
(-div σ + c u = f, as in Führer–Heuer–Stephan 2016), an obstacle χ on Γ_S
(u ≥ χ, (u - χ) σ·n = 0, as in Bartels–Gudi–Kaltenbach 2024), a Neumann
boundary Γ_N (σ·n = g) and no Γ_D at all.  The cone and the active set are the
same in every case.

Discrete problem (MVI)_h: find (σ_h, u_h) ∈ K_h × Q_h with

    (σ_h, τ-σ_h) + (u_h, div(τ-σ_h)) ≥ ⟨u_D, (τ-σ_h)·n⟩_{Γ_D}   ∀ τ ∈ K_h,
    (div σ_h, v)                       = -(f, v)                  ∀ v ∈ Q_h,

with Σ_h = BDM_{k+1}, Q_h = P_k (discontinuous), and the convex cone

    K_h = { τ ∈ Σ_h :  every Bernstein coefficient of τ·n|_F is ≥ 0,  ∀ F ⊂ Γ_S }.

By the convex-hull property (see `BernsteinTrace`), K_h ⊂ {τ·n ≥ 0 on Γ_S}
pointwise: the discrete cone is conforming for every degree, and exact for
k = 0 (linear normal traces).  This is the scalar counterpart of the
`:coefficient` active set of GridapContactV2.jl
(`wip/ThermoContactFEM/src2d/CoefficientContact.jl`).

## How the cone is imposed

On each facet F ⊂ Γ_S the normal trace σ_h·n|_F ∈ P_{k+1}(F) is determined by
the k+2 BDM dofs a_F owned by F.  With W_F the map a_F ↦ c_F (Bernstein
coefficients of σ_h·n|_F), the dofs are re-parametrised as a_F = W_F⁻¹ c_F.
Every coefficient c_{F,i} is either

  * **in contact** (free, its constraint inactive), or
  * **separated** (fixed to c_{F,i} = 0, its constraint active).

All of it is one sparse matrix T with x = T x̃, applied to the assembled
(unconstrained) saddle-point system K x = b:  (Tᵀ K T) x̃ = Tᵀ b.

## The active set

The KKT conditions of (MVI)_h for the bound constraints c ≥ 0 read

    c ≥ 0,   μ ≥ 0,   c_i μ_i = 0,     μ = Tᵀ_c (K x - b),

where μ_{F,i} is the algebraic multiplier of c_{F,i} ≥ 0: the residual of the
full system in the direction of the flux whose normal trace on F is B_i.  Its
continuous counterpart is ∫_F u B_i (≥ 0 is the discrete "u ≥ 0").  It is
exactly zero on coefficients in contact.  The primal-dual active set (PDAS)
update is therefore, coefficient by coefficient,

    in contact  → stays unless   c_i < -tol·max|c|    (flux would be negative),
    separated   → enters iff     μ_i < -tol·max|μ|    (u would be negative),

i.e. c_i = max(0, c_i - ρ μ_i) with a round-off guard; no ϱ is needed because
each side is decided on its own variable.  The iteration stops when the set
does not change; a repeated set is reported as `converged = false`.
"""
module ScalarSignorini

using Gridap
using Gridap.FESpaces
using Gridap.CellData
using Gridap.Arrays
using Gridap.Algebra
using FillArrays
using LinearAlgebra
using SparseArrays

using ..BernsteinTrace

export SignoriniFacets, signorini_facets, cone_transform, cone_multipliers,
       solve_scalar_signorini

# -----------------------------------------------------------------------------
# Facet data
# -----------------------------------------------------------------------------

"""
    SignoriniFacets

For each facet of Γ_S (in the cell order of its `BoundaryTriangulation`):

  * `dofs[F]`  : the `p+1` free BDM dofs owned by F (those with a non-zero
                 normal trace on F);
  * `W[F]`     : `c = W[F] * x[dofs[F]]`, Bernstein coefficients of σ_h·n|_F;
  * `Winv[F]`  : its inverse, `x[dofs[F]] = Winv[F] * c`;
  * `p`        : the degree of the normal trace (= BDM degree = k+1).
"""
struct SignoriniFacets
  dofs :: Vector{Vector{Int}}
  W    :: Vector{Matrix{Float64}}
  Winv :: Vector{Matrix{Float64}}
  p    :: Int
end

Base.length(sf::SignoriniFacets) = length(sf.dofs)

"""
    signorini_facets(Σ, ΓS, p) -> SignoriniFacets

Build the facet data from the BDM space `Σ` of degree `p` and the boundary
triangulation `ΓS`.  The normal traces of the basis functions are *evaluated*
at the Bernstein nodes of each facet instead of being read from the dof
definitions, so nothing is assumed about Gridap's moment functionals, their
orientation or the local dof order.
"""
function signorini_facets(Σ, ΓS, p::Int)
  nS    = get_normal_vector(ΓS)
  nodes = bernstein_nodes(p)
  TB    = bernstein_transform(nodes)
  nF    = num_cells(ΓS)

  φn   = get_fe_basis(Σ) ⋅ nS
  vals = collect(lazy_map(evaluate,
                          get_data(change_domain(φn, ReferenceDomain())),
                          Fill(nodes, nF)))
  ids  = collect(get_cell_dof_ids(Σ, ΓS))

  dofs = Vector{Int}[];  W = Matrix{Float64}[];  Winv = Matrix{Float64}[]
  for F in 1:nF
    Vf    = vals[F]                              # nodes × local dofs of the cell
    scale = maximum(abs, Vf)
    own   = [j for j in 1:size(Vf, 2) if maximum(abs, view(Vf, :, j)) > 1e-10 * scale]
    length(own) == p + 1 ||
      error("Γ_S facet $F owns $(length(own)) dofs with a normal trace, expected $(p+1)")
    d = ids[F][own]
    all(>(0), d) || error("Γ_S facet $F carries a Dirichlet dof")
    WF = TB * Vf[:, own]                         # Bernstein coefficients per dof
    cond(WF) < 1e10 || error("Γ_S facet $F: normal-trace map is singular")
    push!(dofs, d);  push!(W, WF);  push!(Winv, inv(WF))
  end
  SignoriniFacets(dofs, W, Winv, p)
end

# -----------------------------------------------------------------------------
# Cone parametrisation and multipliers
# -----------------------------------------------------------------------------

"""
    cone_transform(nfree, sf, state, off) -> T

Sparse `T` (`nfree × ncols`) with `x = T x̃`.  The dofs owned by Γ_S facets are
replaced by the Bernstein coefficients in contact (`state[F][i] == true`);
separated coefficients are fixed to zero.  Every other dof is kept (identity).
`off` is the offset of the flux block in the multi-field free-dof vector.
"""
function cone_transform(nfree::Int, sf::SignoriniFacets, state, off::Int)
  replaced = falses(nfree)
  for F in 1:length(sf), d in sf.dofs[F]
    replaced[off + d] = true
  end
  I = Int[];  J = Int[];  V = Float64[];  col = 0
  for i in 1:nfree
    replaced[i] && continue
    col += 1;  push!(I, i);  push!(J, col);  push!(V, 1.0)
  end
  for F in 1:length(sf), i in 1:(sf.p + 1)
    state[F][i] || continue                      # separated: c_{F,i} = 0
    col += 1
    for (kd, d) in enumerate(sf.dofs[F])
      push!(I, off + d);  push!(J, col);  push!(V, sf.Winv[F][kd, i])
    end
  end
  sparse(I, J, V, nfree, col)
end

"""
    cone_multipliers(sf, r, off) -> Vector{Vector{Float64}}

Multiplier μ_{F,i} of each constraint c_{F,i} ≥ 0: the residual `r = K x - b`
of the full system in the direction of the column `cone_transform` gives to
c_{F,i} when it is free.  Zero (to round-off) on coefficients in contact.
"""
cone_multipliers(sf::SignoriniFacets, r::AbstractVector, off::Int) =
  [transpose(sf.Winv[F]) * r[off .+ sf.dofs[F]] for F in 1:length(sf)]

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

"""
    solve_scalar_signorini(model; k, f, tag_S, uD, tag_D, kwargs...) -> NamedTuple

Solve (MVI)_h on `model` with Σ_h × Q_h = BDM_{k+1} × P_k by the coefficient
PDAS described in the module docstring.

Arguments
  * `k`             polynomial degree of u_h (k ≥ 0);
  * `f`             source (function of x);
  * `tag_S`         face-labelling tag of Γ_S;
  * `uD`, `tag_D`   Dirichlet datum and tag of Γ_D (`nothing` if there is no Γ_D).

Optional data (defaults reproduce the problem of the module docstring)
  * `reaction`      c ≥ 0 in  -div σ + c u = f  (default 0): adds -(c u, v) to
                    the second equation;
  * `χ`             obstacle on Γ_S, u ≥ χ (function of x, default `nothing` = 0):
                    adds ⟨χ, τ·n⟩_{Γ_S} to the right-hand side; the multipliers
                    become the discrete counterpart of ∫_F (u - χ) B_i;
  * `tag_N`, `gN`   Neumann boundary σ·n = g, imposed essentially on the BDM
                    space; `gN` is a vector field whose normal trace is g
                    (default: no Neumann boundary).

Keywords
  * `qdegree`       quadrature degree (default `2k + 6`);
  * `maxit`         maximum number of PDAS iterations (default 100);
  * `tol`           relative round-off guard of the PDAS update (default 1e-10);
  * `init_contact`  initial state of every coefficient.  Default `false`: every
                    coefficient separated, i.e. σ_h·n = 0 on Γ_S (homogeneous
                    Neumann).  `true` starts from σ_h·n free, i.e. u_h = 0 on
                    Γ_S weakly.  Both reach the same set; on the benchmark of
                    the tests `false` needs fewer iterations.

Returns `(σh, uh, Ω, dΩ, ΓS, dΓS, nS, X, facets, state, c, μ, it, converged,
ncontact, ndofs)`, where `c[F]`, `μ[F]` are the Bernstein coefficients of
σ_h·n and the multipliers on each Γ_S facet, and `ncontact[it]` the number of
coefficients in contact at each iteration.
"""
function solve_scalar_signorini(model; k::Int, f, tag_S, uD = nothing, tag_D = nothing,
                                reaction::Real = 0.0, χ = nothing,
                                tag_N = nothing, gN = nothing,
                                qdegree::Int = 2k + 6, maxit::Int = 100,
                                tol::Real = 1e-10, init_contact::Bool = false)
  p = k + 1                                       # BDM degree = normal-trace degree
  has_D = tag_D !== nothing
  has_D && uD === nothing && error("tag_D given without uD")
  tag_N !== nothing && gN === nothing && error("tag_N given without gN")

  Ω   = Triangulation(model);                       dΩ  = Measure(Ω, qdegree)
  ΓS  = BoundaryTriangulation(model, tags = tag_S); dΓS = Measure(ΓS, qdegree)
  nS  = get_normal_vector(ΓS)

  # Γ_D is natural in the mixed form, Γ_N essential (σ·n = g), and the traces
  # on Γ_S are parametrised by the cone transform.
  reffe_σ = ReferenceFE(bdm, Float64, p)
  Σ_ = tag_N === nothing ?
       TestFESpace(model, reffe_σ, conformity = :HDiv) :
       TestFESpace(model, reffe_σ, conformity = :HDiv, dirichlet_tags = tag_N)
  Q_ = TestFESpace(model, ReferenceFE(lagrangian, Float64, k), conformity = :L2)
  Σ  = tag_N === nothing ? TrialFESpace(Σ_) : TrialFESpace(Σ_, gN)
  Q  = TrialFESpace(Q_)
  Y  = MultiFieldFESpace([Σ_, Q_]);  X = MultiFieldFESpace([Σ, Q])

  function a((σ, u), (τ, v))
    r = ∫(σ ⋅ τ + u * (∇ ⋅ τ) + v * (∇ ⋅ σ))dΩ
    reaction == 0 ? r : r + ∫(-(reaction * u * v))dΩ
  end
  function l((τ, v))
    r = ∫(-(f * v))dΩ
    if has_D
      ΓD = BoundaryTriangulation(model, tags = tag_D)
      r = r + ∫((τ ⋅ get_normal_vector(ΓD)) * uD)Measure(ΓD, qdegree)
    end
    χ === nothing ? r : r + ∫((τ ⋅ nS) * χ)dΓS
  end

  # The unconstrained operator does not depend on the active set: assembled once.
  op = AffineFEOperator(a, l, X, Y)
  K  = get_matrix(op);  b = get_vector(op)
  nfree = num_free_dofs(X)

  sf    = signorini_facets(Σ_, ΓS, p)
  state = [fill(init_contact, p + 1) for _ in 1:length(sf)]
  seen  = Set{Vector{Vector{Bool}}}()
  ncontact = Int[]

  x = zeros(nfree);  c = Vector{Float64}[];  μ = Vector{Float64}[]
  converged = false;  it = 0
  for outer it in 1:maxit
    push!(seen, deepcopy(state))
    push!(ncontact, sum(count, state))

    T  = cone_transform(nfree, sf, state, 0)
    Kt = sparse(transpose(T) * K * T)
    xt = lu(Kt) \ (transpose(T) * b)
    x  = T * xt

    r  = K * x - b
    c  = [sf.W[F] * x[sf.dofs[F]] for F in 1:length(sf)]
    μ  = cone_multipliers(sf, r, 0)

    sc = max(maximum(maximum(abs, cF) for cF in c), eps())
    sμ = max(maximum(maximum(abs, μF) for μF in μ), eps())
    new = [[state[F][i] ? (c[F][i] ≥ -tol * sc) : (μ[F][i] < -tol * sμ)
            for i in 1:p+1] for F in 1:length(sf)]

    if new == state
      converged = true
      break
    end
    if new in seen                                  # the PDAS is cycling
      @warn "coefficient PDAS cycling at iteration $it"
      break
    end
    state = new
  end
  converged || @warn "coefficient PDAS did not converge (it = $it)"

  σh, uh = FEFunction(X, x)
  return (; σh, uh, Ω, dΩ, ΓS, dΓS, nS, X, facets = sf, state, c, μ, it,
            converged, ncontact, ndofs = nfree)
end

end # module ScalarSignorini
