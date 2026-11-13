"""
    Estimators

A posteriori error indicators for the scalar Signorini problem, built on the
postprocessing (3.2) of Muga, Rojas & Vega (SIAM J. Numer. Anal. 2023).

They do not use the momentum equation (no residual f - div q_h): only the
constitutive law q = -∇u, the conformity of ν_h and the boundary conditions,
in the spirit of the improved estimator (3.9) of the paper.  With q_h = -σ_h
and, per element K,

    η_K² = ‖∇ε_K‖²_K                                    built-in, Remark 3.4
         + ‖σ_h - ∇ν_h‖²_K                              constitutive mismatch
         + ½ Σ_{F ⊂ ∂K interior} h_F⁻¹ ‖[[ν_h]]‖²_F       non-conformity of ν_h
         + Σ_{F ⊂ ∂K ∩ Γ_D}      h_F⁻¹ ‖u_D - ν_h‖²_F       Dirichlet data
         + Σ_{F ⊂ ∂K ∩ Γ_S} ( h_F⁻¹ ‖β_F^{1/2} (ν_h - χ)‖²_F               contact: u = χ
                              + h_F⁻¹ ‖(1-β_F)^{1/2} min(ν_h - χ, 0)‖²_F )  separation: u ≥ χ

The first four lines are (3.9).  On Γ_S the trace of u is unknown, so the
Dirichlet term of (3.9) is replaced by the contact terms:

  * β_F = Σ_{i in contact} B_i is the Bernstein partition of unity restricted to
    the coefficients of σ_h·n|_F that the active set put in contact.  On a facet
    fully in contact (β_F = 1) the term is the Dirichlet one with u = χ; on a
    separated facet (β_F = 0) only the violation of u ≥ χ counts; a facet that
    the transition point cuts is weighted coefficient by coefficient;
  * no term for the sign of σ_h·n is needed: the Bernstein cone gives
    σ_h·n ≥ 0 at every point of Γ_S;
  * on a Neumann boundary Γ_N (σ·n = g imposed exactly) there is no term.

The complementarity |⟨σ_h·n, ν_h - χ⟩_F| is computed as a diagnostic but is
NOT part of η_K by default (`complementarity = false`): it is a product of an
O(1) factor and an O(error) factor, not quadratic in the error, so it decays
more slowly than η² and makes the estimator inefficient (measured on the smooth
test benchmark: with it, η/E grows from 0.83 to 1.61 at k = 2; without it, it
stays in 0.83-0.86).

The reliability of this estimator for the variational inequality is not
proven; Theorem 3.9 of the paper covers the linear problem (3.9) only.
"""
module Estimators

using Gridap
using Gridap.CellData
using Gridap.Arrays
using FillArrays
using LinearAlgebra

using ..BernsteinTrace

export contact_estimator

"Gauss–Legendre nodes and weights on [0, 1] (Golub–Welsch)."
function gauss01(n::Int)
  β = [i / sqrt(4i^2 - 1) for i in 1:n-1]
  E = eigen(SymTridiagonal(zeros(n), β))
  (E.values .+ 1) ./ 2, E.vectors[1, :] .^ 2
end

"Values of the facet field `fh` (living on, or restricted to, `Γ`) at the reference points `s`."
facet_values(fh, Γ, s) =
  collect(lazy_map(evaluate, get_data(change_domain(fh, Γ, ReferenceDomain())),
                   Fill([Point(si) for si in s], num_cells(Γ))))

"""
    contact_estimator(model, sol, νh, εh; k, uD, tag_D, χ, complementarity) -> (; η2, parts, cells)

Element indicators of the estimator described in the module docstring.

  * `sol`      the output of `solve_scalar_signorini` (σ_h, Γ_S, active set);
  * `νh`, `εh` the output of `residual_minimization_postprocess`;
  * `uD`, `tag_D` Dirichlet datum and tag (`nothing` if there is no Γ_D);
  * `χ`        the obstacle on Γ_S (`nothing` = 0);
  * `complementarity` include |⟨σ_h·n, ν_h - χ⟩_F| in η_K² (default `false`;
                 it is always returned in `parts` and `cells`).

Returns `η2` (η_K² per cell, for marking), `parts` (the global sum of each
contribution: `tilde`, `flux`, `jump`, `dirichlet`, `active`, `inactive`,
`complementarity`) and `cells` (the same, per cell).  `parts` and `cells` also
carry alternative contact terms, for comparison only (not in `η2`):
`active_facet`/`inactive_facet` (facet-wise classification, no Bernstein
weighting), `active_sigma`/`inactive_sigma` (classification by σ_h·n > 0) and
`flux_sep` = Σ h_F ‖σ_h·n‖²_{F ∩ {ν_h > χ}}.
"""
function contact_estimator(model, sol, νh, εh; k::Int, uD = nothing, tag_D = nothing,
                           χ = nothing, complementarity::Bool = false,
                           qdegree::Int = 2k + 8, nq::Int = k + 6)
  Ω  = sol.Ω;  dΩ = Measure(Ω, qdegree);  nc = num_cells(Ω)
  contrib(dc, trian) = collect(get_contribution(dc, trian))

  # --- cell terms ---------------------------------------------------------------
  tilde = contrib(∫(∇(εh) ⋅ ∇(εh))dΩ, Ω)
  flux  = contrib(∫((sol.σh - ∇(νh)) ⋅ (sol.σh - ∇(νh)))dΩ, Ω)

  # --- interior jumps of ν_h, half to each neighbour --------------------------------
  Λ  = SkeletonTriangulation(model);  dΛ = Measure(Λ, qdegree)
  jF = contrib(∫(jump(νh) * jump(νh))dΛ, Λ) ./ contrib(∫(1.0)dΛ, Λ)
  jump2 = zeros(nc)
  for (F, (cp, cm)) in enumerate(zip(Λ.plus.glue.face_to_cell, Λ.minus.glue.face_to_cell))
    jump2[cp] += jF[F] / 2;  jump2[cm] += jF[F] / 2
  end

  # --- Dirichlet boundary ---------------------------------------------------------------
  dir2 = zeros(nc)
  if tag_D !== nothing
    ΓD = BoundaryTriangulation(model, tags = tag_D);  dΓD = Measure(ΓD, qdegree)
    dF = contrib(∫((uD - νh) * (uD - νh))dΓD, ΓD) ./ contrib(∫(1.0)dΓD, ΓD)
    for (F, c) in enumerate(ΓD.glue.face_to_cell)
      dir2[c] += dF[F]
    end
  end

  # --- contact boundary, by Gauss quadrature on the reference facet ------------------
  ΓS = sol.ΓS;  p = k + 1
  s, w = gauss01(nq)
  B  = [binomial(p, i) * sq^i * (1 - sq)^(p - i) for sq in s, i in 0:p]     # nq × (p+1)
  hS = contrib(∫(1.0)Measure(ΓS, 2), ΓS)
  νS = facet_values(νh, ΓS, s)
  σn = facet_values(sol.σh ⋅ sol.nS, ΓS, s)
  χS = χ === nothing ? [zeros(nq) for _ in 1:num_cells(ΓS)] : facet_values(CellField(χ, ΓS), ΓS, s)
  act2 = zeros(nc);  ina2 = zeros(nc);  comp = zeros(nc)
  # alternative contact terms, returned as diagnostics only (not in η2):
  #   *_facet  a facet is in contact iff all its coefficients are (no Bernstein weighting)
  #   *_sigma  contact where σ_h·n > 0 instead of the discrete active set
  #   flux_sep h_F ‖σ_h·n‖² where ν_h > χ: flux on what ν_h says is separation
  actF = zeros(nc);  inaF = zeros(nc);  actσ = zeros(nc);  inaσ = zeros(nc);  fsep = zeros(nc)
  σmax = max(maximum(maximum(abs, v) for v in σn; init = 0.0), eps())
  for (F, c) in enumerate(ΓS.glue.face_to_cell)
    β = B * Float64.(sol.state[F])                 # Bernstein weights of the contact part
    d = νS[F] .- χS[F]
    # h_F⁻¹ ∫_F g = Σ_q w_q g(s_q): the facet length cancels
    act2[c] += sum(w .* β .* d .^ 2)
    ina2[c] += sum(w .* (1 .- β) .* min.(d, 0) .^ 2)
    comp[c] += abs(hS[F] * sum(w .* σn[F] .* d))
    full = all(sol.state[F])
    actF[c] += full ? sum(w .* d .^ 2) : 0.0
    inaF[c] += full ? 0.0 : sum(w .* min.(d, 0) .^ 2)
    inσ = σn[F] .> 1e-12 * σmax
    actσ[c] += sum(w .* inσ .* d .^ 2)
    inaσ[c] += sum(w .* .!inσ .* min.(d, 0) .^ 2)
    fsep[c] += hS[F]^2 * sum(w .* (d .> 0) .* σn[F] .^ 2)   # h_F ∫_F = h_F² Σ_q w_q
  end

  cells = (; tilde, flux, jump = jump2, dirichlet = dir2, active = act2,
             inactive = ina2, complementarity = comp,
             active_facet = actF, inactive_facet = inaF, active_sigma = actσ,
             inactive_sigma = inaσ, flux_sep = fsep)
  η2 = tilde .+ flux .+ jump2 .+ dir2 .+ act2 .+ ina2
  complementarity && (η2 = η2 .+ comp)
  parts = NamedTuple{keys(cells)}(map(sum, values(cells)))
  return (; η2, parts, cells)
end

end # module Estimators
