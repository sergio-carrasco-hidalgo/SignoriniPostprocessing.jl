###############################################################################
# estimator_study.jl
#
# Is the residual representative ε_h of (3.2) an a posteriori error estimator
# for the scalar Signorini problem?  And does anything change with respect to
# the same problem WITHOUT the Signorini condition?
#
# Built-in estimator (Muga–Rojas–Vega 2023, Remark 3.4):
#
#     η̃² = Σ_K η̃_K²,     η̃_K = ‖∇ε_K‖_K.
#
# What the paper proves, and what is measured here (q = -∇u = -σ, q_h = -σ_h):
#
#   * local efficiency, Thm 3.8 (constant 1):
#         η̃_K ≤ ‖∇(u - ν_h)‖_K + ‖q - q_h‖_{*,K}          → eff_loc_max ≤ 1
#   * reliability, Thm 3.6, under the saturation Assumption 3.5:
#         δ = ‖∇_h(u - θ_h)‖ / ‖∇_h(u - ν_h)‖ < 1,          → delta
#     θ_h ∈ P_{p+2} from (3.3); then ‖∇(u-ν_h)‖ + ‖q-q_h‖_* ≲ η̃.
#     Effectivity index  eff = η̃ / E,  E² = ‖∇_h(u-ν_h)‖² + ‖q-q_h‖²_*.
#   * identity (A.2), an implementation check:  ∇(θ_h - ν_h) = ∇ε_h on every K,
#     relative to max_K ‖∇ν_h‖_K               → a2_dev ≈ round-off
#
# The discrete dual norm (2.2) is computed exactly through its local Riesz
# representative z_K ∈ V_{⋆,K}^{p+2}: (∇z_K, ∇v)_K = (q - q_h, ∇v)_K, so
# ‖q - q_h‖_{*,K} = ‖∇z_K‖_K.  Both θ_h and z_K are local P_{p+2} problems with
# zero-mean tests, i.e. `stenberg_postprocess` called with degree p+1.
#
# Benchmarks (Ω = (-1/2,1/2)×(0,1), Γ_S = {y = 0}, f = -2):
#
#     u = -r^α sin(αθ) + y²,   α = 11/2 ("smooth", the test benchmark)
#                              α =  3/2 ("singular", u ∈ H^{5/2-ε}: the generic
#                                        regularity of Signorini at a free boundary)
#
# Both satisfy u ≥ 0, σ·n ≥ 0, u σ·n = 0 on Γ_S with the free boundary at x = 0.
# Each is solved twice on the same meshes:
#
#     signorini  : the coefficient-PDAS solver of the package;
#     linear     : the same u, Dirichlet data u on the WHOLE boundary
#                  (mixed Poisson, no contact condition).
#
# Usage (from the root of SignoriniPostprocessing.jl):
#
#     julia --project=. studies/aposteriori/estimator_study.jl
#
# Output: results/csv/aposteriori_<bench>_<problem>_k<k>.csv and a table per case.
###############################################################################

using SignoriniPostprocessing
using Gridap
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "test", "manufactured.jl"))
using .Manufactured: square_model, LEVELS, l2, rates, TAG_D, TAG_S

# --- exact solutions ----------------------------------------------------------
"u = -r^α sin(αθ) + y², σ = ∇u (analytic), f = -Δu = -2."
function exact_solution(α)
  θf(x) = atan(abs(x[2]), x[1])          # y = -0.0 must not jump to θ = -π
  rf(x) = sqrt(x[1]^2 + x[2]^2)
  u(x) = -rf(x)^α * sin(α * θf(x)) + x[2]^2
  σ(x) = (ρ = rf(x); t = θf(x);
          -α * ρ^(α - 1) * VectorValue(sin((α - 1) * t), cos((α - 1) * t)) +
          VectorValue(0.0, 2 * x[2]))
  f(x) = -2.0
  (; u, σ, f)
end

const BENCHMARKS = [("smooth", 11 / 2), ("singular", 3 / 2)]

# --- the linear counterpart: mixed Poisson, Dirichlet on all of ∂Ω -------------
function solve_linear(model, k, ex)
  Ω  = Triangulation(model);    dΩ = Measure(Ω, 2k + 6)
  Γ  = BoundaryTriangulation(model);  dΓ = Measure(Γ, 2k + 6);  nΓ = get_normal_vector(Γ)
  Σ  = TestFESpace(model, ReferenceFE(bdm, Float64, k + 1), conformity = :HDiv)
  Q  = TestFESpace(model, ReferenceFE(lagrangian, Float64, k), conformity = :L2)
  X  = MultiFieldFESpace([Σ, Q])
  a((σ, u), (τ, v)) = ∫(σ ⋅ τ + u * (∇ ⋅ τ) + v * (∇ ⋅ σ))dΩ
  l((τ, v))         = ∫((τ ⋅ nΓ) * ex.u)dΓ + ∫(-(ex.f * v))dΩ
  σh, uh = solve(AffineFEOperator(a, l, X, X))
  (; σh, uh, Ω, ndofs = num_free_dofs(X), converged = true)
end

function solve_signorini(model, k, ex)
  sol = solve_scalar_signorini(model; k, f = ex.f, uD = ex.u, tag_D = TAG_D, tag_S = TAG_S)
  (; sol.σh, sol.uh, sol.Ω, sol.ndofs, sol.converged)
end

# --- estimator and error quantities ------------------------------------------------
function estimator_quantities(model, s, k, ex)
  p  = k + 1
  Ω  = s.Ω
  dΩ = Measure(Ω, 2k + 12)
  cell(dc) = collect(Gridap.CellData.get_contribution(dc, Ω))
  ∇u = CellField(ex.σ, Ω)

  pp = residual_minimization_postprocess(model, -s.σh, s.uh, p)          # ν_h, ε_h
  θh = stenberg_postprocess(model, -s.σh, s.uh, p + 1).ũh                # (3.3), P_{p+2}
  # Riesz representative of q - q_h in V_⋆^{p+2}: (∇z, ∇v) = -(σ - σ_h, ∇v)
  zh = stenberg_postprocess(model, ∇u - s.σh, CellField(0.0, Ω), p + 1;
                            qdegree = 2k + 12).ũh

  η2  = cell(∫(∇(pp.εh) ⋅ ∇(pp.εh))dΩ)
  e∇2 = cell(∫((∇u - ∇(pp.νh)) ⋅ (∇u - ∇(pp.νh)))dΩ)
  edual2 = cell(∫(∇(zh) ⋅ ∇(zh))dΩ)
  eθ2 = cell(∫((∇u - ∇(θh)) ⋅ (∇u - ∇(θh)))dΩ)
  dA2 = cell(∫((∇(θh) - ∇(pp.νh) - ∇(pp.εh)) ⋅ (∇(θh) - ∇(pp.νh) - ∇(pp.εh)))dΩ)
  nν2 = cell(∫(∇(pp.νh) ⋅ ∇(pp.νh))dΩ)

  η  = sqrt(sum(η2));  e∇ = sqrt(sum(e∇2));  edual = sqrt(sum(edual2))
  E  = sqrt(e∇^2 + edual^2)
  ηK = sqrt.(abs.(η2))
  loc = maximum(ηK ./ (sqrt.(abs.(e∇2)) .+ sqrt.(abs.(edual2))))
  # (A.2) is ∇θ_h - ∇ν_h - ∇ε_h = 0: measured against the size of the fields
  # that are subtracted, max_K ‖∇ν_h‖_K, so that it reads as round-off.
  (; η, e∇, edual, E, eff = η / E, δ = sqrt(sum(eθ2)) / e∇, loc,
     a2 = sqrt(maximum(abs, dA2)) / sqrt(maximum(nν2)))
end

# --- CSV ---------------------------------------------------------------------------
fmt(x::Integer)       = string(x)
fmt(x::AbstractFloat) = isnan(x) ? "" : @sprintf("%.10e", x)
function write_csv(path, cols)
  open(path, "w") do io
    println(io, join(first.(cols), ","))
    for i in eachindex(last(first(cols)))
      println(io, join((fmt(c[i]) for c in last.(cols)), ","))
    end
  end
end
withnan(r) = vcat(NaN, r)

# --- one case ----------------------------------------------------------------------
function run_case(bench, α, problem, k, outdir)
  ex = exact_solution(α)
  hs = Float64[];  nd = Int[];  R = []
  for n in LEVELS[k]
    model = square_model(n)
    s = problem == "signorini" ? solve_signorini(model, k, ex) : solve_linear(model, k, ex)
    s.converged || @warn "$bench/$problem k=$k n=$n: active set did not converge"
    push!(hs, 1 / n);  push!(nd, s.ndofs)
    push!(R, estimator_quantities(model, s, k, ex))
  end
  g(f) = [getfield(r, f) for r in R]
  η, E = g(:η), g(:E)
  rη, rE = withnan(rates(hs, η)), withnan(rates(hs, E))

  println("\n$bench / $problem, k = $k")
  hdr = @sprintf("%5s %7s %10s %5s %10s %5s %10s %10s %6s %6s %8s %8s",
                 "1/h", "ndof", "η̃", "rate", "E", "rate", "‖∇(u-ν)‖", "‖q-qh‖*",
                 "η̃/E", "δ", "effloc", "A2 dev")
  println(hdr);  println("-"^length(hdr))
  for i in eachindex(hs)
    rt(r) = i == 1 ? "-" : @sprintf("%.2f", r[i])
    @printf("%5d %7d %10.3e %5s %10.3e %5s %10.3e %10.3e %6.3f %6.3f %8.4f %8.1e\n",
            round(Int, 1 / hs[i]), nd[i], η[i], rt(rη), E[i], rt(rE), R[i].e∇, R[i].edual,
            R[i].eff, R[i].δ, R[i].loc, R[i].a2)
  end

  write_csv(joinpath(outdir, "aposteriori_$(bench)_$(problem)_k$(k).csv"),
            Pair{String,Any}["k" => fill(k, length(hs)), "n" => round.(Int, 1 ./ hs),
             "h" => hs, "ndof" => nd,
             "eta" => η, "rate_eta" => rη, "err" => E, "rate_err" => rE,
             "err_grad_nu" => g(:e∇), "err_dual" => g(:edual),
             "eff" => g(:eff), "delta" => g(:δ), "eff_loc_max" => g(:loc),
             "a2_dev" => g(:a2)])
end

let outdir = joinpath(ROOT, "results", "csv")
  mkpath(outdir)
  t0 = time()
  for (bench, α) in BENCHMARKS, problem in ("signorini", "linear"), k in 0:3
    run_case(bench, α, problem, k, outdir)
  end
  println("\ndone in ", round(time() - t0; digits = 1), " s")
end
