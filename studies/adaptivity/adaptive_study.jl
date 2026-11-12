###############################################################################
# adaptive_study.jl
#
# Adaptive refinement driven by the built-in estimator of Muga–Rojas–Vega
# (2023, Remark 3.4),  η̃_K = ‖∇ε_K‖_K,  on a scalar Signorini problem with the
# generic regularity of a contact/separation transition point:
#
#     u = -r^{3/2} sin(3θ/2) + y²  ∈ H^{5/2-ε}(Ω),     Ω = (-1/2,1/2)×(0,1),
#
# Γ_S = {y = 0} (contact for x > 0, separation for x < 0), Γ_D = rest, f = -2.
# Under uniform refinement every error is limited to O(h^{3/2}) = O(N^{-3/4})
# whatever k is (see studies/aposteriori).  The question: does the adaptive
# loop recover the optimal orders, and the superconvergence of ν_h?
#
#     SOLVE    solve_scalar_signorini (coefficient PDAS on the Bernstein cone)
#     ESTIMATE η̃_K from ε_h of (3.2)
#     MARK     Dörfler, Σ_marked η̃_K² ≥ θ Σ η̃_K²   (Gridap.Adaptivity.DorflerMarking)
#     REFINE   newest-vertex bisection             (refinement_method = "nvb")
#
# Optimal orders in N = ndof of the mixed system (2D: h ~ N^{-1/2}):
#     ‖σ-σ_h‖, ‖∇_h(u-ν_h)‖, E, η̃ : N^{-(k+2)/2}      ‖u-u_h‖ : N^{-(k+1)/2}
#     ‖u-ν_h‖                      : N^{-(k+3)/2}   (superconvergence)
#
# Usage (from the root of SignoriniPostprocessing.jl):
#
#     julia --project=. studies/adaptivity/adaptive_study.jl            # k = 0,1,2,3
#     julia --project=. studies/adaptivity/adaptive_study.jl --k=0,1 --theta=0.5 --nmax=50000
#     julia --project=. studies/adaptivity/adaptive_study.jl --problem=linear
#
#   --problem=signorini (default)  the Signorini problem above
#   --problem=linear               the same u WITHOUT contact condition: mixed
#                                  Poisson with Dirichlet data u on all of ∂Ω.
#                                  Same estimator, marking and refinement, to
#                                  see whether anything is specific to Signorini.
#   --bench=singular (default)     transition point at x0 = 0, a mesh vertex at
#                                  every step (uniform or NVB)
#   --bench=offset                 the same u centred at x0 = 1/(2π): irrational,
#                                  so never a vertex of these (dyadic) meshes --
#                                  the transition always lies inside a facet of Γ_S
#
# Output (results/csv/), <s> = "" for signorini, "_linear" for linear:
#     adaptive_<bench><s>_k<k>.csv        one row per adaptive step
#     uniform_<bench><s>_k<k>.csv         one row per uniform level (reference)
#     adaptive_<bench><s>_k<k>_mesh.csv   final adaptive mesh, one triangle per row
###############################################################################

using SignoriniPostprocessing
using Gridap
using Gridap.Adaptivity
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "test", "manufactured.jl"))
using .Manufactured: square_model, l2, TAG_D, TAG_S

# --- exact solution (as in studies/aposteriori/estimator_study.jl, α = 3/2) -------
# Polar coordinates centred at the transition point (x0, 0): contact for x > x0
# (u = 0, σ·n = α r^(α-1) ≥ 0), separation for x < x0 (u = r^α > 0, σ·n = 0).
function exact_solution(α; x0 = 0.0)
  θf(x) = atan(abs(x[2]), x[1] - x0)     # y = -0.0 must not jump to θ = -π
  rf(x) = sqrt((x[1] - x0)^2 + x[2]^2)
  u(x) = -rf(x)^α * sin(α * θf(x)) + x[2]^2
  σ(x) = (ρ = rf(x); t = θf(x);
          -α * ρ^(α - 1) * VectorValue(sin((α - 1) * t), cos((α - 1) * t)) +
          VectorValue(0.0, 2 * x[2]))
  f(x) = -2.0
  (; u, σ, f, x0)
end

const BENCH_X0 = Dict("singular" => 0.0, "offset" => 1 / (2π))
EX = exact_solution(3 / 2)                # reset in the main block from --bench

# --- solvers ----------------------------------------------------------------------------
"Mixed Poisson BDM_{k+1} × P_k with Dirichlet data u on all of ∂Ω (no contact)."
function solve_linear(model, k)
  Ω  = Triangulation(model);    dΩ = Measure(Ω, 2k + 6)
  Γ  = BoundaryTriangulation(model);  dΓ = Measure(Γ, 2k + 6);  nΓ = get_normal_vector(Γ)
  Σ  = TestFESpace(model, ReferenceFE(bdm, Float64, k + 1), conformity = :HDiv)
  Q  = TestFESpace(model, ReferenceFE(lagrangian, Float64, k), conformity = :L2)
  X  = MultiFieldFESpace([Σ, Q])
  a((σ, u), (τ, v)) = ∫(σ ⋅ τ + u * (∇ ⋅ τ) + v * (∇ ⋅ σ))dΩ
  l((τ, v))         = ∫((τ ⋅ nΓ) * EX.u)dΓ + ∫(-(EX.f * v))dΩ
  σh, uh = solve(AffineFEOperator(a, l, X, X))
  (; σh, uh, Ω, ndofs = num_free_dofs(X), it = 0, converged = true)
end

solve_problem(model, k, problem) =
  problem == "signorini" ?
    solve_scalar_signorini(model; k, f = EX.f, uD = EX.u, tag_D = TAG_D, tag_S = TAG_S) :
    solve_linear(model, k)

# --- one solve + estimator + errors ---------------------------------------------------
function step!(model, k, problem)
  sol = solve_problem(model, k, problem)
  p  = k + 1
  Ω  = sol.Ω
  dΩ = Measure(Ω, 2k + 12)
  ∇u = CellField(EX.σ, Ω);  u = CellField(EX.u, Ω)

  pp = residual_minimization_postprocess(model, -sol.σh, sol.uh, p)
  # ‖q - q_h‖_{*,K} through its Riesz representative in V_⋆^{p+2}
  zh = stenberg_postprocess(model, ∇u - sol.σh, CellField(0.0, Ω), p + 1;
                            qdegree = 2k + 12).ũh

  η2 = collect(Gridap.CellData.get_contribution(∫(∇(pp.εh) ⋅ ∇(pp.εh))dΩ, Ω))
  e∇ = l2(∇u - ∇(pp.νh), dΩ)
  E  = sqrt(e∇^2 + l2(∇(zh), dΩ)^2)
  η  = sqrt(sum(η2))
  row = (; ndof = sol.ndofs, ncells = num_cells(Ω), pdas_it = sol.it,
           eta = η, err_E = E, eff = η / E,
           err_sigma_L2 = l2(∇u - sol.σh, dΩ), err_u_L2 = l2(u - sol.uh, dΩ),
           err_nu_L2 = l2(u - pp.νh, dΩ), err_grad_nu_L2 = e∇)
  sol.converged || @warn "active set did not converge" k row.ndof
  return row, η2
end

# --- adaptive loop ----------------------------------------------------------------------
function run_adaptive(k, problem; θ = 0.5, n0 = 4, nmax = 50_000, maxsteps = 60)
  model = square_model(n0)
  rows  = []
  marking = DorflerMarking(θ)
  for s in 0:maxsteps
    row, η2 = step!(model, k, problem)
    push!(rows, (; step = s, row...))
    @printf("  adaptive k=%d step %2d  ndof %6d  η̃ %.3e  E %.3e  ‖u-νh‖ %.3e  eff %.3f  it %d\n",
            k, s, row.ndof, row.eta, row.err_E, row.err_nu_L2, row.eff, row.pdas_it)
    row.ndof ≥ nmax && break
    marked = Gridap.Adaptivity.mark(marking, η2)
    model  = get_model(refine(model; refinement_method = "nvb", cells_to_refine = marked))
  end
  return rows, model
end

function run_uniform(k, problem; nmax = 50_000)
  rows = []
  n = 4
  while true
    row, _ = step!(square_model(n), k, problem)
    push!(rows, (; step = n, row...))
    @printf("  uniform  k=%d n=%3d    ndof %6d  η̃ %.3e  E %.3e  ‖u-νh‖ %.3e\n",
            k, n, row.ndof, row.eta, row.err_E, row.err_nu_L2)
    row.ndof ≥ nmax && break
    n *= 2
  end
  return rows
end

# --- output -----------------------------------------------------------------------------
fmt(x::Integer)       = string(x)
fmt(x::AbstractFloat) = @sprintf("%.10e", x)

function write_rows(path, k, rows)
  cols = keys(first(rows))
  open(path, "w") do io
    println(io, join(("k", cols...), ","))
    for r in rows
      println(io, join((fmt(k), (fmt(getfield(r, c)) for c in cols)...), ","))
    end
  end
  println("  wrote ", relpath(path, ROOT))
end

function write_mesh(path, model)
  open(path, "w") do io
    println(io, "x1,y1,x2,y2,x3,y3")
    for c in get_cell_coordinates(Triangulation(model))
      println(io, join((@sprintf("%.12e", v) for P in c for v in (P[1], P[2])), ","))
    end
  end
  println("  wrote ", relpath(path, ROOT))
end

"Least-squares order in h-units, -2 d log(e)/d log(N), over the steps with N ≥ N_end/frac."
function fitted_order(rows, col; frac = 10)
  N = [r.ndof for r in rows];  e = [getfield(r, col) for r in rows]
  sel = N .≥ N[end] / frac
  x = log.(N[sel]);  y = log.(e[sel])
  -2 * sum((x .- sum(x) / length(x)) .* (y .- sum(y) / length(y))) /
       sum((x .- sum(x) / length(x)) .^ 2)
end

function parse_args(args)
  ks = [0, 1, 2, 3];  θ = 0.5;  nmax = 50_000;  problem = "signorini";  bench = "singular"
  for a in args
    if startswith(a, "--k=");            ks = parse.(Int, split(a[5:end], ","))
    elseif startswith(a, "--theta=");    θ = parse(Float64, a[9:end])
    elseif startswith(a, "--nmax=");     nmax = parse(Int, a[8:end])
    elseif startswith(a, "--problem=");  problem = a[11:end]
    elseif startswith(a, "--bench=");    bench = a[9:end]
    else error("unknown argument $a (expected --k=, --theta=, --nmax=, --problem=, --bench=)")
    end
  end
  problem in ("signorini", "linear") || error("--problem must be signorini or linear")
  haskey(BENCH_X0, bench) || error("--bench must be one of $(collect(keys(BENCH_X0)))")
  ks, θ, nmax, problem, bench
end

let (ks, θ, nmax, problem, bench) = parse_args(ARGS)
  global EX = exact_solution(3 / 2; x0 = BENCH_X0[bench])
  outdir = joinpath(ROOT, "results", "csv");  mkpath(outdir)
  sfx = problem == "signorini" ? "" : "_$(problem)"
  t0 = time()
  summary = []
  for k in ks
    println("\n=== $bench (x0 = $(EX.x0)) / $problem, k = $k: adaptive (Dörfler θ = $θ, NVB) and uniform, N ≤ $nmax ===")
    ad, mesh = run_adaptive(k, problem; θ, nmax)
    un = run_uniform(k, problem; nmax)
    write_rows(joinpath(outdir, "adaptive_$(bench)$(sfx)_k$(k).csv"), k, ad)
    write_rows(joinpath(outdir, "uniform_$(bench)$(sfx)_k$(k).csv"), k, un)
    write_mesh(joinpath(outdir, "adaptive_$(bench)$(sfx)_k$(k)_mesh.csv"), mesh)
    push!(summary, (k, ad, un))
  end

  println("\nFitted orders in h-units (-2 dlog e / dlog N, last decade of N); optimal in ():")
  @printf("%3s %-10s %12s %12s %12s %12s %12s %8s\n", "k", "", "‖σ-σh‖", "‖u-uh‖",
          "‖u-νh‖", "‖∇(u-νh)‖", "η̃", "eff")
  for (k, ad, un) in summary
    for (lab, rows) in (("adaptive", ad), ("uniform", un))
      @printf("%3d %-10s %5.2f (%d)    %5.2f (%d)    %5.2f (%d)    %5.2f (%d)    %5.2f (%d)    %6.3f\n",
              k, lab, fitted_order(rows, :err_sigma_L2), k + 2, fitted_order(rows, :err_u_L2), k + 1,
              fitted_order(rows, :err_nu_L2), k + 3, fitted_order(rows, :err_grad_nu_L2), k + 2,
              fitted_order(rows, :eta), k + 2, rows[end].eff)
    end
  end
  println("\ndone in ", round(time() - t0; digits = 1), " s")
end
