###############################################################################
# compare_estimators.jl
#
# The full estimator of `Estimators.contact_estimator` (the improved estimator
# (3.9) of Muga–Rojas–Vega plus the contact terms on Γ_S) against the built-in
# η̃ = ‖∇ε_h‖, on six examples and k = 0..3.  Both are PDE-independent.
#
#   η² = Σ_K ‖∇ε_K‖² + ‖σ_h - ∇ν_h‖²_K + ½Σ h_F⁻¹‖[[ν_h]]‖² + Σ_{Γ_D} h_F⁻¹‖u_D - ν_h‖²
#        + Σ_{Γ_S} h_F⁻¹‖β_F^{1/2}(ν_h - χ)‖² + h_F⁻¹‖(1-β_F)^{1/2} min(ν_h - χ, 0)‖²
#
# Error norms (when u is known):
#   E_s² = ‖σ - σ_h‖² + ‖∇_h(u - ν_h)‖² + Σ_int h_F⁻¹‖[[ν_h]]‖² + Σ_{Γ_D ∪ contact} h_F⁻¹‖u - ν_h‖²
#          (the norm of Theorem 3.9: ‖u - ν_h‖_{1,h} and ‖q - q_h‖_Ω)  → eff  = η / E_s
#   E²   = ‖∇_h(u - ν_h)‖² + ‖q - q_h‖²_*  (Theorem 3.6)               → effw = η̃ / E
#
# The adaptive loop is driven by η (Dörfler θ = 1/2, newest-vertex bisection).
# The η̃-driven runs of studies/adaptivity and studies/references are the
# reference for "which estimator refines better".
#
# Examples: singular, offset (studies/adaptivity: u = -r^{3/2} sin(3θ/2) + y²
# centred at x0 = 0 and x0 = 1/(2π)), bgk72, fhs42 (k ≤ 2: at k = 3 it is
# reproduced to round-off), and the unknown-solution fhs43, bgk731.
#
# Usage (from the root of SignoriniPostprocessing.jl):
#
#     julia --project=. studies/estimators/compare_estimators.jl
#     julia --project=. studies/estimators/compare_estimators.jl --ex=singular,offset --k=0,1
#
# Output: results/csv/estimators/<ex>_{adaptive,uniform}_k<k>.csv, <ex>_adaptive_k<k>_mesh.csv
###############################################################################

include(joinpath(@__DIR__, "..", "references", "reference_examples.jl"))   # examples, meshes, I/O

# --- the two benchmarks of studies/adaptivity ------------------------------------
function singular_exact(x0)
  θf(x) = atan(abs(x[2]), x[1] - x0)
  rf(x) = sqrt((x[1] - x0)^2 + x[2]^2)
  u(x) = -rf(x)^1.5 * sin(1.5θf(x)) + x[2]^2
  σ(x) = (ρ = rf(x); t = θf(x);
          -1.5 * sqrt(ρ) * VectorValue(sin(t / 2), cos(t / 2)) + VectorValue(0.0, 2x[2]))
  (; u, σ)
end
function square_model(n)
  m = rect_model(-0.5, 0.5, 0, 1, n, n);  lab = get_face_labeling(m)
  add_tag_from_tags!(lab, "GammaS", ["tag_5"])
  add_tag_from_tags!(lab, "GammaD", ["tag_6", "tag_7", "tag_8"])
  m
end
bench(x0) = (ex = singular_exact(x0);
             (model = square_model, n0 = 4,
              data = (f = x -> -2.0, tag_S = "GammaS", uD = ex.u, tag_D = "GammaD"),
              exact = ex))

const ALL = merge(EXAMPLES, Dict("singular" => bench(0.0), "offset" => bench(1 / (2π))))
const ORDER = ["singular", "offset", "bgk72", "fhs42", "fhs43", "bgk731"]
const SKIP  = Dict("fhs42" => [3])

getdata(ex, key) = haskey(ex.data, key) ? getfield(ex.data, key) : nothing

# --- one solve ------------------------------------------------------------------------
function step_full!(model, k, ex)
  sol = solve_scalar_signorini(model; k, ex.data...)
  p   = k + 1
  pp  = residual_minimization_postprocess(model, -sol.σh, sol.uh, p)
  est = contact_estimator(model, sol, pp.νh, pp.εh; k, uD = getdata(ex, :uD),
                          tag_D = getdata(ex, :tag_D), χ = getdata(ex, :χ))
  P   = est.parts
  row = (; ndof = sol.ndofs, ncells = num_cells(sol.Ω), pdas_it = sol.it,
           converged = Int(sol.converged), eta = sqrt(sum(est.η2)), eta_tilde = sqrt(P.tilde),
           part_tilde = P.tilde, part_flux = P.flux, part_jump = P.jump,
           part_dirichlet = P.dirichlet, part_contact = P.active + P.inactive,
           part_complementarity = P.complementarity)
  if ex.exact !== nothing
    Ω  = sol.Ω;  dΩ = Measure(Ω, 2k + 12)
    ∇u = CellField(ex.exact.σ, Ω);  u = CellField(ex.exact.u, Ω)
    c(dc, t) = collect(Gridap.CellData.get_contribution(dc, t))
    Λ  = SkeletonTriangulation(model);  dΛ = Measure(Λ, 2k + 12)
    Γ  = BoundaryTriangulation(model);  dΓ = Measure(Γ, 2k + 12)
    jmp = sum(c(∫(jump(pp.νh) * jump(pp.νh))dΛ, Λ) ./ c(∫(1.0)dΛ, Λ))
    # Boundary part of ‖u - ν_h‖_{1,h}: only where the trace of u is prescribed,
    # i.e. on Γ_D and on the exact contact zone of Γ_S (u = χ).  On the
    # separated part of Γ_S and on Γ_N the trace of u is free: there the term
    # is reported (err_bnd_sep) but is not part of E_s.
    χf  = something(getdata(ex, :χ), x -> 0.0)
    bndF(t, dt, w) = sum(c(∫(w * (ex.exact.u - pp.νh) * (ex.exact.u - pp.νh))dt, t) ./ c(∫(1.0)dt, t))
    ΓSx = sol.ΓS;  dΓSx = Measure(ΓSx, 2k + 12)
    incontact(x) = abs(ex.exact.u(x) - χf(x)) < 1e-12 ? 1.0 : 0.0
    bndD = getdata(ex, :tag_D) === nothing ? 0.0 :
           (ΓDx = BoundaryTriangulation(model, tags = getdata(ex, :tag_D));
            bndF(ΓDx, Measure(ΓDx, 2k + 12), 1.0))
    bndC = bndF(ΓSx, dΓSx, CellField(incontact, ΓSx))
    bndS = bndF(ΓSx, dΓSx, CellField(x -> 1 - incontact(x), ΓSx))
    eσ  = l2(∇u - sol.σh, dΩ);  e∇ = l2(∇u - ∇(pp.νh), dΩ)
    zh  = stenberg_postprocess(model, ∇u - sol.σh, CellField(0.0, Ω), p + 1;
                               qdegree = 2k + 12).ũh
    Es  = sqrt(eσ^2 + e∇^2 + jmp + bndD + bndC)
    E   = sqrt(e∇^2 + l2(∇(zh), dΩ)^2)
    # diagnostic only (not in η, which is PDE-independent): the momentum residual
    # of Muga–Rojas–Vega (3.15), (h_K/π)‖f - div q_h - R ν_h‖_K = (h_K/π)‖f + div σ_h - c ν_h‖_K,
    # with ν_h (not u_h: with a reaction term, c(u - u_h) would swamp it), h_K = 2√|K|.
    # Without reaction it is the oscillation (h_K/π)‖f - Π_k f‖_K.
    cr  = something(getdata(ex, :reaction), 0.0)
    res = ex.data.f + (∇ ⋅ sol.σh) - cr * pp.νh
    hK2 = 4 .* c(∫(1.0)dΩ, Ω)
    osc = sqrt(sum(hK2 ./ π^2 .* c(∫(res * res)dΩ, Ω)))
    row = (; row..., err_strong = Es, eff = row.eta / Es, err_weak = E, eff_tilde = row.eta_tilde / E,
             err_sigma_L2 = eσ, err_u_L2 = l2(u - sol.uh, dΩ), err_nu_L2 = l2(u - pp.νh, dΩ),
             err_bnd_sep = sqrt(bndS), osc = osc)
  end
  sol.converged || @warn "active set did not converge" k row.ndof
  return row, est.η2
end

function adaptive_full(ex, k; θ = 0.5, nmax = 50_000, maxsteps = 60, label = "")
  model = ex.model(ex.n0);  rows = [];  marking = DorflerMarking(θ)
  for s in 0:maxsteps
    row, η2 = step_full!(model, k, ex)
    push!(rows, (; step = s, row...))
    @printf("  %-8s adaptive k=%d step %2d ndof %6d η %.3e η̃ %.3e%s\n", label, k, s, row.ndof,
            row.eta, row.eta_tilde, haskey(row, :eff) ? @sprintf(" eff %.2f effw %.2f", row.eff, row.eff_tilde) : "")
    row.ndof ≥ nmax && break
    model = get_model(refine(model; refinement_method = "nvb",
                             cells_to_refine = Gridap.Adaptivity.mark(marking, η2)))
  end
  rows, model
end

function uniform_full(ex, k; nmax = 50_000, label = "")
  rows = [];  n = ex.n0
  while true
    row, _ = step_full!(ex.model(n), k, ex)
    push!(rows, (; step = n, row...))
    @printf("  %-8s uniform  k=%d n=%3d  ndof %6d η %.3e%s\n", label, k, n, row.ndof, row.eta,
            haskey(row, :eff) ? @sprintf(" eff %.2f effw %.2f", row.eff, row.eff_tilde) : "")
    row.ndof ≥ nmax && break
    n *= 2
  end
  rows
end

function parse_cmp(args)
  exs = copy(ORDER);  ks = [0, 1, 2, 3];  nmax = 50_000
  for a in args
    if startswith(a, "--ex=");       exs = String.(split(a[6:end], ","))
    elseif startswith(a, "--k=");    ks = parse.(Int, split(a[5:end], ","))
    elseif startswith(a, "--nmax="); nmax = parse(Int, a[8:end])
    else error("unknown argument $a (expected --ex=, --k=, --nmax=)")
    end
  end
  all(e -> haskey(ALL, e), exs) || error("examples: $(ORDER)")
  exs, ks, nmax
end

let (exs, ks, nmax) = parse_cmp(ARGS)
  outdir = joinpath(ROOT, "results", "csv", "estimators");  mkpath(outdir)
  t0 = time();  summary = []
  for name in exs, k in ks
    k in get(SKIP, name, Int[]) && continue
    ex = ALL[name]
    println("\n=== $name, k = $k ===")
    ad, mesh = adaptive_full(ex, k; nmax, label = name)
    un = uniform_full(ex, k; nmax, label = name)
    write_rows(joinpath(outdir, "$(name)_adaptive_k$(k).csv"), k, ad)
    write_rows(joinpath(outdir, "$(name)_uniform_k$(k).csv"), k, un)
    write_mesh(joinpath(outdir, "$(name)_adaptive_k$(k)_mesh.csv"), mesh)
    push!(summary, (name, k, ad, un))
  end

  println("\nLast decade of N: slopes s (e ~ N^s) and effectivity ranges.  Optimal: η, σ ~ N^-(k+2)/2")
  rng(v) = @sprintf("[%.2f, %.2f]", minimum(v), maximum(v))
  for (name, k, ad, un) in summary
    for (lab, rows) in (("adaptive", ad), ("uniform", un))
      late = [r for r in rows if r.ndof ≥ rows[end].ndof / 10]
      line = @sprintf("%-8s k=%d %-9s η %6.2f  η̃ %6.2f", name, k, lab, slope(rows, :eta), slope(rows, :eta_tilde))
      if haskey(rows[end], :eff)
        line *= @sprintf("  σ %6.2f  ν %6.2f  E_s %6.2f  η/E_s %s  η̃/E %s",
                         slope(rows, :err_sigma_L2), slope(rows, :err_nu_L2), slope(rows, :err_strong),
                         rng([r.eff for r in late]), rng([r.eff_tilde for r in late]))
      end
      println(line)
    end
  end
  println("\ndone in ", round(time() - t0; digits = 1), " s")
end
