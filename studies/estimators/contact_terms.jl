###############################################################################
# contact_terms.jl
#
# Which contact-boundary term should the (PDE-independent) estimator carry?
#
# Bulk part, common to every variant (Muga–Rojas–Vega (3.9) on Ω and Γ_D):
#     bulk = Σ ‖∇ε_K‖² + ‖σ_h - ∇ν_h‖² + ½Σ h_F⁻¹‖[[ν_h]]‖² + Σ_{Γ_D} h_F⁻¹‖u_D - ν_h‖²
# Contact variants on Γ_S (χ the obstacle, β_F the Bernstein weight of the
# coefficients in contact):
#     A   nothing
#     B   h_F⁻¹‖β^{1/2}(ν_h-χ)‖² + h_F⁻¹‖(1-β)^{1/2} min(ν_h-χ,0)‖²       (default η)
#     Bf  the same, a facet in contact iff all its coefficients are
#     Bs  the same, in contact where σ_h·n > 0
#     C   B + |⟨σ_h·n, ν_h - χ⟩_F|                       (unsquared complementarity)
#     D   B + h_F‖σ_h·n‖²_{F ∩ {ν_h > χ}}                 (flux where ν_h says separation)
#
# Examples with f = -2 ∈ P_k, so that the oscillation of f is zero and the
# only thing a variant can miss is the contact boundary:
#     singular     u = -r^{3/2} sin(3θ/2) + y², transition at x0 = 0 (a mesh vertex)
#     offset       the same at x0 = 1/(2π) (always inside a facet)
#     offset_obst  offset + L(x), L = 0.3x - 0.05, obstacle χ = L on Γ_S
#                  (L is harmonic and ∂_y L = 0: same f, u - χ and σ·n on Γ_S)
#
# Error: E_s² = ‖σ-σ_h‖² + ‖∇_h(u-ν_h)‖² + Σ_int h_F⁻¹‖[[ν_h]]‖² + Σ_{Γ_D ∪ contact} h_F⁻¹‖u-ν_h‖².
# Also the contact-boundary error alone, bndC = Σ_{contact} h_F⁻¹‖u-ν_h‖², to see
# whether the contact term follows it locally and not only in the total.
#
# Adaptive loop driven by η_B (Dörfler θ = 1/2, NVB); all variants are evaluated
# on the same solutions.
#
# Usage:  julia --project=. studies/estimators/contact_terms.jl [--ex=offset --k=0,1 --nmax=50000]
# Output: results/csv/contact_terms/<ex>_{adaptive,uniform}_k<k>.csv
###############################################################################

include(joinpath(@__DIR__, "..", "references", "reference_examples.jl"))   # rect_model, write_rows, slope, l2

function make_example(x0; L = nothing)
  θf(x) = atan(abs(x[2]), x[1] - x0)
  rf(x) = sqrt((x[1] - x0)^2 + x[2]^2)
  Lf  = L === nothing ? (x -> 0.0) : L
  dLx = L === nothing ? 0.0 : (L(Point(1.0, 0.0)) - L(Point(0.0, 0.0)))      # L linear in x
  u(x) = -rf(x)^1.5 * sin(1.5θf(x)) + x[2]^2 + Lf(x)
  σ(x) = (ρ = rf(x); t = θf(x);
          -1.5 * sqrt(ρ) * VectorValue(sin(t / 2), cos(t / 2)) + VectorValue(dLx, 2x[2]))
  model(n) = (m = rect_model(-0.5, 0.5, 0, 1, n, n);  lab = get_face_labeling(m);
              add_tag_from_tags!(lab, "GammaS", ["tag_5"]);
              add_tag_from_tags!(lab, "GammaD", ["tag_6", "tag_7", "tag_8"]);  m)
  data = (f = x -> -2.0, tag_S = "GammaS", uD = u, tag_D = "GammaD")
  L === nothing || (data = (; data..., χ = L))
  (; model, n0 = 4, data, u, σ, χ = Lf)
end

const CT = Dict("singular"    => make_example(0.0),
                "offset"      => make_example(1 / (2π)),
                "offset_obst" => make_example(1 / (2π); L = x -> 0.3x[1] - 0.05))
const CT_ORDER = ["singular", "offset", "offset_obst"]

function step_ct!(model, k, ex)
  sol = solve_scalar_signorini(model; k, ex.data...)
  pp  = residual_minimization_postprocess(model, -sol.σh, sol.uh, k + 1)
  χ   = haskey(ex.data, :χ) ? ex.data.χ : nothing
  est = contact_estimator(model, sol, pp.νh, pp.εh; k, uD = ex.data.uD, tag_D = ex.data.tag_D, χ)
  P   = est.parts
  bulk = P.tilde + P.flux + P.jump + P.dirichlet

  Ω  = sol.Ω;  q = 2k + 12;  dΩ = Measure(Ω, q)
  c(dc, t) = collect(Gridap.CellData.get_contribution(dc, t))
  ∇u = CellField(ex.σ, Ω)
  Λ  = SkeletonTriangulation(model);  dΛ = Measure(Λ, q)
  ΓD = BoundaryTriangulation(model, tags = "GammaD");  dΓD = Measure(ΓD, q)
  ΓS = sol.ΓS;  dΓS = Measure(ΓS, q)
  e2(t, dt, w) = sum(c(∫(w * (ex.u - pp.νh) * (ex.u - pp.νh))dt, t) ./ c(∫(1.0)dt, t))
  incontact(x) = abs(ex.u(x) - ex.χ(x)) < 1e-12 ? 1.0 : 0.0
  jmp  = sum(c(∫(jump(pp.νh) * jump(pp.νh))dΛ, Λ) ./ c(∫(1.0)dΛ, Λ))
  bndD = e2(ΓD, dΓD, 1.0)
  bndC = e2(ΓS, dΓS, CellField(incontact, ΓS))
  eσ   = l2(∇u - sol.σh, dΩ);  e∇ = l2(∇u - ∇(pp.νh), dΩ)
  Es   = sqrt(eσ^2 + e∇^2 + jmp + bndD + bndC)

  B  = P.active + P.inactive
  vars = (A = bulk, B = bulk + B, Bf = bulk + P.active_facet + P.inactive_facet,
          Bs = bulk + P.active_sigma + P.inactive_sigma,
          C = bulk + B + P.complementarity, D = bulk + B + P.flux_sep)
  row = (; ndof = sol.ndofs, pdas_it = sol.it, converged = Int(sol.converged),
           err_strong = Es, err_bndC = sqrt(bndC), err_sigma_L2 = eσ,
           eta_B = sqrt(vars.B), contact_B = sqrt(B),
           part_complementarity = P.complementarity, part_flux_sep = P.flux_sep,
           eff_A = sqrt(vars.A) / Es, eff_B = sqrt(vars.B) / Es, eff_Bf = sqrt(vars.Bf) / Es,
           eff_Bs = sqrt(vars.Bs) / Es, eff_C = sqrt(vars.C) / Es, eff_D = sqrt(vars.D) / Es,
           ratio_contact = sqrt(B) / max(sqrt(bndC), eps()))
  sol.converged || @warn "active set did not converge" k row.ndof
  return row, est.η2
end

function adaptive_ct(ex, k; θ = 0.5, nmax = 50_000, maxsteps = 60, label = "")
  model = ex.model(ex.n0);  rows = [];  marking = DorflerMarking(θ)
  for s in 0:maxsteps
    row, η2 = step_ct!(model, k, ex)
    push!(rows, (; step = s, row...))
    @printf("  %-11s adaptive k=%d step %2d ndof %6d  eff A %.2f B %.2f C %.2f D %.2f  contact/bndC %.2f\n",
            label, k, s, row.ndof, row.eff_A, row.eff_B, row.eff_C, row.eff_D, row.ratio_contact)
    row.ndof ≥ nmax && break
    model = get_model(refine(model; refinement_method = "nvb",
                             cells_to_refine = Gridap.Adaptivity.mark(marking, η2)))
  end
  rows
end

function uniform_ct(ex, k; nmax = 50_000, label = "")
  rows = [];  n = ex.n0
  while true
    row, _ = step_ct!(ex.model(n), k, ex)
    push!(rows, (; step = n, row...))
    @printf("  %-11s uniform  k=%d n=%3d ndof %6d  eff A %.2f B %.2f C %.2f D %.2f  contact/bndC %.2f\n",
            label, k, n, row.ndof, row.eff_A, row.eff_B, row.eff_C, row.eff_D, row.ratio_contact)
    row.ndof ≥ nmax && break
    n *= 2
  end
  rows
end

function parse_ct(args)
  exs = copy(CT_ORDER);  ks = [0, 1, 2, 3];  nmax = 50_000
  for a in args
    if startswith(a, "--ex=");       exs = String.(split(a[6:end], ","))
    elseif startswith(a, "--k=");    ks = parse.(Int, split(a[5:end], ","))
    elseif startswith(a, "--nmax="); nmax = parse(Int, a[8:end])
    else error("unknown argument $a (expected --ex=, --k=, --nmax=)")
    end
  end
  exs, ks, nmax
end

let (exs, ks, nmax) = parse_ct(ARGS)
  outdir = joinpath(ROOT, "results", "csv", "contact_terms");  mkpath(outdir)
  t0 = time();  summary = []
  for name in exs, k in ks
    ex = CT[name]
    println("\n=== $name, k = $k ===")
    ad = adaptive_ct(ex, k; nmax, label = name)
    un = uniform_ct(ex, k; nmax, label = name)
    write_rows(joinpath(outdir, "$(name)_adaptive_k$(k).csv"), k, ad)
    write_rows(joinpath(outdir, "$(name)_uniform_k$(k).csv"), k, un)
    push!(summary, (name, k, ad, un))
  end
  println("\nEffectivity η_variant / E_s over the last decade of N: [min, max]")
  rng(rows, f) = (L = [getfield(r, f) for r in rows if r.ndof ≥ rows[end].ndof / 10];
                  @sprintf("[%.2f,%.2f]", minimum(L), maximum(L)))
  for (name, k, ad, un) in summary, (lab, rows) in (("adaptive", ad), ("uniform", un))
    println(@sprintf("%-11s k=%d %-8s", name, k, lab), "  A ", rng(rows, :eff_A), "  B ", rng(rows, :eff_B),
            "  Bf ", rng(rows, :eff_Bf), "  Bs ", rng(rows, :eff_Bs), "  C ", rng(rows, :eff_C),
            "  D ", rng(rows, :eff_D), "  contact/bndC ", rng(rows, :ratio_contact))
  end
  println("\ndone in ", round(time() - t0; digits = 1), " s")
end
