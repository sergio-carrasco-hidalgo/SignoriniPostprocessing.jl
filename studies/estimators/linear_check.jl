###############################################################################
# linear_check.jl
#
# Is the lower effectivity of η (variant B) with the transition point inside a
# facet a CONTACT effect, or an effect of the singularity r^{3/2} lying inside an
# element (which a problem without contact would show too)?
#
# Same u = -r^{3/2} sin(3θ/2) + y² centred at x0 = 0 ("singular") or
# x0 = 1/(2π) ("offset"), solved WITHOUT contact condition: mixed Poisson
# BDM_{k+1} × P_k with Dirichlet data u on all of ∂Ω.  Same estimator: with no
# Γ_S, η reduces to (3.9) of Muga–Rojas–Vega,
#     η² = Σ ‖∇ε_K‖² + ‖σ_h - ∇ν_h‖² + ½Σ h_F⁻¹‖[[ν_h]]‖² + Σ_{∂Ω} h_F⁻¹‖u - ν_h‖²,
# against the same norm (all of ∂Ω is Dirichlet here):
#     E_s² = ‖σ-σ_h‖² + ‖∇_h(u-ν_h)‖² + Σ_int h_F⁻¹‖[[ν_h]]‖² + Σ_{∂Ω} h_F⁻¹‖u-ν_h‖².
# Uniform refinement and η-driven adaptive refinement (Dörfler θ = 1/2, NVB).
# The Signorini counterpart is in results/csv/contact_terms (column eff_B).
#
# Usage:  julia --project=. studies/estimators/linear_check.jl
# Output: results/csv/linear_check/<bench>_{adaptive,uniform}_k<k>.csv
###############################################################################

include(joinpath(@__DIR__, "..", "references", "reference_examples.jl"))   # rect_model, write_rows, l2

function exact(x0)
  θf(x) = atan(abs(x[2]), x[1] - x0);  rf(x) = sqrt((x[1] - x0)^2 + x[2]^2)
  u(x) = -rf(x)^1.5 * sin(1.5θf(x)) + x[2]^2
  σ(x) = (ρ = rf(x); t = θf(x);
          -1.5 * sqrt(ρ) * VectorValue(sin(t / 2), cos(t / 2)) + VectorValue(0.0, 2x[2]))
  (; u, σ)
end
const LIN = Dict("singular" => exact(0.0), "offset" => exact(1 / (2π)))

function step_lin!(model, k, ex)
  q  = 2k + 12
  Ω  = Triangulation(model);  dΩ = Measure(Ω, 2k + 6);  dΩe = Measure(Ω, q)
  Γ  = BoundaryTriangulation(model);  nΓ = get_normal_vector(Γ)
  Σ  = TestFESpace(model, ReferenceFE(bdm, Float64, k + 1), conformity = :HDiv)
  Q  = TestFESpace(model, ReferenceFE(lagrangian, Float64, k), conformity = :L2)
  X  = MultiFieldFESpace([Σ, Q])
  a((σ, u), (τ, v)) = ∫(σ ⋅ τ + u * (∇ ⋅ τ) + v * (∇ ⋅ σ))dΩ
  l((τ, v))         = ∫((τ ⋅ nΓ) * ex.u)Measure(Γ, 2k + 6) + ∫(-(-2.0 * v))dΩ
  σh, uh = solve(AffineFEOperator(a, l, X, X))
  pp = residual_minimization_postprocess(model, -σh, uh, k + 1)

  c(dc, t) = collect(Gridap.CellData.get_contribution(dc, t))
  Λ = SkeletonTriangulation(model);  dΛ = Measure(Λ, q);  dΓ = Measure(Γ, q)
  # cell indicators: η̃, constitutive mismatch, half jumps, boundary (all Dirichlet)
  η2  = c(∫(∇(pp.εh) ⋅ ∇(pp.εh))dΩe, Ω) .+ c(∫((σh - ∇(pp.νh)) ⋅ (σh - ∇(pp.νh)))dΩe, Ω)
  jF  = c(∫(jump(pp.νh) * jump(pp.νh))dΛ, Λ) ./ c(∫(1.0)dΛ, Λ)
  for (F, (cp, cm)) in enumerate(zip(Λ.plus.glue.face_to_cell, Λ.minus.glue.face_to_cell))
    η2[cp] += jF[F] / 2;  η2[cm] += jF[F] / 2
  end
  bF  = c(∫((ex.u - pp.νh) * (ex.u - pp.νh))dΓ, Γ) ./ c(∫(1.0)dΓ, Γ)
  for (F, cc) in enumerate(Γ.glue.face_to_cell)
    η2[cc] += bF[F]
  end
  ∇u = CellField(ex.σ, Ω)
  Es = sqrt(l2(∇u - σh, dΩe)^2 + l2(∇u - ∇(pp.νh), dΩe)^2 + sum(jF) + sum(bF))
  η  = sqrt(sum(η2))
  (; ndof = num_free_dofs(X), eta = η, err_strong = Es, eff = η / Es), η2
end

function run_lin(ex, k; nmax = 50_000, θ = 0.5)
  ad = [];  model = square_lin(4);  marking = DorflerMarking(θ)
  for s in 0:60
    row, η2 = step_lin!(model, k, ex);  push!(ad, (; step = s, row...))
    row.ndof ≥ nmax && break
    model = get_model(refine(model; refinement_method = "nvb",
                             cells_to_refine = Gridap.Adaptivity.mark(marking, η2)))
  end
  un = [];  n = 4
  while true
    row, _ = step_lin!(square_lin(n), k, ex);  push!(un, (; step = n, row...))
    row.ndof ≥ nmax && break
    n *= 2
  end
  ad, un
end
square_lin(n) = rect_model(-0.5, 0.5, 0, 1, n, n)

let outdir = joinpath(ROOT, "results", "csv", "linear_check");  mkpath(outdir)
  t0 = time()
  rng(rows, f) = (L = [getfield(r, f) for r in rows if r.ndof ≥ rows[end].ndof / 10];
                  @sprintf("[%.2f,%.2f]", minimum(L), maximum(L)))
  for bench in ("singular", "offset"), k in 0:3
    ad, un = run_lin(LIN[bench], k)
    write_rows(joinpath(outdir, "$(bench)_adaptive_k$(k).csv"), k, ad)
    write_rows(joinpath(outdir, "$(bench)_uniform_k$(k).csv"), k, un)
    println(@sprintf("linear %-8s k=%d  η/E_s adaptive %s  uniform %s  (uniform per level: %s)",
                     bench, k, rng(ad, :eff), rng(un, :eff),
                     join([@sprintf("%.2f", r.eff) for r in un], " ")))
  end
  println("done in ", round(time() - t0; digits = 1), " s")
end
