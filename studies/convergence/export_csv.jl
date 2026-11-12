###############################################################################
# export_csv.jl
#
# Export the convergence data of the test run to CSV, one file per study and
# polynomial degree, for LaTeX tables (pgfplotstable) or log-log plots
# (plot_convergence.py, pgfplots, pandas, ...).
#
# Non-invasive: src/ and test/ are not modified.  The script includes the test
# fixture test/manufactured.jl, so the exact solution, the meshes (LEVELS), the
# solver and its defaults are exactly those of the test suite, and the norms
# below are the ones test_signorini_convergence.jl / test_superconvergence.jl
# compute.
#
# Usage (from the root of SignoriniPostprocessing.jl):
#
#     julia --project=. studies/convergence/export_csv.jl
#     julia --project=. studies/convergence/export_csv.jl --k=0,1 --outdir=results/csv
#
#   --k=...       degrees to run (default 0,1,2,3)
#   --outdir=...  output directory (default results/csv, relative to the root)
#
# Output, for every k:
#
#   signorini_k<k>.csv     k, n, h, ndof, err_sigma_L2, rate_sigma_L2,
#                          err_u_L2, rate_u_L2
#   postprocess_k<k>.csv   k, n, h, ndof, ndof_nu, err_u_L2, rate_u_L2,
#                          err_nu_L2, rate_nu_L2, err_grad_nu_L2, rate_grad_nu_L2,
#                          err_sigma_L2, rate_sigma_L2
#   stenberg_k<k>.csv      k, n, h, ndof, err_nu_L2, rate_nu_L2,
#                          err_utilde_L2, rate_utilde_L2, err_grad_nu_L2,
#                          rate_grad_nu_L2, err_grad_utilde_L2,
#                          rate_grad_utilde_L2, reldiff_L2, reldiff_grad
#
#   n               mesh: n×n squares, each split into two triangles (h = 1/n)
#   ndof            free dofs of the mixed system (σ_h, u_h), BDM_{k+1} × P_k
#   ndof_nu         dofs of the postprocessed scalar ν_h ∈ P_{k+2}
#   err_sigma_L2    ‖σ - σ_h‖_{L²(Ω)}
#   err_u_L2        ‖u - u_h‖_{L²(Ω)}
#   err_nu_L2       ‖u - ν_h‖_{L²(Ω)}
#   err_grad_nu_L2  ‖∇_h(u - ν_h)‖_{L²(Ω)}   (broken gradient)
#   *_utilde_*      the same norms for ũ_h of (2.10) (stenberg_postprocess)
#   reldiff_L2      ‖ν_h - ũ_h‖ / ‖ν_h‖          (Proposition 3.1: round-off)
#   reldiff_grad    ‖∇_h(ν_h - ũ_h)‖ / ‖∇_h ν_h‖
#   rate_*          log(e_i/e_{i-1}) / log(h_i/h_{i-1}); empty on the first row
###############################################################################

using SignoriniPostprocessing
using Gridap
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "test", "manufactured.jl"))
using .Manufactured

# --- command line ------------------------------------------------------------
function parse_args(args)
  ks = [0, 1, 2, 3];  outdir = joinpath(ROOT, "results", "csv")
  for a in args
    if startswith(a, "--k=")
      ks = parse.(Int, split(a[5:end], ","))
    elseif startswith(a, "--outdir=")
      d = a[10:end];  outdir = isabspath(d) ? d : joinpath(ROOT, d)
    else
      error("unknown argument $a (expected --k=0,1,... or --outdir=DIR)")
    end
  end
  all(k -> haskey(LEVELS, k), ks) || error("k must be in $(sort(collect(keys(LEVELS))))")
  return ks, outdir
end

# --- CSV writing ---------------------------------------------------------------
fmt(x::Integer)       = string(x)
fmt(x::AbstractFloat) = isnan(x) ? "" : @sprintf("%.10e", x)

"Write `cols` (name => vector, all of the same length) to `path` as CSV."
function write_csv(path, cols)
  open(path, "w") do io
    println(io, join(first.(cols), ","))
    for i in eachindex(last(first(cols)))
      println(io, join((fmt(c[i]) for c in last.(cols)), ","))
    end
  end
  println("  wrote ", relpath(path, ROOT))
end

withnan(r) = vcat(NaN, r)          # no rate on the coarsest level

# --- one degree ------------------------------------------------------------------
function export_degree(k, outdir)
  ns = LEVELS[k]
  hs = Float64[];  nd = Int[];  ndν = Int[]
  eσ = Float64[];  eu = Float64[];  eν = Float64[];  e∇ = Float64[]
  eũ = Float64[];  e∇ũ = Float64[];  dL2 = Float64[];  d∇ = Float64[]

  for n in ns
    (; model, sol, h) = signorini_level(k, n)
    sol.converged || @warn "k = $k, n = $n: active set did not converge"
    Ω  = sol.Ω
    dΩ = Measure(Ω, 2k + 12)                      # as in the tests
    pp = residual_minimization_postprocess(model, -sol.σh, sol.uh, k + 1)
    ũh = stenberg_postprocess(model, -sol.σh, sol.uh, k + 1).ũh
    dν = pp.νh - ũh

    push!(hs, h);  push!(nd, sol.ndofs);  push!(ndν, num_free_dofs(pp.Vν))
    push!(eσ, l2(CellField(σex, Ω) - sol.σh, dΩ))
    push!(eu, l2(CellField(uex, Ω) - sol.uh, dΩ))
    push!(eν, l2(CellField(uex, Ω) - pp.νh, dΩ))
    push!(e∇, l2(CellField(σex, Ω) - ∇(pp.νh), dΩ))
    push!(eũ, l2(CellField(uex, Ω) - ũh, dΩ))
    push!(e∇ũ, l2(CellField(σex, Ω) - ∇(ũh), dΩ))
    push!(dL2, l2(dν, dΩ) / l2(pp.νh, dΩ))
    push!(d∇, l2(∇(dν), dΩ) / l2(∇(pp.νh), dΩ))
    @printf("  k = %d, n = %3d done\n", k, n)
  end

  kk = fill(k, length(ns))
  write_csv(joinpath(outdir, "signorini_k$(k).csv"),
            Pair{String,Any}["k" => kk, "n" => ns, "h" => hs, "ndof" => nd,
             "err_sigma_L2" => eσ, "rate_sigma_L2" => withnan(rates(hs, eσ)),
             "err_u_L2"     => eu, "rate_u_L2"     => withnan(rates(hs, eu))])
  write_csv(joinpath(outdir, "postprocess_k$(k).csv"),
            Pair{String,Any}["k" => kk, "n" => ns, "h" => hs, "ndof" => nd, "ndof_nu" => ndν,
             "err_u_L2"       => eu, "rate_u_L2"       => withnan(rates(hs, eu)),
             "err_nu_L2"      => eν, "rate_nu_L2"      => withnan(rates(hs, eν)),
             "err_grad_nu_L2" => e∇, "rate_grad_nu_L2" => withnan(rates(hs, e∇)),
             "err_sigma_L2"   => eσ, "rate_sigma_L2"   => withnan(rates(hs, eσ))])
  write_csv(joinpath(outdir, "stenberg_k$(k).csv"),
            Pair{String,Any}["k" => kk, "n" => ns, "h" => hs, "ndof" => nd,
             "err_nu_L2"          => eν,  "rate_nu_L2"          => withnan(rates(hs, eν)),
             "err_utilde_L2"      => eũ,  "rate_utilde_L2"      => withnan(rates(hs, eũ)),
             "err_grad_nu_L2"     => e∇,  "rate_grad_nu_L2"     => withnan(rates(hs, e∇)),
             "err_grad_utilde_L2" => e∇ũ, "rate_grad_utilde_L2" => withnan(rates(hs, e∇ũ)),
             "reldiff_L2" => dL2, "reldiff_grad" => d∇])
end

# --- main ------------------------------------------------------------------------
let (ks, outdir) = parse_args(ARGS)
  mkpath(outdir)
  t0 = time()
  for k in ks
    println("degree k = $k, meshes n = $(LEVELS[k])")
    export_degree(k, outdir)
  end
  println("done in ", round(time() - t0; digits = 1), " s")
end
