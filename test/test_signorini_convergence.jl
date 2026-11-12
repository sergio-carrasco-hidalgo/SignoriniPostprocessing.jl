###############################################################################
# test_signorini_convergence.jl
#
# Convergence of the scalar Signorini solver (BDM_{k+1} × P_k on the Bernstein
# cone, coefficient PDAS) for k = 0, 1, 2, 3 against the exact solution of
# manufactured.jl.  Expected orders (Theorem 2.2 of Muga–Rojas–Vega; the
# solution is smooth enough for all of them):
#
#     ‖σ - σ_h‖_{L²} = O(h^{k+2}),      ‖u - u_h‖_{L²} = O(h^{k+1}).
#
# At every level the discrete solution is also checked to BE a solution of the
# discrete Signorini problem:
#
#   * the active set converged by its own criterion;
#   * cone:  every Bernstein coefficient of σ_h·n on Γ_S is ≥ 0 — re-derived
#     independently from σ_h by `facet_bernstein_coeffs`, and confirmed by
#     sampling σ_h·n densely on every facet;
#   * multipliers μ ≥ 0 (the discrete u ≥ 0) and complementarity c_i μ_i = 0;
#   * div σ_h = -f exactly (f is constant, so it lies in P_k).
###############################################################################

function signorini_convergence(k::Int)
  hs = Float64[];  nd = Int[];  eσ = Float64[];  eu = Float64[];  its = Int[]
  p  = k + 1

  for n in LEVELS[k]
    (; sol, h) = signorini_level(k, n)
    Ω  = sol.Ω
    dΩ = Measure(Ω, 2k + 12)

    @testset "k = $k, n = $n: discrete Signorini conditions" begin
      @test sol.converged

      sc = maximum(maximum(abs, cF) for cF in sol.c)
      sμ = maximum(maximum(abs, μF) for μF in sol.μ)
      @test minimum(minimum(cF) for cF in sol.c) ≥ -1e-10 * sc        # cone
      @test minimum(minimum(μF) for μF in sol.μ) ≥ -1e-10 * sμ        # u ≥ 0
      @test maximum(maximum(abs.(sol.c[F] .* sol.μ[F])) for F in eachindex(sol.c)) ≤
            1e-10 * sc * sμ                                            # c μ = 0

      σn = sol.σh ⋅ sol.nS
      cB = facet_bernstein_coeffs(σn, p)
      @test maximum(maximum(abs, cB[F] .- sol.c[F]) for F in eachindex(cB)) ≤ 1e-8 * sc
      @test minimum(minimum(v) for v in sample_on_facets(σn, sol.ΓS, 41)) ≥ -1e-8 * sc

      @test l2((∇ ⋅ sol.σh) + CellField(fex, Ω), dΩ) ≤ 1e-8
    end

    push!(hs, h);  push!(nd, sol.ndofs);  push!(its, sol.it)
    push!(eσ, l2(CellField(σex, Ω) - sol.σh, dΩ))
    push!(eu, l2(CellField(uex, Ω) - sol.uh, dΩ))
  end

  print_table("Scalar Signorini, k = $k (BDM$(k+1) × P$k), PDAS iterations $(its)",
              hs, nd, [("‖σ-σh‖", eσ), ("‖u-uh‖", eu)])

  rσ = rates(hs, eσ);  ru = rates(hs, eu)
  @testset "k = $k: orders" begin
    @test rσ[end] ≥ (k + 2) - 0.2
    @test ru[end] ≥ (k + 1) - 0.2
  end
end

for k in 0:3
  @testset "k = $k" begin
    signorini_convergence(k)
  end
end
