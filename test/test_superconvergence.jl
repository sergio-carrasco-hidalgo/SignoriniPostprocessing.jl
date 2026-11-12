###############################################################################
# test_superconvergence.jl
#
# Superconvergence of the postprocessing based on local residual minimization
# (Muga–Rojas–Vega 2023, Section 3.1, problem (3.2)) applied to the scalar
# Signorini solutions of test_signorini_convergence.jl (same cached solves).
#
# Paper notation: p = k + 1 is the BDM degree and q_h = -σ_h is the flux, since
# here σ = ∇u while the paper uses q = -∇u.  ν_h ∈ P_{k+2}, ε_h ∈ P_{k+3}.
#
# Expected orders (Remark 2.6 + Corollary 3.2; f is constant, so the gain of two
# orders also holds for p = 1):
#
#     ‖u - ν_h‖_{L²}        = O(h^{k+3})    (two orders above ‖u - u_h‖ = O(h^{k+1})),
#     ‖∇_h(u - ν_h)‖_{L²}  = O(h^{k+2}).
#
# Whether this survives the contact condition is precisely the question: the
# a priori theory of the paper is for the linear problem (2.5).
#
# The local problems are also checked at every level for the two constraints
# that close them: (ν_h, 1)_K = (u_h, 1)_K (3.2c) and (ε_h, 1)_K = 0 (ε_K ∈ V_⋆).
###############################################################################

function signorini_superconvergence(k::Int)
  hs = Float64[];  nd = Int[];  eu = Float64[];  eν = Float64[];  e∇ = Float64[]
  p  = k + 1

  for n in LEVELS[k]
    (; model, sol, h) = signorini_level(k, n)
    Ω  = sol.Ω
    dΩ = Measure(Ω, 2k + 12)

    pp = residual_minimization_postprocess(model, -sol.σh, sol.uh, p)

    @testset "k = $k, n = $n: local constraints of (3.2)" begin
      Δmean = collect(Gridap.CellData.get_contribution(∫(pp.νh - sol.uh)dΩ, Ω))
      εmean = collect(Gridap.CellData.get_contribution(∫(pp.εh)dΩ, Ω))
      @test maximum(abs, Δmean) ≤ 1e-12
      @test maximum(abs, εmean) ≤ 1e-12
    end

    push!(hs, h);  push!(nd, num_free_dofs(pp.Vν))
    push!(eu, l2(CellField(uex, Ω) - sol.uh, dΩ))
    push!(eν, l2(CellField(uex, Ω) - pp.νh, dΩ))
    push!(e∇, l2(CellField(σex, Ω) - ∇(pp.νh), dΩ))
  end

  print_table("Postprocessing (3.2), k = $k: uh ∈ P$k, νh ∈ P$(k+2), εh ∈ P$(k+3)" *
              "  (ndofs of νh)", hs, nd,
              [("‖u-uh‖", eu), ("‖u-νh‖", eν), ("‖∇(u-νh)‖", e∇)])

  rν = rates(hs, eν);  r∇ = rates(hs, e∇);  ru = rates(hs, eu)
  @testset "k = $k: orders" begin
    @test rν[end] ≥ (k + 3) - 0.2
    @test r∇[end] ≥ (k + 2) - 0.2
    @test rν[end] - ru[end] ≥ 2 - 0.2          # the gain of two orders
  end
end

for k in 0:3
  @testset "k = $k" begin
    signorini_superconvergence(k)
  end
end
