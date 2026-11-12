###############################################################################
# test_stenberg_equivalence.jl
#
# Proposition 3.1 of Muga–Rojas–Vega (2023): the residual-minimization
# postprocess ν_h of (3.2) and the (modified) Stenberg postprocess ũ_h of (2.10)
# are the same function.  Checked on the cached scalar Signorini solves of
# test_signorini_convergence.jl (p = k + 1, q_h = -σ_h), at every level:
#
#   * identity, to round-off:  ‖ν_h - ũ_h‖ / ‖ν_h‖  and  ‖∇_h(ν_h - ũ_h)‖ / ‖∇_h ν_h‖;
#   * the difference is negligible against the discretisation error,
#     ‖∇_h(ν_h - ũ_h)‖ ≤ 1e-3 ‖∇_h(u - ν_h)‖, so the error seminorms
#     ‖∇_h(u - ν_h)‖ and ‖∇_h(u - ũ_h)‖ agree to at least three digits.
#
# Observed: relative L² difference ≤ 6e-15 and relative H¹ difference ≤ 1e-13
# (growing like h⁻¹, as the gradient of round-off does); the ratio to the error
# reaches ~2e-5 at k = 3, n = 16, where the error itself is ~1e-8.
###############################################################################

using Printf

function stenberg_equivalence(k::Int)
  hs = Float64[];  rL2 = Float64[];  rH1 = Float64[]
  eν = Float64[];  eũ = Float64[];  e∇ν = Float64[];  e∇ũ = Float64[]
  p  = k + 1

  for n in LEVELS[k]
    (; model, sol, h) = signorini_level(k, n)
    Ω  = sol.Ω
    dΩ = Measure(Ω, 2k + 12)

    ν = residual_minimization_postprocess(model, -sol.σh, sol.uh, p).νh
    ũ = stenberg_postprocess(model, -sol.σh, sol.uh, p).ũh
    d = ν - ũ

    push!(hs, h)
    push!(rL2, l2(d, dΩ) / l2(ν, dΩ))
    push!(rH1, l2(∇(d), dΩ) / l2(∇(ν), dΩ))
    push!(eν,  l2(CellField(uex, Ω) - ν, dΩ))
    push!(eũ,  l2(CellField(uex, Ω) - ũ, dΩ))
    push!(e∇ν, l2(CellField(σex, Ω) - ∇(ν), dΩ))
    push!(e∇ũ, l2(CellField(σex, Ω) - ∇(ũ), dΩ))

    @testset "k = $k, n = $n" begin
      @test rL2[end] ≤ 1e-12
      @test rH1[end] ≤ 1e-11
      @test l2(∇(d), dΩ) ≤ 1e-3 * e∇ν[end]
      @test abs(e∇ν[end] - e∇ũ[end]) ≤ 1e-3 * e∇ν[end]
    end
  end

  r∇ν = vcat(NaN, rates(hs, e∇ν));  r∇ũ = vcat(NaN, rates(hs, e∇ũ))
  println("\nProposition 3.1, k = $k: νh of (3.2) vs ũh of (2.10), both in P$(k+2)")
  hdr = @sprintf("%5s %11s %11s %11s %11s %11s %5s %11s %5s",
                 "1/h", "‖ν-ũ‖/‖ν‖", "‖∇(ν-ũ)‖rel", "‖u-ν‖", "‖u-ũ‖",
                 "‖∇(u-ν)‖", "rate", "‖∇(u-ũ)‖", "rate")
  println(hdr);  println("-"^length(hdr))
  for i in eachindex(hs)
    rate(r) = i == 1 ? "-" : @sprintf("%.2f", r[i])
    @printf("%5d %11.2e %11.2e %11.3e %11.3e %11.4e %5s %11.4e %5s\n",
            round(Int, 1 / hs[i]), rL2[i], rH1[i], eν[i], eũ[i],
            e∇ν[i], rate(r∇ν), e∇ũ[i], rate(r∇ũ))
  end
end

for k in 0:3
  @testset "k = $k" begin
    stenberg_equivalence(k)
  end
end
