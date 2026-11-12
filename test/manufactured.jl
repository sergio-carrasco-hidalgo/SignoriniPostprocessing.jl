###############################################################################
# manufactured.jl
#
# Test fixture shared by the convergence and superconvergence tests: the exact
# solution, the meshes, a cache of the Signorini solves (so the postprocessing
# test reuses them) and small helpers for norms, rates and tables.
#
# Exact solution (Ω = (-1/2, 1/2) × (0, 1), Γ_S = {y = 0}, Γ_D = rest):
#
#     u(r, θ) = -r^{11/2} sin(11θ/2) + c y²,       θ ∈ [0, π],
#
# the classical scalar Signorini benchmark plus a bulk term c y².  On Γ_S:
#
#   * x < 0 (θ = π):  u = r^{11/2} > 0,        σ·n = -∂_y u = 0         (separation)
#   * x > 0 (θ = 0):  u = 0,                   σ·n = (11/2) r^{9/2} ≥ 0 (contact)
#
# so u ≥ 0, σ·n ≥ 0 and u σ·n = 0 hold, with the free boundary at the mesh
# vertex x = 0.  The term c y² vanishes on Γ_S together with its normal
# derivative, so it leaves the Signorini conditions untouched and only adds
# the source f = -Δu = -2c.  f is constant (f ∈ P_k for every k), which is the
# case where Remark 2.6 of Muga–Rojas–Vega predicts a gain of two orders for
# the postprocessed scalar also at k = 0 (p = 1).
#
# u ∈ H^{13/2-ε}(Ω): enough for the optimal rates up to k = 3.
###############################################################################

module Manufactured

using Gridap
using Gridap.CellData
using Gridap.Arrays
using FillArrays
using Printf
using SignoriniPostprocessing

export uex, σex, fex, square_model, LEVELS, signorini_level, l2, rates, print_table,
       sample_on_facets, TAG_D, TAG_S

const α  = 11 / 2
const cb = 1.0
const TAG_D = "GammaD"
const TAG_S = "GammaS"

# θ = atan(|y|, x): on y = 0 a quadrature point with y = -0.0 must not jump to θ = -π.
θf(x) = atan(abs(x[2]), x[1])
rf(x) = sqrt(x[1]^2 + x[2]^2)

uex(x) = -rf(x)^α * sin(α * θf(x)) + cb * x[2]^2

# ∇(-r^α sin(αθ)) = -α r^(α-1) (sin((α-1)θ), cos((α-1)θ)), written analytically.
σex(x) = (ρ = rf(x); t = θf(x);
          -α * ρ^(α - 1) * VectorValue(sin((α - 1) * t), cos((α - 1) * t)) +
          VectorValue(0.0, 2cb * x[2]))

fex(x) = -2cb

"Simplicial n×n mesh of Ω with the tags Γ_S (bottom) and Γ_D (top, left, right)."
function square_model(n::Int)
  model  = simplexify(CartesianDiscreteModel((-0.5, 0.5, 0.0, 1.0), (n, n)))
  labels = get_face_labeling(model)
  # Gridap's 2D Cartesian convention: tag_5 bottom, tag_6 top, tag_7 left, tag_8 right
  add_tag_from_tags!(labels, TAG_S, ["tag_5"])
  add_tag_from_tags!(labels, TAG_D, ["tag_6", "tag_7", "tag_8"])
  return model
end

# Mesh levels n per degree k.  For k = 3 the finest level stops at n = 16: at
# n = 32 the BDM_4 flux error (≈ 8e-10) has reached a floor that the plain mixed
# Poisson problem with the same exact solution and Dirichlet data everywhere
# reproduces to three digits, so it is unrelated to the contact condition.
const LEVELS = Dict(0 => [8, 16, 32, 64],
                    1 => [4, 8, 16, 32],
                    2 => [4, 8, 16, 32],
                    3 => [4, 8, 16])

const CACHE = Dict{Tuple{Int,Int},Any}()

"Solve (and cache) the scalar Signorini problem of degree k on the n×n mesh."
signorini_level(k::Int, n::Int) = get!(CACHE, (k, n)) do
  model = square_model(n)
  sol   = solve_scalar_signorini(model; k, f = fex, uD = uex,
                                 tag_D = TAG_D, tag_S = TAG_S)
  (; model, sol, h = 1 / n)
end

l2(e, dΩ) = sqrt(abs(sum(∫(e ⋅ e)dΩ)))

"Observed orders log(e_i/e_{i-1}) / log(h_i/h_{i-1})."
rates(hs, es) = [log(es[i] / es[i-1]) / log(hs[i] / hs[i-1]) for i in 2:length(es)]

"Values of the facet field `fh` at `npts` equispaced points of every facet of Γ."
function sample_on_facets(fh, Γ, npts::Int)
  pts = [Point(s) for s in range(0, 1, length = npts)]
  collect(lazy_map(evaluate, get_data(change_domain(fh, ReferenceDomain())),
                   Fill(pts, num_cells(Γ))))
end

"Print a convergence table: one column pair (error, rate) per entry of `cols`."
function print_table(title, hs, extra, cols)
  println("\n", title)
  hdr = @sprintf("%6s %8s", "1/h", "ndofs") *
        join([@sprintf(" %11s %6s", name, "rate") for (name, _) in cols])
  println(hdr);  println("-"^length(hdr))
  rs = [vcat(NaN, rates(hs, es)) for (_, es) in cols]
  for i in eachindex(hs)
    line = @sprintf("%6d %8d", round(Int, 1 / hs[i]), extra[i])
    for (j, (_, es)) in enumerate(cols)
      line *= @sprintf(" %11.3e %6s", es[i], i == 1 ? "-" : @sprintf("%.2f", rs[j][i]))
    end
    println(line)
  end
end

end # module Manufactured
