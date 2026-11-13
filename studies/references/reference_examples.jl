###############################################################################
# reference_examples.jl
#
# The numerical examples of the two new references, solved with the scalar
# Signorini solver of the package (BDM_{k+1} × P_k on the Bernstein cone,
# coefficient PDAS) and driven / measured by the built-in estimator of
# Muga–Rojas–Vega (2023, Remark 3.4),  η̃_K = ‖∇ε_K‖_K.
#
#   fhs42  Führer, Heuer & Stephan, "On the DPG method for Signorini problems"
#          (arXiv:1609.00765), §4.2: Ω = (-1,1)×(0,1), -Δu + u = f, Signorini on
#          all of ∂Ω, manufactured u (smooth on x < 0 and x > 0, C¹ across x = 0).
#   fhs43  ibid. §4.3: L-shaped Ω = (-1,1)² \ [-1,0]², -Δu + u = f with
#          f = -1 for |x| ≤ 0.8 and 1/2 else, Signorini on all of ∂Ω, unknown u.
#   bgk72  Bartels, Gudi & Kaltenbach, "A priori and a posteriori error
#          identities for the scalar Signorini problem" (arXiv:2407.10912),
#          §7.2: Ω = (0,1)², Γ_C = bottom, Γ_D = rest,
#          u = -10 ψ(r) r^{3/2} sin(3θ/2) centred at (1/2, 0), ψ a C⁴ cut-off.
#   bgk731 ibid. §7.3.1: Ω = (-1,1)², Γ_C = bottom, Γ_D = top ∪ {1}×(0,1),
#          Γ_N = rest (g = 0), f = -1, obstacle χ = min{(|x₁| - 1/2)/2, 0},
#          unknown u.
#
# Not reproduced: FHS §4.4 (singularly perturbed).  Its manufactured solution as
# printed does not satisfy the Signorini conditions on x = 1 (u > 0 with
# ∂ₙu ≠ 0 there), and the example tests the ε-robustness of a DPG formulation,
# which this mixed method is not designed for.
#
# For every example and k = 0..3: uniform refinement and adaptive refinement
# (SOLVE → ESTIMATE η̃_K → MARK Dörfler θ = 1/2 → REFINE newest-vertex bisection).
# With a known solution, also the errors ‖σ-σ_h‖, ‖u-u_h‖, ‖u-ν_h‖ and the
# effectivity η̃/E, E² = ‖∇_h(u-ν_h)‖² + ‖q-q_h‖²_* (the norm of Thm 3.6).
#
# Usage (from the root of SignoriniPostprocessing.jl):
#
#     julia --project=. studies/references/reference_examples.jl
#     julia --project=. studies/references/reference_examples.jl --ex=fhs42,bgk72 --k=0,1 --nmax=50000
#
# Output (results/csv/references/):
#     <ex>_adaptive_k<k>.csv, <ex>_uniform_k<k>.csv, <ex>_adaptive_k<k>_mesh.csv
###############################################################################

using SignoriniPostprocessing
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Adaptivity
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
l2(e, dΩ) = sqrt(abs(sum(∫(e ⋅ e)dΩ)))

# =============================================================================
# Meshes and boundary tags
# =============================================================================
"Simplicial mesh of the rectangle (a,b)×(c,d) with nx × ny squares."
rect_model(a, b, c, d, nx, ny) = simplexify(CartesianDiscreteModel((a, b, c, d), (nx, ny)))

"L-shaped domain (-1,1)² \\ [-1,0]², squares of side 1/n, each split in two."
function lshape_model(n)
  N = 2n;  xs = range(-1.0, 1.0, length = N + 1)
  id(i, j) = i + (j - 1) * (N + 1)
  coords = [Point(xs[i], xs[j]) for j in 1:N+1 for i in 1:N+1]
  cells = Vector{Int}[]
  for j in 1:N, i in 1:N
    (xs[i] + xs[i+1] < 0 && xs[j] + xs[j+1] < 0) && continue      # removed quadrant
    a, b, c, d = id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1)
    push!(cells, [a, b, c]);  push!(cells, [b, d, c])               # counter-clockwise
  end
  used = sort(unique(reduce(vcat, cells)));  new = Dict(o => k for (k, o) in enumerate(used))
  # increasing vertex ids per cell: an Oriented grid, which Gridap's BDM spaces
  # with more than one dof per edge require (as simplexify produces)
  grid = UnstructuredGrid(coords[used], Table([sort([new[v] for v in c]) for c in cells]),
                          [TRI3], fill(Int8(1), length(cells)), Oriented())
  topo = get_grid_topology(UnstructuredDiscreteModel(grid))
  # standard "interior"/"boundary" labelling (the default one has no entities)
  m = UnstructuredDiscreteModel(grid, topo, FaceLabeling(topo))
  add_tag_from_tags!(get_face_labeling(m), "GammaS", ["boundary"])          # all of ∂Ω
  m
end

"""
    tag_boundary_facets!(model, name, pred)

New tag `name` for the boundary facets whose midpoint satisfies `pred`.  They get
a fresh entity id, which newest-vertex bisection passes on to their children.
"""
function tag_boundary_facets!(model, name, pred)
  labels = get_face_labeling(model)
  topo   = get_grid_topology(model)
  D      = num_cell_dims(model)
  f2v    = get_faces(topo, D - 1, 0)
  X      = get_vertex_coordinates(topo)
  ent    = labels.d_to_dface_to_entity[D]
  newid  = maximum(maximum, labels.d_to_dface_to_entity) + 1
  for f in findall(get_isboundary_face(topo, D - 1))
    pred(sum(X[v] for v in f2v[f]) / length(f2v[f])) && (ent[f] = newid)
  end
  add_tag!(labels, name, [newid])
  model
end

# =============================================================================
# Examples
# =============================================================================
# FHS §4.2 ------------------------------------------------------------------
fhs42_u(x) = x[1] ≥ 0 ? -16 * x[1]^2 * (1 - x[1]) * x[2] * (1 - x[2]) :
             (s = x[1] + 1; 2s^3 - 3s^2 + 1)
fhs42_σ(x) = x[1] ≥ 0 ?
  VectorValue(-16 * (2x[1] - 3x[1]^2) * x[2] * (1 - x[2]), -16 * x[1]^2 * (1 - x[1]) * (1 - 2x[2])) :
  (s = x[1] + 1; VectorValue(6s^2 - 6s, 0.0))
fhs42_lap(x) = x[1] ≥ 0 ? -16 * (2 - 6x[1]) * x[2] * (1 - x[2]) + 32 * x[1]^2 * (1 - x[1]) :
               12 * (x[1] + 1) - 6
fhs42_f(x) = -fhs42_lap(x) + fhs42_u(x)

# FHS §4.3 ------------------------------------------------------------------
fhs43_f(x) = sqrt(x[1]^2 + x[2]^2) ≤ 0.8 ? -1.0 : 0.5

# BGK §7.2 ------------------------------------------------------------------
const BGK_R = 0.45
_S1(t) = 630 * t^4 * (1 - t)^4                 # S(t) = t⁵(126 - 420t + 540t² - 315t³ + 70t⁴)
_S(t)  = t^5 * (126 - 420t + 540t^2 - 315t^3 + 70t^4)
_S2(t) = 2520 * t^3 * (1 - t)^3 * (1 - 2t)
ψ(r)   = r < BGK_R ? 1 - _S(r / BGK_R) : 0.0   # ψ(0) = 1, ψ(R) = 0, ψ⁽ⁱ⁾ = 0 at both, i = 1..4
ψ1(r)  = r < BGK_R ? -_S1(r / BGK_R) / BGK_R : 0.0
ψ2(r)  = r < BGK_R ? -_S2(r / BGK_R) / BGK_R^2 : 0.0
_rθ(x) = (sqrt((x[1] - 0.5)^2 + x[2]^2), atan(abs(x[2]), x[1] - 0.5))
bgk72_u(x) = ((r, t) = _rθ(x); -10 * ψ(r) * r^1.5 * sin(1.5t))
bgk72_σ(x) = ((r, t) = _rθ(x);
  -10 * (ψ1(r) * r^1.5 * sin(1.5t) * VectorValue(cos(t), sin(t)) +
         ψ(r) * 1.5 * sqrt(r) * VectorValue(sin(t / 2), cos(t / 2))))
bgk72_f(x) = ((r, t) = _rθ(x); 10 * sin(1.5t) * sqrt(r) * (4ψ1(r) + r * ψ2(r)))

function bgk72_model(n)
  m = rect_model(0, 1, 0, 1, n, n);  lab = get_face_labeling(m)
  add_tag_from_tags!(lab, "GammaS", ["tag_5"])
  add_tag_from_tags!(lab, "GammaD", ["tag_6", "tag_7", "tag_8"])
  m
end

# BGK §7.3.1 ------------------------------------------------------------------
bgk731_χ(x) = min(0.5 * (abs(x[1]) - 0.5), 0.0)
function bgk731_model(n)
  m = rect_model(-1, 1, -1, 1, 2n, 2n);  tol = 1e-12
  tag_boundary_facets!(m, "GammaS", p -> abs(p[2] + 1) < tol)
  tag_boundary_facets!(m, "GammaD", p -> abs(p[2] - 1) < tol || (abs(p[1] - 1) < tol && p[2] > 0))
  tag_boundary_facets!(m, "GammaN", p -> abs(p[1] + 1) < tol || (abs(p[1] - 1) < tol && p[2] < 0))
  m
end

const zero2 = x -> VectorValue(0.0, 0.0)

const EXAMPLES = Dict(
  "fhs42"  => (model = n -> rect_model(-1, 1, 0, 1, 2n, n), n0 = 1,
               data = (f = fhs42_f, tag_S = "boundary", reaction = 1.0),
               exact = (u = fhs42_u, σ = fhs42_σ)),
  "fhs43"  => (model = lshape_model, n0 = 1,
               data = (f = fhs43_f, tag_S = "GammaS", reaction = 1.0),
               exact = nothing),
  "bgk72"  => (model = bgk72_model, n0 = 2,
               data = (f = bgk72_f, tag_S = "GammaS", uD = bgk72_u, tag_D = "GammaD"),
               exact = (u = bgk72_u, σ = bgk72_σ)),
  "bgk731" => (model = bgk731_model, n0 = 1,
               data = (f = x -> -1.0, tag_S = "GammaS", uD = x -> 0.0, tag_D = "GammaD",
                       tag_N = "GammaN", gN = zero2, χ = bgk731_χ),
               exact = nothing),
)

# =============================================================================
# One solve: estimator, and errors when the solution is known
# =============================================================================
function step!(model, k, ex)
  sol = solve_scalar_signorini(model; k, ex.data...)
  p  = k + 1
  Ω  = sol.Ω
  dΩ = Measure(Ω, 2k + 12)
  pp = residual_minimization_postprocess(model, -sol.σh, sol.uh, p)
  η2 = collect(Gridap.CellData.get_contribution(∫(∇(pp.εh) ⋅ ∇(pp.εh))dΩ, Ω))
  η  = sqrt(sum(η2))
  row = (; ndof = sol.ndofs, ncells = num_cells(Ω), pdas_it = sol.it,
           converged = Int(sol.converged), eta = η)
  if ex.exact !== nothing
    ∇u = CellField(ex.exact.σ, Ω);  u = CellField(ex.exact.u, Ω)
    zh = stenberg_postprocess(model, ∇u - sol.σh, CellField(0.0, Ω), p + 1;
                              qdegree = 2k + 12).ũh          # Riesz rep. of q - q_h
    θh = stenberg_postprocess(model, -sol.σh, sol.uh, p + 1).ũh   # (3.3), saturation
    e∇ = l2(∇u - ∇(pp.νh), dΩ);  edual = l2(∇(zh), dΩ)
    E  = sqrt(e∇^2 + edual^2)
    row = (; row..., err_E = E, eff = η / E, delta = l2(∇u - ∇(θh), dΩ) / e∇,
             err_sigma_L2 = l2(∇u - sol.σh, dΩ), err_u_L2 = l2(u - sol.uh, dΩ),
             err_nu_L2 = l2(u - pp.νh, dΩ), err_grad_nu_L2 = e∇, err_dual = edual)
  end
  sol.converged || @warn "active set did not converge" k row.ndof
  return row, η2
end

function run_adaptive(ex, k; θ = 0.5, nmax = 50_000, maxsteps = 60, label = "")
  model = ex.model(ex.n0);  rows = [];  marking = DorflerMarking(θ)
  for s in 0:maxsteps
    row, η2 = step!(model, k, ex)
    push!(rows, (; step = s, row...))
    @printf("  %-7s adaptive k=%d step %2d ndof %6d η̃ %.3e%s it %d\n", label, k, s, row.ndof,
            row.eta, haskey(row, :eff) ? @sprintf(" eff %.3f", row.eff) : "", row.pdas_it)
    row.ndof ≥ nmax && break
    model = get_model(refine(model; refinement_method = "nvb",
                             cells_to_refine = Gridap.Adaptivity.mark(marking, η2)))
  end
  rows, model
end

function run_uniform(ex, k; nmax = 50_000, label = "")
  rows = [];  n = ex.n0
  while true
    row, _ = step!(ex.model(n), k, ex)
    push!(rows, (; step = n, row...))
    @printf("  %-7s uniform  k=%d n=%3d  ndof %6d η̃ %.3e%s\n", label, k, n, row.ndof, row.eta,
            haskey(row, :eff) ? @sprintf(" eff %.3f", row.eff) : "")
    row.ndof ≥ nmax && break
    n *= 2
  end
  rows
end

# =============================================================================
# Output
# =============================================================================
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
end
function write_mesh(path, model)
  open(path, "w") do io
    println(io, "x1,y1,x2,y2,x3,y3")
    for c in get_cell_coordinates(Triangulation(model))
      println(io, join((@sprintf("%.12e", v) for P in c for v in (P[1], P[2])), ","))
    end
  end
end

"Least-squares slope s of log e vs log N over the last decade of N (e ~ N^s)."
function slope(rows, col)
  N = [r.ndof for r in rows];  e = [getfield(r, col) for r in rows]
  m = N .≥ N[end] / 10
  x = log.(N[m]);  y = log.(e[m]);  x̄ = sum(x) / length(x);  ȳ = sum(y) / length(y)
  sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

function parse_args(args)
  exs = ["fhs42", "fhs43", "bgk72", "bgk731"];  ks = [0, 1, 2, 3];  nmax = 50_000
  for a in args
    if startswith(a, "--ex=");       exs = String.(split(a[6:end], ","))
    elseif startswith(a, "--k=");    ks = parse.(Int, split(a[5:end], ","))
    elseif startswith(a, "--nmax="); nmax = parse(Int, a[8:end])
    else error("unknown argument $a (expected --ex=, --k=, --nmax=)")
    end
  end
  all(e -> haskey(EXAMPLES, e), exs) || error("examples: $(sort(collect(keys(EXAMPLES))))")
  exs, ks, nmax
end

# Main block only when run as a script (studies/estimators includes this file).
if abspath(PROGRAM_FILE) == @__FILE__
let (exs, ks, nmax) = parse_args(ARGS)
  outdir = joinpath(ROOT, "results", "csv", "references");  mkpath(outdir)
  t0 = time();  summary = []
  for name in exs, k in ks
    ex = EXAMPLES[name]
    println("\n=== $name, k = $k ===")
    ad, mesh = run_adaptive(ex, k; nmax, label = name)
    un = run_uniform(ex, k; nmax, label = name)
    write_rows(joinpath(outdir, "$(name)_adaptive_k$(k).csv"), k, ad)
    write_rows(joinpath(outdir, "$(name)_uniform_k$(k).csv"), k, un)
    write_mesh(joinpath(outdir, "$(name)_adaptive_k$(k)_mesh.csv"), mesh)
    push!(summary, (name, k, ad, un))
  end

  println("\nSlopes s in e ~ N^s over the last decade of N (optimal adaptive: η̃, σ: -(k+2)/2; ν: -(k+3)/2)")
  for (name, k, ad, un) in summary
    for (lab, rows) in (("adaptive", ad), ("uniform", un))
      line = @sprintf("%-7s k=%d %-9s η̃ %6.2f", name, k, lab, slope(rows, :eta))
      if haskey(rows[end], :eff)
        effs = [r.eff for r in rows if r.ndof ≥ rows[end].ndof / 10]
        line *= @sprintf("  σ %6.2f  u_h %6.2f  ν %6.2f  eff [%.2f, %.2f]",
                         slope(rows, :err_sigma_L2), slope(rows, :err_u_L2),
                         slope(rows, :err_nu_L2), minimum(effs), maximum(effs))
      end
      println(line, "  (PDAS it ≤ $(maximum(r.pdas_it for r in rows)))")
    end
  end
  println("\ndone in ", round(time() - t0; digits = 1), " s")
end
end
