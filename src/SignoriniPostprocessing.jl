"""
    SignoriniPostprocessing

Scalar Signorini problem in dual-mixed form (BDM_{k+1} × P_k), with the contact
condition imposed on a convex cone of Bernstein coefficients and solved by a
primal-dual active set, followed by the superconvergent postprocessing based on
local residual minimization of Muga, Rojas & Vega (SIAM J. Numer. Anal. 2023,
Section 3.1).

Submodules
  * `BernsteinTrace`        Bernstein form of polynomial facet traces (the
                            convex-hull certificate used to define the cone).
  * `ScalarSignorini`       the cone K_h, its parametrisation x = T x̃ and the
                            coefficient-wise active-set solver.
  * `ResidualMinimization`  the local mixed problems (3.2) giving ν_h and ε_h.
  * `StenbergPostprocess`   the local problems (2.10) giving ũ_h (= ν_h, Prop. 3.1).
"""
module SignoriniPostprocessing

include("BernsteinTrace.jl")
include("ScalarSignorini.jl")
include("ResidualMinimization.jl")
include("StenbergPostprocess.jl")

using .BernsteinTrace
using .ScalarSignorini
using .ResidualMinimization
using .StenbergPostprocess

export BernsteinTrace, ScalarSignorini, ResidualMinimization, StenbergPostprocess

export bernstein_nodes, bernstein_transform, bernstein_basis, facet_bernstein_coeffs
export SignoriniFacets, signorini_facets, cone_transform, cone_multipliers,
       solve_scalar_signorini
export residual_minimization_postprocess
export stenberg_postprocess

end # module SignoriniPostprocessing
