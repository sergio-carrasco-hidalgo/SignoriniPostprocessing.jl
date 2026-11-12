###############################################################################
# runtests.jl -- SignoriniPostprocessing
#
#   julia --project -e 'using Pkg; Pkg.test()'
#
#   1. convergence of the scalar Signorini solver, k = 0, 1, 2, 3;
#   2. superconvergence of the Section 3.1 postprocessing on those solutions.
#
# The second group reuses the solves of the first (cache in manufactured.jl).
###############################################################################

using SignoriniPostprocessing
using Gridap
using Test

include("manufactured.jl")
using .Manufactured

const t0 = time()

@testset verbose = true "SignoriniPostprocessing.jl" begin
  @testset verbose = true "scalar Signorini: convergence" begin
    include("test_signorini_convergence.jl")
  end
  @testset verbose = true "Section 3.1 postprocessing: superconvergence" begin
    include("test_superconvergence.jl")
  end
  @testset verbose = true "Proposition 3.1: (3.2) ≡ (2.10)" begin
    include("test_stenberg_equivalence.jl")
  end
end

println("\ntest suite finished in ", round(time() - t0; digits = 1), " s")
