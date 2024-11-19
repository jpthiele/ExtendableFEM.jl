#=

# D01 : Poisson-Problem
([source code](@__SOURCE_URL__))

This example computes the solution ``u`` of the D-dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with right-hand sides ``f(x)=4x^4``,
``f(x,y,z) \equiv 4(x^4+y^4)`` 
and ``f(x,y,z) \equiv 4(x^4+y^4+z^4)``
depending on the dimension and Dirichlet boundary conditions
given by ``g(x)=x^2``, ``g(x,y)=x^2+y^2`` 
and ``g(x,y,z)=x^2+y^2,z^2`` respectively.

on the unit cube domain ``\Omega`` on a given grid. The computed solution for the default
parameters looks like this:

![](exampleD01.png)

=#

module ExampleD01_PoissonProblem

using ExtendableFEM
using ExtendableGrids
using Test #hide

function rhs!(fval, qpinfo)
    fval[1] = 0
    for i ∈ eachindex(qpinfo.x)
        fval[1] += 4*(qpinfo.x[i]^4)
    end
    return nothing
end

function bdata!(fval,qpinfo)
    fval[1] = 0
    for i ∈ eachindex(qpinfo.x)
        fval[1] += (qpinfo.x[i]^2)
    end
    return nothing
end

function grid_hypercube(dim,scale=2,shift=-1)
    if dim == 1
	    return uniform_refine(simplexgrid(shift:scale:1), nrefs)
    elseif dim == 2
	    return uniform_refine(grid_unitsquare(Triangle2D,scale=[scale,scale],shift=[shift,shift]) nrefs)
    else
	    return uniform_refine(grid_unitcube(Tetrahedron3D,scale=[scale,scale,scale],shift=[shift,shift,shift]), nrefs)
    end
end

function main(dim = 2; μ = 1.0, nrefs = 3, Plotter = nothing, kwargs...)

	## problem description
	PD = ProblemDescription()
	u = Unknown("u"; name = "potential")
	assign_unknown!(PD, u)
	assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, kwargs...))
	assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
	assign_operator!(PD, InterpolateBoundaryData(u, bdata!; regions = 1:4))

	## discretize
    xgrid = grid_hypercube(dim)
	FES = FESpace{H1P2{1, 3}}(xgrid)

	## solve
	sol = solve(PD, FES; kwargs...)

	## plot
	plt = plot([id(u)], sol; Plotter = Plotter)

	return sol, plt
end

generateplots = default_generateplots(ExampleD01_PoissonProblem, "exampleD01.png") #hide
function runtests() #hide
	sol, plt = main(;) #hide
	@test sum(sol.entries) ≈ 21.874305144549524 #hide
end #hide
end # module
