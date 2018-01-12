# This package can solve two player dynamic game by collocation method
module DynamicGame

using BasisMatrices
using QuantEcon
using Distributions

export
vmax, coll, collocation_method, residual

# update value function and policy function given the basis coefficient
function vmax(model, 
    x::Array{Float64, 2},　
    colnodes::Array{Float64, 2},　
    b, 
    coef::Array{Float64, 2}, 
    epss::Array{Float64, 2}, 
    weights::Array{Float64, 1}, 
    u,
    ux,
    uxx,
    s,
    sx,
    sxx,
    tol=0.000000001, 
    maxit=10000)
    
    xnew = x
    v = zeros((size(colnodes)[1], 2))
    vnew = v
    for p in 1:2
        xl, xu = 0.0, colnodes[:, p]
        order1 = [0 0]
        order1[1, p] = 1
        order2 = [0 0]
        order2[1, p] = 2
        for it in 1:maxit
            util, util_der1, util_der2 = u(model, colnodes, xnew, p), ux(model, colnodes, xnew, p), uxx(model, colnodes, xnew, p)
            Ev, Evx, Evxx = 0.0, 0.0, 0.0
            for k in 1:size(epss)[1]
                eps, weight= epss[k, :], weights[k]
                transition, transition_der1, transition_der2 = s(model, xnew, eps), sx(model, xnew, eps, p), sxx(model, xnew, eps, p)
                vn = funeval(coef[:, p], b, transition)
                vnder1 =  funeval(coef[:, p], b, transition, order1)
                vnder2 = funeval(coef[:, p], b, transition, order2)
                Ev += weight *vn
                Evx += weight* vnder1.* transition_der1
                Evxx += weight * (vnder1.*transition_der2 + vnder2 .* (transition_der1.^2))
            end
            v[:, p] = util + model.delta*Ev
            delx = -(util_der1 + model.delta * Evx) ./ (util_der2 + model.delta*Evxx)
            delx = min.(max.(delx, xl-xnew[:, p]), xu-xnew[:, p])
            xnew[:, p] = x[:, p] + delx
            if norm(delx) < tol
                break
            end
        end
        vnew[:, p] = v[:, p]
    end
    return vnew, xnew
end

# generate elements for collocation method
function coll(smax::Int64, smin::Int64, n::Int64)
    # smax is the maximum of the state variable
    # smin is the minimum of the state variable
    # n is the number of grid for one dimension
    sgrid0 = linspace(smin, smax, n)
    basis = Basis(SplineParams(sgrid0, 0, 3), SplineParams(sgrid0, 0, 3))
    S, (coordx, coordy) = nodes(basis)
    Φ = BasisMatrix(basis, Expanded(), S, 0)
    return basis, S, Φ, coordx, coordy
end

# collocationmethod main loop
function collocation_method(model, 
    x_initial::Array{Float64, 2}, 
    c_initial::Array{Float64, 2}, 
    S::Array{Float64,2}, 
    basis, 
    Φ, 
    e::Array{Float64,2}, 
    w::Array{Float64,1}, 
    u,
    ux,
    uxx,
    s,
    sx,
    sxx,
    maxit = 1000, 
    tol = 0.000000001)
    
    c = c_initial
    x = x_initial
    c_error = c
    count = 0
    for it in 1:maxit
        cold = c
        vnew, x = vmax(model, x, S, basis, c, e, w, u, ux, uxx, s, sx, sxx)
        c = Φ.vals[1] \ vnew
        v = vnew
        c_error = cold - c
        count += 1
        if maximum(abs, cold - c) < tol 
            print(maximum(abs, cold - c) )
            break
        end
    end
    return x, vnew, c
end

# Bellman_residual calculation
# true_x, c is provided by the above collocation method function
function residual(model, 
    newn::Int64, 
    smin::Int64, 
    smax::Int64, 
    true_x::Array{Float64, 2}, 
    basis, 
    c::Array{Float64, 2}, 
    e::Array{Float64,2}, 
    w::Array{Float64,1},
    u,
    ux,
    uxx,
    s,
    sx,
    sxx)
    
    gri = linspace(smin, smax, nn)
    new_grid = gridmake(gri, gri)
    
    c_x =Φ.vals[1] \ true_x
    newx = funeval(c_x, basis, new_grid)
    
    v_true = vmax(model, newx[:,:, 1], new_grid, basis, c, e, w, u, ux, uxx, s, sx, sxx)[1][:, 1];
    predict_value = funeval(c, basis, new_grid)[:, 1];
    
    resid = reshape(v_true - predict_value, nn, nn)
    return resid
end


end



















