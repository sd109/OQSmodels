
### COULD WE IMPORT MANIFOLDS.JL HERE AND SIMPLY USE THE FUNCTIONS BELOW TO EXTEND FUNCTONALITY?
### This might save time in implementing (or at least provide a fallback for) geodesic tracing and other new functionality


"""
 This file contains a number of functions for calculating various properties of Riemannian metrics.
 The general structure is that each function takes 3 args which are:
    * An OQS model instance which is fed into the two functions in the next two args
    * A local_metric_func(::OQSmodel) function which calculates a Riemannian metric (e.g. an FIM) for the model given in the first arg
    * A diff_params_func(::OQSmodel) model which returns a vector of DiffParam instances which contain info about the params to be varied in the OQSmodel

*local_metric_func(::OQSmodel) should also call/make use of diff_param_func in its definition*
"""

using TensorOperations #For tensor contraction and Einstein summation (through the @tensor macro)

export christoffel_symbols_second, ricci_curvature


#Calculate derivatives of metric
function metric_jacobian(m::OQSmodel, local_metric_func, diff_param_func)
    #Construct diff param instance
    diff_params = diff_param_func(m)
    FIM_dim = length(diff_params)
    #Calculate derivs using (multithreaded?) gradient function
    derivatives = gradient(m, local_metric_func, diff_params)
    #Reshape into 3D tensor
    return reshape(hcat(derivatives...), FIM_dim, FIM_dim, FIM_dim) #3rd idx is derivative param
end


"""
`christoffel_symbols_second(m::OQSmodel, f1::func, f2::func)`

Constructs Chrsitoffel symbols of the second kind following the 
[standard formula](https://www.wikiwand.com/en/List_of_formulas_in_Riemannian_geometry).

Args:

* Function `f1` should take an OQSmodel as sole arg and return a matrix representation of a chosen Riemannian metric at that point.

* Function `f2` should take an OQSmodel as sole arg and return a Vector of `OQSmodels.DiffParam` instances which store information about the parameter space coordinates and functions to vary those parameters in `m`.

"""
function christoffel_symbols_second(m::OQSmodel, local_metric_func, diff_param_func)
   
    #Calculate some essential quantities
    diff_params = diff_param_func(m)
    FIM_dim = length(diff_params)
    FIM = local_metric_func(m) |> Array #TensorOperations.jl doesn't seem to like Symmetric matrix types so drop here
    inv_FIM = inv(FIM)
    
    #Get derivatives of metric    
    metric_J = metric_jacobian(m, local_metric_func, diff_param_func)
    
    #Rearrange metric diffs into christoffel symbols following std formula (3rd idx of metric_J is deriv param)
    @tensor begin
        Γ[m, i, j] := 0.5 * inv_FIM[m, k] * ( metric_J[k, i, j] + metric_J[k, j, i] - metric_J[i, j, k])
    end
    
    return Γ
    
end



function christoffel_symbols_second_jacobian(m::OQSmodel, metric_func, diff_param_func)
    #Construct diff param instance
    diff_params = diff_param_func(m)
    FIM_dim = length(diff_params)
    #Calculate derivs using (multithreaded?) gradient function
    local_christoffel_func(x::OQSmodel) = christoffel_symbols_second(x, metric_func, diff_param_func)
    derivatives = gradient(m, local_christoffel_func, diff_params)
    #Reshape into 4D tensor (can't use simple reshape(...) here so be explicit instead)
    Γ_jacobian = zeros(fill(FIM_dim, 4)...)
    for (i, d) in enumerate(derivatives)
        Γ_jacobian[:, :, :, i] = d
    end
    return Γ_jacobian    
end



function ricci_curvature(m::OQSmodel, local_metric_func, diff_param_func)
    
    #Calculate some essential quantities
    diff_params = diff_param_func(m)
    FIM_dim = length(diff_params)
    FIM = local_metric_func(m) |> Array #TensorOperations.jl doesn't seem to like Symmetric matrix types so drop here
    inv_FIM = inv(FIM)
    
    #Get connection coeffs and their derivatives
    Γ = christoffel_symbols_second(m, local_metric_func, diff_param_func)
    Γ_jacobian = christoffel_symbols_second_jacobian(m, local_metric_func, diff_param_func)
    
    #Combine / contract these into scalar curvature (first Γ idx is the raised one)
    @tensor begin
        S = inv_FIM[a, b] * ( Γ_jacobian[c, a, b, c] - Γ_jacobian[c, a, c, b] + Γ[d, a, b]*Γ[c, c, d] - Γ[d, a, c]*Γ[c, b, d] )
    end
    
    return S
end