# module ModelCalculus


# using ..ModelConstruction, ..ModelTimeEvolution, 
using FiniteDiff, LinearAlgebra, Statistics, IterTools
using FiniteDifferences #Higher order FD formulae

"""
   General functions which are used for all param derivatives:
-----------------------------------------------------------------
"""

export DiffParam
struct DiffParam{N <: Number,F <: Any,S <: Any}
    name::String # Param name for labelling stuff
    param_val::N # Param value at which to diff
    var_func::F # Function which takes model, val as params and sets param in model to val then returns new model
    scaling::S # Value to use for rescaling bare parameters - #Could be a real number or a scaling funcs which take a (modified) model as only arg
end


# Calculate (1/θo - θ/θo^2 dθo/dθ) since, if θo = θo(θ) & η = θ/θo, then df/dη = df/dθ  / (1/θo - θ/θo^2 dθo/dθ)
# function param_deriv_scale(model, p::DiffParam)
#    θ = p.param_val
#    θo = p.scaling(model)
#    scale(x) = p.scaling(p.var_func(model, x))
#    dscale = FiniteDiff.finite_difference_derivative(scale, float(θ)) # == dθo/dθ   (x val needs to be float for finite diff)
#    return 1 / (1 / θo - θ / θo^2 * dscale)
# end


export ModelGradient
struct ModelGradient{X <: OQSmodel,F <: Any,D <: DiffParam,Y <: Any,Z <: Any}
    model::X
    fom::F
    params::Array{D,1}
    scale_vec::Array{Y,1}
    grad_vec::Array{Z,1}
end
# Default constructor
function ModelGradient(model::OQSmodel, fom, params::Vector{D} where D <: DiffParam)
   # Get basic gradient
    bare_grad = gradient(model, fom, params)
   # Return the case where model.options.scaled_params == false
    if !model.options.scaled_params
        return ModelGradient(model, fom, params, ones(length(params)), bare_grad)
    end
   # Otherwise, work out the scaling vec
   # get_scale(p::DiffParam) = typeof(p.scaling) <: Number ? p.scaling : param_deriv_scale(model, p)
   get_scale(p::DiffParam) = p.scaling == :self ? p.param_val : p.scaling
   scale_vec = map(get_scale, params)
   return ModelGradient(model, fom, params, scale_vec, scale_vec .* bare_grad)
end
# Custom show method
Base.show(io::IO, x::ModelGradient) = show(io, x.grad_vec)


# Derivatives of FoM(model) w.r.t. each DiffParam instance in param_list
export gradient
function gradient(model::OQSmodel, FoM, param_list::Vector{P} where P <: DiffParam)

   fx = FoM(model)
   param_list_len = length(param_list)
   grad = fill(zero(fx), param_list_len) # Pre-allocate grad list

   # Use all available threads for derivatives w.r.t. different parameters
   Threads.@threads for i in 1:param_list_len
      if model.options.extrapolate_deriv
         grad[i] = extrapolate_deriv(model, FoM, param_list[i])
      else
         grad[i] = model.options.finite_diff_func(x -> FoM(param_list[i].var_func(model, x)), param_list[i].param_val)
      end
   end

   return grad
end

# Function to return the names associated with each gradient
export named_gradient
named_gradient(model, FoM, param_list) = Dict(name => grad for (name, grad) in zip(getfield.(param_list, :name), gradient(model, FoM, param_list)))


function extrapolate_deriv(model::OQSmodel, FoM, p::DiffParam)
 

   #Catch error with wrong deriv func
   model.options.finite_diff_func == FiniteDiff.finite_difference_derivative && throw(error("Only methods from FiniteDifferences.jl can be used for derivative extrapolation."))

   #Pick a sensible initial step size
   p0 = p.param_val == 0 ? 1e-2 : p.param_val/100

   #Calculate derivative (contraction rates closer to 1 are more accurate, default is 0.125 in FiniteDifferences)
   deriv, err = extrapolate_fdm(model.options.finite_diff_func, v -> FoM(p.var_func(model, v)), p.param_val, p0, contract=0.8, breaktol=100, rtol=1e-10)

   #Warn if error estimate is large - NOT SURE WHAT THE BEST WAY TO DO THIS IS -> maybe compare to rtol value instead??
   ref = maximum(abs.(deriv))
   if ref > 1e-8 && err / ref > 1e-2
      @warn "extrapolate_deriv error = $(round(err, sigdigits=3)) with (reference) deriv value = $(round(ref, sigdigits=3)) - numerical issues?"
   end

   return deriv
end




"""
   Implementation of on-site energy derivatives:
---------------------------------------------------
"""

# Simple functions for constructing a list of site energy DiffParams
export SiteEnergyDeriv 
# Single site, single scale:
function SiteEnergyDeriv(model::OQSmodel, site::Int, scale) 
   param_val = model.Ham.param_dict["E$(site)"]
   var_func = (m, val) -> vary_Hamiltonian_param(m, "E$(site)", val)
   return DiffParam("E$(site)", param_val, var_func, scale)
end
# Multi-site, single scale:
SiteEnergyDeriv(model, sites::AbstractArray, scale) = [SiteEnergyDeriv(model, s, scale) for s in sites]
# Multi-site, diff scales (e.g. log params):
SiteEnergyDeriv(model, sites::AbstractArray, scales::Array{N,1} where N <: Real) = [SiteEnergyDeriv(model, i, j) for (i, j) in zip(sites, scales)]
# All sites, single scale:
SiteEnergyDeriv(model::OQSmodel, scale) = SiteEnergyDeriv(model, 1:numsites(model), scale)




"""
   Implementation of position co-ord derivatives:
----------------------------------------------------
"""

# Simple functions for constructing a list of site coord
export PosDeriv
# Single site, single coord constructor:
function PosDeriv(model, site::Int, coord::Symbol, scale)
   # Check that site 1 is at origin otherwise warn
   x1, y1, z1 = map(x -> model.Ham.param_dict[x], ["x1", "y1", "z1"])
   ([x1, y1, z1] != zeros(3) && model.options.polar_warning) && ( @warn "Site 1 not situated at origin - if this is intentional then silence this warning with by setting model.options.polar_warning = false" )
   return DiffParam("$(coord)$(site)", model.Ham.param_dict["$(coord)$(site)"], (model, val) -> vary_Hamiltonian_param(model, "$(coord)$(site)", val), scale)
end
# Single site, all 3 coords, same scale:
PosDeriv(model, site::Int, scale) = [PosDeriv(model, site, c, scale) for c in [:x, :y, :z]]
# All 3 coords with different scales:
PosDeriv(model, site::Int, scales::Array{N,1} where N <: Real) = [PosDeriv(model, site, c, s) for (c, s) in zip([:x, :y, :z], scales)]
# Multi-site, single coord, different scales:
PosDeriv(model, sites::AbstractArray, coord::Symbol, scales::Array{N,1} where N <: Real) = length(sites) == length(scales) ? [PosDeriv(model, i, coord, j) for (i, j) in zip(sites, scales)] : error("Must provide same number of sites and scale values")
# Multi-site, single coord, same scale:
PosDeriv(model, sites::AbstractArray, coord::Symbol, scale) = [PosDeriv(model, i, coord, scale) for i in sites]
# Multi-site, all coords, same scale:
PosDeriv(model, sites::AbstractArray, scale) = hcat([PosDeriv(model, s, scale) for s in sites]...) 




# Polar co-ords
# Simple functions for constructing a list of site coord derivatives
export PolarPosDeriv
# Single site, single coord version:
function PolarPosDeriv(model, site::Int, coord::Symbol, scale)
   # Check that site 1 is at origin otherwise warn
   x1, y1, z1 = map(x -> model.Ham.param_dict[x], ["x1", "y1", "z1"])
   ([x1, y1, z1] != zeros(3) && model.options.polar_warning) && ( @warn "Site 1 not situated at origin - if this is intentional then silence this warning with by setting model.options.polar_warning = false" )
   return DiffParam("$(coord)1$(site)", model.Ham.param_dict["$(coord)1$(site)"], (model, val) -> vary_Hamiltonian_param(model, "$(coord)1$(site)", val), scale)
end
# Single site, all coords, same scale:
PolarPosDeriv(model, site::Int, scale) = [PolarPosDeriv(model, site, c, scale) for c in [:r, :ϕ, :θ]]
# Single site, all coords, diff scales (e.g. log-params):
PolarPosDeriv(model, site::Int, scales::Array{N,1} where N <: Real) = [PolarPosDeriv(model, site, i, j) for (i, j) in zip([:r, :θ, :ϕ], scales)]
# Multi-site, all coords, same scale
PolarPosDeriv(model, sites::AbstractArray, scale) = [PolarPosDeriv(model, s, scale) for s in sites]



"""
   Implementation of coherent coupling constant deriv:
------------------------------------------------------------------
"""

export CoherentCouplingConstantDeriv
CoherentCouplingConstantDeriv(model::OQSmodel, scale) = DiffParam("J", model.Ham.param_dict["J"], (m, val) -> vary_Hamiltonian_param(m, "J", val), scale)





"""
   Implementation of CollapseOp rate derivatives:
----------------------------------------------------
"""

# Function to vary a single CollapseOp rate
function vary_collapse_rate!(model::OQSmodel, proc_name::String, new_val::Number; update_generator=true)

   # Find idx in env_processes
   proc_idx = findfirst(n -> n == proc_name, getfield.(model.env_processes, :name))
   proc_idx === nothing && throw(error("name not found in env_processes"))
   # Set env rate to new value
   model.env_processes[proc_idx].rate = new_val
    # Recalculate L and overwrite relevant values in model
   if update_generator
      new_model = OQSmodel(model.Ham, model.env_processes, model.ME_type, model.InitState)
      model.C_ops = new_model.C_ops
      model.L = new_model.L
   end

   return model
end

vary_collapse_rate(model::OQSmodel, proc_name::String, new_val::Number) = vary_collapse_rate!(deepcopy(model), proc_name, new_val)


# Simple functions to construct collapse rate DiffParams
export CollapseRateDeriv
# Single proc_name, single scale:
function CollapseRateDeriv(model::OQSmodel, name::String, scale)
   proc_idx = findfirst(x -> x == name, getfield.(model.env_processes, :name))
   proc_idx === nothing && error("Process name not found in model.env_processes")
   return DiffParam(name, model.env_processes[proc_idx].rate, (model, val) -> vary_collapse_rate(model, name, val), scale)
end
# Multi-proc, diff scales:
CollapseRateDeriv(model::OQSmodel, name_list::Array{String,1}, scales::Array{T,1} where T) = [CollapseRateDeriv(model, name, s) for (name, s) in zip(name_list, scales)] 
# Multi-proc, same scale:
CollapseRateDeriv(model::OQSmodel, name_list::Array{String,1}, scale) = [CollapseRateDeriv(model, name, scale) for name in name_list]

# Alternative version for varying inverse rate (e.g. lifetimes) -- NEED TO UPDATE TO WORK WITH PARAM SCALES
# export CollapseInverseRateDeriv
# vary_inv_collapse_rate(model::OQSmodel, proc_name::String, new_val) = vary_collapse_rate(model, proc_name, 1/new_val)
# CollapseInverseRateDeriv(model::OQSmodel, name::String, scale) = DiffParam(name, 1/model.env_processes[findfirst(x -> x == name, getfield.(model.env_processes, :name))].rate, (model, val) -> vary_inv_collapse_rate(model, name, val), scale)
# CollapseInverseRateDeriv(model::OQSmodel, name_list::Array{String, 1}, scales) = [CollapseInverseRateDeriv(model, name, s) for (name, s) in zip(name_list, scales)]


"""
    Implementation of simulaneous derivative of multiple collapse rates
"""
# Simultansiously vary all instances of param_name for each env_proc in proc_list
function vary_multi_collapse_rate!(model::OQSmodel, proc_list::Array{String,1}, new_val::Number)

   # Loop through all procs in proc_list and vary param (only re-calculate L on last modification for efficiency)
   for proc_name in proc_list[1:end - 1]
      vary_collapse_rate!(model, proc_name, new_val, update_generator=false)
   end
   vary_collapse_rate!(model, proc_list[end], new_val, update_generator=true)

   return model
end

# Non-mutating version
vary_multi_collapse_rate(model::OQSmodel, proc_list::Array{String,1}, new_val::Number) = vary_multi_collapse_rate!(copy(model), proc_list, new_val)


# Simple functions for constructing relevant DiffParam instances
export CollapseRateSimDeriv
# Multi-proc, single-param, single scale:
function CollapseRateSimDeriv(model::OQSmodel, name_pattern::String, scale)
   # Get all proc idxs which contain name_pattern
   proc_idxs = [get_env_process_idx(model, proc.name) for proc in model.env_processes if occursin(name_pattern, proc.name)]
   length(proc_idxs) == 0 && throw(error("name_pattern ($(name_pattern)) not found in any env_process names"))
   # Get param val and check it is the same in all procs
   param_vals = [model.env_processes[idx].rate for idx in proc_idxs]
   map(x -> isequal(param_vals[1], x), param_vals[2:end]) |> sum == length(param_vals) - 1 || throw(error("Not all param values to be varied Simultansiously are equal. Param values: $(param_vals)"))
   # If equal check passed then construct DiffParam
   return DiffParam(name_pattern, param_vals[1], (model, val) -> vary_multi_collapse_rate(model, getfield.(model.env_processes[proc_idxs], :name), val), scale)
end




"""
   Implementation of InteractionOp spectrum param derivatives:
-----------------------------------------------------------------
"""

function vary_spectrum_param!(model::OQSmodel, proc_name::String, param_name::Symbol, new_val::Number; update_generator=true)

   #Update env process spectrum (env_processes NamedTuple is immutable but fields of it's elements can still be mutated - is this bad practice?)
   proc = model.env_processes[Symbol(proc_name)]
   model.env_processes[Symbol(proc_name)].spectrum = SpectralDensity(proc.spectrum.func, (; proc.spectrum.args..., [param_name => new_val]...))
   update_generator && (model.L = transport_generator(model))

   # # Get proc idx from env_processes and param idx from spectrum.arg_names
   # proc_idx = findfirst(string.(keys(model.env_processes)) .== proc_name) #get_env_process_idx(model, proc_name)
   # #Set new arg value (need to create new NamedTuples since they're immutable)
   # model.env_processes[proc_idx].spectrum.args = (; model.env_processes[proc_idx].spectrum.args..., [param_name => new_val]...)

   # New NamedTuple implementation
   # proc = model.env_processes[Symbol(proc_name)]
   # proc.spectrum = SpectralDensity(proc.spectrum.func, (; proc.spectrum.args..., [param_name => new_val]...))

   # model.env_processes = (; model.env_processes..., [Symbol(proc_name) => InteractionOp(proc_name, proc.oper, new_spectrum)]...) #Replace env process with updated version

   # Construct new model then set relevant params in model
   # update_generator && (model.L = transport_generator(model))
   # if update_generator
   #    # new_model = OQSmodel(model.Ham, model.env_processes, model.ME_type, model.InitState)
   #    # model.A_ops = new_model.A_ops
   #    # model.L = new_model.L
   #    model.L = transport_generator(model)
   # end

   return model
end

vary_spectrum_param(model::OQSmodel, proc_name::String, param_name::Symbol, new_val::Number) = vary_spectrum_param!(copy(model), proc_name, param_name, new_val)

# Simple functions to construct spectrum param DiffParam instances
export SpectrumParamDeriv
# Single proc, single param, single scale
function SpectrumParamDeriv(model::OQSmodel, proc_name::String, param_name::Symbol, scale)
   # proc_idx = get_env_process_idx(model, proc_name) # Get proc idx from env_processes and param idx from spectrum.arg_names
   # param_val = model.env_processes[proc_idx].spectrum.args[param_name] # Get param val

   # New NamedTuple implementation
   param_val = model.env_processes[Symbol(proc_name)].spectrum.args[param_name]

   # Construct DiffParam
   return DiffParam(proc_name * "-" * string(param_name), param_val, (model, val) -> vary_spectrum_param(model, proc_name, param_name, val), scale)
end

# Single proc, multi-param, single scale:
SpectrumParamDeriv(model::OQSmodel, proc_name::String, param_names::Array{Symbol,1}, scale) = [SpectrumParamDeriv(model, proc_name, n, scale) for n in param_names] 
# Single proc, multi-param, diff scales:
SpectrumParamDeriv(model::OQSmodel, proc_name::String, names::Array{Symbol,1}, scales::Array{T,1} where T) = [SpectrumParamDeriv(model, proc_name, n, s) for (n, s) in zip(names, scales)]



""" 
   Implementation of SIMULTANEOUS spectrum param deriv:
----------------------------------------------------------
"""

# Simultansiously vary all instances of param_name for each env_proc in proc_list
function vary_multi_spectrum_param!(model::OQSmodel, proc_list::Array{String,1}, param_name::Symbol, new_val::Number)

   # Loop through all procs in proc_list and vary param (only re-calculate L on last modification for efficiency)
   for proc_name in proc_list[1:end - 1]
      vary_spectrum_param!(model, proc_name, param_name, new_val, update_generator=false)
   end
   vary_spectrum_param!(model, proc_list[end], param_name, new_val, update_generator=true)

   return model
end

# Non-mutating version
vary_multi_spectrum_param(model::OQSmodel, proc_list::Array{String,1}, param_name::Symbol, new_val::Number) = vary_multi_spectrum_param!(copy(model), proc_list, param_name, new_val)


# Simple functions for constructing relevant DiffParam instances
export SpectrumParamSimDeriv
# Multi-proc, single-param, single scale:
function SpectrumParamSimDeriv(model::OQSmodel, name_pattern::String, param_name::Symbol, scale)
   
   # Old implementation 

   # # Get all proc idxs which contain name_pattern
   # proc_idxs = [get_env_process_idx(model, proc.name) for proc in model.env_processes if occursin(name_pattern, proc.name)]
   # length(proc_idxs) == 0 && throw(error("name_pattern ($(name_pattern)) not found in any env_process names"))
   # # Get param val and check it is the same in all procs
   # param_vals = [model.env_processes[idx].spectrum.args[param_name] for idx in proc_idxs]

   # New NamedTuple implementation
   proc_list = [p for p in model.env_processes if occursin(name_pattern, p.name)]
   length(proc_list) == 0 && throw(error("name_pattern ($(name_pattern)) not found in any env_process names"))
   param_vals = [p.spectrum.args[param_name] for p in proc_list] #[p.spectrum.args[param_name] for p in model.env_processes if occursin(name_pattern, p.name)]

   map(x -> isequal(param_vals[1], x), param_vals[2:end]) |> sum == length(param_vals) - 1 || throw(error("Not all param values to be varied simultansiously are equal. Param values: $(param_vals)"))
   # If equal check passed then construct DiffParam
   return DiffParam(name_pattern * "-" * string(param_name), param_vals[1], (model, val) -> vary_multi_spectrum_param(model, getfield.(proc_list, :name), param_name, val), scale)

end
# Multi-proc, multi-param, single scale:
SpectrumParamSimDeriv(model, name_pattern::String, param_list::Array{Symbol,1}, scale) = [SpectrumParamSimDeriv(model, name_pattern, param_name, scale) for param_name in param_list]
# Multi-proc, multi-param, diff scales:
SpectrumParamSimDeriv(model, name_pattern::String, param_list::Array{Symbol,1}, scales::Array{T,1} where T) = [SpectrumParamSimDeriv(model, name_pattern, param_name, s) for (param_name, s) in zip(param_list, scales)]



# ------------------------ DiffParam implementation for weightings in CollectiveInteractionOp ------------------------ #

export InteractionWeightingsDeriv

function vary_interaction_weightings!(model::OQSmodel, proc_name::String, weight_idx::Int, new_val::Real; update_generator=true)

   proc = model.env_processes[Symbol(proc_name)]
   proc.weightings[weight_idx] = new_val
   model.env_processes = (; model.env_processes..., [proc_name => proc]...) #Replace proc with updated weight value
   update_generator && (model.L = transport_generator(model))

   return model
end
#Non-mutating version
vary_interaction_weightings(m::OQSmodel, p::String, idx::Int, val::Real; kwargs...) = vary_interaction_weightings!(copy(m), p, idx, val; kwargs...)

# function InteractionWeightingsDeriv(model::OQSmodel, proc_name::String)

# function vary_multi_interaction_weighting!(model::OQSmodel, name_pattern::String, weight_idx::Int, new_val::Real; update_generator=true)
# end

# export InteractionWeightingsSimDeriv





# ---------------------- DiffParam constructors for DickeHamiltonian and DickeLadderHamiltonian ---------------------- #

export AtomEnergyDeriv
function AtomEnergyDeriv(m::OQSmodel, scale::Number) 
   return DiffParam(
       "ωz", 
       m.Ham.param_dict["ωz"], 
       (x, y) -> OQSmodels.vary_Hamiltonian_param(x, "ωz", y),
       scale
   )
end
AtomEnergyDeriv(m::OQSmodel) = AtomEnergyDeriv(m, 1)

export CavityEnergyDeriv
function CavityEnergyDeriv(m::OQSmodel, scale::Number) 
   return DiffParam(
       "ωc", 
       m.Ham.param_dict["ωc"], 
       (x, y) -> OQSmodels.vary_Hamiltonian_param(x, "ωc", y),
       scale
   )
end
CavityEnergyDeriv(m::OQSmodel) = CavityEnergyDeriv(m, 1)

export CavityCouplingDeriv
function CavityCouplingDeriv(m::OQSmodel, scale::Number) 
   return DiffParam(
       "g", 
       m.Ham.param_dict["g"], 
       (x, y) -> OQSmodels.vary_Hamiltonian_param(x, "g", y),
       scale
   )
end
CavityCouplingDeriv(m::OQSmodel) = CavityCouplingDeriv(m, 1)

