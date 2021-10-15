



# Export lots of types
export EnvState, EnvProcess, CollapseOp, InteractionOp
export SpectralDensity #, arg_names, arg_values, get_arg_value, set_arg_value!
export OQSmodel, ModelOptions
export MasterEquation, LindbladME, BlochRedfieldME, PauliME
# Export some functions too
export coupling_operator, transport_generator
export data, get_env_state_idx, get_env_state #, get_env_process_idx, get_env_process

using LinearAlgebra, IterTools, QuantumOptics, PauliMasterEquation
using FiniteDiff, FiniteDifferences # For default finite_diff_func in ModelOptions


"""
    SYSTEM HAMILTONIAN & ENV STATES:
========================================
"""
# -------------------------------------------------------
### NOW DEFINED IN SEPARATE ModelHamiltonians.jl FILE
# -------------------------------------------------------



""" 
    SYSTEM-ENV INTERACTIONS:
================================
"""

abstract type EnvProcess end

# Lindblad-type collapse operators
mutable struct CollapseOp{O <: AbstractOperator,R <: Number} <: EnvProcess
    name::String
    oper::O
    rate::R
end



mutable struct SpectralDensity{F <: Any,N <: NamedTuple}
    func::F
    args::N
end

#Custom show method
function Base.show(io::IO, S::SpectralDensity)
    println(io, "Function:  ", S.func)
    println(io, "Additional args:  ", S.args)
end

#Provide a 1 arg function call which splats all extra args into S.func
(S::SpectralDensity)(ω::Real) = S.func(ω, values(S.args)...)

# Convenience constructor to ease transition from old SpectralDensity type `(; zip(keys, vals)...)` syntax is recommended in NamedTuple docstring
SpectralDensity(f, K::Vector{Symbol}, V::Vector) = SpectralDensity(f, (; zip(K, V)...))

# Bloch-Redfield-type system-env interaction operators
mutable struct InteractionOp{O <: AbstractOperator,S <: SpectralDensity} <: EnvProcess
    name::String
    oper::O # We should probably be checking Hermitianity of oper during construction
    spectrum::S
end





""" 
    MASTER EQUATION TYPES:
=============================
"""

abstract type MasterEquation end

struct LindbladME <: MasterEquation end

Base.@kwdef struct BlochRedfieldME <: MasterEquation
    use_secular::Bool = false
    secular_cutoff::Float64 = 1e-12
end

struct PauliME <: MasterEquation end


""" 
    MODEL OPTIONS TYPE:
===========================
"""
# Must be defined before main OQSmodel type

mutable struct ModelOptions #{F <: Any}

    nonphysical_tol::Float64
    err_on_nonphysical::Bool
    ss_eval_tol::Float64
    ss_time_tol::Float64
    polar_warning::Bool # If true, warn during polar site pos deriv if site 1 is not at origin
    scaled_params::Bool # Whether or not to use dimensionless parameters in ModelCalculus (so that η' = η/η0    ⟺   d/dη' = η0 * d/dη)
    finite_diff_func #::F # Function to be used in ModelCalculus.jl for finite difference derivatives - must take form: fd_func(f, val)
    extrapolate_deriv::Bool #Toggle use of Richardson extrapolation (via FiniteDifferences.jl) in ModelCalculus

end

# Default constructor using kwargs
function ModelOptions(; nonphysical_tol=1e-10, 
                        err_on_nonphysical=false, 
                        ss_eval_tol=1e-10, 
                        ss_time_tol=1e-6, 
                        polar_warning=true, 
                        scaled_params=true,
                        finite_diff_func=central_fdm(2, 1), #FiniteDiff.finite_difference_derivative,
                        extrapolate_deriv=true,
                        )
    return ModelOptions(
                nonphysical_tol, 
                err_on_nonphysical, 
                ss_eval_tol, 
                ss_time_tol, 
                polar_warning, 
                scaled_params, 
                finite_diff_func,
                extrapolate_deriv,
            )
end

# Custom show function
function Base.show(io::IO, x::MO where MO <: ModelOptions)
    names = fieldnames(ModelOptions)
    vals = map(f -> getfield(x, f), names)
    print(io, "OQS Model options: ")
    for (n, v) in zip(names, vals)
        print(io, "$n=$v, ")
    end
    # print(io, ["$n : $v" for (n, v) in zip(names, vals)])
end




""" 
    MAIN OQS MODEL IMPLEMENTATION:
=====================================
"""

# mutable struct OQSmodel{H <: SystemHamiltonian,I <: AbstractOperator,E <: Array{T,1} where T <: EnvProcess,C <: AbstractOperator,A <: AbstractOperator,M <: MasterEquation,U <: Union{Array{R,2} where R,SuperOperator},MO <: ModelOptions} # L is an array if ME_type = Pauli()
#     NoSites::Int
#     Ham::H
#     InitState::I            # Initial density matrix
#     env_processes::E        # Needed for model reconstruction in calculus but not used in any calculations - This is technically an abstract field so should try to get rid
#     C_ops::Array{C,1}       # Liouvillian collapse operators
#     A_ops::Array{A,1}       # System part of system-environment interaction operators
#     spectral_funcs          # ::Array{F, 1} #Functions describe system-env operator rates
#     ME_type::M              # Master equation choice
#     L::U                    # Dynamics generator (Liouvillian or BR tensor etc)
#     options::MO
# end

# mutable struct OQSmodel{H <: SystemHamiltonian,I <: AbstractOperator,E <: NamedTuple,M <: MasterEquation,U <: Union{Array{R,2} where R,SuperOperator},MO <: ModelOptions}
mutable struct OQSmodel{H <: SystemHamiltonian,I <: AbstractOperator,E <: Any,M <: MasterEquation,U <: Union{Array{R,2} where R,SuperOperator},MO <: ModelOptions}
    Ham::H                  # System Hamiltonian
    env_processes::E        # List of all system-env interactions (combination of CollapseOp & InteractionOp instances)
    ME_type::M              # Master equation choice
    InitState::I            # Initial system state (density matrix)
    L::U                    # Dynamics generator (Liouvillian, BR tensor or Pauli generator) -- L is an array iff ME_type = Pauli()
    options::MO             
end

#Method overload for faster copying
Base.copy(m::OQSmodel) = OQSmodel(copy(m.Ham), deepcopy(m.env_processes), m.ME_type, copy(m.InitState), copy(m.L), deepcopy(m.options))

# Getter functions to replace unnecessary model fields 
get_C_ops(env_processes) = [sqrt(op.rate) * op.oper for op in env_processes if typeof(op) <: CollapseOp]
get_A_ops(env_processes) = [op.oper for op in env_processes if typeof(op) <: InteractionOp]
get_spectral_funcs(env_processes) = getproperty.(A_ops(env_processes), :spectrum)
get_A_ops_and_spectral_funcs(env_processes) = [[op.oper, op.spectrum] for op in env_processes if typeof(op) <: InteractionOp]



# --------------------------------------------------- CONSTRUCTORS --------------------------------------------------- #

#Constructor which creates NamedTuple
function OQSmodel(
        Ham::SystemHamiltonian, env_processes::Vector, ME_type::MasterEquation, InitState::AbstractOperator; 
        L = transport_generator(Ham.op, env_processes, ME_type), options=ModelOptions(),
    )
    return OQSmodel(Ham, NamedTuple{Tuple(Symbol.(getfield.(env_processes, :name)))}(env_processes), ME_type, InitState; options=options, L=L)
end

# Constructor which calculates L automatically
function OQSmodel(
        Ham::SystemHamiltonian, env_processes::NamedTuple, ME_type::MasterEquation, InitState::AbstractOperator; 
        L = transport_generator(Ham.op, env_processes, ME_type), options=ModelOptions(),
    )
    tr(InitState) != 1 && throw(error("Initial state not correctly normalized!"))
    return OQSmodel(Ham, env_processes, ME_type, InitState, L, options)
end

# Constructor which converts Ket to Density Matrix
function OQSmodel(
        Ham::SystemHamiltonian, env_processes::Array, ME_type::MasterEquation, InitState::Ket; 
        L = transport_generator(Ham.op, env_processes, ME_type), options=ModelOptions(),
    )
    return OQSmodel(Ham, env_processes, ME_type, dm(InitState); options=options, L=L)
end

# Constructor which creates localized initial state
function OQSmodel(
        Ham::SystemHamiltonian, env_processes::Vector, ME_type::MasterEquation, InitSite::Int; 
        L = transport_generator(Ham.op, env_processes, ME_type), options=ModelOptions(),
    )

    if typeof(basis(Ham)) <: NLevelBasis
        InitState = transition(Float64, basis(Ham), InitSite, InitSite) #nlevelstate(basis(Ham), InitSite) |> dm |> sparse

    elseif typeof(basis(Ham)) <: CompositeBasis
        states = fockstate.(bases(Ham), 0)
        states[InitSite] = fockstate(bases(Ham)[InitSite], 1)
        InitState = tensor(states...) |> dm |> sparse

    else
        error("Hamiltonian basis not recognized during initial state construction")
    end

    return OQSmodel(Ham, env_processes, ME_type, InitState, options=options, L=L)
end


# Custom show function 
function Base.show(io::IO, x::OQSmodel)
    println(io, "  $(numsites(x.Ham)) site OQS model with:")
    println(io, "---------------------------\n")
    println(io, " -> $(length(get_C_ops(x))) Lindblad collapse operators")
    println(io, " -> $(length(get_A_ops(x))) Bloch-Redfield interaction operators")
    println(io, "\nSystem Hamiltonian:\n$(x.Ham.op)")
end



# METHOD OVERLOADS AND MORE CONVENIENCE FUNCTIONS

numsites(model::OQSmodel) = numsites(model.Ham)
site_positions(model::OQSmodel) = site_positions(model.Ham)
QuantumOptics.basis(model::OQSmodel) = basis(model.Ham)
bases(model::OQSmodel) = bases(model.Ham)
get_C_ops(m::OQSmodel) = get_C_ops(m.env_processes)
get_A_ops(m::OQSmodel) = get_A_ops(m.env_processes)
get_spectral_funcs(m::OQSmodel) = get_spectral_funcs(m.env_processes)
get_A_ops_and_spectral_funcs(m::OQSmodel) = get_A_ops_and_spectral_funcs(m.env_processes)



function transport_generator(H::AbstractOperator, env_processes, ME_type::LindbladME)
    C_ops = get_C_ops(env_processes)
    # length(get_A_ops(env_processes)) != 0 && error("Lindblad master equation model can't have A_ops.")
    length(C_ops) != length(env_processes) && error("Lindblad master equation model can't have A_ops.")
    return liouvillian(H, C_ops)
end

function transport_generator(H::AbstractOperator, env_processes, ME_type::BlochRedfieldME)
    return timeevolution.bloch_redfield_tensor(H, get_A_ops_and_spectral_funcs(env_processes), J=get_C_ops(env_processes), use_secular=ME_type.use_secular, secular_cutoff=ME_type.secular_cutoff)[1]
end

function transport_generator(H::AbstractOperator, env_processes, ME_type::PauliME)
    # length(get_C_ops(env_processes)) != 0 && error("Pauli master equation model can't have C_ops.")
    A_ops_and_spectra = get_A_ops_and_spectral_funcs(env_processes)
    length(A_ops_and_spectra) != length(env_processes) && error("Pauli master equation model can't have C_ops.")
    return pauli_generator(H, A_ops_and_spectra)[1]
end

transport_generator(model::OQSmodel) = transport_generator(model.Ham.op, model.env_processes, model.ME_type)




# Functions which update the model's Hamiltonian by calling `vary_Hamiltonian_param(::SystemHamiltonian, name, val)``
function vary_Hamiltonian_param!(model::OQSmodel, name::String, val::Real; update_generator=true)
    # model.Ham = vary_Hamiltonian_param(model.Ham, name, val)
    model.Ham = vary_Hamiltonian_param!(model.Ham, name, val) #Despite the '!', this method is only mutation for certain Hamiltonian types, so need to assign directly to model.Ham just in case
    update_generator && (model.L = transport_generator(model))
    return model
end
# Non-mutating version
# vary_Hamiltonian_param(model::OQSmodel, name::String, val::Real; kwargs...) = vary_Hamiltonian_param!(deepcopy(model), name, val; kwargs...)
vary_Hamiltonian_param(model::OQSmodel, name::String, val::Real; kwargs...) = vary_Hamiltonian_param!(copy(model), name, val; kwargs...)



