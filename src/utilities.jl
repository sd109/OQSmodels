

# Define constants for converting between units - values taken from scipy.constants module
const hbar = 6.62607015e-34 / (2 * pi)# constants.hbar
const c = 299792458.0 # constants.c
const e = 1.602176634e-19
const kb_eV = 1.380649e-23 / e # constants.k / constants.e

export from_nm, to_nm
from_nm(L) = L * 1e-9 * e / (hbar * c)
to_nm(L) = L * 1e9 * hbar * c / e

""" A note on units:
By working in units of eV we are implcitly setting the value of hbar[eV] = hbar[J⋅s]/e = 6.63e-34 / (2π*1.6e-19).
Then, by setting hbar = c = 1, we are implcitly dividing the equation E = ħω = ħc/λ through by the numerical value hbar[eV⋅s]*c[m/s] = 6.63e-34*3e8 / (2π*1.6e-19).
Therefore, to get from meters to eV^-1 (i.e. 'natural' length units) we use r[eV^-1] = e/(hbar[J]*c[m/s]) * r[m] and to get back we do the opposite r[m] = r[eV^-1] * hbar[J]*c[m/s] / e
"""

# const single_ex_basis = Operator{T,T,D} where T <: NLevelBasis where D <: AbstractArray
# const multi_ex_basis = Operator{T,T,D} where T <: CompositeBasis where D <: AbstractArray
# export single_ex_basis, multi_ex_basis

"""
    Define some commonly used functions: 
-------------------------------------------
"""

export nbe, Sw_superohmic
nbe(w, T) = ( exp(abs(w) / (kb_eV * T)) - 1 )^-1.0
Sw_superohmic(w, rate, cutoff, Tph) = w == 0 ? 0.0 : rate * (abs(w) / cutoff)^3 * exp(-(w / cutoff)^2) * (nbe(w, Tph) + (w > 0))

export gibbs_state
function gibbs_state(H::DataOperator, T)
    st = exp(-dense(H) / (kb_eV * T))
    return st / tr(st)
end
gibbs_state(H::SystemHamiltonian, T) = gibbs_state(H.op, T)
gibbs_state(m::OQSmodel, T) = gibbs_state(m.Ham, T)


export ground_state
function ground_state(H::DataOperator) 
    ens, states = eigenstates(dense(H))
    return states[1]
end
ground_state(H::SystemHamiltonian) = ground_state(H.op)
ground_state(m::OQSmodel) = ground_state(m.Ham.op)


# Type-stability helpers
data(O::Operator) = O.data
data(S::SuperOperator) = S.data
data(M::Matrix) = M #Fail-safe in case I call data when not needed

# Simple function for extracting the populations from a state
export populations
populations(st::Operator{T,T,D} where T <: NLevelBasis where D <: AbstractArray) = real(diag(st.data))
populations(st::AbstractArray) = real(diag(st))

# What's the equivalent function for fockstate/multi-excitation Hamiltonians?
function populations(st::Operator{T,T,D} where T <: CompositeBasis where D <: AbstractArray)
    # full_basis = basis(st)
    # ops = [embed(full_basis, i, number(full_basis.bases[i])) for i in 1:length(full_basis.bases)]
    # return map(op -> expect(op, st) |> real, ops)
    b = basis(st)
    map(i -> real(expect(i, number(b.bases[i]), st)), 1:length(b.bases))
end


### Dipole coupling functions

using IterTools
export dd_coupling
function dd_coupling(pos, J)

    N_sites = size(pos, 1)
    pairs = subsets(1:N_sites, 2)

    #Simpler implementation
    f(pos, i, j, J) = J / norm(pos[i, :] - pos[j, :])^3
    couplings = map(p -> f(pos, p[1], p[2], J), pairs)
    
    return zip(pairs, couplings)
end


export nn_coupling
function nn_coupling(pos, val)
    N_sites = size(pos, 1)
    return [[i, i + 1, val] for i in 1:N_sites - 1]
end


# Functions for constructing system-env interactions

export site_dephasing_ops
site_dephasing_ops(H::SystemHamiltonian, B::NLevelBasis) = [transition(Float64, B, i, i) for i in 1:numsites(H)]
site_dephasing_ops(H::SystemHamiltonian, B::CompositeBasis) = [embed(B, i, number(B.bases[i])) for i in 1:numsites(H)]
site_dephasing_ops(H::SystemHamiltonian) = site_dephasing_ops(H, basis(H)) # Dispatch on basis type


site_energies(m::OQSmodel) = site_energies(m.Ham)


function get_env_state_idx(Ham::SystemHamiltonian, st_name::String)

    all_names = getfield.(Ham.env_states, :name)
    idx = findfirst(all_names .== st_name)
    idx === nothing && error("Name not found in env states")
    return idx
end

get_env_state_idx(model::OQSmodel) = get_env_state_idx(model.Ham)
get_env_state(Ham::SystemHamiltonian, name::String) = Ham.env_states[get_env_state_idx(Ham, name)]
get_env_state(model::OQSmodel) = get_env_state(model.Ham)



# Utility function for using average eigen-energy detuning as E0 scale in dimensionless parameter derivatives
function avg_eigenenergy_detunings(model; inc_env=true, avg_func=mean)
    H = inc_env ? Array(model.Ham.op.data) : Array(model.Ham.op.data[1:model.NoSites, 1:model.NoSites])
    Es = eigvals(H)
    pairs = subsets(Es, 2)
    detunings = abs.(map(p -> diff(p)[1], pairs))
    return avg_func(detunings) 
end

# Utility function for getting average distance of all sites from origin
function avg_site_displacement(model)
    pos = parse_coords(model.Ham.param_dict)
    pos[1, :] != zeros(3) && (@warn "Site 1 not at origin - is this intentional...?")
    return sum(mapslices(x -> norm(x), pos[2:end, :], dims=2)) / (numsites(model) - 1) # Average site distance from origin (excluding site 1)
 end



### None of the functions below should be needed now that env_processes and SpectralDensity utilize NamedTuples

# function get_env_process_idx(model::OQSmodel, proc_name::String)

#     names = getfield.(model.env_processes, :name)
#     idx = findfirst(names .== proc_name)
#     idx === nothing && error("Name not found in transport model env_processes") # Catch error
#     return idx
# end

# get_env_process(model::OQSmodel, name::String) = model.env_processes[get_env_process_idx(model, name)]
# get_multi_env_process_idx(model::OQSmodel, name_pattern::String) = findall(occursin.(name_pattern, getfield.(model.env_processes, :name)))
# get_multi_env_process(model::OQSmodel, name_pattern::String) = model.env_processes[get_multi_env_process_idx(model, name_pattern)]

# Some useful methods for accessing various model properties

# None of these are needed with new SpectralDensity definition
# arg_names(S::SpectralDensity) = S.arg_names
# arg_values(S::SpectralDensity) = S.arg_vals
# Sw(S::SpectralDensity) = S.func
# get_arg_value(S::SpectralDensity, name::Symbol) = S.arg_vals[findfirst(arg_names(S) .== name)]
# set_arg_value!(S::SpectralDensity, name::Symbol, value::Real) = S.arg_vals[findfirst(arg_names(S) .== name)] = value

