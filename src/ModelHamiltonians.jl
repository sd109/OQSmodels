

# Separate Hamiltonian stuff from ModelConstruction.jl and include it here instead

using LinearAlgebra, QuantumOptics
export SystemHamiltonian, SingleExcitationHamiltonian, MultiExcitationHamiltonian, DickeHamiltonian, DickeLadderHamiltonian


# Container for information about a particular env state (i.e. those with no coherent coupling to any other site)
mutable struct EnvState
    name::String
    energy::Float64
    idx::Int
end


"""
`SystemHamiltonian`

Abstract type for all Hamiltonian structures. 
    
To work correctly with finite difference derivatives, all subtypes must look like:

```
mutable struct ExampleHamiltonian <: SystemHamiltonian
    param_dict
    op (<: AbstractOperator)
    (Any additional fields)
end
```

and implement a method `SystemHamiltonian(H::ExampleHamiltonian) = ExampleHamiltonian(args...)`
"""
abstract type SystemHamiltonian end

"""In hindsight, naming this method 'SystemHamiltonian' is confusing and unneccesary... Start using update_H! instead in other places, and define a compatibility version here."""

export update_H!
update_H!(H::SystemHamiltonian) = SystemHamiltonian(H) #Dispatches on specific sub-types of SystemHamiltonian




# Generic methods for all sub-types

# Method overloads 
export bases, numsites
QuantumOptics.basis(Ham::SystemHamiltonian) = basis(Ham.op)
bases(Ham::SystemHamiltonian) = [basis(Ham).bases...]
LinearAlgebra.eigen(Ham::SystemHamiltonian) = eigen(dense(Ham.op).data)

# Custom show function
function Base.show(io::IO, x::SystemHamiltonian)
    println(io, "\n  System Hamiltonian with system size = $(numsites(x)):")
    println(io, "===============================================")
    println(io, x.op)
end



# Generic function which can be supplied to FiniteDiff for easy derivatives of any parameter in Ham.param_dict
function vary_Hamiltonian_param(Ham::SystemHamiltonian, name::String, val::Real)
    Ham = deepcopy(Ham)
    Ham.param_dict[name] = val
    return SystemHamiltonian(Ham)
end

""" Generic fallback method which is inefficient and *doesn't* actually mutate Ham. More efficient methods can be implemented for specific sub-types of SystemHamiltonian which will then take precedence due to multiple dispatch. """
vary_Hamiltonian_param!(Ham::SystemHamiltonian, name::String, val::Real) = vary_Hamiltonian_param(Ham, name, val)


"""
`SingleExcitationHamiltonian`

Implementation of 'tight-binding' type Hamiltonian where the system can hold at most 1 excitations.
This is implemented as an NLevelBasis with each state corresponding to a single 'site'.
"""
mutable struct SingleExcitationHamiltonian{F <: Any,H <: DataOperator} <: SystemHamiltonian
    param_dict::Dict{String,Float64}
    hopping_func::F
    env_states::Vector{EnvState}
    op::H
end

# Extracts either cartesian or polar site coords from param_dict
function parse_coords(p::Dict)

    # Check for cartesian coords first
    xs = [val for (key, val) in p if occursin(r"x\d+", key)]
    ys = [val for (key, val) in p if occursin(r"y\d+", key)]
    zs = [val for (key, val) in p if occursin(r"z\d+", key)]

    if length(xs) != 0 && (length(xs) == length(ys) == length(zs))
        return vcat([[p["x$i"] p["y$i"] p["z$i"]] for i in 1:length(xs)]...) # Need to make sure pos are in order #hcat(xs, ys, zs)
    end

    # Check for polar coords instead
    rs = [val for (key, val) in p if occursin(r"r1\d+", key)]
    ϕs = [val for (key, val) in p if occursin(r"ϕ1\d+", key)]
    θs = [val for (key, val) in p if occursin(r"θ1\d+", key)]

    if length(rs) != 0 && (length(rs) == length(ϕs) == length(θs))
        # return vcat(zeros(3), hcat(rs, ϕs, θs)) #Make sure first site is at origin of coords
        return vcat(zeros(3), [[p["r1$i"] p["ϕ1$i"] p["θ1$i"]] for i in 2:length(xs) + 1]...) # Need to make sure pos are in order and site 1 is at origin
    end

    # Default error
    throw(error("Could not extract either cartesian or polar site coords from param dict"))
end

# Utility functions for converting between cartesian and spherical co-ordinates
to_spherical(x, y, z) = (r = norm([x, y, z]); [r, atan(y, x), r == 0 ? 0 : acos(z / r)])
from_spherical(r, θ, ϕ) = [r * cos(θ) * sin(ϕ), r * sin(θ) * sin(ϕ), r * cos(ϕ)]


### Constructors

function SingleExcitationHamiltonian(p::Dict, hopping_func; force_real=false)

    # Extract param dict quantities with regex
    env_Es = [val for (key, val) in p if occursin(r"E_\D.*", key)] # Catches any key where key is"E_{some name}" ('\D' = single non-digit, '.*' = anything but new line)

    N_sites = count(key -> occursin(r"E\d+", key), keys(p)) #length(Es)
    N = N_sites + length(env_Es)

    pos = parse_coords(p)::Matrix{Float64}
    N_sites == size(pos, 1) || error("Different number of energies and positions provided - total number of sites is ambiguous.") # Check that we have coords for each site

    # Construct env states (drop "E_" from name)
    names = [key for (key, val) in p if occursin(r"E\D.*", key)]
    env_states = [EnvState(n[3:end], p[n],  p[n[3:end]*"_idx"]) for n in names]

    # Create basis and Operator
    b = NLevelBasis(N)

    # Add site energies
    # H = sum(p["E$(i)"] * transition(b, i, i) for i in 1:N_sites)
    H = SparseOperator(b, diagm(vcat([p["E$(i)"] for i in 1:N_sites], zeros(length(env_Es)))))

    # Add inter-site couplings
    couplings = hopping_func(pos, p["J"])
    for (pair, val) in couplings
        i, j = pair
        H.data[i, j] = val
        H.data[j, i] = val
    end

    # Add env state energies
    for st in env_states
        H.data[st.idx, st.idx] = st.energy
    end

    # Discard imag part for efficiency if it's zero - this causes errors in QuantumOptics.timeevolution.bloch_redfield_master
    force_real && (H = Operator(basis(H), real(H.data)))
    # Convert to dense array if there are lots of non-zero elements
    length(H.data.nzval) > N^2/3 && (H = dense(H))

    return SingleExcitationHamiltonian(p, hopping_func, env_states, H)

end

# Convenience constructor which takes same args as previous implementation for easier transition to this new setup
function SingleExcitationHamiltonian(
    site_energies::Array{T,1} where T <: Number, 
    pos::Array{P,2} where P <: Number, 
    coupling_func, J::Real, 
    env_states::Array{E,1} where E <: EnvState; 
    polar=false, force_real=false)

    N = length(site_energies)
    param_dict = Dict{String,Float64}()
    # Extract site quantities
    for i in 1:N
        param_dict["E$i"] = site_energies[i]
        if polar
            param_dict["r$i"] = pos[i, 1]
            param_dict["ϕ$i"] = pos[i, 2]
            param_dict["θ$i"] = pos[i, 3]
        else
            param_dict["x$i"] = pos[i, 1]
            param_dict["y$i"] = pos[i, 2]
            param_dict["z$i"] = pos[i, 3]
        end
    end
    param_dict["J"] = J #Coherent coupling constant
    # Extract env state energies and indices
    for st in env_states
        param_dict["E_" * st.name] = st.energy
        param_dict[st.name * "_idx"] = st.idx
    end

    # return SingleExcitationHamiltonian(param_dict, coupling_func)
    return SingleExcitationHamiltonian(param_dict, coupling_func; force_real=force_real)

end

# Constructor for finite diff compatibility
SystemHamiltonian(Ham::SingleExcitationHamiltonian; force_real=eltype(Ham.op.data) <: Real) = SingleExcitationHamiltonian(Ham.param_dict, Ham.hopping_func; force_real=force_real)

#Method overload for faster copying
Base.copy(H::SingleExcitationHamiltonian) = SingleExcitationHamiltonian(copy(H.param_dict), H.hopping_func, deepcopy(H.env_states), copy(H.op))

# Specific method for varying H params (more efficient than generic fallback)
function vary_Hamiltonian_param!(Ham::SingleExcitationHamiltonian, name::String, val::Real)

    Ham.param_dict[name] = val #Update param dict entry
    if match(r"E\d+", name) !== nothing #Site energy variation
        site = parse(Int, name[2:end])
        Ham.op.data[site, site] = val
    elseif match(r"E_.+", name) #Env energy variation
        idx = Ham.param_dict["$(name[3:end])_idx"]
        Ham.op.data[idx, idx] = val

    # elseif # Add more cases here as required
    
    else #Generic (slow) fallback which creates new Ham instance and sets appropriate field
        new_H = vary_Hamiltonian_param(Ham, name, val)
        for f in fieldnames(SingleExcitationHamiltonian)
            setfield!(Ham, f, getfield(new_H, f))
        end
    end

    return Ham
end





"""
`MultiExcitationHamiltonian`

Extension of `SingleExcitationHamiltonian` where each site is a fock space with upper cutoff dimension 'L'.

An 'N' site system can hold N⋅L excitations and all env states have fock space cutoff N⋅L.
"""
mutable struct MultiExcitationHamiltonian{F <: Any,H <: AbstractOperator} <: SystemHamiltonian
    param_dict::Dict{String,Float64}
    hopping_func::F
    env_states::Vector{EnvState}
    fock_cutoff::Int
    full_env_states::Bool
    op::H
end


### Constructors

function MultiExcitationHamiltonian(p::Dict, hopping_func, L_cutoff, full_env_states)

    # Extract param dict quantities with regex
    N_sites = count(key -> occursin(r"E\d+", key), keys(p))
 
    env_Es = [val for (key, val) in p if occursin(r"E_\D.*", key)] # Catches any key where key is"E_{some name}" ('\D' = single non-digit, '.*' = anything but new line)
    N_env = length(env_Es)
 
    N = N_sites + N_env

    # xs = [val for (key, val) in p if occursin(r"x\d+", key)]
    # ys = [val for (key, val) in p if occursin(r"y\d+", key)]
    # zs = [val for (key, val) in p if occursin(r"z\d+", key)]
    pos = parse_coords(p) #hcat(xs, ys, zs)

    # Sanity check on number of parameter provided
    N_sites == size(pos, 1) || error("Different number of energies and positions provided - total number of sites is ambiguous.") # Check that we have same number of site energies and coords


    
    # Construct env states (drop "E_" from name using [3:end] indexing)
    names = [key for key in keys(p) if occursin(r"E\D.*", key)]
    env_states = [EnvState(n[3:end], p[n], N_sites + i) for (i, n) in enumerate(names)]

    # Create basis and Operator
    b_site = FockBasis(L_cutoff)
    env_dim = full_env_states ? N_sites * L_cutoff : L_cutoff #Decide whether to include space for ALL excitations simultaneously in each env state
    b_env = FockBasis(env_dim)
    b = tensor(fill(b_site, N_sites)..., fill(b_env, N_env)...)

    # Add site energies
    H = sum(embed(b, i, p["E$(i)"] * number(b_site)) for i in 1:N_sites)

    # Add inter-site couplings
    couplings = hopping_func(pos, p["J"])
    # for (i, j, val) in couplings
        # H += val * (embed(b, Int.([i, j]), [create(b_site), destroy(b_site)]) + embed(b, Int.([i, j]), [destroy(b_site), create(b_site)]))
    for (p, val) in couplings
        i, j = p
        H += val * (embed(b, [i, j], [create(b_site), destroy(b_site)]) + embed(b, [i, j], [destroy(b_site), create(b_site)]))
    end

    # Add env state energies
    for st in env_states
        H += embed(b, st.idx, st.energy * number(b_env))
    end

    # Discard imag part for efficiency if it's zero
    sum(imag(H.data)) == 0 && (H = Operator(basis(H), real(H.data)))

    return MultiExcitationHamiltonian(p, hopping_func, env_states, L_cutoff, full_env_states, H)

end

# Convenience constructor which takes same args as previous implementation for easier transition to this new setup
function MultiExcitationHamiltonian(site_energies::Array{T,1} where T <: Number, pos::Array{P,2} where P <: Number, coupling_func, J::Real, env_states::Array{E,1} where E <: EnvState; 
                                        L_cutoff=1, polar=false, full_env_states=true)

    N = length(site_energies)
    param_dict = Dict{String,Float64}()

    # Extract site quantities
    for i in 1:N
        param_dict["E$i"] = site_energies[i]
        if polar
            param_dict["r$i"] = pos[i, 1]
            param_dict["ϕ$i"] = pos[i, 2]
            param_dict["θ$i"] = pos[i, 3]
        else
            param_dict["x$i"] = pos[i, 1]
            param_dict["y$i"] = pos[i, 2]
            param_dict["z$i"] = pos[i, 3]
        end
    end
    param_dict["J"] = J #Coherent coupling constant
    # Extract env state energies 
    for st in env_states
        param_dict["E_" * st.name] = st.energy
    end

    # return SingleExcitationHamiltonian(param_dict, coupling_func)
    return MultiExcitationHamiltonian(param_dict, coupling_func, L_cutoff, full_env_states)

end

# Constructor for finite diff compatibility
SystemHamiltonian(Ham::MultiExcitationHamiltonian) = MultiExcitationHamiltonian(Ham.param_dict, Ham.hopping_func, Ham.fock_cutoff, Ham.full_env_states)


# Other useful methods

numsites(H::SingleExcitationHamiltonian) = size(H.op, 1) - length(H.env_states)
numsites(H::MultiExcitationHamiltonian) = count(x -> occursin(r"E\d+", x), keys(H.param_dict))

export site_energies, site_positions
site_energies(H::Union{SingleExcitationHamiltonian, MultiExcitationHamiltonian}) = [H.param_dict["E$i"] for i in 1:numsites(H)]
site_positions(H::Union{SingleExcitationHamiltonian, MultiExcitationHamiltonian}) = vcat([[H.param_dict["x$i"] H.param_dict["y$i"] H.param_dict["z$i"]] for i in 1:numsites(H)]...)


"""
`DickeHamiltonian`

Hamiltonian for the Dicke model, given by:

H = ωc a^† a + ωz ∑ σ_z^j + 2g/√N (a + a^†) ∑ σ_x^j
"""
mutable struct DickeHamiltonian{H <: AbstractOperator} <: SystemHamiltonian
    param_dict::Dict{String,Float64}
    N::Int
    op::H
end

### Constructors
"""
Param dict must constain only the keys ωz, ωc & g
"""
function DickeHamiltonian(p::Dict, N)

    p = Dict{String,Float64}(p...) # Convert all param values to floats for simplicity

    b_atom = SpinBasis(1 // 2)
    b_cavity = FockBasis(N)
    b = tensor(fill(b_atom, N)..., b_cavity)

    H = embed(b, N + 1, p["ωc"] * number(b_cavity)) # Cavity term
    H += sum(embed(b, i, p["ωz"] * sigmaz(b_atom)) for i in 1:N) # Atom terms
    H += 2 * p["g"] / sqrt(N) * sum(embed(b, [i, N + 1], [sigmax(b_atom), destroy(b_cavity) + create(b_cavity)]) for i in 1:N)

    # return DickeHamiltonian(p, N, H)
    return DickeHamiltonian(p, N, Operator(basis(H), real(H.data)))
end

# Constructor for finite diff compatibility
SystemHamiltonian(Ham::DickeHamiltonian) = DickeHamiltonian(Ham.param_dict, Ham.N)

# Other useful methods
numsites(H::DickeHamiltonian) = H.N




"""
`DickeLadderHamiltonian`

Hamiltonian for modelling the symmetric Dicke model as an spin-N/2 system with Hamiltonian given by:

H = ωc a^† a + ωz S_z + 2g/√N (a + a^†) S_x
"""
mutable struct DickeLadderHamiltonian{H <: AbstractOperator} <: SystemHamiltonian
    param_dict::Dict{String,Float64}
    N::Int
    op::H
end

### Constructors

"""
Param dict must constain only the keys ωz, ωc & g
"""
function DickeLadderHamiltonian(p::Dict, N)

    p = Dict{String,Float64}(p...) # Convert all param values to floats for simplicity

    b_atom = SpinBasis(N // 2)
    b_cavity = FockBasis(N)
    b = tensor(b_atom, b_cavity)

    H = embed(b, 1, p["ωz"] * sigmaz(b_atom)) # Atom terms
    H += embed(b, 2, p["ωc"] * number(b_cavity)) # Cavity term
    H += 2 * p["g"] / sqrt(N) * embed(b, [1, 2], [sigmax(b_atom), destroy(b_cavity) + create(b_cavity)])

    # return DickeHamiltonian(p, N, H)
    return DickeLadderHamiltonian(p, N, Operator(basis(H), real(H.data)))
end

# Constructor for finite diff compatibility
SystemHamiltonian(Ham::DickeLadderHamiltonian) = DickeLadderHamiltonian(Ham.param_dict, Ham.N)

# Other useful methods
numsites(H::DickeLadderHamiltonian) = H.N


### HOW DO WE CREATE A SIMILAR IMPLEMENTATION (BASED ON PARAM_DICTS) FOR LIOUVILLIANS?

### More importantly, do we even need to?
### I could just rewrite DiffParam constructors for Hamiltonian parameters instead and leave EnvProcess diff params as is.
### This solves the initial problem with Hamiltonian params without over complicating the rest of the existing code.

# We'd want something like:
# > SiteEnergyDeriv(model, sites) = [DiffParam("E$i", model.Ham.param_dict["E$i"], (m, val) -> OQSmodel(vary_Hamiltonian_param(m.Ham, "E$i", val), model.(other-args)...) for i in sites]
# > AtomEnergyDeriv(model) = DiffParam("ωz", model.Ham.param_dict["ωz"], (m, val) -> OQSmodel(vary_Hamiltonian_param(m.Ham, "ωz", val), model.(other-args)...))



