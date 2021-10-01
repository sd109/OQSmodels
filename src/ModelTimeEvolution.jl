
# module ModelTimeEvolution

# using ..ModelConstruction,

using LinearAlgebra, QuantumOptics, PauliMasterEquation
using Roots # For finding steady state time

export dynamics, expLt_solve, steady_state, steady_state_time

# NonPhysicalState error type
struct NonPhysicalStateError <: Exception
    populations::Array{Float64,1}
end
Base.show(io::IO, err::NonPhysicalStateError) = println("Non-physical state encountered. (1 - Tr[state] = $(1 - sum(err.populations))) \nState populations: $(err.populations)")




""" Functions to find steady state of transport model """

"""
`exp_L_nullspace(L::SuperOperator, tol::Real)`

This function is used to find the part of the liouvillian corresponding 
to the zero eigenvalued subspace (i.e. the bit relevant to the OQS steady state).
It is used in both LindbladME and BlochRedfieldME model types for steady
state calculation.
"""
function exp_L_nullspace(L::SuperOperator, tol)
    L_dim = size(L.data, 1)
    T = eltype(L.data)
    expL_eigvals, expL_transf = eigen(Array(L.data))::Eigen{T,T,Array{T,2},Array{T,1}}
    # Get all approx zero eigvals and set them = 1 (since exp(0)=1), discard all other eigvals
    ss_eigval_idxs = findall(map(x -> isapprox(x, 0, atol=tol), expL_eigvals))
    # Put these eigvals into the diagonals of an L_dim x L_dim sparse array
    expL_nullspace_Leb = sparse(ss_eigval_idxs, ss_eigval_idxs, ones(length(ss_eigval_idxs)), L_dim, L_dim)
    # Un-diagonalize expL
    return expL_transf * expL_nullspace_Leb * inv(expL_transf)    
end


# Convenience function
steady_state(model::OQSmodel) = steady_state(model::OQSmodel, model.ME_type::MasterEquation)


# Lindblad steady state
function steady_state(model::OQSmodel, ME_type::LindbladME)

    # Check model actually reaches steady state
    length(model.env_processes) == 0 && error("transport model has no env interactions so won't reach a steady state")

    # Using nullspace method: 
    
    L = Array(model.L.data)
    ρ0_vec = reshape(model.InitState.data, :)

    N = size(model.Ham.op.data, 1) # Hilbert space dim #Int(sqrt(size(L, 1)))
    steady_state = zeros(N, N)
    for v in eachcol(nullspace(L))
        steady_state += reshape((v * v') * ρ0_vec, N, N)
    end
    steady_state /= tr(steady_state) #Make sure state is normalized

    ### OLD METHOD:
    # # Exponentiate the zero-eigenvalued subspace of L
    # expL_ss = exp_L_nullspace(model.L, model.options.ss_eval_tol)

    # # Act expLt_ss on vectorized initial state
    # init_state_vec = reshape(Array(model.InitState.data), :)
    # steady_state_vec = expL_ss * init_state_vec

    # # Convert back from Liouville space (i.e. reshape) and make into Operator
    # H_dim = size(model.Ham.op, 1)
    # steady_state = reshape(steady_state_vec, H_dim, H_dim)

    # ADD NON-HERMITIAN / POSITIVITY CHECK HERE?

    steady_state = steady_state |> Hermitian |> Array # Also ensure state is exactly hermitian (but not of type Hermitian)
    # steady_state /= tr(steady_state) # This is only really needed if inter-site coupling is zero in system Hamiltonian

    return Operator(basis(model), steady_state)
end


# Bloch-Redfield steady state
function steady_state(model::OQSmodel, ME_type::BlochRedfieldME)

    # Check model actually reaches steady state
    length(model.env_processes) == 0 && error("transport model has no env interactions so won't reach a steady state")

    if ME_type.use_secular && isapprox(ME_type.secular_cutoff, 0, atol=1e-10)

        # Using nullspace method 
        # (THIS IS NOT GOOD FOR NON-SECULAR BRME SINCE IT REQUIRES NORMALIZING FINAL STATE WHICH COULD MASK NON_PHYSICAL RESULTS): 

        L = Array(model.L.data)
        U = eigvecs(Array(model.Ham.op.data))
        ρ0_vec = reshape(inv(U) * model.InitState.data * U, :)

        N = size(U, 1) # Hilbert space dim #Int(sqrt(size(L, 1)))
        steady_state = zeros(ComplexF64, N, N)
        NS = nullspace(L)
        #Faster nullspace based on QR decomp? - needs more testing 
        # Q, R = qr(X)
        # Q, R = Array(Q), Array(R) 
        # NS = Q[:, map(all, eachrow(R .== 0))]

        if size(NS, 2) == 1 #If there's only 1 possible steady state then no need to worry about initial state
            steady_state = reshape(NS, N, N)
        else
            for v in eachcol(NS) #, atol=1e-3))
                steady_state += reshape((v * v') * ρ0_vec, N, N)
            end
        end

        #Skip return statement and proceed to exp method if nullspace method looks like it failed
        if isapprox(tr(steady_state), 0, atol=1e-10)
            @warn "Nullspace method failed (true steady state is probably orthogonal to initial state) \n -> falling back on slower exponential method."
        else
            steady_state /= tr(steady_state) #Make sure state is correctly normalized
            steady_state = U * steady_state * inv(U) #Convert back to original basis
            steady_state = Operator(basis(model), steady_state) #Convert back to QO.jl object
            return steady_state    
        end
    end

    ### (OLD)
    # Exponentiate the zero-eigenvalued subspace of L
    expL_ss = exp_L_nullspace(model.L, model.options.ss_eval_tol)

    # Prep eigenbasis transf (since BR tensor is calculated in Hamiltonian eigenbasis)
    H = Hermitian(Array(data(model.Ham.op))) # Convert to dense Hermitian array (Herm ensures eigen is type stable)
    transf_mat = eigvecs(H)
    inv_transf_mat = inv(transf_mat)

    # Convert init state to Heb
    init_state_Heb = inv_transf_mat * data(model.InitState) * transf_mat
    
    # Act expLt_ss on vectorized initial state
    init_state_vec_Heb = reshape(init_state_Heb, :)
    steady_state_vec = expL_ss * init_state_vec_Heb

    # Convert back from Liouville space (i.e. reshape) and make into Operator
    H_dim = size(model.Ham.op, 1)
    steady_state_Heb = reshape(steady_state_vec, H_dim, H_dim)

    # And finally, convert back to site basis
    steady_state = Operator(basis(model), transf_mat * steady_state_Heb * inv_transf_mat)

    # Check state is physical if using non-secular BRME
    if !ME_type.use_secular
        pops = real(diag(steady_state.data)) #populations(steady_state)
        if !isapprox(sum(pops), 1, atol=model.options.nonphysical_tol) || any(pops .< -model.options.nonphysical_tol)
            model.options.err_on_nonphysical ? throw(NonPhysicalStateError(pops)) : (@warn "Non-physical steady state encountered. \nState populations: $(pops) \n1 - Tr[state] = $(1 - sum(pops))")
        end
    end

    # ADD NON-HERMITIAN / POSITIVITY CHECK HERE?

    steady_state.data = steady_state.data |> Hermitian |> Array # Also ensure state is exactly hermitian (but not of type Hermitian)
    # steady_state /= tr(steady_state) # This is only really needed if inter-site coupling is zero in system Hamiltonian

    return steady_state
end


# Pauli steady state
function steady_state(model::OQSmodel, ME_type::PauliME)

    # H = Hermitian(model.Ham.op.data) # Make Herm for type stable eigen
    # _, transf_mat = eigen(Array(H)) # Need transf to pass to pauli_steady_state (array call ensures dense matrix)
    transf_mat = eigvecs(Hermitian(Array(model.Ham.op.data))) # Need transf to pass to pauli_steady_state (array call ensures dense matrix)

    return pauli_steady_state(model.Ham.op, get_A_ops_and_spectral_funcs(model), model.InitState, model.L, transf_mat; tol=model.options.ss_eval_tol)
end



""" Functions to calculate time dynamics of Transport Model by solving system of ODEs """

dynamics(model::OQSmodel, tspan; kwargs...) = dynamics(model, tspan, model.ME_type; kwargs...)

# Lindblad version
function dynamics(model::OQSmodel, tspan, ME_type::LindbladME; kwargs...) 
    # Deal with non-zero start times
    init_state = tspan[1] == 0 ? model.InitState : sparse(expLt_solve(model, tspan[1]))    
    return timeevolution.master(tspan, init_state, model.Ham.op, get_C_ops(model); kwargs...)
end

# Bloch-Redfield version
function dynamics(model::OQSmodel, tspan, ME_type::BlochRedfieldME; kwargs...) # kwargs passed to ODE solver

    # Deal with non-zero start times
    init_state = tspan[1] == 0 ? model.InitState : expLt_solve(model, tspan[1])

    times, states =  timeevolution.master_bloch_redfield(
        float.(tspan), init_state, model.L, model.Ham.op,
        J=get_C_ops(model), use_secular=ME_type.use_secular, secular_cutoff=ME_type.secular_cutoff; 
        kwargs... # Passed to DiffEq ODE solver
    )
                                                            
    # If using non-secular BRME, check states are physically valid (choose 100 equally spaced states throughout time evolution for efficiency)
    if !ME_type.use_secular

        N_times = length(states)
        dt = N_times < 100 ? Int(round(times[2] - times[1])) : (N_times ÷ 100)
        pops = hcat(populations.(states[1:dt:N_times])...) # Gives H_dim x 100 array of site populations
        min_pop, pop_idx = findmin(pops)
        trace_deviation, tr_idx = abs.(1 .- sum(pops, dims=1)) |> findmax

        # If non-physical states found, find time idx at which non-physicality is most severe
        if min_pop < -model.options.nonphysical_tol # Check if populations are negative
            t_idx = 1 + (pop_idx[2] - 1) * dt # Convert pops slice index back to full times list index
            model.options.err_on_nonphysical ? throw(NonPhysicalStateError(pops[:, pop_idx[2]])) : (@warn "Non-physical state encountered at time $(times[t_idx]). \nState populations: $(pops[:, pop_idx[2]]) \n1 - Tr[state] = $(1 - sum(pops[:, pop_idx[2]]))")
        elseif trace_deviation > model.options.nonphysical_tol
            t_idx = 1 + (tr_idx[2] - 1) * dt # Convert pops slice index back to full times list index
            model.options.err_on_nonphysical ? throw(NonPhysicalStateError(pops[:, tr_idx[2]])) : (@warn "Non-physical state encountered at time $(times[t_idx]). \nState populations: $(pops[:, tr_idx[2]]) \n1 - Tr[state] = $(1 - sum(pops[:, tr_idx[2]]))")
        end

    end

    return times, states
end

# Pauli version
function dynamics(model::OQSmodel, tspan, ME_type::PauliME; kwargs...) 

    # Deal with non-zero start times
    init_state = tspan[1] == 0 ? model.InitState : expLt_solve(model, tspan[1])

    return pauli_dynamics((tspan[1], tspan[end]), model.Ham.op, get_A_ops_and_spectral_funcs(model), init_state; kwargs...)
end




""" Functions to calculate state of Transport Model at single time t without solving the system of ODEs """

expLt_solve(model::OQSmodel, t::Real) = expLt_solve(model, t, model.ME_type)

# Lindblad version
function expLt_solve(model::OQSmodel, t::Real, ME_type::LindbladME)

    # Prep eigenbasis transf
    H = Hermitian(Array(data(model.Ham.op))) # Convert to dense Hermitian array (Herm ensures eigen is type stable)
    evals, transf_mat = eigen(H)

    # Unpack some useful quantities
    N = size(model.Ham.op, 1) # model.Ham.dim
    L = Array(model.L.data) # data(dense(model.L))

    # Calculate matrix exponential
    exp_Lt = exp(L * t)
    # Vectorize initial state
    vec_init_state = reshape(Array(model.InitState.data), :)
    # Get ρ(t) by exp(Lt)*ρ(0)
    vec_state = exp_Lt * vec_init_state
    # Convert from Liouville space back to Hilbert space (vec to 2D array)

    # ADD NON-HERMITIAN / POSITIVITY CHECK HERE?

    state = reshape(vec_state, N, N) |> Hermitian |> Array # Also ensure state is exactly hermitian (but not type Hermitian)
    # Convert to QO state
    state = Operator(basis(model), state)

    return state
end



# Bloch-Redfield version - includes eigenbasis transformations unlike Lindblad
function expLt_solve(model, t::Real, ME_type::BlochRedfieldME)

    # Prep eigenbasis transf
    H = Hermitian(Array(data(model.Ham.op))) # Convert to dense Hermitian array (Herm ensures eigen is type stable)
    evals, transf_mat = eigen(H)

    # transf_mat = diagm(ones(size(model.Ham.op, 1))) # FOR DEBUGGING

    # Unpack some useful quantities
    N = size(model.Ham.op, 1) # model.Ham.dim
    L = Array(model.L.data) # data(dense(model.L))

    # Calculate matrix exponential
    exp_Lt = exp(L * t)
    # In Bloch-Redfield model we need to transform init state to H eigenbasis before vectorizing
    init_state_eb = inv(transf_mat) * data(model.InitState) * transf_mat    
    # Vectorize initial state
    vec_init_state = reshape(init_state_eb, N^2)
    # Get ρ(t) by exp(Lt)*ρ(0)
    vec_state = exp_Lt * vec_init_state
    
    # Convert from Liouville space back to Hilbert space (vec to 2D array)
    state = reshape(vec_state, N, N) 
    # Change back to site basis and convert to Operator
    state = Operator(basis(model), transf_mat * state * inv(transf_mat))

    # Check state is physical if using non-secular BRME
    if !ME_type.use_secular
        pops = populations(state)
        if !isapprox(sum(pops), 1, atol=model.options.nonphysical_tol) || any(pops .< -model.options.nonphysical_tol)
            model.options.err_on_nonphysical ? throw(NonPhysicalStateError(pops)) : (@warn "Non-physical state encountered at time $t. \nState populations: $(pops) \n1 - Tr[state] = $(1 - sum(pops))")
        end
    end

    # ADD NON-HERMITIAN / POSITIVITY CHECK HERE?

    state.data = state.data |> Hermitian |> Array # Also ensure state is exactly hermitian (but not of type Hermitian)

    return state
end


# Pauli version - DOENS'T WORK AS PME CAN'T INCLUDE UNITARY DYNAMICS
# expLt_solve(model::OQSmodel, t::Real, ME_type::PauliME; kwargs...) = pauli_expLt_solve(model.Ham.op, [[op, func] for (op, func) in zip(model.A_ops, model.spectral_funcs)], model.InitState, t; kwargs...)
expLt_solve(model::OQSmodel, t::Real, ME_type::PauliME; kwargs...) = throw(error("Pauli master equation can't be used for time evolution."))




""" Functions to find time at which steady state is reached """

# TODO - Re-write steady state time functionality to fall back on ODE method and solve until ODE solution matches solution reached by exp method (maybe QO.jl steady_state function would be useful?)

function steady_state_time(model::OQSmodel) # , ME_type::MasterEquation, state_idx::Int)

    length(model.env_processes) == 0 && error("transport model has no env interactions so won't reach a steady state")

    # We use the trace distance between ρ(t) and ρ_ss and find the point where this is ≈ 0 using root finding methods -> would we be quicker using gradient descent here?
    ss = steady_state(model) # Pre-calc steady state
    t_guess = 100
    ss_time = find_zero(t -> tracedistance(expLt_solve(model, abs(t)), ss), t_guess, atol=model.options.ss_time_tol, ) # Use abs(t) to enforce root > 0

    return abs(ss_time)
end




""" Using above steady state time function we can write an even simpler site dynamics
function which defaults to solving the full time range from t0 to tss """
function dynamics(model::OQSmodel; t0=0.0, t_end=nothing, L=10^4, kwargs...)

    # If no t_end given, use steady state time
    t_end === nothing && (t_end = steady_state_time(model))
    times = range(t0, t_end, length=L)
    times, states = dynamics(model, times; kwargs...)

    return times, states
end



# end # Module
