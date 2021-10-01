
# using ExcitonTransportModels
# include(MyPackageDir * "/ExcitonTransportModels/test/test_TransportModels.jl");


# """===========================================================================
#  THESE NEED UPDATED TO WORK WITH NEW TRANSPORT MODEL CALCULUS IMPLEMENTATION
# =============================================================================="""



# # Define a simple transport model and a steady state sink population figure of merit
# NoSites = 4
# env_states = [ EnvState("sink", 0.5, NoSites+1); EnvState("ground", 0.1, NoSites+2)];
# energies = Float64[2 for i in 1:NoSites];
# H = TransportHamiltonian(energies, NextNearestNeighbour(0.01, 0.002), env_states[1:end]);
# #Construct env processes
# extraction = CollapseOp("extraction", transition(NLevelBasis(H.dim), NoSites+1, NoSites), 0.0005);
# loss = [CollapseOp("Site$(i)_loss", transition(NLevelBasis(H.dim), NoSites+2, i), 1e-5) for i in 1:NoSites];
# dephasing = [CollapseOp("Site$(i)_deph", transition(NLevelBasis(H.dim), i, i), 1e-3) for i in 1:NoSites];
# local_phonons = [InteractionOp("Site$(i)_phonons", transition(NLevelBasis(H.dim), i, i), SpectralDensity(Sw_superohmic_Gaussian, (beta=1/(k_boltzmann*300), cutoff=0.1, coupling=1e-2))) for i in 1:NoSites]

# #Construct transport model
# tm = ExcitonTransportModel(H, [local_phonons...; loss...; extraction], BlochRedfieldME(), 1);
# FoM(x::ExcitonTransportModel) = real(data(steady_state(x))[x.NoSites+1, x.NoSites+1])
# @show FoM(tm)

# #Test various derivative functions
# @time named_gradient(tm, FoM, SiteEnergiesDeriv(1:NoSites))
# @time gradient(tm, FoM, SiteEnergiesDeriv(1:2))
# @time gradient(tm, FoM, SiteEnergiesDeriv(3:4))

# get_param_values(tm, SiteEnergiesDeriv(1:3))
# get_param_values(tm, CoherentCouplingDeriv(1:NoSites))

# @time named_gradient(tm, FoM, CoherentCouplingDeriv(1:NoSites))

# H = TransportHamiltonian(energies, NearestNeighbour(0.01), env_states[1:end]);
# tm = ExcitonTransportModel(H, [dephasing...; loss...; extraction], LindbladME(), 1);
# @time named_gradient(tm, FoM, CoherentCouplingDeriv(1:NoSites))


# dist_dep(x1, x2; J=1e-2) = J / norm(x1 - x2)^3;
# pos = zeros(NoSites, 3);
# pos[:, 1] = 1:NoSites;
# # pos = rand(NoSites, 3)
# H = TransportHamiltonian(energies, DistanceDependent(pos, dist_dep), env_states[1:end]);
# tm = ExcitonTransportModel(H, [dephasing...; loss...; extraction], LindbladME(), 1);
# @time gradient(tm, FoM, CoherentCouplingDeriv(1:NoSites))
# @time gradient(tm, FoM, CoherentCouplingDeriv(3:4))
# @time named_gradient(tm, FoM, CoherentCouplingDeriv(1:NoSites))

# get_param_values(tm, CoherentCouplingDeriv(1:3))
# get_param_names(tm, CoherentCouplingDeriv(1:3))

# #Test env energy derivs
# @time gradient(tm, FoM, EnvEnergyDeriv("sink"))


# #Test env rate derivative functions
# H = TransportHamiltonian(energies, NearestNeighbour(0.01), env_states[1:end]);
# tm = ExcitonTransportModel(H, [dephasing...; loss...; extraction], LindbladME(), 1);
# @time gradient(tm, FoM, EnvRateDeriv("extraction"))
# @time for i in 1:NoSites
#     println(named_gradient(tm, FoM, EnvRateDeriv("Site$(i)_loss")))
#     println(named_gradient(tm, FoM, EnvRateDeriv("Site$(i)_deph")))
#     # println("d/d($(i)_loss) = ", gradient(tm, FoM, EnvRateDeriv("Site$(i)_loss")))
#     # println("d/d($(i)_deph) = ", gradient(tm, FoM, EnvRateDeriv("Site$(i)_deph")))
# end

# #Test env param derivative functions
# tm = ExcitonTransportModel(H, [local_phonons; extraction], BlochRedfieldME(), 1);
# @time for i in 1:NoSites
#     # println("d/d($(i)_ph) = ", gradient(tm, FoM, EnvParamDeriv("Site$(i)_phonons")))
#     println(named_gradient(tm, FoM, EnvParamDeriv("Site$(i)_phonons")))
# end



# """ The above tests seem to work when the FoM returns a scalar value - what about a discrete distribution (i.e. an array) instead? """

# function FoM_dist(tm)
#     ss = steady_state(tm)
#     return Float64[abs2(expect(transition(tm.basis, i, i), ss)) for i in 1:tm.Ham.dim]
# end;
# @show FoM_dist(tm)

# gradient(tm, FoM_dist, SiteEnergiesDeriv(1:NoSites))
# named_gradient(tm, FoM_dist, SiteEnergiesDeriv(1:NoSites))
# gradient(tm, FoM_dist, SiteEnergiesDeriv(2:NoSites-1))

# H = TransportHamiltonian(energies, NearestNeighbour(0.01), env_states[1:end]);
# tm = ExcitonTransportModel(H, [dephasing...; loss...; extraction], LindbladME(), 1);
# gradient(tm, FoM_dist, CoherentCouplingDeriv(1:NoSites))
# named_gradient(tm, FoM_dist, CoherentCouplingDeriv(1:NoSites))

# H = TransportHamiltonian(energies, NextNearestNeighbour(0.01, 0.002), env_states[1:end]);
# tm = ExcitonTransportModel(H, [dephasing...; loss...; extraction], LindbladME(), 1);
# gradient(tm, FoM_dist, CoherentCouplingDeriv(1:NoSites))
# named_gradient(tm, FoM_dist, CoherentCouplingDeriv(1:NoSites))


# H = TransportHamiltonian(energies, DistanceDependent(pos, dist_dep), env_states[1:end]);
# tm = ExcitonTransportModel(H, [dephasing...; loss...; extraction], LindbladME(), 1);
# gradient(tm, FoM_dist, CoherentCouplingDeriv(1:NoSites))
# gradient(tm, FoM_dist, CoherentCouplingDeriv(1:2))
# named_gradient(tm, FoM_dist, CoherentCouplingDeriv(1:NoSites))



# named_gradient(tm, FoM_dist, EnvRateDeriv("extraction"))
# tm = ExcitonTransportModel(H, [local_phonons...; loss...; extraction], BlochRedfieldME(), 1);
# named_gradient(tm, FoM_dist, EnvParamDeriv("Site1_phonons"))
# named_gradient(tm, FoM_dist, EnvParamDeriv("Site1_phonons", [:beta]))



# """ Test full transport model gradient function """

# dist_dep(x1, x2; J=1e-2) = J / norm(x1 - x2)^3;
# pos = zeros(NoSites, 3);
# pos[:, 1] = 1:NoSites;
# H = TransportHamiltonian(energies, DistanceDependent(pos, dist_dep), env_states[1:end]);
# tm = ExcitonTransportModel(H, [local_phonons...; loss...; extraction], BlochRedfieldME(), 1);
# deriv_params = [SiteEnergiesDeriv(1:NoSites), CoherentCouplingDeriv(1:NoSites), EnvRateDeriv("extraction"), EnvParamDeriv("Site1_phonons")]

# # For simple steady state figure of merit
# @time grad_vec = transport_model_gradient(tm, FoM, deriv_params)
# @time grad_vec = transport_model_named_gradient(tm, FoM, deriv_params)
# # And for FoM_dist figure of merit
# @time grad_vec = transport_model_gradient(tm, FoM_dist, deriv_params)
# @time grad_vec = transport_model_named_gradient(tm, FoM_dist, deriv_params)

# @code_warntype transport_model_gradient(tm, FoM_dist, FoM_dist(tm), deriv_params)
