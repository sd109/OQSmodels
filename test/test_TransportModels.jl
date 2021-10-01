
# """ Definitions needed for tests: """

# using QuantumOptics, ExcitonTransportModels
# const k_boltzmann = 1.38064852e-23 / 1.6021766208e-19 #eV     from python's scipy.constants module for k/e
# #Super-ohmic spectral density with Gaussian cutoff
# function Sw_superohmic_Gaussian(w::Real, beta::Real, w_cutoff::Real, phonon_coupling::Real)
#     NBE = (exp(abs(w)*beta)-1)^-1.0 #Bose-Einstein dist for phonons
#     A = phonon_coupling * abs(w/w_cutoff)^3.0
#     B = exp(-(w/w_cutoff)^2.0)
#     #Absorb a phonon
#     if w < 0.0
#         return (NBE * A * B)::Float64 #Annotate return type
#     #Emit a phonon
#     elseif w > 0.0
#         return ((NBE+1) * A * B)::Float64 #Annotate return type
#     else w == 0.0
#         return 0.0
#     end
# end

# NoSites = 4;
# NoEnvStates = 1;
# N = NoSites + NoEnvStates;
# bas = NLevelBasis(N);

# #Test coupling constructors
# coupling_operator(NoSites, NearestNeighbour(0.1), extra_dims=NoEnvStates)
# coupling_operator(NoSites, NextNearestNeighbour(0.1, 0.02), extra_dims=NoEnvStates)
# f(pos1, pos2; J=1e-2) = J / norm(pos1 - pos2)^3;
# positions = zeros(NoSites, 3);
# positions[:, 1] = 1:NoSites;
# coupling_operator(NoSites, DistanceDependent(positions, f), extra_dims=NoEnvStates)

# #Test TransportHamiltonian constructors
# env_states = [EnvState("ground", 0.5, NoSites+1)]
# energies = Float64[2-0.001*i for i in 1:NoSites]
# # energies = Float64[2+0.1*rand() for i in 1:NoSites]
# H = TransportHamiltonian(energies, NextNearestNeighbour(0.1, 0.02), env_states)

# #Create and env process
# extraction = CollapseOp("Extraction", transition(bas, NoSites+1, NoSites), 0.0001);
# injection = CollapseOp("Injection", transition(bas, 1, NoSites+1), 0.02);
# # Sw = w -> Sw_superohmic_Gaussian(w, 1/(k_boltzmann*0.1), 0.1, 1e-1)
# local_phonons = [InteractionOp("Phonons", transition(bas, i, i), SpectralDensity(Sw_superohmic_Gaussian, (beta=1/(k_boltzmann*0.1), cutoff=0.1, coupling=1e-1))) for i in 1:NoSites]
# env_processes = [local_phonons; extraction; injection];

# #Test transport model constructors
# tm = ExcitonTransportModel(H, [injection, extraction], LindbladME(), 1); # InitSite = 1
# tm = ExcitonTransportModel(H, env_processes, BlochRedfieldME(), 1);
# tm = ExcitonTransportModel(H, [local_phonons...], PauliME(), 1);
