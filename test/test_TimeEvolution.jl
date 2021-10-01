
include("./test_TransportModels.jl"); #Runs all transport model tests as pre-requisite for tests in this file
# using TransportModelTimeEvolution


""" Test steady state functions """
#Case 1 - env state should be steady state if C_ops included
tm_ss1 = ExcitonTransportModel(H, [extraction], LindbladME(), 1); # Works correctly
steady_state(tm_ss1)

tm_ss1 = ExcitonTransportModel(H, [extraction], BlochRedfieldME(), 1); # Works correctly
steady_state(tm_ss1)

tm_ss1 = ExcitonTransportModel(H, [extraction], PauliME(), 1); # Errors as it should since tm has C_ops


#Case 2 - Steady state should be close to lowest energy eigenstate but not quite (due to finite T)
tm_ss2 = ExcitonTransportModel(H, [local_phonons...], LindbladME(), 1) # Errors as it should since tm has A_ops

tm_ss2 = ExcitonTransportModel(H, [local_phonons...], BlochRedfieldME(), 1);
real(steady_state(tm_ss2).data) #Looks sensible and populations are right

tm_ss2 = ExcitonTransportModel(H, [local_phonons...], PauliME(), 1);
steady_state(tm_ss2).data |> real # Finds a sensible steady state (close to lowest energy eigenstate)

#Check lowest energy eigenstate to see if it matches steady state
evals, U = eigen(H.H.data);
ss_pops = abs2.(U[:, 2])



#Case 3 - no steady state
tm_ss3 = ExcitonTransportModel(H, [injection], LindbladME(), 1);
steady_state(tm_ss3) #Ideally should error

tm_ss3 = ExcitonTransportModel(H, [injection;], BlochRedfieldME(), 1);
steady_state(tm_ss3) # I'm not sure if there's a good way of making this error - might just have to use common sense for ss with BRME...

tm_ss3 = ExcitonTransportModel(H, [injection], PauliME(), 1); # Errors as it should since C_ops provided



""" Test time dynamics ODE functions """
using Plots

#Simple extraction model
tm = ExcitonTransportModel(H, [extraction], LindbladME(), 1); # InitSite = 1
tspan = range(0, 1e3, length=10000);
times, states = dynamics(tm, tspan)
#Plot the results
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])


#Local phonon model with extraction
tm = ExcitonTransportModel(H, [extraction; local_phonons;], BlochRedfieldME(), 1); # InitSite = 1
tspan = range(0, 1e3, length=10000);
times, states = dynamics(tm, tspan)
#Plot the results
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])


#Local phonon only model - steady state looks right but initial state is wrong... why? - Because PME can't do time dynamics... It only works for steady states!
# tm = ExcitonTransportModel(H, [local_phonons...], PauliME(), 1); # InitSite = 1
# tspan = range(0, 1e3, length=10000);
# times, states = dynamics(tm, tspan)
# #Plot the results
# P = populations.(states)
# plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])

#Local phonon only model - BRME
tm = ExcitonTransportModel(H, [local_phonons...], BlochRedfieldME(), 1); # InitSite = 1
tspan = range(0, 1e3, length=10000);
times, states = dynamics(tm, tspan)
#Plot the results
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])



""" Test expLt_solve functions """

#Lindblad model
tm = ExcitonTransportModel(H, [extraction], LindbladME(), 1); # InitSite = 1
expLt_solve(tm, 0) #Should recover initial state
expLt_solve(tm, 1e8) #Should recover steady state


#BlochRedfield model
tm = ExcitonTransportModel(H, [local_phonons...], BlochRedfieldME(), 1); # InitSite = 1
expLt_solve(tm, 0) #Should recover initial state
expLt_solve(tm, 1e8) #Should recover steady state

#Pauli model - doesn't work since PME can't do non-steady state stuff
# tm = ExcitonTransportModel(H, [local_phonons...], PauliME(), 1); # InitSite = 1
# expLt_solve(tm, 0) #Should recover initial state
# expLt_solve(tm, 1000000) #Should recover steady state



""" Compare exp and ODE results """

tm = ExcitonTransportModel(H, [extraction], LindbladME(), 1); # InitSite = 1

times = range(0, 400000, length=10000)
exp_states = map( t -> expLt_solve(tm, t), times )
exp_Ps = [real(expect(transition(tm.basis, i, i), exp_states)) for i in 1:tm.Ham.dim]
times, ode_states = dynamics(tm, times)
ode_Ps =  [real(expect(transition(tm.basis, i, i), ode_states)) for i in 1:tm.Ham.dim]

using Plots
exp_plot = plot(fill(times, 5), exp_Ps);
ode_plot = plot(fill(times, 5), ode_Ps);
plot(exp_plot, ode_plot, layout=(2, 1))








""" Test steady_state_time functions """
using Plots

#Lindblad version
tm = ExcitonTransportModel(H, [extraction], LindbladME(), 1); # InitSite = 1
tss = steady_state_time(tm)
#Compare to time dynamics plot
times, states = dynamics(tm, range(0, tss, length=10^5))
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])

#BRME version
tm = ExcitonTransportModel(H, [extraction, local_phonons...], BlochRedfieldME(), 1); # InitSite = 1
tss = steady_state_time(tm)
#Compare to time dynamics plot
times, states = dynamics(tm, range(0, tss, length=10^5))
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])

#Pauli version - doesn't work as PME can't do intermediate time dynamics
# tm = ExcitonTransportModel(H, [local_phonons...], PauliME(), 1); # InitSite = 1
# tss = steady_state_time(tm, eb_warn=false) #Doesn't work for PME - need to think about why...
# times, states = dynamics(tm, range(0, 30000, length=10^5), eb_warn=false)
# P = populations.(states)
# plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])



""" Test dynamics func with auto ss_time calc """
using Plots
gr()

tm = ExcitonTransportModel(H, [extraction], LindbladME(), 1); # InitSite = 1
times, states = dynamics(tm, ss_time_tol=1e-5, silent=true)
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])

tm = ExcitonTransportModel(H, [local_phonons...], BlochRedfieldME(), 1); # InitSite = 1
times, states = dynamics(tm, ss_time_tol=1e-3)
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])



tm = ExcitonTransportModel(H, [local_phonons...], PauliME(), 1); # InitSite = 1
times, states = dynamics(tm; eb_warn=false, silent=true)
P = populations.(states)
plot(fill(times, 4), [[P[i][j] for i in 1:length(times)] for j in 1:H.dim])
