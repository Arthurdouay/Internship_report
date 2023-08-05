import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as anim
from matplotlib import cm

##common variables
#initial starting time, total time of the simulation, size of the grid
t0 = 0
N = 1500
#size of time incrementation
dt = 0.0005
T = dt*N                                  #s

time = np.linspace(0,T,N)

#constants and parameters
R = 100*10**5            #ohm
C = 100*10**-12        #farad
tm = R*C
#average threshold
thr = 80*10**-3        #V
#tension t=0
u0 = -80*10**-3          #V
#tension reset
ur = -80*10**-3          #V
#absolute refractory period
dabs = 2*10**-3          #s

##construction of the patch

#number of neurons in the patch
m = 1000

def sd(t):
    return 0.1

def g(u):
    return u



def Eulerintfpatch(U0, Vbar, g, g1, sd):
    random_thr = np.random.normal(thr,10**-2,m)            #randomized thresholds
    spikes = np.zeros((m,N))
    #translation of absolute refractory period in dt iterations (returns 1 in case int(dabs/dt) is zero st the program runs)
    l = max(1,int(dabs/dt))
    #variable I use in the loop
    spike_countdown = np.zeros(m)
    #membrane potential
    U = np.zeros((m,N))
    U[:,0] = U0

    for k in range(1,N):
        for j in range(m):
            if spike_countdown[j] != 0:
                spike_countdown[j] += -1
                U[j,k] = ur

            else :
                if U[j,k-1] > random_thr[j]:
                    spike_countdown[j] = l
                    spikes[j,k] = 1
                    U[j,k] = ur

                else :
                    In = np.random.normal(0,sd(t0 + k*dt),1)                                     #centered noise current inside this neuron
                    U[j,k] = U[j,k-1] + dt*(-1/tm*g(U[j,k-1]) + g1*Vbar) + g1*np.sqrt(dt)*In


    #temporary cheap solution to avoid last iteration big spike
    U[:,N-1] = U[:,N-2]
    return U, spikes



U0 = ur*np.ones(m) + thr*np.random.rand(m)                          #irregular starting condition

##To test the Eulerintfpatch


U, spikes = Eulerintfpatch(U0,10,g,10**-9/C,sd)


Umean = np.mean(U,axis = 0)

plt.figure()
plt.clf()
plt.plot(time*10**3, Umean*10**3)
plt.xlabel("time in ms")
plt.ylabel("tension in mV")
plt.title("Average tension in cell membrane in the patch of neurons")
plt.show()

plt.figure()
plt.clf()
plt.plot(time,U[0,:])
plt.show()

firing_rate = np.count_nonzero(spikes, axis = 0)/dt
plt.figure()
plt.clf()
plt.plot(time*10**3, firing_rate*10**-3)
plt.xlabel("time in ms")
plt.ylabel("average firing rate in kHz")
plt.title("Average firing rate of the patch of neurons")
plt.show()


##Average firing rate depending on input current

firing_rate = np.zeros(300)
Vbar_vect = np.linspace(-5,25,300)

for i,Vbar in enumerate(Vbar_vect):
    print(i)
    U, spikes = Eulerintfpatch(U0,Vbar,g,10**-9/C,sd)
    if i == 180:
        Utest = U
    firing_rate[i] = np.count_nonzero(spikes)/T

##Sigmoid to fit the curve


def sigm(x,V):
    return x[0] + x[1]/((x[2] + np.exp(-x[3]*V*np.log(x[4])))**x[5])  #formula 2



nbr_particles = 100


def diff_func(x):
    sigm_vect = np.zeros((nbr_particles,np.size(Vbar_vect)))
    difference = np.zeros(nbr_particles)
    for i in range(nbr_particles):
        sigm_vect[i,:] = sigm(x[i,:],Vbar_vect)

    for i in range(nbr_particles):
        difference[i] = np.sum((firing_rate*10**-3/m - sigm_vect[i,:])**2)
    return difference



options = {'c1':0.7, 'c2':0.4, 'w':0.9}
optimizer = GlobalBestPSO(n_particles = nbr_particles, dimensions = 7, options = options)

cost, pos = optimizer.optimize(diff_func, 1000)

print(cost, pos)

"""
##Threshold to fit the curve

def sigm_thr(x,V):
    V = np.atleast_1d(V)
    Vreturn = np.zeros(np.size(V))
    for k in range(np.size(V)):
        if V[k] >= 1.0:
           Vreturn[k] = x[5]/((x[0] + x[1]*np.log(np.abs(x[2]*V[k]/(x[3]*V[k] - x[4]))))**x[6])
        else:
            Vreturn[k] = 0
    return Vreturn

nbr_particles = 50


def diff_func_thr(x):
    sigm_thr_vect = np.zeros((nbr_particles,np.size(Vbar_vect)))
    difference = np.zeros(nbr_particles)
    for i in range(nbr_particles):
        for k,Vbar in enumerate(Vbar_vect):
            sigm_thr_vect[i,k] = sigm_thr(x[i,:],Vbar)

    for i in range(nbr_particles):
        difference[i] = np.max(np.abs(firing_rate*10**-3/m - sigm_thr_vect[i,:]))
    return difference


x_max = 2*np.ones(5)
x_min = -1*x_max
bounds = (x_min,x_max)
options = {'c1':0.7, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles = nbr_particles, dimensions = 7, options = options)

cost, pos_thr = optimizer.optimize(diff_func_thr, 500)
"""

##Load saved curve and fit

firing_rate = np.loadtxt("Curve_data.txt")

Vbar_vect = np.linspace(-5,25,200)

pos = np.array([ 0.31076342, -0.51088103, 16.74378733,  1.37668199,  0.48519008,  0.17654685])

def sigm(x,V):
    return x[0] + x[1]/((x[2] + np.exp(-x[3]*V*np.log(x[4])))**x[5])

##plot
plt.figure()
plt.clf()
#plt.plot(Vbar_vect,firing_rate*10**-3/m*50, color = "blue", label = "$S[\overline{V}(t)]$")
#plt.plot(Vbar_vect, sigm_thr(pos_thr,Vbar_vect)*50, color = "darkgreen", label = "sigmoid fit with threshold")
plt.plot(Vbar_vect, sigm(pos,Vbar_vect)*50, color = "orange", label = "sigmoid fit")
plt.plot(Vbar_vect,Vbar_vect, color = "red", label = "$\overline{V}$")
plt.xlabel("Average dendritic current")
plt.ylabel("Average firing rate")
plt.title("Average firing rate depending on average dendritic current")
plt.legend()
plt.show()

##Why isn't Vbar smooth enough ?
plt.figure()
plt.clf()
plt.plot(time,U[0,:], color = "blue", label = "neuron 0")
plt.plot(time,U[20,:], color = "red", label = "neuron 20" )
plt.plot(time,U[60,:], color = "green", label = "neuron 60")
plt.plot(time,U[100,:], color = "purple", label = "neuron 100")
plt.legend()
plt.title("Les plateaux ?")
plt.show()



##Study of Vbar
tau = 10**-1       #s


def h(Vbar):                                     #left hand side of equation (7)
    return -Vbar/tau + 50*sigm(pos,Vbar)/tau

def EulerVbar(Vbar0):                            #Computation of the function with equation (7) and sigmoid fit of vbar
    Vbar = np.zeros(N)
    Vbar[0] = Vbar0
    dt = T/N

    for k in range(1,N):
        Vbar[k] = Vbar[k-1] + dt*h(Vbar[k-1])

    return Vbar

def VbarODE_7(Vbar0):
    Vbar = np.zeros(N)
    Vbar[0] = Vbar0


    for k in range(1,N):
        firing_rate_index = int((Vbar[k-1] + 5)/30*200)
        Vbar[k] = Vbar[k-1] + dt*(-Vbar[k-1]/tau + 50*firing_rate[firing_rate_index]*10**-3/(m*tau))

    return Vbar



def VbarODE_1(V0, U0, g1, g, sd):                      #Computation of Vbar with equation (1)
    dt = T/N
    m_ode = np.size(U0)
    spikes = np.zeros((m_ode,N))           #accounts for the number of spikes at each time period
    l = max(1,int(dabs/dt))            #translation of absolute refractory period in dt iterations (returns 1 in case int(dabs/dt) is zero st the program runs)
    spikes_countdown = np.zeros(m_ode)
    tk = t0

    random_thr = np.random.normal(thr,5*10**-3,m)

    #membrane potential (calculated at each time st it depends on Vbar(t))
    U = np.zeros((m_ode,N))
    V = np.zeros((m_ode,N))
    V[:,0] = V0
    U[:,0] = U0
    Vbar = np.zeros(N)
    Vbar[0] = np.mean(V0)
    V_index = 1

    for k in range(1,N):
        for j in range(m_ode):
            if spikes_countdown[j] != 0:
                U[j,k] = ur
                spikes_countdown[j] += -1


            else :
                if U[j,k-1] > random_thr[j]:
                    U[j,k] = ur
                    spikes_countdown[j] = l
                    spikes[j,k] = 1

                else :
                    In = np.random.normal(0,sd(t0 + k*dt),1)                                     #centered noise current inside this neuron
                    U[j,k] = U[j,k-1] + dt*(-1/tm*g(U[j,k-1]) + g1*V[j,k-1]) + g1*np.sqrt(dt)*In

        V[:,k] = V[:,k-1] + dt*(-V[:,k-1]/tau + 0.1*np.sum(spikes[:,k-1])/(tau))        #Computing V_is individually (how to put the right scaling factor ?)

    return np.mean(V, axis = 0), U, spikes                           #this returns the average dendritic current

Vbari = 3
Vbari_neural = Vbari*np.ones(m)
U0 = ur*np.ones(m) + thr*np.random.rand(m)

Vbar_estimate = EulerVbar(Vbari)
Vbar_ODE_1_vect, U, synch_spikes = VbarODE_1(Vbari_neural, U0, 10**-9/C, g, sd)
Vbar_ODE_7_vect = VbarODE_7(Vbari)

#stable fixed points for sigmoid fit : 0, 13
#unstable fixed points for sigmoid fit : 5.85

plt.figure()
plt.clf()
plt.plot(time, Vbar_estimate, color = "orange", label = r"$\overline{V}$ with S[$\overline{V}$]")
plt.plot(time, Vbar_ODE_1_vect, color = "blue", label = r"$\overline{V}$ with ODE (1)")
plt.plot(time, Vbar_ODE_7_vect, color = "purple", label = r"$\overline{V}$ with ODE (7)")
plt.xlabel("Time")
plt.ylabel("Average dendritic current")
plt.legend()
plt.title("Evolution of dendritic current with $V_0$ = " + str(Vbari))
plt.show()

##measure of neuron synchrony

"""
U_bar_var = []
U_var = []

m_min = 100
m_max = 500
m_step = 10
Vbari = 10

for m_loop in range(m_min,m_max,m_step):
    U0 = -80*10**-3*np.ones(m_loop) + 80*10**-3*np.random.rand(m_loop)
    U = VbarODE_1(Vbari*np.ones(m_loop) + np.random.normal(0,1,m_loop), U0, 10**-9/C, g, sd)[1]
    Ubar = np.mean(U, axis = 0)                                                  #calcul de Vbar
    U_bar_var.append(np.mean(Ubar**2) - np.mean(Ubar)**2)                        #calcul de la variance
    U_var.append(np.mean(U**2,axis = 1) - np.mean(U, axis=1)**2)

synch = np.zeros(len(U_bar_var))

for k in range(np.size(synch)):
    synch[k] = U_bar_var[k]/np.mean(U_var[k])

plt.figure()
plt.clf()
plt.plot(np.linspace(m_min,m_max,int(m_max-m_min/m_step)), np.sqrt(synch))
plt.title("measure of synchrony depending on the number of neurons")
plt.xlabel("number of neurons")
plt.ylabel("measure of synchrony")
plt.show()

"""
##Raster plot

plt.figure()
plt.clf()
for j in range(0,m,20):
    for k in range(N):
        if synch_spikes[j,k] == 0 :
            synch_spikes[j,k] = -30
    plt.scatter(time, max(j,1)*synch_spikes[j,:])
plt.title("Raster plot of the number of spikes in the neural field")
plt.ylim(bottom = 0 ,top = m)
plt.xlabel("time")
plt.ylabel("index of the neuron that fires")
plt.show()

##To test fixed points
f1, f2 = 0.12, 12.9
fu = 5

colors = ["red", "orangered", "chocolate", "orange", "yellowgreen", "limegreen", "darkturquoise", "steelblue", "darkslateblue", "crimson"]

plt.figure()
plt.clf()
for col in colors:
    plt.plot(time, VbarODE_1(np.random.normal(f1,1,m),U0, 10**-9/C, g, sd)[0], color = col)
plt.title("10 trajectories with random initial condition around " + str(f1))
plt.show()

plt.figure()
plt.clf()
for col in colors:
    plt.plot(time, VbarODE_1(np.random.normal(f2,1,m),U0, 10**-9/C, g, sd)[0], color = col)
plt.title("10 trajectories with random initial condition around "  + str(f2))
plt.show()

plt.figure()
plt.clf()
for col in colors:
    plt.plot(time, VbarODE_1(np.random.normal(fu,1,m),U0,10**-9/C, g, sd)[0], color = col)
plt.title("10 trajectories with random initial condition around "  + str(fu))
plt.show()

##Random coupling of neurons
tau = 10**-1       #s
Vbar = 4
U0 = np.random.normal(ur,10**-2,1000)
V0 = np.random.normal(Vbar, 1, 1000)


def Vbar_coupled(V0, U0, g1, g, sd, p):                      #Computation of Vbar with equation (1)
    dt = T/N
    m_ode = np.size(U0)
    spikes = np.zeros((m_ode,N))           #accounts for the number of spikes at each time period
    l = max(1,int(dabs/dt))            #translation of absolute refractory period in dt iterations (returns 1 in case int(dabs/dt) is zero st the program runs)
    spikes_countdown = np.zeros(m_ode)
    tk = t0

    K = np.random.binomial(1,p,(m_ode,m_ode))                       #random coupling matrix (Erdös-Renyi)

    random_thr = np.random.normal(thr,5*10**-3,m)

    #membrane potential (calculated at each time st it depends on Vbar(t))
    U = np.zeros((m_ode,N))
    V = np.zeros((m_ode,N))
    V[:,0] = V0
    U[:,0] = U0
    Vbar = np.zeros(N)
    Vbar[0] = np.mean(V0)
    V_index = 1

    for k in range(1,N):
        for j in range(m_ode):
            if spikes_countdown[j] != 0:
                U[j,k] = ur
                spikes_countdown[j] += -1


            else :
                if U[j,k-1] > random_thr[j]:
                    U[j,k] = ur
                    spikes_countdown[j] = l
                    spikes[j,k] = 1

                else :
                    In = np.random.normal(0,sd(t0 + k*dt),1)                                     #centered noise current inside this neuron
                    U[j,k] = U[j,k-1] + dt*(-1/tm*g(U[j,k-1]) + g1*V[j,k-1]) + g1*np.sqrt(dt)*In

        for j in range(m_ode):
            V[j,k] = V[j,k-1] + dt*(-V[j,k-1]/tau + 0.1*np.sum(K[:,j]*spikes[:,k-1])/(tau))      #Computing V_is individually (how to put the right scaling factor ?)

    return np.mean(V, axis = 0), U, spikes                           #this returns the average dendritic current



Vp = np.zeros(100)
for i,p in enumerate(np.linspace(0,1,100)):
    Vp[i] = Vbar_coupled(V0,U0,10**-9/C,g,sd,p)[0][-1]
    print(i)

plt.figure()
plt.clf()
plt.plot(np.linspace(0,1,100), Vp)
plt.title(r"Average dendritic current in stationary state for $p \in [ 0,1 ]$" )
plt.xlabel("p")
plt.ylabel("Average dendritic current")
plt.show()




























