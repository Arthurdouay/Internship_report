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

##Load saved curve and fit

firing_rate = np.loadtxt("Curve_data.txt")

Vbar_vect = np.linspace(-5,25,200)

pos = np.array([ 0.31076342, -0.51088103, 16.74378733,  1.37668199,  0.48519008,  0.17654685])

def sigm(x,V):
    return x[0] + x[1]/((x[2] + np.exp(-x[3]*V*np.log(x[4])))**x[5])

##plot
plt.figure()
plt.clf()
plt.plot(Vbar_vect,firing_rate*10**-3/m*50, color = "blue", label = "$S[\overline{V}(t)]$")
plt.plot(Vbar_vect, sigm_thr(pos_thr,Vbar_vect)*50, color = "darkgreen", label = "sigmoid fit with threshold")
plt.plot(Vbar_vect, sigm(pos,Vbar_vect)*50, color = "orange", label = "sigmoid fit")
plt.plot(Vbar_vect,Vbar_vect, color = "red", label = "$\overline{V}$")
plt.xlabel("Average dendritic current")
plt.ylabel("Average firing rate")
plt.title("Average firing rate depending on average dendritic current")
plt.legend()
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























