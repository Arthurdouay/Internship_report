import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------
##common variables
#initial starting time, total time of the simulation, size of the grid
t0 = 0
N = 10000
sd = 1

#noise current in the neuron (random gaussian vector)
I =  np.random.normal(0,sd,N)

#constant current in the neuron
I0 = 0

#-----------------------------------------------------------------------------------------------
##Integrate and fire model but with noise

Ti = 0.5
#size of time incrementation
dti = Ti/N

timei = np.linspace(0,Ti,N)

#constants and parameters
R = 1             #ohm
C = 0.1         #farad
tm = R*C
#threshold
thr = -20         #mV
#tension t=0
u0 = -80          #mV
#tension reset
ur = -80          #mV
#absolute refractory period
dabs = 0.004


def Eulerintf(u0,ur,thr,I0,In):
    spikes = np.empty(0)
    #translation of abs refr per in dti iterations
    l = int(dabs/dti)
    #variable I use in the loop
    k = 1
    tk = t0
    u = np.zeros(N)
    u[0] = u0

    while k < N:

        tk = t0 + k*dti

        if u[k-1] > thr:

            for m in range(l):
                if k+m < N:
                    u[k+m] = ur

            spikes = np.append(spikes,tk)
            k = k+l

        else :
            u[k] = u[k-1] + dti*(-1/tm*u[k-1] + I0) + 1/C*np.sqrt(dti)*In[k-1]
            k = k+1

    return u, spikes


u, spikesintf = Eulerintf(u0,ur,thr,I0,I)

plt.figure()
plt.clf()
plt.plot(timei, u, label = "potential u(t)")
plt.title("Integrate and fire model with sd = " + str(sd))
plt.xlabel("t")
plt.ylabel("u(t)")
plt.show()


#-------------------------------------------------------------------------------------------
##FitzHugh-Nagumo model but with noise

Tf = 100                  #ms
#size of time incrementation
dtf = Tf/N                #ms
eps = 0.1
b0 = 2
b1 = 1.5
u0 = -3
w0 = -1

#U = (u,w)
def G(U,I0):
    return np.array([U[0]-1/3*U[0]**3-U[1]+ I0,eps*(b0 + b1*U[0] - U[1])])

def Eulerfitznag(u0,w0,I0,In):
    spikes = np.empty(0)
    spikelag = 200                    #this variable controls the size of the window we use to test if there is a spike at time tk or not
    U = np.zeros((2,N))
    U[:,0] = np.array([u0,w0])

    for k in range(1,N):
        tk = t0 + k*dtf

        U[:,k] = U[:,k-1] + dtf*G(U[:,k-1],I0) + np.sqrt(dtf)*np.array([I[k-1],0])

    for k in range(N):
        if U[0,k] == np.max(U[0,np.max(np.array([0,k-spikelag])):np.min(np.array([N,k+spikelag]))]) and U[0,k] > 1.5: #looks kind of bad to detect spikes, 1.5 is arbitrary
            spikes = np.append(spikes,t0 + k*dtf)

    return U, spikes

U, spikesnagu = Eulerfitznag(u0,w0,I0,I)


#FitzHugh-Nagumo plot

timef = np.linspace(0,Tf,N)

plt.figure()
plt.clf()
plt.plot(timef, U[0,:], color = "red")
plt.xlabel("t")
plt.ylabel("u(t)")
#plt.scatter(spikesnagu,np.ones(np.size(spikesnagu)), label = "spikes")
plt.title("FitzHugh-Nagumo model with sd = " + str(sd))
plt.legend()
plt.show()

#phase plane plot

plt.figure()
plt.clf()
plt.plot(U[0,:],U[1,:])
plt.xlabel("u")
plt.ylabel("w")
plt.title("Phase plane of noisy FitzHugh-Nagumo")
plt.show()


#--------------------------------------------------------------------------------------------
##firing rate

rate_window = 0.5
#translation of rate_window in terms of dt iterations
l = int(rate_window/dti)


def rate(rate_window,T,t0,N,spikes):
    rate = np.zeros(N-l)
    dt = T/N

    if rate_window != T:
        for m in range(N-l):
            rate[m] = np.count_nonzero((spikes > t0 + m*dt)*(spikes < t0 + (m+l)*dt))/rate_window
        tspikes = np.linspace(t0,t0 + (N-l)*dt, N-l)
        fig3 = plt.figure(3)
        plt.clf()
        plt.plot(tspikes, rate)
        plt.xlabel("time in ms")
        plt.ylabel("firing rate in Hz")
        plt.title("firing rate")
        plt.show()
        return
    else:
        return np.size(spikes)/rate_window

















































