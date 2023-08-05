import numpy as np
import matplotlib.pyplot as plt
##common variables
#initial starting time, total time of the simulation, size of the grid
t0 = 0
N = 10000


#current flowing through the neuron
def I(t):
    return 1.3

##Integrate and fire model

Ti = 0.25
#size of time incrementation
dti = Ti/N

#constants and parameters
R = 1
C = 0.01
tm = R*C
#threshold
thr = 1.0
#tension t=0
u0 = 0.0
#tension reset
ur = 0.0
#absolute refractory period
dabs = 0.004



#right hand side of the eq
def F(t,u,I):
    return -1/tm*u + 1/C*I(t)







def Eulerintfire():
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
                u[k] = ur

            spikes = np.append(spikes,tk)
            k = k+l

        else :
            u[k] = u[k-1] + dti*F(tk + dti/2,u[k-1] + dti/2*F(tk,u[k-1],I),I)
            k = k+1

    return u, spikes



#tension plot

u, spikes = Eulerintfire()
timei = np.linspace(0,Ti,N)

fig = plt.figure(1)
plt.clf()
plt.plot(timei, u)
#plt.scatter(spikes, thr*np.ones(len(spikes)), color = "purple", label = "spikes") #useless but pretty
plt.title("Integrate and fire model")
plt.ylabel("u(t)")
plt.xlabel("t")
plt.legend()
plt.show()

##FitzHugh-Nagumo model
Tf = 100           #ms
#size of time incrementation
dtf = Tf/N
eps = 0.1
b0 = 2
b1 = 1.5

#U = (u,w)
def G(t,U,I):
    return np.array([U[0]-1/3*U[0]**3-U[1]+I(t),eps*(b0 + b1*U[0] - U[1])])

def RK2fitznag(u0,w0):
    spikes = np.empty(0)
    U = np.zeros((2,N))
    U[:,0] = np.array([u0,w0])

    for k in range(1,N):
        tk = t0 + k*dtf

        U[:,k] = U[:,k-1] + dtf*G(tk + dtf/2, U[:,k-1] + dtf/2*G(tk,U[:,k-1],I),I)

    for k in range(1,N-1):
        if U[0,k] == max([U[0,k-1],U[0,k],U[0,k+1]]):
            spikes = np.append(spikes,t0 + k*dtf)

    return U, spikes

U, spikesnagu = RK2fitznag(-3,-1)


fig2 = plt.figure(2)
plt.clf()

#plot of the curve
#NB : un cycle veut dire qu'on a une période régulière de tension, i.e. le neurone tire de façon régulière
plt.plot(U[0,:],U[1,:], label = "trajectory of (w(t),u(t))")
plt.xlabel("u(t)")
plt.xlim([-3,3])
plt.ylabel("w(t)")
plt.ylim([-3,5])
plt.title("FitzHugh-Nagumo")

x = np.linspace(-3,3,N)
plt.plot(x,x - 1/3*x**3 + I(0), label = "u nullcline")
plt.plot(x,b0 + b1*x, label = "w nullcline")

plt.legend()
plt.show()


#plot of the tension
timef = np.linspace(0,Tf,N)

fig4 = plt.figure(4)
plt.clf()
plt.plot(timef, U[0,:], color = 'red')
plt.title("tension with FitzHugh-Nagumo model")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.show()




##firing rate function and plot


rate_window = 0.25
#translation of rate_window in terms of dt iterations
l = int(rate_window/dti)


def rate(rate_window,T,t0,N,spikes):
    rate = np.zeros(N-l)
    dt = T/N

    for m in range(N-l):
        rate[m] = np.count_nonzero((spikes > t0 + m*dt)*(spikes < t0 + (m+l)*dt))/rate_window

    tspikes = np.linspace(t0,t0 + (N-l)*dt, N-l)

    return tspikes, rate


u, spikesintf = RK2intfire()


fig3 = plt.figure(3)
plt.clf()
plt.plot(rate(rate_window,Ti,t0,N,spikesintf)[0], rate(rate_window,Ti,t0,N,spikesintf)[1])
plt.xlabel("time in ms")
plt.ylabel("firing rate in Hz")
plt.title("firing rate")
plt.show()




























