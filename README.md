# Internship_report
The python codes I did during my internship at iCube

# Internship_models.py contains :
-the implementation of the integrate and fire model
-the implementation of the FitzHugh-Nagumo model 

# Internship_noisy_models.py contains :
-the implementation of the integrate and fire model but with a random input current
-the implementation of the FitzHugh-Nagumo model but with a random input current

# Internship_patch_model.py contains :
-the simulation of a patch of 1000 neurons, which then calculates the average firing rate of the patch depending on the input current
-a sigmoid fit of the average firing rate depending on the input current, calculated with swarm optimization
-The comparison of the evolution of the average postsynaptic potential in the patch, modeled by 3 different equations
-a Raster plot of the neuron activity to check that it is actually asynchronous
-a plot of random trajectories given by one of the previous differential equation to assess analytic properties of the numerical solution
