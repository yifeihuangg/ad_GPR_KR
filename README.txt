Umbrella sampling data.  The raw trajectories (already burned in) are saved in trajs.npy, and the window centers are in centers.npy.  You can load these files in python using

>>> trajs = np.load('trajs.npy')
>>> traj_window_1 = trajs[0] # Trajectory of first window

The force constant of each window is 24.9668912741 in each dimension.  The Boltzmann factor (k_B*T) is 0.616033281788. 
The data is periodic, ranging from -pi to pi.
