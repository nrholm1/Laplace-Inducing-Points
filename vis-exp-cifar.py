#%%
import numpy as np
#%%
#! M = 50 # alpha = 10

# NLL
# ACC
# BRIER
# ECE

alpha10 = np.array([
    [1.38857,1.37924,1.38523],
    [55.809 , 56.430, 56.470],
    [0.66119,0.65939,0.66117],
    [0.23126,0.23613,0.23618]
])

alpha10.mean(axis=1), alpha10.std(axis=1)

#%%
#! M = 50 # alpha = 50

# NLL
# ACC
# BRIER
# ECE

alpha50 = np.array([
    [1.43654,1.43728,1.43698],
    [ 48.327, 47.917, 48.067],
    [0.68757,0.68709,0.68687],
    [0.09609,0.08898,0.09200]
])

alpha50.mean(axis=1), alpha50.std(axis=1)

#%%
#! M = 50 # alpha = 100

# NLL
# ACC
# BRIER
# ECE

alpha100 = np.array([
    [],
    [],
    [],
    []
])

alpha100.mean(axis=1), alpha100.std(axis=1)
#%%
#! M = 50 # alpha = 1

# NLL
# ACC
# BRIER
# ECE

alpha1 = np.array([
    [2.48886, 2.70564, 2.84322],
    [ 36.458,  36.689,  35.397],
    [0.76851, 0.77069, 0.77455],
    [0.05742, 0.05352, 0.04956]
])
#%%
#! M = 50 # alpha = 0.1

# NLL
# ACC
# BRIER
# ECE

alpha0_1 = np.array([
    [4.28611, 4.07341, 4.48362],
    [ 31.060,  31.661,  30.278],
    [0.81487, 0.81092, 0.81561],
    [0.11516, 0.10392, 0.12511]
])

alpha0_1.mean(axis=1), alpha0_1.std(axis=1)

#%%
#! M = 50 # alpha = 15 # 240 epochs

# NLL
# ACC
# BRIER
# ECE

alpha15 = np.array([
    [1.32160, 1.31490, 1.31551],
    [ 63.311,  63.702,  63.431],
    [0.63594, 0.63388, 0.63383],
    [0.30869, 0.31267, 0.30840]
])

alpha15.mean(axis=1), alpha15.std(axis=1)