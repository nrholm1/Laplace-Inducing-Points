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
    [],
    [],
    [],
    []
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