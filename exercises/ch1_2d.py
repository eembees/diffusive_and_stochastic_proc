# coding=utf-8
# Perform the computer simulation of 1-dimensional random walk on a lattice, with ∆x = 1 and
# ∆t = 1. confirm that the mean and variance obeys what we have seen.
import numpy as np
from matplotlib import pyplot as plt

n_steps = 2048
n_particles = 1024
dx = 1
dt = 1
d = 2


diff_coefficient = dx ** 2 / (2 * dt)

available_dx = np.array([
    [0,dx],[dx,0],
    [0, -dx], [-dx, 0]
])

## setup position matrix
positions = np.zeros(shape=(n_steps, n_particles, 2))

steps = np.arange(1, n_steps)

for i in steps:
    updates = available_dx[np.random.randint(available_dx.shape[0], size = n_particles)]
    # updates = np.random.choice(available_dx, size=n_particles)
    # print(updates.shape)
    # exit()
    positions[i] = positions[i - 1] + updates

# find mean of all positions


# for i in range(0,n_steps, 100):
#     print(f"at step {i}: mean pos : {np.mean(positions[i])}")
#     print(f"at step {i}: var pos : {np.var(positions[i])}")


# plotting arrays here
pos_means = np.mean(positions, axis=1)
pos_vars = np.var(positions, axis=1)
anal_vars = steps * 2 * diff_coefficient * d

fig, ax = plt.subplots(nrows=2)

ax[0].plot(pos_means, label=[f"Means, {i}" for i in range(d)])
ax[0].axhline(0, 0, steps[-1], label='0')

ax[1].plot(pos_vars, label=[f"Variance, {i}" for i in range(d)])
ax[1].plot(anal_vars, label="2Dt (analytical)")

for a in ax:
    a.legend()

fig.suptitle(f"2D Brownian Motion, N: {n_particles}")

plt.show()
