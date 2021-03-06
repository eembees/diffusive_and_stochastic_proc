# coding=utf-8
# Perform the computer simulation of 1-dimensional random walk on a lattice, with ∆x = 1 and
# ∆t = 1. confirm that the mean and variance obeys what we have seen.
import numpy as np
from matplotlib import pyplot as plt

n_steps = 2048
n_particles = 1024
dx = 1
dt = 1

diff_coefficient = dt ** 2 / (2 * dt)

available_dx = np.array([1., -1.])

## setup position matrix
positions = np.zeros(shape=(n_steps, n_particles))

steps = np.arange(1, n_steps)

for i in steps:
    updates = np.random.choice(available_dx, size=n_particles)
    positions[i] = positions[i - 1] + updates

# find mean of all positions


# for i in range(0,n_steps, 100):
#     print(f"at step {i}: mean pos : {np.mean(positions[i])}")
#     print(f"at step {i}: var pos : {np.var(positions[i])}")


# plotting arrays here
pos_means = np.mean(positions, axis=1)
pos_vars = np.var(positions, axis=1)
anal_vars = steps * 2 * diff_coefficient

fig, ax = plt.subplots(nrows=2)

ax[0].plot(pos_means, label="Means")
ax[0].axhline(0, 0, steps[-1], label='0')

ax[1].plot(pos_vars, label="Variance")
ax[1].plot(anal_vars, label="2Dt (analytical)")

for a in ax:
    a.legend()

fig.suptitle("1D Brownian Motion")

plt.show()
