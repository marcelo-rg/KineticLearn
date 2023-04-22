import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps


def lorenz(xyz, *, s=10, r=28, b=2.667):

    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.01
num_steps = 10000
t = np.arange(0,(num_steps+1)*dt,dt).T

xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (0., 1., 1.05)  # Set initial values
# xyzs[0] = (1., 0.05, 0.05)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\NeuralODEs\\lorenz_attractor.png")

# print(np.shape(xyzs), np.shape(t))
# Use SINDy sparse regression
model = ps.SINDy(
    differentiation_method= ps.FiniteDifference(order=2),
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer= ps.STLSQ(threshold=0.2),
    feature_names=["x", "y", "z"],
)

print("\n")
model.fit(xyzs, t)
model.print()
print("\n")