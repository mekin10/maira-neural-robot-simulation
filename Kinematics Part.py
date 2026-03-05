#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, norm

class Link:
    def __init__(self, d, a, alpha):
        self.d = d
        self.a = a
        self.alpha = alpha

    def A(self, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        return np.array([
            [ct, -st * ca, st * sa, self.a * ct],
            [st, ct * ca, -ct * sa, self.a * st],
            [0, sa, ca, self.d],
            [0, 0, 0, 1]
        ])

class SerialLink:
    def __init__(self, links, name):
        self.links = links
        self.name = name
        self.n = len(links)

    def fkine(self, q):
        T = np.eye(4)
        for i in range(self.n):
            theta = q[i]
            A = self.links[i].A(theta)
            T = T @ A
        return T

    def jacob0(self, q):
        n = self.n
        J = np.zeros((6, n))
        T = np.eye(4)
        o = np.zeros(3)
        for j in range(n):
            A = self.links[j].A(q[j])
            T_new = T @ A
            o_new = T_new[:3, 3]
            z = T[:3, 2]
            J[:3, j] = np.cross(z, o_new - o)
            J[3:, j] = z
            T = T_new
            o = o_new
        return J

    def jacobe(self, q):
        J0 = self.jacob0(q)
        T = self.fkine(q)
        R = T[:3, :3]
        Z = np.zeros((3, 3))
        Rot = np.block([[R, Z], [Z, R]])
        Je = Rot.T @ J0
        return Je

    def plot(self, q, ax=None, title=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            show = True
        else:
            show = False
        T = np.eye(4)
        pos = [np.array([0, 0, 0])]
        for i in range(self.n):
            A = self.links[i].A(q[i])
            T = T @ A
            pos.append(T[:3, 3])
        pos = np.array(pos)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'bo-')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if title:
            ax.set_title(title)
        if show:
            plt.show()

# DH Parameters for MAiRA Neura Robot (7-DOF, in meters)
L1 = Link(0.429, 0, -np.pi / 2)
L2 = Link(0, 0, np.pi / 2)
L3 = Link(0.406, 0.106, np.pi / 2)
L4 = Link(0, -0.106, -np.pi / 2)
L5 = Link(0.494, 0, np.pi / 2)
L6 = Link(0, 0.113, np.pi / 2)
L7 = Link(0.138, 0, 0)

# Create the SerialLink robot object (7-DOF)
maira = SerialLink([L1, L2, L3, L4, L5, L6, L7], 'MAiRA Neura')

# Display the robot model for verification (basic print)
print('Robot Model: MAiRA Neura (7-DOF)')

# Define a sample home configuration (all zeros)
q_home = np.zeros(7)

# Forward Kinematics at home position
T_home = maira.fkine(q_home)
print('-----------------------------------------------')
print('Home Pose T_home (4x4 Homogeneous Matrix):')
print(T_home)
print('-----------------------------------------------')

# Plot the robot at home position
maira.plot(q_home, title='MAiRA Neura at Home Configuration')

# Inverse Kinematics (Numerical method for redundant 7-DOF)
# --- 1. Define Target Pose (T_target) ---
q_target_deg = np.array([0, 30, 0, 60, 0, 0, 45])  # Example reachable configuration (degrees)
q_target_rad = q_target_deg * (np.pi / 180)
T_target = maira.fkine(q_target_rad)
print('-----------------------------------------------')
print('Target Pose T_target (4x4 Homogeneous Matrix):')
print(T_target)
print('-----------------------------------------------')

# --- 2. Numerical IK Solver (Damped Pseudoinverse with optional joint centering) ---
q_guess = np.zeros(7)  # Initial guess
q = q_guess.copy()
tol = 1e-8  # Convergence tolerance
max_iter = 1000
lambda_ = 0.01  # Damping factor for stability near singularities
alpha = 0.5  # Step size
k_n = 0.05  # Gain for secondary task (joint centering)
q_desired = np.zeros(7)  # Desired joint positions for centering (home)

for iter in range(1, max_iter + 1):
    # Current pose
    T_current = maira.fkine(q)
    p_current = T_current[:3, 3]
    R_current = T_current[:3, :3]

    # Target
    p_target = T_target[:3, 3]
    R_target = T_target[:3, :3]

    # Position error
    Delta_p = p_target - p_current

    # Orientation error (approximate axis-angle vector for small errors)
    err_R = R_target.T @ R_current
    rx = err_R[:, 0]
    ry = err_R[:, 1]
    rz = err_R[:, 2]
    Delta_omega = 0.5 * (np.cross(rx, ry) + np.cross(ry, rz) + np.cross(rz, rx))

    # Total task error
    e = np.concatenate([Delta_p, Delta_omega])

    if norm(e) < tol:
        print(f'Converged in {iter} iterations')
        break

    # Jacobian in base frame (6x7)
    J = maira.jacob0(q)

    # Damped pseudoinverse
    Jt = J.T
    JJt = J @ Jt
    Jp = Jt @ inv(JJt + lambda_**2 * np.eye(6))

    # Primary task
    Delta_q_primary = Jp @ e  # 7x1

    # Secondary task: joint centering (null-space projection)
    q_error = q - q_desired
    Delta_q_secondary = -k_n * q_error
    null_proj = np.eye(7) - Jp @ J
    Delta_q_secondary = null_proj @ Delta_q_secondary

    # Total joint increment
    Delta_q = Delta_q_primary + Delta_q_secondary

    # Update
    q = q + alpha * Delta_q

# --- 3. Display Results ---
q_solution_deg = q * (180 / np.pi)
print('Inverse Kinematics Solution q (Joint Angles in degrees):')
print('[J1, J2, J3, J4, J5, J6, J7]')
print(' '.join(f'{val:.4f}' for val in q_solution_deg))

# Verification
T_verification = maira.fkine(q)
print('-----------------------------------------------')
print('FK Verification (Pose achieved by IK solution):')
print(T_verification)

# Pose error
error = norm(T_target - T_verification, 'fro')
print(f'\nPose Error Magnitude (Frobenius norm, should be near zero): {error:.6e}')

# Plot the solution
maira.plot(q, title='MAiRA Neura at Inverse Kinematics Solution')

# Jacobian
# --- 1. Define a Test Configuration ---
q_test_deg = np.array([0, 30, 0, 60, 0, 0, 45])
q_test_rad = q_test_deg * (np.pi / 180)

# --- 2. Calculate Jacobians ---
# Jacobian in base frame (6x7 for 7-DOF robot)
J_BaseFrame = maira.jacob0(q_test_rad)
print('----------------------------------------------------')
print('J_BaseFrame (J_0): 6x7 matrix (End-effector twist in BASE frame)')
print('Rows 1-3: Linear velocity, Rows 4-6: Angular velocity')
print(J_BaseFrame)

# Jacobian in end-effector frame (6x7)
J_EndEffectorFrame = maira.jacobe(q_test_rad)
print('----------------------------------------------------')
print('J_EndEffectorFrame (J_E): 6x7 matrix (End-effector twist in EE frame)')
print(J_EndEffectorFrame)

# Plot at test configuration
maira.plot(q_test_rad, title='MAiRA Neura at Jacobian Test Configuration')

# MAiRA Neura Kinematic Trajectory Generation (Joint Space)
# --- 1. Define Start and End Configurations ---
q_start_rad = np.zeros(7)  # Home position
q_end_deg = np.array([30, 20, 40, -30, 10, 20, 45])  # Example end configuration
q_end_rad = q_end_deg * (np.pi / 180)

# --- 2. Trajectory Parameters ---
T_f = 5  # Duration (seconds)
steps = 100  # Number of points
t = np.linspace(0, T_f, steps)  # Time vector

# --- 3. Generate Joint Space Trajectory (quintic polynomial) ---
def jtraj(q0, qf, t):
    N = len(t)
    n = len(q0)
    Q = np.zeros((N, n))
    QD = np.zeros((N, n))
    QDD = np.zeros((N, n))
    tf = t[-1]
    for j in range(n):
        q0j = q0[j]
        qfj = qf[j]
        delta = qfj - q0j
        if tf == 0:
            Q[:, j] = q0j
            continue
        M = np.array([
            [tf**3, tf**4, tf**5],
            [3 * tf**2, 4 * tf**3, 5 * tf**4],
            [6 * tf, 12 * tf**2, 20 * tf**3]
        ])
        a345 = np.linalg.solve(M, np.array([delta, 0, 0]))
        a3, a4, a5 = a345
        tt = t
        Q[:, j] = q0j + a3 * tt**3 + a4 * tt**4 + a5 * tt**5
        QD[:, j] = 3 * a3 * tt**2 + 4 * a4 * tt**3 + 5 * a5 * tt**4
        QDD[:, j] = 6 * a3 * tt + 12 * a4 * tt**2 + 20 * a5 * tt**3
    return Q, QD, QDD

Q_rad, QD_rad, QDD_rad = jtraj(q_start_rad, q_end_rad, t)

# Convert position to degrees for display
Q_deg = Q_rad * (180 / np.pi)

# --- 4. Visualization ---
# Animate the trajectory (in Jupyter, use %matplotlib notebook for interactive)
# Note: If not interactive, this will create multiple figures or use pause for animation
get_ipython().run_line_magic('matplotlib', 'notebook  # Uncomment this in Jupyter for interactive plots')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('MAiRA Neura Joint-Space Trajectory')

for i in range(steps):
    ax.cla()
    maira.plot(Q_rad[i], ax=ax)
    fig.canvas.draw()
    plt.pause(0.01)

# Plot kinematic profiles for Joint 1 (example)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, Q_deg[:, 0])
plt.ylabel('Position (deg)')
plt.title('Joint 1 Profile')

plt.subplot(3, 1, 2)
plt.plot(t, QD_rad[:, 0] * 180 / np.pi)
plt.ylabel('Velocity (deg/s)')

plt.subplot(3, 1, 3)
plt.plot(t, QDD_rad[:, 0] * 180 / np.pi)
plt.ylabel('Acceleration (deg/s^2)')
plt.xlabel('Time (s)')
plt.show()


# In[ ]:





# In[2]:


# Note: This extends the previous kinematics code with a simple controller part.
# We'll implement a basic joint-space PD controller for trajectory tracking in simulation.
# Assumption: Robot joints are modeled as double integrators (simple dynamics: torque = I * qdd, assume I=1 for simplicity).
# Control law: tau = Kp * (q_des - q) + Kd * (qd_des - qd) + qdd_des (feedforward, but since I=1, tau = qdd_des + ...)
# Then integrate to get positions.

# Add these imports if not already there
import time  # For animation delay

# ... (Previous code for SerialLink, fkine, etc., goes here)

# Controller Section: Simple Simulated PD Controller for Joint Trajectory Tracking

# --- 1. Define Control Parameters ---
Kp = np.diag([100.0] * 7)  # Proportional gain (7x7 diagonal)
Kd = np.diag([20.0] * 7)   # Derivative gain (7x7 diagonal)
dt = 0.01  # Simulation time step (seconds)
sim_duration = T_f  # Same as trajectory duration
sim_steps = int(sim_duration / dt)

# --- 2. Initialize Simulation States ---
q_sim = q_start_rad.copy()  # Initial position
qd_sim = np.zeros(7)       # Initial velocity

# Precompute desired trajectory at higher resolution for simulation
t_sim = np.linspace(0, T_f, sim_steps)
Q_des, QD_des, QDD_des = jtraj(q_start_rad, q_end_rad, t_sim)

# Storage for simulation results
Q_sim = np.zeros((sim_steps, 7))
QD_sim = np.zeros((sim_steps, 7))
Tau_sim = np.zeros((sim_steps, 7))  # Torques

# --- 3. Simulation Loop (PD + Feedforward Control) ---
for i in range(sim_steps):
    # Desired values
    q_des = Q_des[i]
    qd_des = QD_des[i]
    qdd_des = QDD_des[i]
    
    # Errors
    e = q_des - q_sim
    ed = qd_des - qd_sim
    
    # Control torque (assuming unit inertia: tau = qdd_des + Kp*e + Kd*ed)
    tau = qdd_des + np.dot(Kp, e) + np.dot(Kd, ed)
    
    # Dynamics (double integrator: qdd = tau, since I=1)
    qdd_sim = tau  # Simplified, no coriolis/gravity
    
    # Integrate
    qd_sim += qdd_sim * dt
    q_sim += qd_sim * dt
    
    # Store
    Q_sim[i] = q_sim
    QD_sim[i] = qd_sim
    Tau_sim[i] = tau

# --- 4. Visualization of Simulation ---
# Plot simulated vs desired positions for Joint 1 (example)
plt.figure()
plt.plot(t_sim, Q_des[:, 0] * 180 / np.pi, label='Desired')
plt.plot(t_sim, Q_sim[:, 0] * 180 / np.pi, label='Simulated')
plt.ylabel('Position (deg)')
plt.title('Joint 1 Position Tracking')
plt.legend()
plt.xlabel('Time (s)')

# Plot torques for Joint 1
plt.figure()
plt.plot(t_sim, Tau_sim[:, 0])
plt.ylabel('Torque')
plt.title('Joint 1 Control Torque')
plt.xlabel('Time (s)')

# Animate the simulated trajectory
get_ipython().run_line_magic('matplotlib', 'notebook')
fig_sim = plt.figure()
ax_sim = fig_sim.add_subplot(111, projection='3d')
ax_sim.set_title('MAiRA Neura Simulated Controlled Trajectory')

for i in range(0, sim_steps, 10):  # Step by 10 for faster animation
    ax_sim.cla()
    maira.plot(Q_sim[i], ax=ax_sim)
    fig_sim.canvas.draw()
    plt.pause(0.01)

plt.show()


# In[ ]:




