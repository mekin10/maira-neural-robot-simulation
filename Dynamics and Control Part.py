import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from roboticstoolbox import jtraj

# ===============================================================
#  MAiRA Neura Robot – Full Kinematics + Dynamics + Control
# ===============================================================

deg = np.pi / 180  # Degree to radian

# ===============================================================
# 1. LINK PARAMETERS (YOUR PROVIDED DATA)
# ===============================================================

d1 = 0.429
d2 = 0
d3 = 0.406
d4 = 0
d5 = 0.494
d6 = 0

# -------- Link 1 --------
L1 = RevoluteDH(
    d=d1, a=0, alpha=-np.pi/2, offset=0,
    m=1.5, r=[d1/2, 0, -0.03],
    I=[0.0029525, 0.0060091, 0.0058821, 0, 0, 0],
    Jm=13.5e-6, G=1/156, B=1.48e-3,
    qlim=np.array([-360, 360])*deg
)

# -------- Link 2 --------
L2 = RevoluteDH(
    d=d2, a=0, alpha=np.pi/2, offset=0,
    m=1.5, r=[d1/2, 0, 0],
    I=[0.0029525, 0.0060091, 0.0058821, 0, 0, 0],
    Jm=13.5e-6, G=1/156, B=1.48e-3,
    qlim=np.array([-180, 180])*deg
)

# -------- Link 3 --------
L3 = RevoluteDH(
    d=d2, a=0.106, alpha=np.pi/2, offset=0,
    m=1.6, r=[d1/2, 0, 0],
    I=[0.00172767, 0.00041967, 0.0018468, 0, 0, 0],
    Jm=13.5e-6, G=1/100, B=1.48e-3,
    qlim=np.array([-360, 360])*deg
)

# -------- Link 4 --------
L4 = RevoluteDH(
    d=d3, a=-0.106, alpha=-np.pi/2,
    m=1.8, r=[0, d1/2, 0],
    I=[0.0006764, 0.0010573, 0.0006610, 0, 0, 0],
    Jm=9.25e-6, G=1/71, B=1.48e-3,
    qlim=np.array([-180, 180])*deg
)

# -------- Link 5 --------
L5 = RevoluteDH(
    d=d5, a=0, alpha=np.pi/2, offset=0,
    m=1, r=[0, 0, 0],
    I=[0.0001934, 0.0001602, 0.0000689, 0, 0, 0],
    Jm=3.5e-6, G=1/71, B=1.48e-3,
    qlim=np.array([-360, 360])*deg
)

# -------- Link 6 --------
L6 = RevoluteDH(
    d=d6, a=0.113, alpha=np.pi/2,
    m=0.5, r=[0, 0, 0],
    I=[0.0001934, 0.0001602, 0.0000689, 0, 0, 0],
    Jm=3.5e-6, G=1/71, B=1.48e-3,
    qlim=np.array([-90, 90])*deg
)

# -------- Link 7 (End-effector) --------
L7 = RevoluteDH(
    d=0.138, a=0, alpha=0,
    m=0.3, r=[0, 0, 0.05],
    I=[1e-4, 1e-4, 1e-4, 0, 0, 0],
    qlim=np.array([-360, 360])*deg
)

# ===============================================================
# 2. CREATE MAiRA ROBOT
# ===============================================================

maira = DHRobot([L1, L2, L3, L4, L5, L6, L7], name='MAiRA Neura')

print("MAiRA Robot Model with Full Dynamics:")
print(maira)

# Home configuration
maira.plot(np.zeros(7))
plt.title('MAiRA Neura – Home Configuration')
plt.show()

# ===============================================================
# 3. TRAJECTORY GENERATION
# ===============================================================

q_start = np.zeros(7)
q_end = np.array([30, 20, 40, -30, 10, 20, 45]) * deg

Tf = 5
t = np.linspace(0, Tf, 100)

traj = jtraj(q_start, q_end, t)
q = traj.q
qd = traj.qd
qdd = traj.qdd

# ===============================================================
# 4. COMPUTED TORQUE CONTROL (PD + DYNAMICS)
# ===============================================================

Kp = np.diag([120, 120, 120, 100, 80, 60, 40])
Kd = np.diag([25, 25, 25, 20, 15, 10, 8])

q_act = q_start.copy()
qd_act = np.zeros(7)

tau_log = np.zeros((len(t), 7))
dt = t[1] - t[0]

for k in range(len(t)):
    e = q[k, :] - q_act
    ed = qd[k, :] - qd_act

    M = maira.inertia(q_act)
    C = maira.coriolis(q_act, qd_act)
    G = maira.gravload(q_act)

    tau = M @ (qdd[k, :] + Kd @ ed + Kp @ e) + C @ qd_act + G

    qdd_act = np.linalg.solve(M, tau - C @ qd_act - G)

    qd_act = qd_act + qdd_act * dt
    q_act = q_act + qd_act * dt

    tau_log[k, :] = tau

# Plot joint torques
plt.figure()
plt.plot(t, tau_log)
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Computed Torque Control – Joint Torques')
plt.grid(True)
plt.show()

# ===============================================================
# 5. ENERGY & POWER ANALYSIS
# ===============================================================

power = np.sum(np.abs(tau_log * qd), axis=1)
energy = np.cumtrapz(power, t, initial=0)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, power, linewidth=1.5)
plt.ylabel('Power (W)')
plt.title('Instantaneous Power')

plt.subplot(2, 1, 2)
plt.plot(t, energy, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Total Energy Consumption')
plt.show()

# ===============================================================
# 6. FINAL TRAJECTORY ANIMATION
# ===============================================================

maira.plot(q)
plt.title('MAiRA Neura – Controlled Motion')
plt.show()
