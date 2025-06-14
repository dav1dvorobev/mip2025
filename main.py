import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

dt = 1/240
max_time = 100.0
log_t = np.arange(0.0, max_time, dt)
g = 10.0
L1 = L2 = 0.8
m = 1.0
xd, zd = 0.5, 1.0
T_traj = 2.0 
th0 = 0.5 

def main():
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    box_id = p.loadURDF("two-link.urdf.xml", useFixedBase=True)
    p.changeDynamics(box_id, 1, linearDamping=0, angularDamping=0)
    p.changeDynamics(box_id, 2, linearDamping=0, angularDamping=0)
    p.setJointMotorControl2(bodyIndex=box_id, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
    for _ in range(1000):
        p.stepSimulation()
    p.setJointMotorControl2(box_id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    pos0 = p.getLinkState(box_id, 4)[0]
    X0 = np.array([pos0[0], pos0[2]])
    log_x = np.zeros_like(log_t)
    log_z = np.zeros_like(log_t)
    for i, t in enumerate(log_t):
        th1, vel1 = p.getJointState(box_id, 1)[0:2]
        th2, vel2 = p.getJointState(box_id, 3)[0:2]
        pos = p.getLinkState(box_id, 4)[0]
        log_x[i], log_z[i] = pos[0], pos[2]
        # Calculation of the Jacobian via PyBullet
        joint_positions = [th1, th2]
        joint_velocities = [vel1, vel2]
        joint_accs = [0.0, 0.0]
        jac_t, _ = p.calculateJacobian(
            bodyUniqueId=box_id,
            linkIndex=4,
            localPosition=[0, 0, 0],
            objPositions=joint_positions,
            objVelocities=joint_velocities,
            objAccelerations=joint_accs
        )
        J_lin = np.array(jac_t)[[0, 2], :]
        J_inv = np.linalg.pinv(J_lin)
        X = np.array([pos[0], pos[2]])
        Xd = np.array([xd, zd])
        s = 1.0
        if t < T_traj:
            s =  (3.0 / T_traj**2) * t**2 - (2.0 / T_traj**3) * t**3
        Xd_curr = X0 + s * (Xd - X0)
        K_space = 100.0
        dX = Xd_curr - X
        dth = K_space * J_lin @ dX
        p.setJointMotorControlArray(
            bodyIndex=box_id,
            jointIndices=[1, 3],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=dth.flatten(),
            forces=[100.0, 100.0]
        )
        p.stepSimulation()
        time.sleep(dt)
    p.disconnect()

if __name__ == "__main__":
    main()