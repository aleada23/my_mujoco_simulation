#!/usr/bin/env python3
"""
Spot: position-hold controller using PD + gravity compensation for leg motors.

Assumptions:
- You use the 'mujoco' python package (modern bindings).
- Simulation class returns (model, data) as used in your previous code.
- Leg actuators are named exactly as in your XML:
  fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn
- The first 6 qpos/qvel entries belong to floating base.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from my_mujoco_simulation.simulation.simulation import Simulation

# ---------------------------
# Paths (change if needed)
# ---------------------------
ROBOT_PATH = "../robot/models/managerie_spot/robot.xml"
MODEL_PATH = "../environment/models/scene/spot_scene.xml"

# ---------------------------
# Controller gains / limits
# ---------------------------
Kp = 480.0         # proportional gain (Nm/rad)
Kd = 20.5          # derivative gain (Nm/(rad/s))
TAU_MAX = 120.0   # saturation for actuator torque (Nm)

# Leg actuator / joint names (from your XML)
leg_actuator_names = [
    "robot0_fl_hx", "robot0_fl_hy", "robot0_fl_kn",
    "robot0_fr_hx", "robot0_fr_hy", "robot0_fr_kn",
    "robot0_hl_hx", "robot0_hl_hy", "robot0_hl_kn",
    "robot0_hr_hx", "robot0_hr_hy", "robot0_hr_kn",
]

# ---------------------------
# Setup simulation
# ---------------------------
sim = Simulation(env_path=MODEL_PATH)
sim.add_robot(ROBOT_PATH)
model, data = sim.launch(pretty_xml=False)
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "robot0_home")
qpos_home = [0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, -3.14, 3.06, 0, 0, 0, 0]

# Small sanity checks
nv = model.nv    # dimension of velocity space (dofs)
nq = model.nq    # dimension of qpos (including floating base quaternion if present)
nact = model.nu  # number of actuators (ctrl dimension)

print(f"Model nv={nv}, nq={nq}, n_actuators={nact}")

# Map actuators -> joints -> qpos/qvel indices
actuator_ids = []
joint_ids = []
qpos_indices = []   # index into data.qpos for each joint (start)
dof_indices = []    # index into data.qvel / data.qfrc_bias for each joint (start)

for act_name in leg_actuator_names:
    # actuator exists?
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    if aid == -1:
        raise RuntimeError(f"Actuator '{act_name}' not found in model.")
    actuator_ids.append(aid)

    # actuator has a 'joint' property with same name in your XML; find that joint
    # in your XML you named actuators equal to joint names, so joint name = act_name
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, act_name)
    if jid == -1:
        raise RuntimeError(f"Joint '{act_name}' for actuator '{act_name}' not found in model.")
    joint_ids.append(jid)

    # qpos address (start index in qpos)
    try:
        qpos_addr = model.jnt_qposadr[jid]
    except Exception as e:
        raise RuntimeError(f"Could not read qpos address for joint '{act_name}': {e}")
    qpos_indices.append(qpos_addr)

    # dof address (start index in velocity/dof space)
    try:
        dof_addr = model.jnt_dofadr[jid]
    except Exception as e:
        # fallback: compute dof index by searching model.jnt_qposadr -> approximate
        raise RuntimeError(f"Could not read dof address for joint '{act_name}': {e}")
    dof_indices.append(dof_addr)

# Save desired (home) joint positions from current sim_data.qpos (we keep initial pose)
# Note: qpos array contains base (floating) + joint qpos. We extract per-joint qpos
q_des = qpos_home.copy() + 0.01*np.ones(len(qpos_home))
dq_des = np.zeros(len(joint_ids))



print("Desired joint positions (home):", q_des)

# ---------------------------
# Control loop (viewer)
# ---------------------------
try:
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        t0 = data.time
        last_time = t0
        while viewer.is_running():
            # Simple fixed-timestep PD + gravity compensation per-leg joint
            # Read up-to-date bias forces (contains gravity + Coriolis + centrifugal) in dof space
            qfrc_bias = np.array(data.qfrc_bias)  # length nv (dof space)
            # Current qpos and qvel arrays
            qpos = np.array(data.qpos)   # length nq
            qvel = np.array(data.qvel)   # length nv

            # Build a ctrl vector (length model.nu)
            ctrl = np.zeros(model.nu)

            # For each leg joint compute torque
            for i, act_id in enumerate(actuator_ids):
                jid = joint_ids[i]
                qidx = qpos_indices[i]    # index into qpos
                didx = dof_indices[i]     # index into qvel / qfrc_bias

                # current joint states
                q_curr = qpos[qidx]
                dq_curr = qvel[didx]

                # PD error (we keep q_des fixed)
                e = q_des[i] - q_curr
                de = dq_des[i] - dq_curr

                tau_pd = Kp * e + Kd * de

                # gravity + bias compensation (from data.qfrc_bias)
                # qfrc_bias is the generalized bias force (shape nv). For the joint dof we take that entry.
                tau_bias = qfrc_bias[didx]

                # desired torque (joint-space)
                tau_des = tau_bias + tau_pd

                # saturate
                if tau_des > TAU_MAX:
                    tau_des = TAU_MAX
                elif tau_des < -TAU_MAX:
                    tau_des = -TAU_MAX

                # assign to corresponding actuator index (ctrl is actuator-level)
                # The mapping actuator id -> ctrl slot is simply the actuator id (0..nu-1)
                ctrl[act_id] = tau_des

            # Write control to data and step
            data.ctrl[:12] = ctrl[:12]
            data.ctrl[12:] = [0, -3.14, 3.06, 0, 0, 0, 0]
            mujoco.mj_step(model, data)

            # Sync viewer and sleep minimal (viewer.sync handles real-time)
            viewer.sync()

except KeyboardInterrupt:
    print("Interrupted by user, shutting down.")
