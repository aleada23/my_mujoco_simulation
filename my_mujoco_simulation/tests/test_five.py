import mujoco
import mujoco.viewer
import numpy as np
import time
from my_mujoco_simulation.simulation.simulation import Simulation


# --- USER-DEFINED COMMANDS (Planar Velocities) ---
# Desired translational velocity (x, y) and turning rate (psi_dot) 
WANTED_VELOCITY = np.array([0.3, 0.0, 0.0]) # [Vx, Vy, V_turn] m/s and rad/s
NOMINAL_HEIGHT = 0.46 # Robot's desired body height (from keyframe)
GAINS_STANCE_KP = 100.0 # Stance: Proportional gain for Z-force impedance
GAINS_SWING_KP = 50.0   # Swing: Proportional gain for Cartesian PD
GAINS_SWING_KD = 5.0    # Swing: Derivative gain for Cartesian PD

# --- GAIT PARAMETERS ---
CYCLE_TIME = 0.5        # Total step period (seconds)
STANCE_RATIO = 0.6      # Ratio of cycle time spent in stance (0.0 to 1.0)
STEP_HEIGHT = 0.08      # Maximum foot clearance in swing (meters)
INIT_PHASES = [0.0, 0.5, 0.5, 0.0] # Phase offset for [FL, FR, HL, HR] (Trot)

# --- MUJOCO SETUP ---
ROBOT_PATH = "../robot/models/managerie_spot/robot.xml"
MODEL_PATH = "../environment/models/scene/spot_scene.xml"

# Placeholder for your custom simulation class and launch
sim = Simulation(env_path=MODEL_PATH)

sim.add_robot(ROBOT_PATH)

# Launch simulation
sim_model, sim_data = sim.launch(pretty_xml=False)

# --- MuJoCo Name Mapping ---
# Joints are: fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, ...
JOINT_IDS = [sim_model.joint('robot0_fl_hx').id, sim_model.joint('robot0_fl_hy').id, sim_model.joint('robot0_fl_kn').id,
             sim_model.joint('robot0_fr_hx').id, sim_model.joint('robot0_fr_hy').id, sim_model.joint('robot0_fr_kn').id,
             sim_model.joint('robot0_hl_hx').id, sim_model.joint('robot0_hl_hy').id, sim_model.joint('robot0_hl_kn').id,
             sim_model.joint('robot0_hr_hx').id, sim_model.joint('robot0_hr_hy').id, sim_model.joint('robot0_hr_kn').id]
             
FOOT_NAMES = ["robot0_FL", "robot0_FR", "robot0_HL", "robot0_HR"]
HIP_NAMES = ["robot0_fl_hip", "robot0_fr_hip", "robot0_hl_hip", "robot0_hr_hip"]
FOOT_IDS = [sim_model.geom(name).id for name in FOOT_NAMES]
HIP_IDS = [sim_model.body(name).id for name in HIP_NAMES]

# --- CONTROL LOOP ---
try:
    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        while viewer.is_running():
            time_ = sim_data.time
            phase = (time_ % CYCLE_TIME) / CYCLE_TIME
            
            # Reset leg control torque array for 12 DoF (3 per leg)
            legs_control_torque = np.zeros(12) 
            
            # --- State Estimation Placeholder ---
            # In a real implementation (Sec III-H), you would run a Kalman Filter here.
            # For simulation, we can directly read the body state.
            body_pos = sim_data.qpos[:3]
            body_vel = sim_data.qvel[:3]
            body_mat = sim_data.xmat[1:10].reshape(3,3) # Assumes body is the first movable body (id=1)
            
            # 3D Velocity (linear x, y, z)
            Vx, Vy = body_vel[:2]
            
            # --- Gravity Vector (constant) ---
            gravity = sim_model.opt.gravity # (0, 0, -9.81)

            for i in range(4): # Loop through [FL, FR, HL, HR]
                leg_phase = (phase + INIT_PHASES[i]) % 1.0
                
                # Joint index offset for this leg in the full 12-joint control array
                joint_idx_offset = i * 3

                # --- 1. Contact Detection / Gait Scheduling (Sec III-A) ---
                is_stance = leg_phase < STANCE_RATIO
                is_front = (i == 0 or i == 1)
                is_left = (i == 0 or i == 2)
                
                # --- Get Kinematics for Current Leg ---
                hip_id = HIP_IDS[i]
                foot_id = FOOT_IDS[i]

                # Foot position and velocity in world frame
                p_foot_world = sim_data.geom_xpos[foot_id]
                
                # Foot position relative to hip (in hip frame for simpler control/IK)
                # You can get this with MuJoCo's helper functions, but a quick way:
                p_hip_world = sim_data.xpos[hip_id]
                p_current_hip_frame = body_mat.T @ (p_foot_world - p_hip_world)

                # Get Jacobian for the current foot (Linear part only for position control)
                Jp_full = np.zeros((3, sim_model.nv)) 
                Jr_full = np.zeros((3, sim_model.nv)) 
                mujoco.mj_jacGeom(sim_model, sim_data, Jp_full, Jr_full, foot_id)
                # J_i is the 3x3 Jacobian corresponding to the leg's 3 joints (hx, hy, kn)
                J_i = Jp_full[:, JOINT_IDS[joint_idx_offset:joint_idx_offset+3]]


                if is_stance:
                    # --- 2. STANCE PHASE: Force Control Placeholder (Sec III-C) ---
                    
                    # *************************************************************
                    # This block requires a QP Solver for true MIT Cheetah 3 control.
                    # This is a SIMPLIFIED IMPEDANCE/SPRING-DAMPER placeholder.
                    # *************************************************************

                    # Simple vertical force control (Kp on height error)
                    # The goal is to produce a force F_z proportional to body weight and height error
                    F_z_target = sim_model.body_mass[1] * -gravity[2] / (STANCE_RATIO * 4) # Nominal static vertical load
                    F_z_error = (NOMINAL_HEIGHT - body_pos[2]) * GAINS_STANCE_KP

                    F_target_world = np.array([0, 0, F_z_target + F_z_error])
                    
                    # Convert desired force F_target_world to joint torques
                    # F_target_world = J_i @ tau_i * (J_i is for position, so we need full matrix for force)
                    # For simplicity, we assume an inverse relationship for torque: tau = J^T * F_world
                    tau_i_stance = J_i.T @ F_target_world
                    legs_control_torque[joint_idx_offset:joint_idx_offset+3] = tau_i_stance

                else:
                    # --- 3. SWING PHASE: Footstep Planning and Cartesian PD (Sec III-E) ---
                    
                    # A. Footstep Location Planning (Raibert Heuristic + Capture Point (Sec III-E, Eq. 6))
                    
                    # Compute Raibert Heuristic (T_c_phi * V_desired)
                    # T_c_phi is the stance duration for a full trot (CYCLE_TIME * STANCE_RATIO)
                    T_c_phi = CYCLE_TIME * STANCE_RATIO
                    V_desired = WANTED_VELOCITY[:2] # [Vx_d, Vy_d]
                    p_raibert = 0.5 * T_c_phi * V_desired
                    
                    # Compute Capture Point Feedback (sqrt(z0/||g||) * (V_current - V_desired))
                    # V_current is approximated by body_vel[:2]
                    g_scalar = -gravity[2] # 9.81
                    p_capture_point = np.sqrt(NOMINAL_HEIGHT / g_scalar) * (body_vel[:2] - V_desired)

                    # Nominal hip position offset (Approximation based on body structure)
                    p_h_i = sim_data.xpos[hip_id] - body_pos
                    
                    # Footstep target relative to CoM (2D horizontal plane)
                    p_step_CoM_2D = p_raibert + p_capture_point
                    
                    # Convert to target relative to hip in body frame
                    # Nominal hip x-offset (front/rear) and y-offset (side)
                    nominal_hip_x = 0.29785 * (1 if is_front else -1)
                    nominal_hip_y = 0.1108 * (1 if is_left else -1)
                    
                    # Target foot position relative to the hip in the HIP FRAME
                    # Assumes a flat ground plane at z=0 (Sec III-E)
                    p_target_hip_frame = np.array([
                        p_step_CoM_2D[0] + nominal_hip_x, 
                        p_step_CoM_2D[1] + nominal_hip_y,
                        -NOMINAL_HEIGHT + p_h_i[2] # Nominal Z-height
                    ])
                    
                    # B. Trajectory Interpolation (S-curve or simple parabola for swing clearance)
                    s = (leg_phase - STANCE_RATIO) / (1.0 - STANCE_RATIO) # Swing progress [0, 1]
                    z_clearance = STEP_HEIGHT * 4 * s * (1.0 - s) 

                    # X and Y follow a simple linear path from liftoff to touchdown
                    p_liftoff_hip_frame = p_current_hip_frame # Use current position at liftoff
                    p_touchdown_hip_frame = p_target_hip_frame

                    # Swing reference position in HIP FRAME
                    p_ref_swing = (1-s) * p_liftoff_hip_frame + s * p_touchdown_hip_frame
                    p_ref_swing[2] += z_clearance # Add vertical clearance

                    # C. Cartesian PD Control (Similar to Eq. 8, simplified)
                    # Error in Cartesian space (HIP FRAME)
                    delta_p = p_ref_swing - p_current_hip_frame

                    # Calculate joint torques using Jacobian Transpose (approximation of Cartesian force control)
                    F_cartesian_command = GAINS_SWING_KP * delta_p 
                    
                    # Optional: Add damping using Jacobian and joint velocities (Kd term)
                    # J_dot_q_dot = mujoco.mj_dof_velocity(J_i)
                    # F_cartesian_command += GAINS_SWING_KD * (V_ref_swing - V_current) # full PD required velocity V_current
                    
                    # Convert desired Cartesian force to joint torque
                    tau_i_swing = J_i.T @ F_cartesian_command
                    legs_control_torque[joint_idx_offset:joint_idx_offset+3] = tau_i_swing

                # Increment joint index
                joint_idx_offset += 3

            # --- Apply Torques to Motors ---
            # MuJoCo actuators are already set to type="motor" in your XML.
            # We apply the torque directly to the 12 leg motors (first 12 controls)
            sim_data.ctrl[:12] = legs_control_torque 

            # Step the simulation
            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
    pass
except Exception as e:
    print(f"An error occurred: {e}")
    pass