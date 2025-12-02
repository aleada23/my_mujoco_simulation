from my_mujoco_simulation.simulation.simulation import Simulation
from my_mujoco_simulation.robot.controller.controller import Controller
from my_mujoco_simulation.behaviortree.tree_manager import BehaviorTreeManager
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

import py_trees


robot_PATH = "../robot/models/managerie_spot/robot.xml"


MODEL_PATH = "../environment/models/scene/spot_scene.xml"


torque_data = []
force_data = []
sim = Simulation(env_path=MODEL_PATH)

sim.add_robot(robot_PATH)

# Launch simulation
sim_model, sim_data = sim.launch(pretty_xml=False)
sim_data.qpos = [0, 0, 0.46, 1, 0, 0, 0, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, -3.14, 3.06, 0, 0, 0, 0]# legs start at [7]
#sim_data.qpos = [0, 0, 0.46, 1, 0, 0, 0, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8]# legs start at [7]
sim_data.ctrl = [0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, -3.14, 3.06, 0, 0, 0, 0] #[0:3] fl, [3:6] fr, [6:9] rl, [9:12] rr, 
#sim_data.ctrl = [0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8] #[0:3] fl, [3:6] fr, [6:9] rl, [9:12] rr, 

model_legs = ["fl", "fr", "hl", "hr"]
geom_Jac = ["robot0_FL","robot0_FR","robot0_HL","robot0_HR"]
model_Jac_names = ["robot0_fl_lleg", "robot0_fr_lleg", "robot0_hl_lleg", "robot0_hr_lleg"]
model_ref_Jac_names = ["robot0_fl_hip", "robot0_fr_hip", "robot0_hl_hip", "robot0_hr_hip"]
init_phases = [0.2, 0.7, 0.7, 0.2]

wanted_velocity = [0.3, 0.0, 0.0]
step_length = 0.2
step_height = 0.05
step_data = []
try:
    with mujoco.viewer.launch_passive(sim_model, sim_data, show_left_ui = True, show_right_ui = True) as viewer:
        init_time = sim_data.time
        while viewer.is_running():
            legs_control = np.zeros(19)
            #legs_control = np.zeros(12)
            jac_idx = 0

            time_ = sim_data.time % 2
            phase = time_/2
            
            for i in range(len(model_legs)):
                leg_phase = phase + init_phases[i]
                cycle_time = 0.5         # total step period
                stance_ratio = 0.5     # 60% stance, 40% swing
                #phase = (sim_data.time % cycle_time) 

               
                leg_phase = (phase + init_phases[i]) % 1.0
                if leg_phase < stance_ratio:
                    s = leg_phase / stance_ratio
                    x = (s - 1.0) * step_length / 2
                    z = 0.0
                else:
                    s = (leg_phase - stance_ratio) / (1 - stance_ratio)
                    x = -step_length/2 + s * step_length
                    z = step_height * 4 * s * (1 - s) 
                if i == 0 or i ==2:
                    y = 0.12
                else:
                    y = -0.12
                p_target_hip = np.array([x, y, z-0.409])
                if i ==0:
                    step_data.append([x,z])
                body_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_GEOM, geom_Jac[i])
                nv = sim_model.nv
                Jp_full = np.zeros((3, nv))  # Linear velocity Jacobian
                Jr_full = np.zeros((3, nv))  # Angular velocity Jacobian
                mujoco.mj_jacGeom(sim_model, sim_data, Jp_full, Jr_full, body_id)
                J_full = np.vstack((Jp_full, Jr_full))
                base_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, model_ref_Jac_names[i])
                base_rot = sim_data.xmat[base_id].reshape(3,3)
                R = np.zeros((6,6))
                R[:3,:3] = base_rot.T
                R[3:, 3:] = base_rot.T
                #obtain leg Jacobian wrt to body 
                Jac = np.zeros((6, 3))
                Jac[:, :] = J_full[:, jac_idx+6 : jac_idx+6 + 3]
                Jac_R =  R @ Jac
                #q_leg = np.linalg.pinv(Jac[:3,:])@ np.hstack((x,0,z))
                p_current = sim_data.geom_xpos[body_id] - sim_data.xpos[base_id]  # foot relative to hip
                delta = (R[:3,:3]@p_target_hip - R[:3,:3]@p_current)
                q_leg = np.linalg.pinv(Jac_R[:3,:]) @ delta
                if i == 0 or i ==1:
                    gain = 7
                legs_control[jac_idx:jac_idx+3] = q_leg*gain

                jac_idx += 3
            #print(legs_control)
            sim_data.ctrl = sim_data.ctrl + legs_control*0.001
            #sim_data.ctrl = sim_data.qpos[7:] + legs_control*0.1            
            #sim_data.ctrl = [0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, 1.04, -1.8, 0, -3.14, 3.06, 0, 0, 0, 0.0]

            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()
            
            

except KeyboardInterrupt:
    
    step_data= np.array(step_data)
    time_plot = np.arange(len(step_data)) * sim_model.opt.timestep
    plt.figure(figsize=(10, 4))
    plt.plot(time_plot, step_data[:, 0], label="X")
    plt.plot(time_plot, step_data[:, 1], label="Y")
    

    plt.title("Sensor Torques over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.grid(True)
    plt.legend(title="Components")  # Add legend with title
    plt.show()
    