import mujoco
import random
import numpy as np
from my_mujoco_simulation.simulation.simulation import Simulation
import my_mujoco_simulation.robot.robot_action_utils as action_utils
from my_mujoco_simulation.robot.controller.controller import Controller


ENVIRONMENT_PATH = "../environment/models/mobot_lab/mobot_lab.xml"
ROBOT_PATH = "../robot/models/managerie_panda/robot.xml"
TABLE_PATH = "../object/table/models/simple_table_with_target.xml"
PEDESTAL_PATH = "../object/pedestal/models/box_pedestal.xml"
time_step = 0.001

#Create simulation with environment only
sim = Simulation(env_path=ENVIRONMENT_PATH)

#Add robot and objects to the simulation
min_joints = 0.2 * np.array([-2.9, -1.76, -2.9, -3.0718, -2.9, -0.0175, -2.9])
max_joints = 0.2 * np.array([2.9, 1.76, 2.9, -3.0718, -0.0698, 3.7525, 2.9])
home_config = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
robot_config = np.clip(np.random.normal(home_config, (np.array(max_joints)-np.array(min_joints))/6), np.array(min_joints), np.array(max_joints))
#Robot rest position [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
sim.add_robot(ROBOT_PATH, position="0.0 0.0 0.5", orientation="1.0 0.0 0.0 0.0", init_config = robot_config)
sim.add_object(TABLE_PATH, pos="1.2 0.0 0.0")
sim.add_object(PEDESTAL_PATH, pos = "0.0 0.0 0.25", size = "0.15 0.15 0.25")

#RANDOMIZE OBJECT INITIAL POSITION
obj_pos = np.array([0.5, 0.0, 1.0])
#obj_pos = np.array([0.5 + random.uniform(-0.2, 0.2), 0.0 + random.uniform(-0.2, 0.2), 1.0])
obj_pos_str = ' '.join(map(str, obj_pos))
#RANDOMIZE MASS
obj_mass = random.uniform(0, 0.5)
obj_mass_str = (str, obj_mass)
#RANDOMIZE FRICTION
obj_fric = np.array([0.007, 0.0, 0.0])
#obj_fric = np.array([random.uniform(0, 0.1), 0.0, 0.0])
obj_fric_str = ' '.join(map(str, obj_fric))
sim.add_object("../object/geomobj/models/box.xml", pos = obj_pos_str, size = "0.03 0.03 0.03", mass = "0.1", color= "0 1 0 1", friction = obj_fric_str)

#Create Mujoco model and data
sim_model, sim_data = sim.launch(pretty_xml=False)
sim_model.opt.timestep = time_step
robot = sim.get_robot(0)
controller = Controller(sim_model, sim_data, robot)

target_poses = [[0.3, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0], [0.8, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]]
idx = 0

try:
    with mujoco.viewer.launch_passive(sim_model, sim_data, show_left_ui = True, show_right_ui = False) as viewer:

        while viewer.is_running():
            #ROBOT STATE 7 JOINTS
            #Position
            robot_state_pos = robot.get_arm_joint_positions(sim_model, sim_data)
            #Velocities
            robot_state_vel = robot.get_arm_joint_velocities(sim_model, sim_data)
            #END-EFFECTOR POSE [x,y,z,w,a_x,a_y,a_z] WRT ROBOT BASE FRAME
            #Pose is centererd in the blue dot in the center of the end-effector
            ee_pose = robot.get_end_effector_pose(sim_model, sim_data)
            #GET END-EFFECTOR VELOCITIES IN ROBOT BASE FRAME
            ee_vel = robot.get_end_effector_velocity(sim_model, sim_data)
            #OBJECT POSE WRT ROBOT FRAME 
            object_pose = robot.map_pose_into_robot(sim_model, sim_data, sim.get_object(2).get_pose(sim_model, sim_data))
            object_lin_velocities = robot.map_velocity_into_robot(sim_model, sim_data, sim_data.body(sim.get_object(2).get_body_name()).subtree_linvel)
            object_ang_velocities = robot.map_velocity_into_robot(sim_model, sim_data, sim_data.body(sim.get_object(2).get_body_name()).subtree_angmom)
            goal_target_pose = robot.map_pose_into_robot(sim_model, sim_data, np.array([1.0-0.5, 0, 0.85, 1.0, 0.0, 0.0, 0.0]))

            #VELOCITY CONTROLLER with target cartesian pose
            target_pose = target_poses[idx]
            gain = 1.0 #Keep between 0 and 2 to avoid instabilities
            dq, error = action_utils.inverse_kinematic(sim_model, sim_data, robot, robot.get_end_effector_name(sim_model), target_pose, Kp = gain)
            if np.allclose(error[:], 0, atol=1e-2):
                idx = idx + 1
            #VELOCITY CONTROLLER with cartesian velocity
            #target_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
            #dq = action_utils.velocity_cart2joint(sim_model, sim_data, robot, robot.get_end_effector_name(sim_model), target_vel)

            #APPLY VELOCITIES TO THE CONTROLLER
            controller.set_joint_velocity(dq)
            
            #MUJOCO SERVICE FUNCTIONS
            mujoco.mj_step(sim_model, sim_data)

            object_contact = False
            table_contact = False
            for i in range(sim_data.ncon):
                contact = sim_data.contact[i]
                if "hand" in mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "finger" in mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2):
                    if mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) == sim.get_object(2).get_call_geom_name():
                        object_contact = True
                    elif mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) == sim.get_object(0).get_call_geom_name():
                        table_contact = True
                if "hand" in mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "finger" in mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1):
                    if mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) == sim.get_object(2).get_call_geom_name():
                        object_contact = True
                    elif mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) == sim.get_object(0).get_call_geom_name():
                        table_contact = True

            viewer.sync()

except KeyboardInterrupt:
    print("Simulation interrupted by user.")