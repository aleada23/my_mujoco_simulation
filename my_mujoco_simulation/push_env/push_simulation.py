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

#Create simulation with environment only
sim = Simulation(env_path=ENVIRONMENT_PATH)

#Add robot and objects to the simulation
sim.add_robot(ROBOT_PATH, position="0.0 0.0 0.5", orientation="1.0 0.0 0.0 0.0", init_config = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
sim.add_object(TABLE_PATH, pos="1.2 0.0 0.0")
sim.add_object(PEDESTAL_PATH, pos = "0.0 0.0 0.25", size = "0.15 0.15 0.25")

#RANDOMIZE OBJECT INITIAL POSITION
obj_pos = np.array([0.5 + random.uniform(-0.2, 0.2), 0.0 + random.uniform(-0.2, 0.2), 1.0])
obj_pos_str = ' '.join(map(str, obj_pos))
#RANDOMIZE MASS
obj_mass = random.uniform(0, 0.5)
obj_mass_str = (str, obj_mass)
#RANDOMIZE FRICTION
obj_fric = np.array([random.uniform(0, 1), 0.0, 0.0])
obj_fric_str = ' '.join(map(str, obj_fric))
sim.add_object("../object/geomobj/models/box.xml", pos = obj_pos_str, size = "0.03 0.03 0.03", mass = "0.1", color= "0 1 0 1", friction = "0.5 0.0 0.0")

#Create Mujoco model and data
sim_model, sim_data = sim.launch(pretty_xml=False)
robot = sim.get_robot(0)
controller = Controller(sim_model, sim_data, robot)

target_poses = [[0.3, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], [0.65, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0], [0.8, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]]
idx = 0

try:
    with mujoco.viewer.launch_passive(sim_model, sim_data, show_left_ui = False, show_right_ui = True) as viewer:

        while viewer.is_running():
            #ROBOT STATE 7 JOINTS
            #Position
            robot_state_pos = robot.get_arm_joint_positions(sim_model, sim_data)
            #Velocities
            robot_state_vel = robot.get_arm_joint_velocities(sim_model, sim_data)
            #END-EFFECTOR POSE [x,y,z,w,a_x,a_y,a_z] WRT ROBOT BASE FRAME
            #Pose is centererd in the blue dot in the ecnter of the end-effector
            ee_pose = robot.get_end_effector_pose(sim_model, sim_data)
            #OBJECT POSE WRT ROBOT FRAME 
            #Target is:
            object_pose = robot.map_pose_into_robot(sim_model, sim_data, sim.get_object(2).get_pose(sim_model, sim_data))

            #VELOCITY CONTROLLER with target cartesian pose
            #
            #target_pose = target_poses[idx]
            #gain = 1.0
            #dq, error = action_utils.inverse_kinematic(sim_model, sim_data, robot, robot.get_end_effector_name(sim_model), target_pose, Kp = gain)
            #if np.allclose(error[:], 0, atol=1e-2):
            #    idx = idx + 1
            #VELOCITY CONTROLLER with cartesian velocity
            #
            target_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
            dq = action_utils.velocity_cart2joint(sim_model, sim_data, robot, robot.get_end_effector_name(sim_model), target_vel)

            #APPLY VELOCITIES TO THE CONTROLLER
            controller.set_joint_velocity(dq)
            
            #MUJOCO SERVICE FUNCTIONS
            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()

except KeyboardInterrupt:
    print("Simulation interrupted by user.")