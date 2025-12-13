from my_mujoco_simulation.simulation.simulation import Simulation
from my_mujoco_simulation.robot.controller.controller import Controller
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time


from my_mujoco_simulation.genetic_programming import panda_simb_jac, panda_task, population



Franka_PATH = "../robot/models/managerie_panda/robot.xml"

pedestal_PATH = "../object/pedestal/models/box_pedestal.xml"

MODEL_PATH = "../environment/models/mobot_lab/mobot_lab.xml"


torque_data = []
force_data = []
sim = Simulation(env_path=MODEL_PATH)

sim.add_robot(Franka_PATH, position="0.5 -0.8 0.5", orientation="0.7071068 0.0 0.0 0.7071068", init_config = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


sim.add_object(pedestal_PATH, pos = "0.5 -0.8 0", size = "0.15 0.15 0.5")


# Launch simulation
sim_model, sim_data = sim.launch(pretty_xml=False)




try:
    with mujoco.viewer.launch_passive(sim_model, sim_data, show_left_ui = False, show_right_ui = False) as viewer:

        while viewer.is_running():
            q = sim.get_robot(0).get_arm_joint_positions(sim_model, sim_data)
            
            
            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()
            
            

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
    