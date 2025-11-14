from my_mujoco_simulation.simulation.simulation import Simulation
from my_mujoco_simulation.robot.controller.controller import Controller
from my_mujoco_simulation.behaviortree.tree_manager import BehaviorTreeManager
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


Gofa_PATH = "../robot/models/gofa_model/robot.xml"

table_PATH = "../object/table/models/simple_table.xml"
pedestal_PATH = "../object/pedestal/models/box_pedestal.xml"

MODEL_PATH = "../environment/models/mobot_lab/mobot_lab.xml"

torque_data = []
force_data = []
sim = Simulation(env_path=MODEL_PATH)

sim.add_robot(Gofa_PATH, position="0.5 -0.8 1.0", orientation="0.7071068 0.0 0.0 0.7071068", init_config = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])



#sim.add_object(table_PATH)
sim.add_object(pedestal_PATH, pos = "0.5 -90.8 0", size = "0.15 0.15 0.5")

#sim.add_object("../object/geomobj/models/box.xml", pos = "0.35 0.0 2", size = "0.02 0.02 0.12", mass = "0.5", color= "1 0 0 1")

# Launch simulation
sim_model, sim_data = sim.launch(pretty_xml=False)
timestep = 0.001

try:
    with mujoco.viewer.launch_passive(sim_model, sim_data, show_left_ui = True, show_right_ui = True) as viewer:

        while viewer.is_running():

            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()
            
            

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
    