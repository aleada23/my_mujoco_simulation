from my_mujoco_simulation.simulation.simulation import Simulation
from my_mujoco_simulation.robot.controller.controller import Controller

import mujoco
import mujoco.viewer


Franka_PATH = "../robot/models/managerie_panda/robot.xml"
Kuka_PATH = "../robot/models/managerie_kuka/robot.xml"

table_PATH = "../object/table/models/simple_table.xml"

MODEL_PATH = "../environment/models/mobot_lab/mobot_lab.xml"


sim = Simulation(env_path=MODEL_PATH)

#sim.add_robot(Kuka_PATH, position="0 0 0.1", orientation="1 0 0 0")
sim.add_robot(Franka_PATH, position="0.0 0 0.8", orientation="1 0 0 0")

sim.add_object(table_PATH)
sim.add_object("../object/geomobj/models/box.xml", pos = "0.5 0 1", size = "0.02 0.02 0.12", mass = "0.5", color= "0 1 0 1")


# Launch simulation
sim_model, sim_data = sim.launch(pretty_xml=True)
controller = Controller(sim.get_robot(0), sim.sim_model, sim.sim_data)
timestep = 0.001
try:
    with mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
        while viewer.is_running():
            controller.set_joint_velocity([0.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0])
            sim.get_robot(0).get_Jacobian(sim_model, sim_data)
            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()
            

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
