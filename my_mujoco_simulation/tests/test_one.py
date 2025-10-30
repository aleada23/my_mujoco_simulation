from my_mujoco_simulation.simulation.simulation import Simulation
from my_mujoco_simulation.robot.controller.controller import Controller
from my_mujoco_simulation.behaviortree.tree_manager import BehaviorTreeManager
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

import py_trees


Franka_PATH = "../robot/models/managerie_panda/robot.xml"
Kuka_PATH = "../robot/models/managerie_kuka/robot.xml"

table_PATH = "../object/table/models/simple_table.xml"
pedestal_PATH = "../object/pedestal/models/box_pedestal.xml"

MODEL_PATH = "../environment/models/mobot_lab/mobot_lab.xml"

CAMERA_PATH = "../camera/models/camera_example.xml"

torque_data = []
force_data = []
sim = Simulation(env_path=MODEL_PATH)

#sim.add_robot(Kuka_PATH, position="0 0 0.1", orientation="1 0 0 0")
sim.add_robot(Franka_PATH, position="0.5 -0.8 0.5", orientation="0.7071068 0.0 0.0 0.7071068", init_config = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

#sim.add_robot(Franka_PATH, position="10 0 0.8", orientation="1 0 0 0", init_config = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


sim.add_object(table_PATH)
sim.add_object(CAMERA_PATH)
sim.add_object(pedestal_PATH, pos = "0.5 -0.8 0", size = "0.15 0.15 0.5")

sim.add_object("../object/geomobj/models/box.xml", pos = "0.5 0.0 2", size = "0.02 0.02 0.12", mass = "0.5", color= "0 1 0 1")
sim.add_object("../object/geomobj/models/box.xml", pos = "0.35 0.0 2", size = "0.02 0.02 0.12", mass = "0.5", color= "1 0 0 1")
sim.add_object("../object/geomobj/models/box.xml", pos = "0.65 0.0 2", size = "0.02 0.02 0.12", mass = "0.5", color= "0 0 1 1")



# Launch simulation
sim_model, sim_data = sim.launch(pretty_xml=False)
timestep = 0.001
#tree_manager = BehaviorTreeManager(sim_model, sim_data, sim.get_robot(0))
#tree_manager.print_tree()
home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
object1_pose = [0.8, 0.0, 0.55, 0.7071068, 0.0, 0.7071068, 0.0]
offset_pose = [0.8, 0.0, 0.7, 0.7071068, 0.0, 0.7071068, 0.0]
offset_lift_pose = [0.8, 0.0, 0.7, 0.7071068, 0.0, 0.7071068, 0.0]
above_table = [0.5, 0.2, 0.5, 0.0, 0.7071068, 0.7071068, 0.0]
bt_definition = [
    "Sequence",
    ["MoveJoints", {"target_pos": home}],
    ["Sequence", ["MovePose", {"target_pose": offset_pose}],["OpenGripper", {}],["MovePose", {"target_pose": object1_pose}],["CloseGripper", {}],["MeasureGripperOpnening", {}],["MovePose", {"target_pose": offset_lift_pose}],["MeasureMassWithTorque", {}],["MovePose", {"target_pose": object1_pose}],["OpenGripper", {}]],
        ["Sequence",["MovePose", {"target_pose": above_table}],["MoveDownUntillContact", {"target_pose": above_table}],["MeasureGripperSites", {}]],["MoveJoints", {"target_pos": home}]]

bt_builder = BehaviorTreeManager(sim_model, sim_data, sim.get_robot(0))
tree = bt_builder.build_tree(bt_definition)
bt_builder.print_tree()
try:
    with mujoco.viewer.launch_passive(sim_model, sim_data, show_left_ui = False, show_right_ui = False) as viewer:
        #time.sleep(10)

        while viewer.is_running():
            bt_builder.tick(display_tree = False)
            torque_data.append(np.array([sim.get_robot(0).get_sensor_data(sim_data, 3), sim.get_robot(0).get_sensor_data(sim_data, 4), sim.get_robot(0).get_sensor_data(sim_data, 5)]))
            #force_data.append(np.array([sim.get_robot(0).get_sensor_data(sim_data, 0), sim.get_robot(0).get_sensor_data(sim_data, 1), sim.get_robot(0).get_sensor_data(sim_data, 2)]))

            mujoco.mj_step(sim_model, sim_data)
            viewer.sync()
            
            

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
    py_trees.blackboard.Blackboard.enable_activity_stream()
    reader = py_trees.blackboard.Client()
    print(py_trees.blackboard.Blackboard.storage)
    """force_data= np.array(torque_data)
    time_plot = np.arange(len(torque_data)) * sim_model.opt.timestep
    plt.figure(figsize=(10, 4))
    plt.plot(time_plot, force_data[:, 0], label="Torque X")
    plt.plot(time_plot, force_data[:, 1], label="Torque Y")
    plt.plot(time_plot, force_data[:, 2], label="Torque Z")

    plt.title("Sensor Torques over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.grid(True)
    plt.legend(title="Components")  # Add legend with title
    plt.show()"""
    