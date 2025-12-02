import py_trees
from my_mujoco_simulation.robot import robot_actions
from my_mujoco_simulation.robot import robot_conditions
class BehaviorTreeManager:
    def __init__(self, model, data, robot):
        self.model = model
        self.data = data
        self.robot = robot
        self.tree = self._build_tree()

    def _build_tree(self):
        root = py_trees.composites.Sequence("Root", memory=True)

        #Move actions
        #for table
        move_pose_table = robot_actions.MovePose("MovePose", self.model, self.data, self.robot, target_pose=[0.5, 0.2, 0.5, 0.0, 0.7071068, 0.7071068, 0.0]) 
        #for object
        move_approach_object = robot_actions.MovePose("MovePose", self.model, self.data, self.robot, target_pose=[0.8, 0.0, 0.7, 0.7071068, 0.0, 0.7071068, 0.0], kp=10.0, tolerance = 0.01)
        move_pose_object = robot_actions.MovePose("MovePose", self.model, self.data, self.robot, target_pose=[0.8, 0.0, 0.55, 0.7071068, 0.0, 0.7071068, 0.0], kp=10.0, tolerance = 0.01)
        move_pose_object_end = robot_actions.MovePose("MovePose", self.model, self.data, self.robot, target_pose=[0.8, 0.0, 0.55, 0.7071068, 0.0, 0.7071068, 0.0], kp=10.0, tolerance = 0.01)

        move_home = robot_actions.MoveJoints("MoveJoints", self.model, self.data, self.robot, target_pos=self.robot.get_robot_dict()["configuration"])

        move_down = robot_actions.MoveDownUntillContact("MoveDownUntillContact", self.model, self.data, self.robot, target_pose=[0.5, 0.2, 0.3, 0.0, 0.7071068, 0.7071068, 0.0])

        lift = robot_actions.MovePose("Lift", self.model, self.data, self.robot, target_pose=[0.8, 0.0, 0.7, 0.7071068, 0.0, 0.7071068, 0.0])
        #Gripper actions
        open_gripper = robot_actions.OpenGripper("OpenGripper", self.model, self.data, self.robot)
        open_gripper_end = robot_actions.OpenGripper("OpenGripper", self.model, self.data, self.robot)

        close_gripper = robot_actions.CloseGripper("CloseGripper", self.model, self.data, self.robot)

        #Measurement actions
        measure_table = robot_actions.MeasureGripperSites("MeasureGripperSites", self.model, self.data, self.robot)
        measure_mass = robot_actions.MeasureMassWithTorque("MeasureMassWithTorque", self.model, self.data, self.robot)
        measure_mass_1 = robot_actions.MeasureMassWithTorque("MeasureMassWithTorque_1", self.model, self.data, self.robot)

        #CONDITIONS
        validation_mass = robot_conditions.IsObjectMassValid("IsObjectMassValid", self.model, self.data, self.robot)

        table_seq = py_trees.composites.Sequence(name="Table", memory=True)
        object_seq = py_trees.composites.Sequence(name="Object", memory=True)
    
        object_seq.add_children([move_home, move_approach_object, open_gripper, move_pose_object, measure_mass, close_gripper, lift, measure_mass_1, move_pose_object_end, open_gripper_end])
        table_seq.add_children([move_pose_table, move_down, measure_table])

        root.add_children([table_seq,object_seq])

        return py_trees.trees.BehaviourTree(root)

    def tick(self, display_tree = False, display_blackboard = False):
        if display_tree:
            print(py_trees.display.unicode_tree(self.tree.root, show_status=True))
        if display_blackboard:
            print(py_trees.display.unicode_blackboard())
        if self.tree.root.status in [py_trees.common.Status.RUNNING,py_trees.common.Status.INVALID]:
            self.tree.tick()
        
        return self.tree.root.status

    def print_tree(self):
        py_trees.display.render_dot_tree(self.tree.root)
