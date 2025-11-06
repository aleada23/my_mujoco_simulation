import mujoco
import numpy as np
import my_mujoco_simulation.simulation.transformation_utils as transformation_utils
 
def forward_kinematic(model, data, robot, ee_name, frame = "world"):
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
    ee_pos = data.xpos[ee_id].copy()
    ee_quat = data.xquat[ee_id].copy()
    ee_R = data.xmat[ee_id].copy().reshape(3,3)
    
    if frame == "world":
        return ee_pos, ee_quat

    elif frame == "base":
        bodies = [b.attrib.get("name") for b in root.findall(".//body[@name]")]
        base_frame = f"{robot.get_robot_dict()["name"]}_{bodies[0]}" if bodies else None
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_frame)
        base_R = data.xmat[base_id].reshape(3,3)
        base_pos = data.xpos[base_id].copy()
        ee_pos_b = base_R.transpose()@(ee_pos-base_pos)
        ee_R_b = base_R.transpose()@ee_R
        temp = transformation_utils.mat2quat(ee_R_b)
        ee_quat_b = [temp[3], temp[0], temp[1], temp[2]]
        return ee_pos_b, ee_quat_b

    else:
        arm_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
        base_R = data.xmat[arm_base_id].reshape(3,3)
        base_pos = data.xpos[arm_base_id].copy()
        ee_pos_a = base_R.transpose()@(ee_pos-base_pos)
        ee_R_a = base_R.transpose()@ee_R
        temp = transformation_utils.mat2quat(ee_R_a)
        ee_quat_a = [temp[3], temp[0], temp[1], temp[2]]
        return ee_pos_a, ee_quat_a

def inverse_kinematic(model, data, robot, ee_name, ee_target_pose, Kp = 1.0, frame = "base", type_IK = "pose"):
    
    if frame == "base":
        J_full = robot.get_Jacobian_in_base(model, data, frame_name = ee_name)
        ee_pos, ee_orn = robot.get_body_pose_with_frame(model, data, body_name = ee_name)
    else: 
        ee_pos, ee_orn = forward_kinematic(model, data, robot, ee_name, frame)
        J_full = robot.get_Jacobian(model, data, ee_name)
    ee_pose = np.hstack([ee_pos, ee_orn])
    if type_IK == "position":
        position_error = ee_target_pose[:3] - ee_pose[:3]
        J_arm = J_full[:3, :]
        v_des = Kp * position_error
        dq = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T) @ v_des
    elif type_IK == "orientation":
        skew_des = np.array([[int(0), -ee_target_pose[5], ee_target_pose[4]], [ee_target_pose[5], int(0) , -ee_target_pose[3]], [-ee_target_pose[4], ee_target_pose[3], int(0)]])
        orn_err = (ee_pose[6]*np.array([ee_target_pose[3], ee_target_pose[4], ee_target_pose[5]])) - (ee_target_pose[6]*np.array([ee_pose[3], ee_pose[4], ee_pose[5]])) - np.matmul(skew_des,np.array([ee_pose[3], ee_pose[4], ee_pose[5]])) 
        J_arm = J_full[3:, :]
        v_des = Kp * orn_err
        dq = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T) @ v_des

    if type_IK == "pose":
        position_error = ee_target_pose[:3] - ee_pose[:3]
        skew_des = np.array([[int(0), -ee_target_pose[6], ee_target_pose[5]], [ee_target_pose[6], int(0) , -ee_target_pose[4]], [-ee_target_pose[5], ee_target_pose[4], int(0)]])
        orn_err = (ee_pose[3]*np.array([ee_target_pose[4], ee_target_pose[5], ee_target_pose[6]])) - (ee_target_pose[3]*np.array([ee_pose[4], ee_pose[5], ee_pose[6]])) - np.matmul(skew_des,np.array([ee_pose[4], ee_pose[5], ee_pose[6]])) 
        error = np.hstack([position_error, orn_err])
        v_des = Kp * error
        dq = np.linalg.pinv(J_full) @ v_des
    return dq, error

def velocity_cart2joint(model, data, robot, ee_name, v_des, frame = "base"):
    if frame == "base":
        J_full = robot.get_Jacobian_in_base(model, data, frame_name = ee_name)
        ee_pos, ee_orn = robot.get_body_pose_with_frame(model, data, body_name = ee_name)
    else: 
        ee_pos, ee_orn = forward_kinematic(model, data, robot, ee_name, frame)
        J_full = robot.get_Jacobian(model, data, ee_name)
    dq = np.linalg.pinv(J_full) @ v_des

    return dq

def jacobian_in_base_frame(model, data, robot, ee_id, base_body):
    jacp = robot.get_Jacobian(model, data, frame)[:3, :]
    jacr = robot.get_Jacobian(model, data, frame)[3:, :]
    # Base orientation in world
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    R_wb = data.xmat[base_id].reshape(3, 3)   # base in world
    R_bw = R_wb.T                             # world to base rotation

    # Rotate Jacobian rows
    jacp_base = R_bw @ jacp
    jacr_base = R_bw @ jacr

    J_base = np.vstack((jacp_base, jacr_base))
    return J_base

def get_child_pose_from_parent(model, data, parent_pos, parent_body_name, child_body_name):
    # Get IDs
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_name)
    child_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_body_name)

    # Child rotation relative to parent in world frame
    # MuJoCo stores xmat as row-major 3x3 rotation matrix
    R_child_world = data.xmat[child_id].reshape(3,3)

    # Child position in world frame
    child_pos_world = data.xpos[child_id].copy()

    return child_pos_world, R_child_world

def get_mass_from_torques(model, data, robot):
    joint_torques = robot.get_arm_joint_torque(model, data)
    jac = robot.get_Jacobian_in_base(model, data)
    J_inv_T = np.linalg.pinv(jac.T)
    h_e = J_inv_T @ joint_torques
    print(joint_torques, h_e)
    return None

