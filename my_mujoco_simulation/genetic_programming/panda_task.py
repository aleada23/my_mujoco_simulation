import numpy as np

def ik_task(model, data, robot, ee_name, ee_target_pose, Kp=1.0, frame="base", type_IK="pose"):
    """Compute IK for a single task (position/orientation/pose)"""
    if frame == "base":
        J_full = robot.get_Jacobian_in_base(model, data, frame_name=ee_name)
        ee_pos, ee_orn = robot.get_body_pose_with_frame(model, data, body_name=ee_name)
    else:
        ee_pos, ee_orn = robot.get_end_effector_pose(model, data)
        J_full = robot.get_Jacobian(model, data, ee_name)
    ee_pose = np.hstack([ee_pos, ee_orn])

    if type_IK == "position":
        pos_err = ee_target_pose[:3] - ee_pose[:3]
        J_task = J_full[:3, :]
        dq = np.linalg.pinv(J_task) @ (Kp * pos_err)
        error = np.linalg.norm(pos_err)
    elif type_IK == "orientation":
        qd = ee_target_pose[3:]
        qe = ee_pose[3:]
        orn_err = 0.5 * np.array([qe[1]*qd[0]-qe[0]*qd[1],
                                  qe[2]*qd[0]-qe[0]*qd[2],
                                  qe[3]*qd[0]-qe[0]*qd[3]])
        J_task = J_full[3:, :]
        dq = np.linalg.pinv(J_task) @ (Kp * orn_err)
        error = np.linalg.norm(orn_err)
    elif type_IK == "pose":
        pos_err = ee_target_pose[:3] - ee_pose[:3]
        qd = ee_target_pose[3:]
        qe = ee_pose[3:]
        orn_err = 0.5 * np.array([qe[1]*qd[0]-qe[0]*qd[1],
                                  qe[2]*qd[0]-qe[0]*qd[2],
                                  qe[3]*qd[0]-qe[0]*qd[3]])
        full_err = np.hstack([pos_err, orn_err])
        J_task = J_full
        dq = np.linalg.pinv(J_task) @ (Kp * full_err)
        error = np.linalg.norm(full_err)
    else:
        raise ValueError(f"Unknown IK type: {type_IK}")

    return dq, J_task, error


def manipulability_task(model, data, robot, ee_name):
    """Compute manipulability and its Jacobian"""
    J_full = robot.get_Jacobian_in_base(model, data, frame_name=ee_name)
    w = np.sqrt(np.linalg.det(J_full @ J_full.T) + 1e-12)

    # Approximate gradient of manipulability w.r.t joints (1x7)
    dq_grad = np.zeros(J_full.shape[1])
    eps = 1e-6
    qpos = data.qpos.copy()
    for i in range(J_full.shape[1]):
        data.qpos[i] += eps
        J_plus = robot.get_Jacobian_in_base(model, data, frame_name=ee_name)
        w_plus = np.sqrt(np.linalg.det(J_plus @ J_plus.T) + 1e-12)
        dq_grad[i] = (w_plus - w) / eps
        data.qpos[i] = qpos[i]  # reset

    return dq_grad, None, w  # Jacobian not needed, gradient is dq vector


def joint_limit_task(model, data):
    """Compute joint-limit score and identity Jacobian"""
    q = data.qpos
    qmin = model.jnt_range[:,0]
    qmax = model.jnt_range[:,1]
    d = [( (qmax[i]-q[i]) - (q[i]-qmin[i]) ) / (qmax[i]-qmin[i]) for i in range(len(q))]
    score = np.sum(d) / len(q)
    J_task = np.eye(len(q))  # joint-space identity
    dq = np.array(d)
    return dq, J_task, score


def tool_alignment_task(robot, model, data, ee_name, dir_target):
    """Keep a tool aligned along a desired direction"""
    _, ee_orn = robot.get_body_pose_with_frame(model, data, body_name=ee_name)
    ee_dir = ee_orn[:3]
    error_vec = dir_target - ee_dir
    J_task = robot.get_Jacobian(model, data, ee_name)[3:, :]  # rotational part
    dq = np.linalg.pinv(J_task) @ error_vec
    error = np.linalg.norm(error_vec)
    return dq, J_task, error


def com_task(robot, model, data, com_target):
    """Control the robot’s CoM position"""
    J_com = robot.get_CoM_Jacobian(model, data)
    dq = np.linalg.pinv(J_com) @ (com_target - robot.get_CoM(model, data))
    error = np.linalg.norm(com_target - robot.get_CoM(model, data))
    return dq, J_com, error


def posture_task(data, q_pref = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]):
    """Keep joints near a preferred posture"""
    dq_posture = q_pref - data.qpos
    J_task = np.eye(len(q_pref))
    error = np.linalg.norm(dq_posture)
    return dq_posture, J_task, error

import numpy as np

def null_space_projector(J):
    """Compute null-space projector N = I - J^dagger J"""
    n = J.shape[1]
    if J is None or J.size == 0:
        return np.eye(n)
    J_pinv = np.linalg.pinv(J)
    N = np.eye(n) - J_pinv @ J
    return N

def stack_of_tasks_combination(tasks):
    """
    Correct null-space projection for an arbitrary number of tasks.
    
    tasks: list of tuples (dq_desired, J)
        dq_desired: desired joint velocity for the task
        J: task Jacobian
    Returns:
        dq_total: joint velocity combining all tasks respecting priorities
    """
    if len(tasks) == 0:
        return None

    n = tasks[0][0].shape[0]  # number of joints
    dq_total = np.zeros(n)
    N_cumulative = np.eye(n)  # cumulative null-space

    for i, (dq_des, J) in enumerate(tasks):
        if J is None or J.size == 0:
            # task has no Jacobian, just add projected dq
            dq_task = N_cumulative @ dq_des
        else:
            # project Jacobian into current null-space
            J_proj = J @ N_cumulative
            dq_task = np.linalg.pinv(J_proj) @ (dq_des - J @ dq_total)
            dq_task = N_cumulative @ dq_task  # final projection

        dq_total += dq_task
        # update cumulative null-space
        if J is None or J.size == 0:
            continue
        N_cumulative = N_cumulative @ null_space_projector(J_proj)

    return dq_total
