import numpy as np
import copy

EPS = 1e-9

def apply_dq_predict_qnext(data, dq_total, alpha=1.0):
    """Return q_next without modifying permanently (caller may set/restore)."""
    q_curr = data.qpos.copy()
    q_next = q_curr + alpha * dq_total
    return q_next

def compute_postmotion_metrics(model, data, robot, ee_name, dq_total, alpha=1.0, q_pref=None):
    """
    Apply a single-step linearized prediction q_next = q + alpha * dq_total,
    recompute common metrics and return a dict with:
      q_next, ee_pose_next, J_full_next, manipulability_next, jointlimit_score_next,
      posture_err_next, tool_dir_next, com_next, any raw vectors needed (e.g. ee_err_vec).
    This function TEMPORARILY sets data.qpos to q_next and restores it at exit.
    """
    # save
    q_saved = data.qpos.copy()

    # compute q_next and set it
    q_next = apply_dq_predict_qnext(data, dq_total, alpha=alpha)
    data.qpos[:] = q_next

    # recompute robot values at q_next
    # End-effector pose (position + orientation quaternion)
    ee_pos_next, ee_orn_next = robot.get_body_pose_with_frame(model, data, body_name=ee_name)
    # Jacobian (6x n) in base
    try:
        J_full_next = robot.get_Jacobian_in_base(model, data, frame_name=ee_name)
    except Exception:
        # fall back
        J_full_next = robot.get_Jacobian(model, data, ee_name)
    # manipulability
    try:
        w_next = np.sqrt(np.linalg.det(J_full_next @ J_full_next.T) + EPS)
    except Exception:
        w_next = 0.0

    # joint limits metric (average normalized distance from mid-range)
    q = q_next
    qmin = model.jnt_range[:, 0]
    qmax = model.jnt_range[:, 1]
    # avoid divide by zero
    denom = np.where((qmax - qmin) == 0, 1.0, (qmax - qmin))
    d = [((qmax[i] - q[i]) - (q[i] - qmin[i])) / denom[i] for i in range(len(q))]
    jointlimit_score_next = float(np.sum(d) / len(q))

    # posture error (norm to preferred)
    if q_pref is None:
        q_pref = np.zeros_like(q)
    posture_err_next = float(np.linalg.norm(q_pref - q))

    # tool direction (assume first axis of ee_orn_next encodes tool direction; change if your robot wrapper returns a quaternion)
    # if ee_orn_next is quaternion: compute axis vector from orientation: easiest is to convert to rotation matrix if helper exists
    # We will assume robot.get_body_pose_with_frame returns pos, quat; if you instead get rotation matrix adapt accordingly.
    # For now keep ee_orn_next as quaternion and convert first axis via simple rotation to vector if robot helper exists; fallback to zeros.
    try:
        # assume robot has method to compute rotation matrix or a helper: robot.quat_to_matrix
        R = robot.quat_to_matrix(ee_orn_next)  # implement in your wrapper if not present
        tool_dir_next = R[:, 0]  # first column: x-axis of end-effector frame
    except Exception:
        # if no helper, keep NaN vector (user should provide tool direction computation)
        tool_dir_next = np.zeros(3)

    # CoM position
    try:
        com_next = robot.get_CoM(model, data)
    except Exception:
        com_next = np.zeros(3)

    # EE target error vector not known here; caller should compute using ee_target_pose if needed
    # restore
    data.qpos[:] = q_saved

    metrics = {
        'q_next': q_next,
        'ee_pos_next': ee_pos_next,
        'ee_orn_next': ee_orn_next,
        'J_full_next': J_full_next,
        'manipulability_next': w_next,
        'jointlimit_score_next': jointlimit_score_next,
        'posture_err_next': posture_err_next,
        'tool_dir_next': tool_dir_next,
        'com_next': com_next
    }
    return metrics

# --- cost functions built on q_next metrics --- #

def cost_from_postmotion_ik(model, data, robot, ee_name, ee_target_pose, dq_total, alpha=1.0, Kp=1.0):
    """Predict next IK error after applying dq_total and return cost and e_next vector."""
    # temporarily set q_next to get accurate forward-kin
    q_saved = data.qpos.copy()
    q_next = apply_dq_predict_qnext(data, dq_total, alpha=alpha)
    data.qpos[:] = q_next

    # recompute ee pose at q_next
    ee_pos_next, ee_orn_next = robot.get_body_pose_with_frame(model, data, body_name=ee_name)
    ee_pose_next = np.hstack([ee_pos_next, ee_orn_next])

    # compute error vector (pose or per type)
    # If ee_target_pose is 7-element (pos + quat)
    # build error vector: position diff + quaternion "vector" error (use small-angle approx)
    pos_err = ee_target_pose[:3] - ee_pose_next[:3]
    # quaternion part: use same small quaternion vector error used earlier (works for small errors)
    qd = ee_target_pose[3:]
    qe = ee_pose_next[3:]
    orn_err = 0.5 * np.array([qe[1]*qd[0]-qe[0]*qd[1],
                              qe[2]*qd[0]-qe[0]*qd[2],
                              qe[3]*qd[0]-qe[0]*qd[3]])
    full_err = np.hstack([pos_err, orn_err])
    cost = float(np.dot(full_err, full_err))  # squared norm

    # restore
    data.qpos[:] = q_saved
    return cost, full_err

def cost_from_postmotion_manipulability(model, data, robot, ee_name, dq_total, alpha=1.0):
    # compute J at q_next and manipulability
    metrics = compute_postmotion_metrics(model, data, robot, ee_name, dq_total, alpha=alpha)
    w_next = metrics['manipulability_next']
    cost = 1.0 / (w_next + 1e-9)
    return cost, w_next

def cost_from_postmotion_joint_limits(model, data, dq_total, alpha=1.0):
    metrics = compute_postmotion_metrics(model, data, None, None, dq_total, alpha=alpha)
    score = metrics['jointlimit_score_next']
    cost = 1.0 / (score + 1e-9)
    return cost, score

def cost_from_postmotion_posture(model, data, dq_total, q_pref=None, alpha=1.0):
    metrics = compute_postmotion_metrics(model, data, None, None, dq_total, alpha=alpha, q_pref=q_pref)
    err = metrics['posture_err_next']
    cost = err**2
    return cost, err

def cost_from_postmotion_tool_align(model, data, robot, ee_name, dir_target, dq_total, alpha=1.0):
    metrics = compute_postmotion_metrics(model, data, robot, ee_name, dq_total, alpha=alpha)
    tool_dir = metrics['tool_dir_next']
    err_vec = dir_target - tool_dir
    cost = float(np.dot(err_vec, err_vec))
    return cost, err_vec

def cost_from_postmotion_com(model, data, robot, com_target, dq_total, alpha=1.0):
    metrics = compute_postmotion_metrics(model, data, robot, None, dq_total, alpha=alpha)
    com_next = metrics['com_next']
    err = np.linalg.norm(com_target - com_next)
    cost = err**2
    return cost, err

def cost_time_surrogate_from_dq(model, data, robot, ee_name, ee_target_pose, dq_total, alpha=1.0, t_max=10.0):
    # compute e_next_norm and speed produced by dq_total (in primary task space)
    # We use IK primary as first task's Jacobian: recompute J at current and at q_next if needed
    q_saved = data.qpos.copy()
    # current ee data
    ee_pos, ee_orn = robot.get_body_pose_with_frame(model, data, body_name=ee_name)
    ee_pose = np.hstack([ee_pos, ee_orn])
    # current error norm
    pos_err = ee_target_pose[:3] - ee_pose[:3]
    qd = ee_target_pose[3:]
    qe = ee_pose[3:]
    orn_err = 0.5 * np.array([qe[1]*qd[0]-qe[0]*qd[1],
                              qe[2]*qd[0]-qe[0]*qd[2],
                              qe[3]*qd[0]-qe[0]*qd[3]])
    e_curr_norm = float(np.linalg.norm(np.hstack([pos_err, orn_err])))

    # compute speed: J_primary(q) @ dq_total (use J at current q)
    try:
        J_full = robot.get_Jacobian_in_base(model, data, frame_name=ee_name)
    except Exception:
        J_full = robot.get_Jacobian(model, data, ee_name)
    task_vel = J_full @ dq_total
    speed = np.linalg.norm(task_vel) + 1e-12

    pred_t = (e_curr_norm - np.linalg.norm(J_full @ dq_total)) / (speed + 1e-12)
    # fallback if negative or NaN:
    if not np.isfinite(pred_t) or pred_t < 0:
        pred_t = e_curr_norm / (speed + 1e-12)
    cost_time = min(pred_t / t_max, 1.0)

    data.qpos[:] = q_saved
    return cost_time, pred_t
