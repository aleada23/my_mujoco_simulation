import numpy as np

# --- Individual cost functions --- #

def cost_ik(error):
    """Cost for IK task"""
    return error**2

def cost_manipulability(w):
    """Cost for manipulability task (higher w better)"""
    return 1.0 / (w + 1e-6)

def cost_joint_limit(score):
    """Cost for joint-limit avoidance task (higher score better)"""
    return 1.0 / (score + 1e-6)

def cost_tool_alignment(error):
    """Cost for tool alignment"""
    return error**2

def cost_com(error):
    """Cost for CoM task"""
    return error**2

def cost_posture(error):
    """Cost for posture task"""
    return error**2
def cost_execution_time(t_start, t_current, t_max=10.0):
    """
    Compute cost based on execution time.
    
    Shorter time = lower cost.
    t_max: maximum expected duration for normalization.
    """
    t_elapsed = t_current - t_start
    c_time = min(t_elapsed / t_max, 1.0)  # clip at 1.0
    return c_time


# --- Master cost function --- #

def total_cost(metrics, flags=None, weights=None, t_start=None, t_current=None, t_max=10.0):
    """
    Compute weighted combination of task costs, now including execution time.
    
    metrics : dict
        Evaluation metrics for tasks
    flags : dict
        Boolean flags indicating which tasks are included
    weights : dict
        Weights for each task
    t_start, t_current : float
        Start and current execution time
    t_max : float
        Maximum expected duration for normalization
    """
    all_tasks = ['ik', 'manip', 'jl', 'tool', 'com', 'posture', 'time']
    if flags is None:
        flags = {task: True for task in all_tasks}
    if weights is None:
        enabled_count = sum(flags[task] for task in all_tasks)
        weights = {task: flags[task]/enabled_count for task in all_tasks}  # convex

    c_total = 0.0
    for task in all_tasks:
        if flags[task]:
            if task == 'ik':
                c = cost_ik(metrics['ik'])
            elif task == 'manip':
                c = cost_manipulability(metrics['manip'])
            elif task == 'jl':
                c = cost_joint_limit(metrics['jl'])
            elif task == 'tool':
                c = cost_tool_alignment(metrics['tool'])
            elif task == 'com':
                c = cost_com(metrics['com'])
            elif task == 'posture':
                c = cost_posture(metrics['posture'])
            elif task == 'time':
                if t_start is None or t_current is None:
                    raise ValueError("Execution time requires t_start and t_current")
                c = cost_execution_time(t_start, t_current, t_max=t_max)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            c_total += weights[task] * c

    return c_total
