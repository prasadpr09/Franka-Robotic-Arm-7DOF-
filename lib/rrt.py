import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK


def check_robot_collision(configuration, env_obstacles):
    """
    Validates if robot pose intersects with environment obstacles
    :param configuration: joint angles configuration
    :param env_obstacles: environment obstacle definitions
    :return: boolean indicating collision status
    """
    kinematics = FK()
    positions, _ = kinematics.forward(configuration)
    
    # Safety parameters for collision
    arm_thickness = 0.07
    joint_size = 0.07
    
    # Validate joint positions
    for pos in positions:
        for obstacle in env_obstacles:
            if detectCollision(pos.reshape(1, 3), pos.reshape(1, 3), obstacle)[0]:
                return True
            if compute_obstacle_distance(pos, obstacle) <= joint_size:
                return True

    # Validate links between joints
    for idx in range(len(positions)-1):
        start_pos = positions[idx]
        end_pos = positions[idx + 1]
        
        segment_vec = end_pos - start_pos
        segment_length = np.linalg.norm(segment_vec)
        
        if segment_length > 0:
            for obstacle in env_obstacles:
                if compute_obstacle_distance(start_pos, obstacle) <= arm_thickness:
                    return True
    
    return False

def compute_obstacle_distance(point, obstacle_bounds):
    """
    Determines minimum separation between point and obstacle
    """
    bounded_x = np.clip(point[0], obstacle_bounds[0], obstacle_bounds[3])
    bounded_y = np.clip(point[1], obstacle_bounds[1], obstacle_bounds[4])
    bounded_z = np.clip(point[2], obstacle_bounds[2], obstacle_bounds[5])
    
    nearest_point = np.array([bounded_x, bounded_y, bounded_z])
    return np.linalg.norm(point - nearest_point)


def generate_random_pose(joint_min, joint_max):
    """Generate valid random joint configuration"""
    return np.random.uniform(joint_min, joint_max, size=7)

def find_closest_node(tree_nodes, target_pose):
    """Locate nearest configuration in tree"""
    tree_array = np.array(tree_nodes)
    pose_distances = np.linalg.norm(tree_array - target_pose, axis=1)
    return np.argmin(pose_distances)

def step_towards_target(current_pose, target_pose, max_step=0.45):
    """
    Take a step from current pose towards target pose
    """
    distance = np.linalg.norm(target_pose - current_pose)
    if distance < max_step:
        return target_pose
    return current_pose + max_step * (target_pose - current_pose) / distance

def validate_connection(pose, tree_nodes, threshold=0.01):
    """Verify if pose can connect to tree"""
    return any(np.linalg.norm(pose - node) < threshold for node in tree_nodes)

def extract_path(tree_nodes, final_idx):
    """Extract path from tree structure"""
    trajectory = [tree_nodes[final_idx]]
    current_idx = final_idx
    
    while current_idx > 0:
        current_idx = find_closest_node(tree_nodes[:current_idx], tree_nodes[current_idx])
        trajectory.append(tree_nodes[current_idx])
    
    return trajectory[::-1]

def rrt(environment, initial_pose, target_pose):
    """
    Bidirectional RRT implementation for motion planning
    :param environment: collision environment definition
    :param initial_pose: robot's starting configuration
    :param target_pose: desired final configuration
    :return: sequence of configurations from start to goal
    """
    # Initialize trees
    forward_tree = [initial_pose]
    backward_tree = [target_pose]
    
    # Planning parameters
    max_iterations = 15000
    joint_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    step_length = 0.5
    goal_bias = 0.3

    # Main planning loop
    for _ in range(max_iterations):
        # Sample configuration
        random_pose = target_pose if random.random() < goal_bias else generate_random_pose(joint_min, joint_max)
        
        # Extend both trees
        for tree, other_tree, is_forward in [(forward_tree, backward_tree, True), (backward_tree, forward_tree, False)]:
            nearest_idx = find_closest_node(tree, random_pose)
            new_pose = step_towards_target(tree[nearest_idx], random_pose)
            
            if not check_robot_collision(new_pose, environment.obstacles):
                tree.append(new_pose)
                
                if validate_connection(new_pose, other_tree):
                    print("Path found!")
                    if is_forward:
                        path = extract_path(forward_tree, len(forward_tree) - 1) + \
                              extract_path(backward_tree, find_closest_node(backward_tree, new_pose))[::-1]
                    else:
                        path = extract_path(forward_tree, find_closest_node(forward_tree, new_pose)) + \
                              extract_path(backward_tree, len(backward_tree) - 1)[::-1]
                    return np.array(path)
    
    return np.array([]).reshape(0, 7)


        