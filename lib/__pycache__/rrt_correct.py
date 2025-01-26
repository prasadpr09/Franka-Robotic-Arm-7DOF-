import numpy as np
import random
from copy import deepcopy
from calculateFK import FK  
from detectCollision import detectCollision 
from loadmap import loadmap
# from calculateFK import FK  
# from detectCollision import detectCollision  
# from loadmap import loadmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fk = FK()

class Node:
    def _init_(self, config):
        self.config = config
        self.parent = None

def rrt(map, start, goal):
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                         -2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                         2.8973, 3.7525, 2.8973])

    if not is_config_valid(start, map) or not is_config_valid(goal, map):
        return np.empty((0, len(start)))

    tree = [Node(start)]
    max_iterations = 100000  # Adjusted to prevent timeouts
    step_size = 0.4
    goal_bias = 0.1

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(lowerLim[0], upperLim[0])
    ax.set_ylim(lowerLim[1], upperLim[1])
    ax.set_zlim(lowerLim[2], upperLim[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(start[0], start[1], start[2], c='g', s=50, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='r', s=50, label='Goal')

    for iteration in range(max_iterations):
        # Randomly sample a configuration with goal bias
        q_rand = goal if random.random() < goal_bias else generate_random_config(lowerLim, upperLim)
        # Find the nearest node in the tree to the sampled configuration
        nearest_node = find_nearest_node(tree, q_rand)
        # Extend the tree in the direction of the sampled configuration
        q_new = extend(nearest_node.config, q_rand, step_size)
        
        
        print("q random", q_rand)
        print("nearest node", nearest_node)
        print("q new", q_new)

        # Check if the new configuration is valid and the path to it is collision-free
        if is_config_valid(q_new, map) and is_path_collision_free(nearest_node.config, q_new, map):
            new_node = Node(q_new)
            new_node.parent = nearest_node
            tree.append(new_node)

            ax.plot([nearest_node.config[0], q_new[0]],
                    [nearest_node.config[1], q_new[1]],
                    [nearest_node.config[2], q_new[2]], 'b-', alpha=0.3)

            # Check if the new node is close enough to the goal
            if np.linalg.norm(q_new - goal) < step_size:
                if is_path_collision_free(q_new, goal, map):
                    goal_node = Node(goal)
                    goal_node.parent = new_node
                    path = extract_path(goal_node)

                    path_array = np.array(path)
                    ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'g-', linewidth=2)
                    plt.legend()
                    plt.title(f'RRT Path Found (Iterations: {iteration + 1})')
                    plt.show()

                    return path

        if iteration % 100 == 0:
            plt.title(f'RRT Progress (Iterations: {iteration})')
            plt.draw()
            plt.pause(0.001)

    # Return an empty array if no path is found
    print("No path found after maximum iterations.")
    plt.legend()
    plt.title(f'RRT No Path Found (Iterations: {max_iterations})')
    plt.show()
    return np.empty((0, len(start)))

def generate_random_config(lower_lim, upper_lim):
    # Generate a random configuration within the joint limits
    return np.random.uniform(lower_lim, upper_lim)

def find_nearest_node(tree, q):
    # Find the node in the tree that is closest to the given configuration
    return min(tree, key=lambda node: np.linalg.norm(node.config - q))

def extend(q_near, q_rand, step_size):
    # Extend from q_near towards q_rand by step_size
    direction = q_rand - q_near
    distance = np.linalg.norm(direction)
    q_new = q_rand if distance <= step_size else q_near + (direction / distance) * step_size
    return q_new

def extract_path(goal_node):
    # Extract the path from the goal node to the start node
    path = []
    current = goal_node
    while current is not None:
        path.append(current.config)
        current = current.parent
    path.reverse()
    return np.array(path)

def is_config_valid(q, map):
    # Joint limits
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                         -2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                         2.8973, 3.7525, 2.8973])
    # Check if the configuration is within the joint limits
    if np.any(q < lowerLim) or np.any(q > upperLim):
        return False

    # Collision checking
    jointPositions, T0e = fk.forward(q)
    for obstacle in map.obstacles:
        for i in range(len(jointPositions) - 1):
            start = jointPositions[i].reshape(1, 3)
            end = jointPositions[i + 1].reshape(1, 3)
            # Check for collisions between each segment of the arm and the obstacles
            if detectCollision(start, end, obstacle, 1e-5)[0]:
                return False
    return True

def is_path_collision_free(q_start, q_end, map, step_size=0.2):
    # Check if the path from q_start to q_end is collision-free
    distance = np.linalg.norm(q_end - q_start)
    steps = int(np.ceil(distance / step_size))
    for i in range(1, steps + 1):
        alpha = i / steps
        q_intermediate = (1 - alpha) * q_start + alpha * q_end
        if not is_config_valid(q_intermediate, map):
            return False
    return True

if __name__ == '_main_':
    map_struct = loadmap("../maps/map1.txt")
    # map_struct = loadmap("E:\Documents\Masters_Courses\MEAM 5200\Meam_520_Haiyue\meam520_labs\maps\map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal = np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    if not is_config_valid(start, map_struct) or not is_config_valid(goal, map_struct):
        print("Invalid start or goal configuration.")
    else:
        path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
        
        if len(path) > 0:
            print("Path found:")
            print(path)
        else:
            print("No valid path could be found.")