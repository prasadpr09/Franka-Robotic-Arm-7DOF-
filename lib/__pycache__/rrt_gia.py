import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
import matplotlib.pyplot as plt

# take a start position
# create a node- parent- use lists, linkedlists or class and config 
# define lower limits and upper limits
# check if config is valid, should be in bounds
# start making a tree, max iterations, step size and goal bias are hyper params, 
# iterate over random samples, what to sample - till goal points ?  we look for nearest node from the random sample


# to detect collision:
# increase tolerance and check for collision, detect collision
# every step that we take we need to check for collsiion-
# once its valid that is within join limits then check for collision, 
# like once we move the robot- check if final path is is in collision or not


def isRobotCollided(q, obstacles):
    link_lengths = [0.333, 0.316, 0.384, 0.384, 0.384, 0.088, 0.107]
    joint_positions = np.zeros((8, 3))
    joint_positions[0] = [0, 0, 0]
    current_pos = np.array([0, 0, 0])
    current_angle = 0
    
    for i in range(7):
        if i % 2 == 0:
            current_pos[0] += link_lengths[i] * np.cos(current_angle)
            current_pos[2] += link_lengths[i] * np.sin(current_angle)
            current_angle += q[i]
        else:
            current_pos[1] += link_lengths[i] * np.cos(current_angle)
            current_pos[2] += link_lengths[i] * np.sin(current_angle)
            current_angle += q[i]
        joint_positions[i + 1] = current_pos.copy()

    for i in range(7):
        pt1 = joint_positions[i]
        pt2 = joint_positions[i + 1]
        for obstacle in obstacles:
            if detectCollision(pt1.reshape(1, 3), pt2.reshape(1, 3), obstacle)[0]:
                return True
    return False

def getRandomConfig(lowerLim, upperLim):
    return np.array([random.uniform(lowerLim[i], upperLim[i]) for i in range(7)])

def findNearestNode(q_rand, nodes):
    distances = np.linalg.norm(nodes - q_rand, axis=1)
    return np.argmin(distances)

def interpolateConfig(q1, q2, step_size=0.05):
    dist = np.linalg.norm(q2 - q1)
    if dist < step_size:
        return q2
    return q1 + step_size * (q2 - q1) / dist

def isEdgeCollisionFree(q_start, q_end, obstacles, step_size=0.05):
    distance = np.linalg.norm(q_end - q_start)
    num_steps = int(distance / step_size)
    for i in range(num_steps + 1):
        intermediate_q = q_start + (i / num_steps) * (q_end - q_start)
        if isRobotCollided(intermediate_q, obstacles):
            return False
    return True

# check if path is collision free

# def is_collision_free():
    




def rrt(map, start, goal):
    
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
       
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(lowerLim[0], upperLim[0])
    ax.set_xlim(lowerLim[1], upperLim[1])
    ax.set_xlim(lowerLim[2], upperLim[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(start[0], start[1], start[2], c='g', s=50, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='r', s=50, label='Goal')

    

    if isRobotCollided(start, map.obstacles) or isRobotCollided(goal, map.obstacles):
        return np.array([])

    nodes = [start]  # lit 
    parents = [0]
    max_iterations = 10000
    goal_sample_rate = 0.2
    step_size = 0.2
    goal_threshold = 0.3

    for i in range(max_iterations):
        print(i)
        if random.random() < goal_sample_rate:
            q_rand = goal
        else:
            lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            q_rand = getRandomConfig(lowerLim, upperLim)

        nearest_idx = findNearestNode(q_rand, np.array(nodes))
        q_near = nodes[nearest_idx]
        q_new = interpolateConfig(q_near, q_rand, step_size)

        if not isRobotCollided(q_new, map.obstacles) and isEdgeCollisionFree(q_near, q_new, map.obstacles, step_size=0.05):
            nodes.append(q_new)
            parents.append(nearest_idx)

            if np.linalg.norm(q_new - goal) < goal_threshold and isEdgeCollisionFree(q_new, goal, map.obstacles, step_size=0.05):
                path = [goal]
                current_idx = len(nodes) - 1
                while current_idx != 0:
                    path.append(nodes[current_idx])
                    current_idx = parents[current_idx]
                path.append(start)
                
                path_array= np.array(nodes)    
                # print("search Tree:", search_tree)
                
                ax.plot(path_array[:,0], path_array[:,1], path_array[:,2], 'g-', linewidth=2)
                plt.legend()
                # plt.title()
                plt.show()

                return np.array(path[::-1])




    return np.array([])

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
