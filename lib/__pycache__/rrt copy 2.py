import numpy as np
import random
from detectCollision import detectCollision
from loadmap import loadmap
from copy import deepcopy
from calculateFK import FK
import matplotlib.pyplot as plt

STEP_LENGTH = 0.01

def get_random_point(lower_lim, upper_lim):
    return np.array([random.uniform(lower_lim[i], upper_lim[i]) for i in range(7)])

def get_nearest_point(search_tree, q_random):
    # q_distance = np.linalg.norm(search_tree - q_random, axis=1)
    # return np.argmin(q_distance)
    
    closest_point = search_tree[0]
    min_distance = np.inf
    for q in search_tree:
        q_distance = np.linalg.norm(np.array(q) - np.array(q_random))
        if min_distance > q_distance:
            min_distance = q_distance
            closest_point = q
    return closest_point    



def extend_tree(search_tree, q_nearest, q_random, lower_lim, upper_lim):
    q_step_random = get_random_point(lower_lim, upper_lim)
    if STEP_LENGTH < np.linalg.norm(q_nearest - q_step_random):
        return q_nearest + STEP_LENGTH * ((q_nearest - q_step_random)/np.linalg.norm(q_nearest - q_step_random))
    # elif q_step_random.any(search_tree):
    #     return extend_tree(search_tree, q_nearest, q_random, lower_lim, upper_lim)
    else:    
        return q_step_random
    
    
def is_valid_config(q):
    # also check if q was valid config 
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718,-2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    if np.any(q < lowerLim) or np.any(q > upperLim):
        return False
    
    
def is_robot_collided(q, obstacle_map):
    
    q = q.astype(np.float64)
    fk = FK()
    q_new_coordinates ,_ = fk.forward(q)
    
    for i in range(7):
        for obstacle in obstacle_map:
            if True in detectCollision(q_new_coordinates[ i, 0:3].reshape((1,3)), q_new_coordinates[ i+1 , 0:3 ].reshape((1,3)), obstacle)[0]:
                return True
            else:  
                return False



def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    # get joint limits
    lower_lim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper_lim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    iterations = 5000
    i = 0
    q_nearest = start
    is_goal_reached = False
    search_tree = [start]
    
    

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(lower_lim[0], upper_lim[0])
    ax.set_xlim(lower_lim[1], upper_lim[1])
    ax.set_xlim(lower_lim[2], upper_lim[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(start[0], start[1], start[2], c='g', s=50, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='r', s=50, label='Goal')

    
    while i < iterations:
        print(i)
        
        q_random = get_random_point(lower_lim, upper_lim)
        q_nearest = get_nearest_point(search_tree, q_random)
        q_new = extend_tree(search_tree, q_nearest, q_random, lower_lim, upper_lim)
        
        print("q_new:",q_new)
        print("q_nearest:",q_nearest)
        print("q_random:",q_random)

        i+=1
        
        # check if path between q_nearest and q_new contains collisions
        if is_robot_collided(q_nearest, map.obstacles) and is_robot_collided(q_new, map.obstacles):
            # print("q_new:",q_new)
            print("in is_robot_collision check")
            continue
        
        elif is_valid_config(q_nearest) and is_valid_config(q_new):
            # print("q_new:",q_new)
            print("in is_valid_check")
            continue
        
        else:
            print("q_new:",q_new)
            print("q_nearest:",q_nearest)
            print("q_random:",q_random)
            
            print("not collided and config is valid")
            
            search_tree.append(q_new)
            nearest_goal_point = get_nearest_point(search_tree, goal)
            print("nearest_goal_point", nearest_goal_point)
            
            if STEP_LENGTH >= np.linalg.norm(nearest_goal_point - goal):
                if not is_robot_collided(nearest_goal_point, map.obstacles) and not is_robot_collided(goal, map.obstacles) :
                    search_tree.append(goal)
                    print("goal reached!")
                    break
            else:
                q_new = extend_tree(search_tree, nearest_goal_point, goal, lower_lim, upper_lim)
                print(q_new)
                if not is_robot_collided(q_new, map.obstacles) and not is_robot_collided(goal, map.obstacles) :
                    search_tree.append(goal)
                    print("goal reached!")
                    break

    path_array= np.array(search_tree)    
    print("search Tree:", search_tree)
    
    ax.plot(path_array[:,0], path_array[:,1], path_array[:,2], 'g-', linewidth=2)
    plt.legend()
    # plt.title()
    plt.show()
    
    return path_array

if __name__ == '__main__':
    map_struct = loadmap("../maps/emptyMap.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))