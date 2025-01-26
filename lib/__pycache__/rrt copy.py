import numpy as np
import random
from detectCollision import detectCollision
from loadmap import loadmap
from copy import deepcopy
from calculateFK import FK




def nearest_point(q_rand, q_near):
    distance =  np.linalg.norm(q_rand - q_near)
    



def getRandomConfig(lowerLim, upperLim, map, q_near):
    
    q_rand = np.zeros((1,7))
    q_rand = np.array([random.uniform(lowerLim[i], upperLim[i]) for i in range(7)])
    
    distance = nearest_point(q_rand, q_near)
    
    # jp_next,_ = FK.forward(q_next) # 8 x 3
    # # check if every joint is colliding with the obstacle
    
    # # check for nearest node , choose among the sampled points - distance , and also check the min
    # # if q_new = q_rand if dist <=  step_size - this is the extend function. 
    
    for i in range(7):
        for obs in map.obstacle:
            if detectCollision(jp_next[i, 0:3].reshape((1,3)), jp_next[i+1, 0:3].reshape((1,3)), obs) == False:
                q_path = q_next # return that  config for the step taken
            else:
                getRandomConfig(lowerLim,upperLim,map)
    
    return q_path

# ispathvalid - path should be collision free
# norm between q_rand and q_near - is the direction 
# steps is the distance / step size
# dist can be greater than step_size, so use that % of distance 
# how to find step size? - 0.2
# alpha = some i / steps , i is the time to go from 1 to steps 1/5 = 0.2 
# next inter = 2/5 = 0.4 -this is how we r incrementing 
# we need if every q _intermediate is valid while alpha <
# q_intermediate = (1 - alpha) = 1 - 0.2 = 0.8 , 0.8 * 30 = 24  +0.2 * 40 - went from 30 to 40 by inc 2

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
    fk = FK()

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    
    limit = 10
    counter = 0
    print(map.obstacles)
    q_near = start
    qGoal = goal
    qNew = np.zeros((7))
    T = []
    
    while counter < limit:
        # qNew[0], qNew[1], qNew[2] = np.random.uniform(lowerLim[0], upperLim[0]),np.random.uniform(lowerLim[1], upperLim[1]),np.random.uniform(lowerLim[2], upperLim[2])

        # we use c-space 
        
        qNew = getRandomConfig(lowerLim, upperLim, map, q_near)
        
        
        # check if goal is valid- check if goal is not inside obs, and goal is within joint limits
        # config valid - check if the randomly generated point is within the limits, if not uda do.


        for obstacle in map.obstacles:
            
            # vector coordinates of each joint in 3d space in start config and nw config
            # checking for collisions between start and new and obstacle - which joint position? - all
            # if obs in map obstacle 
            
            for i in range(8):
                qNext = getRandomConfig(lowerLim, upperLim, obstacle)



            # also check if the start position is in collision with obstacle

    return np.array(path)

if __name__ == '__main__':#
    map_struct = loadmap("../maps/emptyMap.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))