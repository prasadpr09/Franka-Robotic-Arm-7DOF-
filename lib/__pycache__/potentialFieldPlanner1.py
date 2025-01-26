import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from calculateFKJac import FK_Jac
from detectCollision import detectCollision
from loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """
        
        e = 0.5
        att_f = np.zeros((3, 1)) 

        d = np.linalg.norm(target - current)
        
        if  d < 0.5:
            att_f = - e *(target - current)
        else:
            att_f = (-0.5 * e * (target - current))/d

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        p0 = 6
        eta = 0.5
        
        rep_f = np.zeros((3, 1)) 
        # current = current.reshape((1,3))

        print(current.shape)

        dist,unit = PotentialFieldPlanner.dist_point2box(current, obstacle.T)
        
        if dist < p0:
            rep_f = eta* ((1/dist)- (1/p0)) * (1/dist**2) * unit
        
        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        num_joints = 9
        joint_forces = np.zeros((3, 9)) 
        
        current = current.T
        
        for i in range (num_joints) :
            att_f_each = PotentialFieldPlanner.attractive_force(target[:,i].reshape((3,1)), current[:,i])
            rep_f = PotentialFieldPlanner.repulsive_force( obstacle[i,:].reshape((1,6)) , current[:,i])

            joint_forces[:,i] = (att_f_each + rep_f)  #flatten- if needed
        
        print("joint forces", joint_forces)

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        # Initialize Jacobian and joint torques for all 9 joints
        fk = FK_Jac()
        joint_torques = np.zeros((1, 9))  # 9 torques for all joints including any virtual ones

        # Compute forward kinematics and joint positions
        joint_positions, T0e = fk.forward_expanded(q)  # joint_positions: (3,10), T0e: (10,4,4)

        # Compute Jacobians and torques for each joint
        for i in range(9):
            # Extract Z-axis and position of joint i from the transformation matrix
            z_i = T0e[i, 0:3, 2]
            o_i = T0e[i, 0:3, 3]
            
            # Compute linear Jacobian for joint i
            jv_i = np.zeros((3, 9))
            for j in range(i + 1):  # Only up to the current joint
                z_j = T0e[j, 0:3, 2]
                o_j = T0e[j, 0:3, 3]
                jv_i[:, j] = np.cross(z_j, (o_i - o_j))

            # Calculate torque for joint i as the dot product of the force and Jacobian
            joint_torques[0, i] = np.sum(jv_i[:, :9].T @ joint_forces[:, i])  # Sum contributions

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        distance = 0
        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """
        dq = np.zeros((1, 7))
        
        # Use self.fk.forward_expanded(q) to calculate joint positions and transforms
        joint_positions, T = PotentialFieldPlanner.fk.forward_expanded(q)
        target_positions, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        
        # is doing a transpose necessary?
        joint_forces = PotentialFieldPlanner.compute_forces(target_positions.T, map_struct.obstacles, joint_positions.T)

        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)

        
        # current_positions = np.array([T[i][:3] for i in range(len(joint_positions))]).T
        # target_positions = np.array([target_positions[i][:3] for i in range(len(target_positions))]).T

        # print(current_positions.shape)

        # Calculate forces acting on each joint

        
        # Compute joint torques
        
        tau_prime = joint_torques.squeeze()[:7]
        
        # Normalize tau_prime to compute dq
        if np.linalg.norm(tau_prime) > 0:
            dq = tau_prime / np.linalg.norm(tau_prime)
        else:
            dq = tau_prime  # Handle case where tau_prime is zero vector

        return dq
        

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)

        q_path = np.array([start])  # Initialize path with the starting configuration
        q_current = start.copy()
        
        for step in range(self.max_steps):
            # Compute the gradient (joint velocity) to move towards the goal
            dq = self.compute_gradient(q_current, goal, map_struct)
            
            # Calculate the new joint configuration
            q_next = q_current + dq
            
            # Ensure the joint angles stay within limits
            q_next = np.clip(q_next, self.lower, self.upper)
            
            # Check for collision with obstacles
            joint_positions, _ = self.fk.forward_expanded(q_next)
            collision = False
            for obstacle in map_struct.obstacles:
                # Check each link for collision with the obstacle
                for i in range(len(joint_positions) - 1):
                    if detectCollision(joint_positions[i], joint_positions[i + 1], obstacle):
                        collision = True
                        break
                if collision:
                    break

            # If there is a collision, exit or take corrective measures
            if collision:
                print("Collision detected, halting the plan.")
                break
            
            # Add new configuration to path
            q_path = np.vstack((q_path, q_next))
            
            # Update current configuration
            q_current = q_next
            
            # Check termination conditions
            dist_to_goal = self.q_distance(goal, q_current)
            if dist_to_goal < self.tol:
                print(f"Converged to goal within tolerance after {step+1} steps.")
                break
            elif np.linalg.norm(dq) < self.min_step_size:
                print(f"Step size below minimum threshold after {step+1} steps.")
                break
            
            # Local minima handling (random perturbation if dq is too small)
            if np.linalg.norm(dq) < self.min_step_size:
                q_current += np.random.uniform(-0.05, 0.05, q_current.shape)
                print("Local minima detected, applying random perturbation.")
        
        # Ensure the final configuration is at the goal
        if self.q_distance(goal, q_path[-1]) > self.tol:
            q_path = np.vstack((q_path, goal))
        
        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
