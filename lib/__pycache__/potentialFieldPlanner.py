import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from  calculateFKJac import FK_Jac
from  detectCollision import detectCollision
from  loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = lower + (upper - lower) / 2  # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-2, max_steps=5000, min_step_size=1e-3, alpha=0.1):
        """
        Constructs a potential field planner with solver parameters.
        """
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size
        self.alpha = alpha  # Initialize the alpha value for gradient steps

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attractive force.
        """
        e = 1.5 # this param should be greater than the repulsive for param , otherwise we arent going anywhere 
        att_f = np.zeros((3, 1))
        d = np.linalg.norm(target - current)
        
        if d < 0.6:
            att_f = -e * (target - current)
        else:
            att_f = (-0.5 * e * (target - current)) / d

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3, 1))):
        """
        Helper function for computing the repulsive force.
        """
        p0 = 0.25
        eta = 0.1 #try smaller than 1 

        # Ensure obstacle is correctly reshaped if it has 6 elements
        if obstacle.size == 6:
            obstacle = obstacle.reshape((6,))
        else:
            print("Skipping obstacle due to incorrect size:", obstacle)
            return np.zeros(3)

        rep_f = np.zeros(3)
        current = current.reshape((1, 3))
        dist, unit = PotentialFieldPlanner.dist_point2box(current, obstacle)

        if dist < p0 and dist >0:
            rep_f = - eta * ((1 / dist) - (1 / p0)) * (1 / dist**2) * unit.flatten()
            
        # elif dist == 0:
        #     rep_f = eta * np.random.randn(3,1)
        #     rep_f /= np.linalg.norm(rep_f)
        
        return rep_f
    
    
    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces.
        """
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin * 0.5 + boxMax * 0.5
        p = np.array(p)

        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        signs = np.sign(boxCenter - p)
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joint.
        """
        num_joints = 9
        joint_forces = np.zeros((3, 9))

        for i in range(num_joints):
            # Compute attractive force for the i-th joint
            att_f_each = PotentialFieldPlanner.attractive_force(target[:3, i].reshape((3, 1)), current[:3, i].reshape((3, 1)))

            # Initialize repulsive force to zero for the i-th joint
            rep_f = np.zeros(3)  # Ensure rep_f is (3,)

            # Calculate repulsive force for all obstacles
            for obs in obstacle:
                try:
                    obs_reshaped = obs.reshape((1, 6))  # Reshape each obstacle as needed
                    rep_f += PotentialFieldPlanner.repulsive_force(obs_reshaped, current[:3, i].reshape((3, 1))).flatten()
                except ValueError:
                    print("Skipping invalid obstacle shape:", obs)

            # Combine attractive and repulsive forces for the joint
            joint_forces[:, i] = (att_f_each.flatten() + rep_f)

        return joint_forces

    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques.
        """
        fk = FK_Jac()
        joint_torques = np.zeros((1, 9))

        joint_positions, T0e = fk.forward_expanded(q)

        for i in range(1,9):
            z_i = T0e[i, 0:3, 2]
            o_i = T0e[i, 0:3, 3]
            
            jv_i = np.zeros((3, 9))
            
            for j in range(i + 1):
                z_j = T0e[j, 0:3, 2]
                o_j = T0e[j, 0:3, 3]
                jv_i[:, j] = np.cross(z_j, (o_i - o_j))

            joint_torques[0, i] = np.sum(jv_i[:, :9].T @ joint_forces[:, i])

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two vectors.
        """
        return np.linalg.norm(target - current)

    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move closer to the goal configuration.
        """
        dq = np.zeros((1, 7))
        
        joint_positions, T = PotentialFieldPlanner.fk.forward_expanded(q)
        target_positions, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        
        joint_forces = PotentialFieldPlanner.compute_forces(target_positions.T, map_struct.obstacles, joint_positions.T)
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        
        tau_prime = joint_torques.squeeze()[:7]
        
        # Adaptive alpha based on the distance to the goal
       
          # Increase alpha as the goal gets farther

        if np.linalg.norm(tau_prime) > 0:
            dq =  tau_prime / np.linalg.norm(tau_prime)
        else:
            dq = tau_prime

        return dq
    
    # check for these three collisions
    # check self_collision 
    # path collision - when its executing its path 
    # and robot collision - check for collision with obstacle


    def within_joint_limits(self, position):
        return np.all(position >= self.lower) and np.all(position <= self.upper)
        
        

    def self_collision_check(self, q, goal, map_struct):
        """
        Checks for self-collisions between non-adjacent links in the robot arm.
        """
        box = map_struct.obstacles  # Ensure obstacles are passed from the map structure
        # Loop through pairs of non-adjacent links
        
        # If no obstacles are present, skip collision checks
        if isinstance(box, (list, np.ndarray)) and len(box) == 0:
            return False

        interpolate_percent = 0.02  # interpolation step percentage
        num_steps = int(1 / interpolate_percent)

        # Generate interpolated configurations from q_start to q_goal
        configs = [q + (goal - q) * t for t in np.linspace(0, 1, num_steps + 1)]
            
        
        for i in configs:
            joint_positions, _ = self.fk.forward_expanded(q)  
            
            for obs in box:
                
                for j in range(len(joint_positions)-1):
                    
                    # Detect collision between link i and link j using each obstacle
                    if detectCollision(joint_positions[j].reshape((1,-1)), joint_positions[j+1].reshape((1,-1)), obs) == True:
                        print(f"Self-collision detected between link {i} and link {j}.")
                        
                        return True  # Self-collision detected
                    
        return False  # No self-collision detected

    def check_self_collision(self, q, map_struct):
        """
        Checks for self-collisions between non-adjacent links in the robot.
        """
        
        obstacles = map_struct.obstacles  # List of obstacles (boxes)
        
        # Get the positions of each joint in the current configuration
        joint_positions, _ = self.fk.forward_expanded(q)

        # Define non-adjacent link pairs for self-collision checking
        link_pairs = [
            (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),  
            (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 4), (2, 5), (2, 6),
            (3, 5), (3, 6),
            (4, 6)
        ]

        # Check each pair for self-collision
        for (i, j) in link_pairs:
            start_i = joint_positions[i].reshape(1, -1)
            end_i = joint_positions[i + 1].reshape(1, -1)
            start_j = joint_positions[j].reshape(1, -1)
            end_j = joint_positions[j + 1].reshape(1, -1)
            
            # Check collision with each obstacle (box)
            for box in obstacles:
                # Check if link i collides with link j and if they intersect with the obstacles
                if (True in detectCollision(start_i, end_i, box) and 
                    True in detectCollision(start_j, end_j, box)):
                    print(f"Self-collision detected between link {i} and link {j}.")
                    return True  # Collision found

        # No self-collisions detected
        return False

        
    
    def random_walk(self, q_current, goal, map_struct):
        """
        Generates a small random step in joint configuration space.
        """
        random_step = np.random.uniform(-0.05, 0.05, q_current.shape)
        q_random = q_current + random_step
        q_random = np.clip(q_random, self.lower, self.upper)
        
        # joint_positions_random, _ = self.fk.forward_expanded(q_random)
        
        if self.self_collision_check(q_random, goal, map_struct) == False:
            return q_random
        


    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from start to goal configuration.
        """
        q_path = [start]
        q_current = start.copy()

        # Ensure start and goal are within joint limits
        if not self.within_joint_limits(start) or not self.within_joint_limits(goal):
            print("Start or goal configuration out of joint limits.")
            return np.array([])
        
        if self.check_self_collision(q_current,map_struct):
            print("Self-collision detected, aborting plan.")
            return np.array([])

        step_count = 0

        while True:
            dist_to_goal = self.q_distance(goal, q_current)

            # Check if we are within tolerance of the goal to terminate
            if dist_to_goal < self.tol:
                q_path.append(goal)
                print("Goal reached within tolerance.")
                break
            elif step_count >= self.max_steps:
                print("Max steps reached; path planning terminated.")
                break

            # Dynamically adjust alpha to reduce overshooting near the goal
            adaptive_alpha = self.alpha * (dist_to_goal + 0.1)
            dq = self.compute_gradient(q_current, goal, map_struct)
            q_next = q_current + adaptive_alpha * dq
            q_next = np.clip(q_next, self.lower, self.upper)

            # Check for local minima and take a random step if gradient step is too small
            step_size = np.linalg.norm(q_next - q_current)
            if step_size < self.min_step_size:
                q_next = self.random_walk(q_current, goal, map_struct)
                q_next = np.clip(q_next, self.lower, self.upper)

            # Collision avoidance
            if self.self_collision_check(q_current, goal, map_struct):
                print("Collision detected; applying random walk to avoid obstacle.")
                q_next = self.random_walk(q_current, goal, map_struct)

            # Add q_next to path if no collision and move to next step
            q_path.append(q_next)
            q_current = q_next
            step_count += 1

        # Ensure the final configuration in q_path is exactly at goal if close enough
        if self.q_distance(q_path[-1], goal) > self.tol:
            print("Path did not converge exactly at goal; adding goal as final configuration.")
            q_path.append(goal)

        return np.array(q_path)

    

    
if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    planner = PotentialFieldPlanner()
    map_struct = loadmap("../maps/emptyMap.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:', i, ' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)


















# import numpy as np
# from math import pi, acos
# from scipy.linalg import null_space
# from copy import deepcopy
# from  lib.calculateFKJac import FK_Jac
# from  lib.detectCollision import detectCollision
# from  lib.loadmap import loadmap


# class PotentialFieldPlanner:

#     # JOINT LIMITS
#     lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
#     upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

#     center = lower + (upper - lower) / 2  # compute middle of range of motion of each joint
#     fk = FK_Jac()

#     def __init__(self, tol=1e-1, max_steps=5000, min_step_size=1e-3, alpha=0.1):
#         """
#         Constructs a potential field planner with solver parameters.
#         """
#         self.tol = tol
#         self.max_steps = max_steps
#         self.min_step_size = min_step_size
#         self.alpha = alpha  # Initialize the alpha value for gradient steps


#     def attractive_force(target, current):
#         """
#         Helper function for computing the attractive force.
#         """
#         e = 100 # this param should be greater than the repulsive for param , otherwise we arent going anywhere 
#         att_f = np.zeros((3, 1))
     
#         if np.linalg.norm(current - target) > 0.12:
#             # att_f = - (current - target)/ np.linalg.norm(current - target)
#             att_f = (current - target )/ np.linalg.norm(current - target)
            
#         else:
#             att_f = -0.5 * 30 * np.linalg.norm(current - target)**2

#         return att_f


#     @staticmethod
#     def repulsive_force(obstacle, current, unitvec=np.zeros((3, 1))):
#         """
#         Helper function for computing the repulsive force.
#         """
#         p0 = 0.25
#         eta = 50 #try smaller than 1 

#         # Ensure obstacle is correctly reshaped if it has 6 elements
#         if obstacle.size == 6:
#             obstacle = obstacle.reshape((6,))
#         else:
#             print("Skipping obstacle due to incorrect size:", obstacle)
#             return np.zeros(3)

#         rep_f = np.zeros(3)
#         current = current.reshape((1, 3))
#         dist, unit = PotentialFieldPlanner.dist_point2box(current, obstacle)

#         if dist < p0 and dist >0:
#             rep_f = - eta * ((1 / dist) - (1 / p0)) * (1 / dist**2) * unit.flatten()
            
#         # elif dist == 0:
#         #     rep_f = eta * np.random.randn(3,1)
#         #     rep_f /= np.linalg.norm(rep_f)
        
#         return rep_f
    
    
#     @staticmethod
#     def dist_point2box(p, box):
#         """
#         Helper function for the computation of repulsive forces.
#         """
#         boxMin = np.array([box[0], box[1], box[2]])
#         boxMax = np.array([box[3], box[4], box[5]])
#         boxCenter = boxMin * 0.5 + boxMax * 0.5
#         p = np.array(p)

#         dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
#         dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
#         dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

#         distances = np.vstack([dx, dy, dz]).T
#         dist = np.linalg.norm(distances, axis=1)

#         signs = np.sign(boxCenter - p)
#         unit = distances / dist[:, np.newaxis] * signs
#         unit[np.isnan(unit)] = 0
#         unit[np.isinf(unit)] = 0
#         return dist, unit

#     @staticmethod
#     def compute_forces(target, obstacle, current):
#         """
#         The base joint(world frame- 0th) isnt included in the computation. 
#         """
#         num_joints = 9
#         joint_forces = np.zeros((3, 9))

#         for i in range(num_joints):
#             # Compute attractive force for the i-th joint
            
#             if i ==0:
#                 att_f_each = np.zeros((3, 1))
                
#             else: 
#                 att_f_each = PotentialFieldPlanner.attractive_force(target[:3, i].reshape((3, 1)), current[:3, i].reshape((3, 1)))

#             # Initialize repulsive force to zero for the i-th joint
#             rep_f = np.zeros(3)  # Ensure rep_f is (3,)

#             # Calculate repulsive force for all obstacles
#             for obs in obstacle:
#                 try:
#                     obs_reshaped = obs.reshape((1, 6))  # Reshape each obstacle as needed
#                     # rep_f += PotentialFieldPlanner.repulsive_force(obs_reshaped, current[:3, i].reshape((3, 1))).flatten()
#                 except ValueError:
#                     print("Skipping invalid obstacle shape:", obs)

#             # Combine attractive and repulsive forces for the joint
#             joint_forces[:, i] = (att_f_each.flatten())

#         return joint_forces

#     @staticmethod
#     def compute_torques(joint_forces, q):
#         """
#         Here we include the base, or the 0th ,torque - starts from i +1 th joint and remove the end effector torque 
#         (2 virtual joints are included)
        
#         INPUTS:
#         joint_forces - 3x9 numpy array representing the force vectors on each 
#         joint/end effector
#         q - 1x7 numpy array representing the current joint angles

#         OUTPUTS:
#         joint_torques - 1x9 numpy array representing the torques on each joint    
         
#         """
#         fk = FK_Jac()    
#         joint_torques = np.zeros((1, 9))    
#         joint_positions, T0e = fk.forward_expanded(q)
        
#         # Initialize arrays for joint origins and z-axes in world frame
#         o_i = np.zeros((3, 9))
#         z_i = np.zeros((3, 9))

#         # Set joint positions and z-axes
#         joint_positions = joint_positions.reshape((3, 10))
#         for i in range(9):
#             o_i[:, i] = joint_positions[:, i]
            
            
#         z_i[:, 0] = [0, 0, 1]  # Base joint z-axis aligned with world z-axis
#         for i in range(8):
#             z_i[:, i+1] = T0e[i, 0:3, 2]  # Assign z-axis of each joint

#         # Calculate Jacobian-based torques for each joint
#         for i in range(1, 9):
#             # Calculate jv_i for joint i
#             jv_i = np.zeros((3, 9))
#             for j in range(i):
#                 jv_i[:, j] = np.cross(z_i[:, j], (o_i[:, i] - o_i[:, j]))

#             # Compute torque for joint i
#             joint_torques[0, i] = np.sum(jv_i[:, :9].T @ joint_forces[:, i])

#         return joint_torques

#     @staticmethod
#     def q_distance(target, current):
#         """
#         Helper function which computes the distance between any two vectors.
#         """
#         return np.linalg.norm(target - current)

#     @staticmethod
#     def compute_gradient(q, target, map_struct):
#         """
#         Computes the joint gradient step to move closer to the goal configuration.
#         """
#         dq = np.zeros((1, 7))
        
#         joint_positions, T = PotentialFieldPlanner.fk.forward_expanded(q)
#         target_positions, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        
#         joint_forces = PotentialFieldPlanner.compute_forces(target_positions.T, map_struct.obstacles, joint_positions.T)
#         joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        
#         tau_prime = joint_torques.squeeze()[:7]
        
#         # Adaptive alpha based on the distance to the goal
       
#           # Increase alpha as the goal gets farther

#         if np.linalg.norm(tau_prime) > 0:
#             dq =  (tau_prime / np.linalg.norm(tau_prime)) 
#         else:
#             dq = tau_prime

#         return dq
    
#     # check for these three collisions
#     # check self_collision 
#     # path collision - when its executing its path 
#     # and robot collision - check for collision with obstacle+6523869+
    


#     def within_joint_limits(self, position):
#         return np.all(position >= self.lower) and np.all(position <= self.upper)
    
        
        

#     # def self_collision_check(self, q, goal, map_struct):
#     #     """
#     #     Checks for self-collisions between non-adjacent links in the robot arm.
#     #     """
#     #     box = map_struct.obstacles  # Ensure obstacles are passed from the map structure
#     #     # Loop through pairs of non-adjacent links
        
#     #     # If no obstacles are present, skip collision checks
#     #     if isinstance(box, (list, np.ndarray)) and len(box) == 0:
#     #         return False

#     #     interpolate_percent = 0.02  # interpolation step percentage
#     #     num_steps = int(1 / interpolate_percent)

#     #     # Generate interpolated configurations from q_start to q_goal
#     #     configs = [q + (goal - q) * t for t in np.linspace(0, 1, num_steps + 1)]
            
        
#     #     for i in configs:
#     #         joint_positions, _ = self.fk.forward_expanded(q)  
            
#     #         for obs in box:
                
#     #             for j in range(len(joint_positions)-1):
                    
#     #                 # Detect collision between link i and link j using each obstacle
#     #                 if detectCollision(joint_positions[j].reshape((1,-1)), joint_positions[j+1].reshape((1,-1)), obs) == True:
#     #                     print(f"Self-collision detected between link {i} and link {j}.")
                        
#     #                     return True  # Self-collision detected
                    
#     #     return False  # No self-collision detected
    
    
#     def check_link_collisions(self, q, map_struct):
#         """
#         Checks if any link of the robot arm in a given configuration q collides with obstacles.
#         """
#         box = map_struct.obstacles  # Retrieve obstacles
        
#         if isinstance(box, (list, np.ndarray)) and len(box) == 0:
#             return False  # No obstacles to check against

#         # Get joint positions for the configuration
#         joint_positions, _ = self.fk.forward_expanded(q)
        
#         # Check each link for collision with each obstacle
#         for obs in box:
#             for j in range(len(joint_positions) - 1):
#                 # Detect collision between link j and j+1 with the obstacle
#                 if detectCollision(joint_positions[j].reshape((1, -1)), joint_positions[j + 1].reshape((1, -1)), obs)[0]:
#                     print(f"Collision detected between link {j} and obstacle.")
#                     return True  # Collision found with obstacle

#         return False  # No collision with obstacles


#     def check_path_collisions(self, q_start, q_goal, map_struct, step_size=0.02):
#         """
#         Checks if there are any collisions along the interpolated path from q_start to q_goal.
#         """
#         # Generate interpolated configurations from q_start to q_goal
#         num_steps = int(1 / step_size)
#         configs = [q_start + (q_goal - q_start) * t for t in np.linspace(0, 1, num_steps + 1)]
        
#         # Check each configuration for collisions along the path
#         for config in configs:
#             if self.check_link_collisions(config, map_struct):
#                 print("Collision detected along the path.")
#                 return True  # Collision found in the path

#         return False  # No collisions along the path


#     # def check_self_collision(self, q, map_struct):
#     #     """
#     #     Checks for self-collisions between non-adjacent links in the robot.
#     #     """
        
#     #     obstacles = map_struct.obstacles  # List of obstacles (boxes)
        
#     #     # Get the positions of each joint in the current configuration
#     #     joint_positions, _ = self.fk.forward_expanded(q)

#     #     # Define non-adjacent link pairs for self-collision checking
#     #     link_pairs = [
#     #         (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),  
#     #         (1, 3), (1, 4), (1, 5), (1, 6),
#     #         (2, 4), (2, 5), (2, 6),
#     #         (3, 5), (3, 6),
#     #         (4, 6)
#     #     ]

#     #     # Check each pair for self-collision
#     #     for (i, j) in link_pairs:
#     #         start_i = joint_positions[i].reshape(1, -1)
#     #         end_i = joint_positions[i + 1].reshape(1, -1)
#     #         start_j = joint_positions[j].reshape(1, -1)
#     #         end_j = joint_positions[j + 1].reshape(1, -1)
            
#     #         # Check collision with each obstacle (box)
#     #         for box in obstacles:
#     #             # Check if link i collides with link j and if they intersect with the obstacles
#     #             if (True in detectCollision(start_i, end_i, box) and 
#     #                 True in detectCollision(start_j, end_j, box)):
#     #                 print(f"Self-collision detected between link {i} and link {j}.")
#     #                 return True  # Collision found

#     #     # No self-collisions detected
#     #     return False

        
    
#     def random_walk(self, q_current, goal, map_struct):
#         """
#         Generates a small random step in joint configuration space that is collision-free.
#         """
#         max_attempts = 50  # Set a limit on the number of attempts to find a valid configuration

#         for _ in range(max_attempts):
#             # Generate a small random step in configuration space
#             random_step = np.random.uniform(-0.05, 0.05, q_current.shape)
#             q_random = q_current + random_step
#             q_random = np.clip(q_random, self.lower, self.upper)

#             # Check if the new configuration is collision-free
#             if not self.check_link_collisions(q_random, map_struct):
#                 # Optionally, check if the path from q_current to q_random is collision-free
#                 if not self.check_path_collisions(q_current, q_random, map_struct):
#                     return q_random

#         print("Unable to find a collision-free configuration after multiple attempts.")
#         return q_current  # Return the current configuration if no collision-free step is found



#     def plan(self, map_struct, start, goal, step_size= 0.1):
#         """
#         Uses potential field to move the Panda robot arm from start to goal configuration.
#         """
#         new_start = np.linspace(start, goal, num= 20)
#         # start_path = self.random_walk(start,goal,map_struct)
#         q_path = [start]
#         q_current = start.copy()

#         # Ensure start and goal are within joint limits
#         if not self.within_joint_limits(start) or not self.within_joint_limits(goal):
#             print("Start or goal configuration out of joint limits.")
#             return np.array([])

#         # # Check initial configuration for self-collisions or collisions with obstacles
#         # if self.check_link_collisions(q_current, map_struct):
#         #     print("Initial configuration is in collision, aborting plan.")
#         #     return np.array([])

#         for k in new_start:
#             if self.check_link_collisions(k ,map_struct) and self.check_path_collisions(q_current,k ,map_struct) and self.within_joint_limits(k) == True:
#                 q_path.append(k)
#                 q_current = k
#             else:
#                 break

#         while True:
        

#             if self.check_path_collisions(q_current,goal,map_struct) and self.within_joint_limits(goal):
#                 print("path to goal is feasible")
#                 q_path.append(goal)
#                 break
            
#             dist_to_goal = self.q_distance(goal, q_current)

          
#             # Dynamically adjust alpha to reduce overshooting near the goal
#             adaptive_alpha = self.alpha * (dist_to_goal )
#             dq = self.compute_gradient(q_current, goal, map_struct)
#             q_next = q_current + adaptive_alpha * dq
#             q_next = np.clip(q_next, self.lower, self.upper)
                
#             if self.q_distance(q_next,goal)< self.tol:
#                 q_path.append(goal)
#                 break
#             elif len(q_path)>= self.max_steps:
#                 break
            
#             # Check for collisions between the current and next configuration
#             if self.check_path_collisions(q_current, q_next, map_struct):
#                 print("Collision detected along the path; applying random walk to avoid obstacle.")
#                 q_next = self.random_walk(q_current, goal, map_struct)    

#             # # Check for local minima and take a random step if gradient step is too small
#             # step_size = np.linalg.norm(q_next - q_current)
#             if step_size < self.min_step_size:
#                 q_next = self.random_walk(q_current, goal, map_struct)
#                 q_next = np.clip(q_next, self.lower, self.upper)


#             step_size= np.linalg.norm(q_next - q_current)
#             if step_size< self.min_step_size:
#                 q_next = self.random_walk(q_current, goal, map_struct)


#             # # Add q_next to path if no collision and move to next step
#             q_path.append(q_next)
#             q_current = q_next

#         q_path =  np.array(q_path)
        
#         return np.array(q_path)

    
# if __name__ == "__main__":

#     np.set_printoptions(suppress=True, precision=5)

#     planner = PotentialFieldPlanner()
#     map_struct = loadmap("../maps/map5.txt")
#     start = np.array([0, -1, 0, -2, 0, 1.57, 0])
#     goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    
#     q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
#     for i in range(q_path.shape[0]):
#         error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
#         print('iteration:', i, ' q =', q_path[i, :], ' error={error}'.format(error=error))

#     print("q path: ", q_path)