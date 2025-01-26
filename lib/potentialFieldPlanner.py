import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = lower + (upper - lower) / 2  # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-2, max_steps=500, min_step_size=1e-3, alpha=0.1):
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
        e = 0.4
        att_f = np.zeros((3, 1))
        d = np.linalg.norm(target - current)
        
        if d < 0.3:
            att_f = -e * (target - current)
        else:
            att_f = (- e * (target - current)) / d

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3, 1))):
        """
        Helper function for computing the repulsive force.
        """
        p0 = 2
        eta = 0.1

        # Ensure obstacle is correctly reshaped if it has 6 elements
        if obstacle.size == 6:
            obstacle = obstacle.reshape((6,))
        else:
            print("Skipping obstacle due to incorrect size:", obstacle)
            return np.zeros(3)

        rep_f = np.zeros(3)
        current = current.reshape((1, 3))
        dist, unit = PotentialFieldPlanner.dist_point2box(current, obstacle)

        if dist < p0:
            rep_f = eta * ((1 / dist) - (1 / p0)) * (1 / dist**2) * unit.flatten()

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
        dist_to_goal = PotentialFieldPlanner.q_distance(q, target)
        alpha = 0.2 * (dist_to_goal + 0.2)  # Increase alpha as the goal gets farther

        if np.linalg.norm(tau_prime) > 0:
            dq = alpha * tau_prime / np.linalg.norm(tau_prime)
        else:
            dq = tau_prime

        return dq

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from start to goal configuration.
        """
        q_path = np.array([start])
        q_current = start.copy()

        for step in range(self.max_steps):
            dq = self.compute_gradient(q_current, goal, map_struct)
            q_next = q_current + dq
            q_next = np.clip(q_next, self.lower, self.upper)

            # Check for NaN values in q_next and reset if necessary
            if np.isnan(q_next).any():
                print("NaN detected in configuration, resetting step.")
                q_next = q_current.copy() + np.random.uniform(-0.05, 0.05, q_next.shape)
                q_next = np.clip(q_next, self.lower, self.upper)

            # Get joint positions for the next step
            joint_positions, _ = self.fk.forward_expanded(q_next)

            # Collision check with recovery attempts
            max_recovery_attempts = 3  # Reduced number of recovery attempts
            collision = True  # Assume collision until confirmed clear

            # Attempt to find a collision-free path with smaller perturbations if needed
            recovery_attempt = 0
            while collision and recovery_attempt < max_recovery_attempts:
                collision = any(
                    detectCollision(
                        np.array([joint_positions[i]]),    # Start of each link
                        np.array([joint_positions[i + 1]]), # End of each link
                        obstacle
                    )
                    for obstacle in map_struct.obstacles
                    for i in range(len(joint_positions) - 1)
                )

                if collision:
                    print(f"Collision detected at step {step}, attempt {recovery_attempt + 1}. Applying random perturbation.")
                    # Apply a random perturbation to `q_next` to try moving away from the obstacle
                    q_next += np.random.uniform(-0.05, 0.05, q_next.shape)
                    q_next = np.clip(q_next, self.lower, self.upper)
                    joint_positions, _ = self.fk.forward_expanded(q_next)
                    recovery_attempt += 1

            # Update the path and configuration if no collision remains
            q_path = np.vstack((q_path, q_next))
            q_current = q_next

            # Check if the goal has been reached or if the gradient step size is too small
            dist_to_goal = self.q_distance(goal, q_current)
            if dist_to_goal < self.tol:
                print(f"Converged to goal within tolerance after {step + 1} steps.")
                break
            elif np.linalg.norm(dq) < self.min_step_size:
                print(f"Step size below minimum threshold after {step + 1} steps.")
                break

        # Ensure final position is added if near the goal but not within tolerance
        if self.q_distance(goal, q_path[-1]) > self.tol:
            q_path = np.vstack((q_path, goal))

        return q_path

    
    
if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    planner = PotentialFieldPlanner()
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:', i, ' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)