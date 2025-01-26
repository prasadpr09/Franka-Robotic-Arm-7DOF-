
import numpy as np 
from calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """


    dq = np.zeros((1, 7))
    J = calcJacobian(q_in)

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    
    desired_velocity = np.vstack((v_in, omega_in))


    # Find indices of non-NaN values (constrained velocities)
    valid_indices = ~np.isnan(desired_velocity).flatten()
    
    # Filter out NaN values from desired velocity and the corresponding rows in the Jacobian
    filtered_velocity = desired_velocity[valid_indices]
    filtered_jacobian = J[valid_indices, :]
    
    # Solve for joint velocities using least squares
    dq, residuals, rank, s = np.linalg.lstsq(filtered_jacobian, filtered_velocity, rcond=None)
    
    # Ensure dq is a 1x7 vector instead of a (7, 1) array
    dq = dq.flatten()

    return dq