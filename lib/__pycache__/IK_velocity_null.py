import numpy as np
from IK_velocity import IK_velocity
from calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null_vector = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))

    dq = IK_velocity(q_in,v_in,omega_in) # 1 x 7
    j= calcJacobian(q_in)  # 6 x 7

    desired_velocity = np.vstack((v_in, omega_in))

    # Find indices of non-NaN values (constrained velocities)
    valid_indices = ~np.isnan(desired_velocity).flatten()
    
    # Filter out NaN values from desired velocity and the corresponding rows in the Jacobian
    filtered_velocity = desired_velocity[valid_indices]
    j = j[valid_indices, :]
    
    # Solve for joint velocities using least squares
    #dq, residuals, rank, s = np.linalg.lstsq(j, filtered_velocity, rcond=None)
    

    J_star = np.linalg.pinv(j)
    null_vector = (np.eye(7) - (J_star @ j)) @ b  # 7 x 1
    
    null_vector =null_vector.reshape((1,7)) # 1 x 7

    return dq + null_vector
