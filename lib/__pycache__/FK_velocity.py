
import numpy as np 
from calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    dq = dq.reshape(-1, 1)

 
    velocity = np.zeros((6, 1))
    J = calcJacobian(q_in)
    velocity = np.zeros((6, 1))
    
    velocity = J @ dq
    

    return velocity
