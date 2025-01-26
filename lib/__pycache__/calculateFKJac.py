import numpy as np
from math import pi
# from calculateFK import FK

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        pass

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """
        z_off = [0.141, 0.0, 0.195, 0.0, 0.125, -0.015, 0.051, 0.0]
        q = q.squeeze()
        
        T = np.zeros((7, 4, 4))
        H = np.zeros((4,4))
        
        #distance a
        A = [0, 0 , 0.0825, 0.0825, 0, 0.088, 0] 
        # alpha
        a = [-pi/2, pi/2 , pi/2, pi/2, -pi/2, pi/2, 0]  
        # distance d
        d = [0.192, 0, 0.121+0.195, 0, 0.125+0.259, 0, 0.159+0.051]  
        #angles q
        q = q.astype(np.float64)
        q[3]=  pi + q[3]
        q[5] = q[5] - pi        
        q[6] = q[6] - (pi/4)
        

        for i in range(7):
            T[i] = np.array([[ np.cos(q[i]), - np.sin(q[i])*np.cos(a[i]) , np.sin(q[i])*np.sin(a[i]) , A[i]*np.cos(q[i])], 
                            [ np.sin(q[i]),    np.cos(q[i])*np.cos(a[i]) ,- np.cos(q[i])*np.sin(a[i]), A[i]*np.sin(q[i])],
                            [ 0           ,          np.sin(a[i])       ,        np.cos(a[i])       ,         d[i]     ],
                            [     0       ,                0            ,              0            ,         1        ]]) 
            
            
        offset = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0.141],
                            [0, 0, 0, 1]]) 
            
            
        # print(np.round(T[0],4))            
        H = offset @ T[0] @ T[1] @ T [2] @ T[3] @ T[4] @ T[5] @ T[6]        
    
        transform_matrix = np.zeros((7, 4, 4))
        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))
        
        
        transform_matrix[0] = offset
        jp = np.zeros((8, 3))
        jp[0] = [0,0,0.141]
        
        T0e[0] = offset
        jointPositions[0]= jp[0]

        
        for i in range(7):
            T[i] = np.array([
                [np.cos(q[i]), -np.sin(q[i])*np.cos(a[i]), np.sin(q[i])*np.sin(a[i]), A[i]*np.cos(q[i])],
                [np.sin(q[i]), np.cos(q[i])*np.cos(a[i]), -np.cos(q[i])*np.sin(a[i]), A[i]*np.sin(q[i])],
                [0, np.sin(a[i]), np.cos(a[i]), d[i]],
                [0, 0, 0, 1]
             
            ])
            
            T0e[i+1]= T[i]
    
        
        current_transform = offset
        for i in range(7):
            current_transform = current_transform @ T[i]
            jp[i + 1] = (current_transform @ ([0, 0, z_off[i + 1], 1]))[:3]
            jointPositions[i+1] = jp[i+1]
        
        # the last two elements in the transformation matrix is the same as the end effector
        T0e[8] = T0e[7]
        T0e[9] = T0e[7]
        
        # joint position changes for each         
        # Compute the positions of the last two virtual joints
        
        # is this correct ??????????????????????
        # jp1 = (T0e[8] @ [0, -0.1, 0.159, 1])[:3]  # 9th element
        # jp2 = (T0e[8] @ [0, 0.1, 0.159, 1])[:3]   # 10th element
        
        jp1 = (T0e[8] @ [0, -0.1, -0.549, 1])[:3]  # 9th element
        jp2 = (T0e[8] @ [0, 0.2, 0, 1])[:3]   # 10th element

        # Assign the computed positions to jointPositions
        jointPositions[8] = jp1
        jointPositions[9] = jp2


        # print("jp is", jp.round(5))
        
        
        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return()
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",joint_positions.round(5))
    print("End Effector Pose:\n",T0e.round(5))
