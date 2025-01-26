import numpy as np
from math import pi

class FK():
    
    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        pass

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """
        q = q.squeeze()
        # z_off = [0.141, 0.0, 0.195, 0.0, 0.125, -0.015, 0.051, 0.0]
        T = np.zeros((7, 4, 4))
        H = np.zeros((4,4))
        A = [0, 0 , 0.0825, 0.0825, 0, 0.088, 0] #distance a
        a = [-pi/2, pi/2 , pi/2, pi/2, -pi/2, pi/2, 0]  # alpha
        d = [0.192, 0, 0.121+0.195, 0, 0.125+0.259, 0, 0.159+0.051]  #d
        
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
        # print(np.round(H,5))
  
        jp = np.zeros((8,3))
        jp[0] = [0,0,0.141]
        # print(jp[0])
        
     
        jp[1]= (offset @ T[0])[0:3, 3]  
        # print(jp[1])
        
        # jp[2]= ((((offset @ T[0]) @ T[1])[0:3, 3]) + np.array([0,0,0.195])) 
        # print(jp[2]) 
        jp[2] = (((offset @ T[0]) @ T[1]) @ ([0, 0, 0.195, 1]))[:3] 
        
        jp[3]= ((offset @ T[0]) @ T[1]  @ T [2] )[0:3, 3] 
    #     # print(jp[3])

    #     # jp[4]= (np.matmul((((offset @ T[0]) @ T[1]  @ T [2] @ T[3] )[0:3, 3]),np.array([0,0,0.125])))
    #     # print(np.round(jp[4],5) ) 
        jp[4] = ((offset @ T[0] @ T[1]  @ T [2] @ T[3] ) @ ([0, 0,  0.125, 1]))[:3] 

  
    #     # jp[5]= ((((offset @ T[0]) @ T[1]  @ T [2] @ T[3] @ T[4] )[0:3, 3]) + np.array([0,0.015,0]))
    #     # print(np.round(jp[5],5)) 
        jp[5] = ((offset @ T[0] @ T[1]  @ T [2] @ T[3] @ T[4]) @ ([0, 0,  -0.015, 1]))[:3]
    #     # jp[6]= ((((offset @ T[0]) @ T[1]  @ T [2] @ T[3] @ T[4] @ T[5])[0:3, 3]) + np.array([0,0,-0.051]))
    #     # print(np.round(jp[6],5)) 
    
        jp[6] = ((offset @ T[0] @ T[1]  @ T [2] @ T[3] @ T[4] @ T[5]) @ ([0, 0,  0.051, 1]))[:3]
        
        jp[7]= (((offset @ T[0]) @ T[1]  @ T [2] @ T[3] @ T[4] @ T[5] @ T[6])[0:3, 3])
    #     # print(np.round(jp[7],5))
    
        return jp.round(4), H
    

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,0,0,pi/2,pi/4])
    
    # T0e = np.zeros((7, 4, 4))

    joint_positions,H = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",H)