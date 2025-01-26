import numpy as np
import sys
sys.path.append("/meam520_labs/lib")
# from lib.calculateFK import FK
from math import pi

def calcJacobian(q):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
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
    # print(np.round(H,5))

    jp = np.zeros((7,3))
    jp[0] = [0,0,0.141]
    # print(jp[0])

    
    jp[1]= (offset @ T[0])[0:3, 3]  
    # print(jp[1])
    
    # jp[2]= ((((offset @ T[0]) @ T[1])[0:3, 3]) + np.array([0,0,0.195])) 
    jp[2] = (((offset @ T[0]) @ T[1]) @ ([0, 0, 0.195, 1]))[:3] 
    # print(jp[2]) 

    
    jp[3]= ((offset @ T[0]) @ T[1]  @ T [2] )[0:3, 3] 
    # print(jp[3])

#     # jp[4]= (np.matmul((((offset @ T[0]) @ T[1]  @ T [2] @ T[3] )[0:3, 3]),np.array([0,0,0.125])))
    jp[4] = ((offset @ T[0] @ T[1]  @ T [2] @ T[3] ) @ ([0, 0,  0.125, 1]))[:3] 
    # print(np.round(jp[4],5) ) 


#     # jp[5]= ((((offset @ T[0]) @ T[1]  @ T [2] @ T[3] @ T[4] )[0:3, 3]) + np.array([0,0.015,0]))
    jp[5] = ((offset @ T[0] @ T[1]  @ T [2] @ T[3] @ T[4]) @ ([0, 0,  -0.015, 1]))[:3]
    # print(np.round(jp[5],5)) 

#     # jp[6]= ((((offset @ T[0]) @ T[1]  @ T [2] @ T[3] @ T[4] @ T[5])[0:3, 3]) + np.array([0,0,-0.051]))
#     # print(np.round(jp[6],5)) 

    jp[6] = ((offset @ T[0] @ T[1]  @ T [2] @ T[3] @ T[4] @ T[5]) @ ([0, 0,  0.051, 1]))[:3]
    # print(np.round(jp[6],5)) 

    transform_matrix = np.zeros((7, 4, 4))
    transform_matrix[0] = offset
    
    Jv = np.zeros((3, 7))
    Jw = np.zeros((3, 7))
    J = np.zeros((6, 7))
    
    o = np.zeros((3, 7)) 
    ee =  (((offset @ T[0]) @ T[1]  @ T [2] @ T[3] @ T[4] @ T[5] @ T[6])[0:3, 3]) #8th = end effector
    z = np. zeros((3,7))

    for i in range(7):    
        o[:,i] = np.reshape(jp[i],-1)
    # print("o is" ,np.round(o,5))

    z[:,0] = [0,0,1] 
    current_transform = offset
    for i in range(6):
        current_transform = current_transform @ T[i]
        z[:,i+1] =  current_transform[:3,2]

    # print(np.round(z))
    
    for i in range(7):
        Jv[:,i] = np.cross((z[:,i]),(ee - o[:,i]))
        Jw[:,i] =  z[:,i]
    
    J = np.vstack((Jv, Jw))

    # print("Jv:", np.round(Jv, 5))
    # print("Jw:", np.round(Jw, 5))
    # print("Jacobian matrix J:", np.round(J, 5))

    return J

 
if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
