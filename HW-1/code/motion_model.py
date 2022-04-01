'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def WrapToPi(self,angle):
        return (angle - 2*np.pi * np.floor((angle + np.pi) / (2*np.pi)))
    
    def sample(self,mu,sigma):
        return np.random.normal(mu,sigma)
    
    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        
        if (u_t1[0] == u_t0[0]) and (u_t1[1] == u_t0[1]) and (u_t1[2] == u_t0[2]):
            # This implies that the state has not changed from time [t -> t+1 -> t+2], hence the robot is motionless
            x_t1 = x_t0
            return x_t1
        
        
        else:
            x_t1 = np.zeros(x_t0.shape)
            
            
            delta_R1 = self.WrapToPi(np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2])        # Desired Angle calculation (tan inverse of slope) - initial angle
            delta_T = np.sqrt((u_t1[0] - u_t0[0])**2 + (u_t1[1] - u_t0[1])**2)                          # Desired Distance to traverse (Distance Formula)
            delta_R2 = self.WrapToPi(u_t1[2] - u_t0[2] - delta_R1)
        
            
            R_1 = self.WrapToPi(delta_R1 - self.sample(0, self._alpha1 * delta_R1**2 + self._alpha2 * delta_T**2))
            T = delta_T - self.sample(0,self._alpha3 * delta_T**2 + self._alpha4 * delta_R1**2 + self._alpha4*delta_R2**2)
            R_2 = self.WrapToPi(delta_R2 - self.sample(0, self._alpha1 * delta_R2**2 + self._alpha2 * delta_T**2))
            

            # Updation of x_t1[]
            x_t1[0] = x_t0[0] + T * np.cos(x_t0[2] + R_1)
            x_t1[1] = x_t0[1] + T * np.sin(x_t0[2] + R_1)
            x_t1[2] = x_t0[2] + R_1 + R_2

            return x_t1
