'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2
        
        
        # User defined occupancy map data
        self.OMap = occupancy_map
        self.OMapSize = np.size(occupancy_map)
        self.resolution = 10
        
        print("Occupancy Map size : \n", np.shape(self.OMap))
        
    def __get_prob(self,z_star,z_t):
        # Hit
        if 0 <= z_t <= self._max_range:
            p_Hit = (np.exp(-1 / 2 * (z_t - z_star) ** 2 / (self._sigma_hit ** 2))) / (np.sqrt(2 * np.pi * self._sigma_hit ** 2))
        else:
            p_Hit = 0
            
        # Short
        if 0 <= z_t <= z_star:
            eta = 1.0/(1-np.exp(-self._lambda_short*z_star))
            pShort = eta * self._lambda_short * np.exp(-self._lambda_short * z_t)
        else: 
            p_Short = 0
        
        # Max
        if z_t >= self._max_range:
            p_Max = 1
        else:
            p_Max = 0
            
        # Rand
        if 0 <= z_t < self._max_range:
            p_Rand = 1 / self._max_range
        else:
            p_Rand = 0
        
        
        #p_total = self._z_hit * p_Hit + self._z_short * p_Short + self._z_max * p_Max + self._z_rand * p_Rand
        #p_total /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        
        
        return p_total, p_Hit, p_Short, p_Max, p_Rand
    
    
    
    def WrapToPi(self,angle):
        return (angle - 2*np.pi * np.floor((angle + np.pi) / (2*np.pi)))

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 1
        
        z_t = [z_t1_arr[n] for n in range(0, 180, self._subsampling)]

        z_t_star = self.__ray_cast(x_t1)

        probs = np.zeros(self._subsampling)
        for i in range(self._subsampling):
            p[i], pHit, pShort, pMax, pRand = self.getProbability(z_t_star[i], z_t[i])
            q += np.log(probs[i])

        prob_zt1 = self._subsampling / np.abs(q)

        return prob_zt1
