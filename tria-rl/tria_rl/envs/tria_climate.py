import gym
from gym import spaces
from gym.error import DependencyNotInstalled
import numpy as np
import random
import math
from typing import Optional, Union

class TriaClimateEnv(gym.Env):
    '''
      scale between -1 and 1 
    '''
    metadata = {
                'render_modes': ['human', 'rgb_array'],
                'render_fps': 50,
                'test' : 0,

                # initial Values for observation space
                't_ini': 55, 'h_ini': 50, 'a_ini': 2000,
                
                # minimum and maximum values for observation space
                't_min':0, 'h_min':0,   'a_min':0,
                't_max':120,   'h_max':100, 'a_max':2000,

                # random abbration setting and episode length
                'stat_rand_min':-1, 'stat_rand_max':1, 'equilibrium_cycles':100,

                # rewards definitions
                'reward1': -5, 'reward2': -1, 'reward3': 20, 'nreward': -10,

                # action weights and action status
                'weight_vec': [.3, .3, .5, .3, .3], 
                'weight_vector': [1,1,1],
                'action_states' : 19,
                
                #range for reward computation
                'reward_calc_range' : [[65,80], [40,60], [0,200]],

                # reward decision constants
                'range_dict': {
                            0 : [65, 80, 50, 85, 40, 90],
                            1 : [40, 60, 30, 70, 20, 80],
                            2 : [0, 200, 201, 500, 501, 800]
                            },    

                'actions' : {
                            0 : np.array([-.1,-.1,.1]),
                            1 : np.array([-.1,-.5,.1]),
                            2 : np.array([-.5,-.1,.1]),
                            3 : np.array([-.5,-.5,.1]),
                            4 : np.array([-.1,-.1,-.5]),
                            5 : np.array([-.1,-.5,.1]),
                            6 : np.array([.5,.5,.1]),
                            7 : np.array([.5,-.1,-.5]),
                            8 : np.array([.5,-.5,.1]),
                            9 : np.array([-.1,.5,-.5]),
                            10 : np.array([-.5,.5,.1]),
                            11 : np.array([.5,.5,-.5]),                   
                            12 : np.array([-.5,.5,-.5]),
                            13 : np.array([-.5,-.5,-.5])                                                                                      
                        }                  
                }

    def __init__(self, render_mode: Optional[str] = None):

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        #self.pre_state = [self.metadata['t_ini'], self.metadata['h_ini'], self.metadata['a_ini']]
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None

        self.scale_range = [(self.metadata['t_min'], self.metadata['t_max']), (self.metadata['h_min'],self.metadata['h_max']),(self.metadata['h_min'],self.metadata['h_max'])]
        
        low = np.array(
            [
                self.metadata['t_min'], 
                self.metadata['h_min'], 
                self.metadata['a_min']
            ], 
            dtype=np.float32,
        )
        high = np.array(
            [
                self.metadata['t_max'], 
                self.metadata['h_max'], 
                self.metadata['a_max']
                ],
                dtype=np.float32,
        )

        self.observation_space = spaces.Box(low, high) #, shape=(3,), dtype=np.float32)

        # We have 2 actions, corresponding to "on", "off"
        '''
        self.action_space = spaces.Tuple( 
            [
                    spaces.Discrete(2), 
                    spaces.Discrete(2),
                    spaces.Discrete(2), 
                    spaces.Discrete(2), 
                    spaces.Discrete(2) 
            ]
                    )
        
        self.action_space = spaces.MultiDiscrete(
            [
                self.metadata['action_states'], 
                self.metadata['action_states'], 
                self.metadata['action_states'], 
                self.metadata['action_states'], 
                self.metadata['action_states']
            ]
        )
        '''
        #self.action_space = gym.spaces.MultiBinary(n=5)
        #self.action_space = tuple((spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2)))
        #a_low = np.array([0, 0, 0, 0, 0])#.astype(np.int32)
        #a_high = np.array([1, 1, 1, 1, 1])#.astype(np.int32)    
        #self.action_space = spaces.Box(a_low, a_high, shape=(5,), dtype=np.int32)


        self.action_space_meta = [[-.1,-.1,.1],[-.1,-.7,.1],[-.7,-.1,.1],[-.7,-.7,.1],[-.1,-.1,-.7],[-.1,-.7,-.7],[-.7,-.1,-.7],
                            [-.7,-.7,-.7],[-.1,.7,.1],[-.7,.7,.1], [-.1,.7,-.7],
                            [-.7,.7,-.7], [.7,-.1,.1], [.7,-.7,.1],[.7,-.1,-.7],
                            [.7,-.7,-.7],[.7,.7,.1],[.7,.7,-.7],[0,0,0]]

        self.action_space = gym.spaces.Discrete(19)


        self.mean = [self.metadata['range_dict'][0][0] + self.metadata['range_dict'][0][1] // 2,
                     self.metadata['range_dict'][1][0] + self.metadata['range_dict'][1][1] // 2,
                     self.metadata['range_dict'][2][0] + self.metadata['range_dict'][2][1] // 2
                     ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        #self.state = [self.metadata['t_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
        #              self.metadata['h_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
        #              self.metadata['a_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max'])
        #              ]
        #self.state =[self.metadata['t_ini'],self.metadata['h_ini'], self.metadata['a_ini']]
        self.state = [np.random.randint(self.metadata['t_min'], self.metadata['t_max']),
                      np.random.randint(self.metadata['h_min'], self.metadata['h_max']),
                      np.random.randint(self.metadata['a_min'], self.metadata['a_max'])]

        self.equilibrium_cycles = self.metadata['equilibrium_cycles']

    def _get_obs(self):
        return {"state": self.state, "sample": self.observation_space.sample()}

    def _get_info(self):
        return {
          "triaClimateEnv"
        }
    
    def c_reward(self, rList, cNum):
        if rList[0] <= cNum <= rList[1]:
           rDist = 100
        else:     
           rNear = min(rList, key=lambda x:abs(x-cNum))
           rDist = abs(rNear - cNum) * -0.05
        return rDist 

    def reset(self, seed=1234, options=None):

        #super().reset()
        #self.state = [self.metadata['t_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
        #              self.metadata['h_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
        #              self.metadata['a_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max'])
        #              ]
        #self.state =[self.metadata['t_ini'],self.metadata['h_ini'], self.metadata['a_ini']]

        #self.state = [np.random.randint(self.metadata['t_min'] + 20, self.metadata['t_max'] - 20),
        #              np.random.randint(self.metadata['h_min'] + 20, self.metadata['h_max'] - 20),
        #              np.random.randint(self.metadata['a_min'] + 200, self.metadata['a_max'] - 1000)]
        
        if self.metadata['test'] == 0:
          self.state = [np.random.randint(self.metadata['t_min'], self.metadata['t_max']),
                      np.random.randint(self.metadata['h_min'], self.metadata['h_max']),
                      np.random.randint(self.metadata['a_min'], self.metadata['a_max'])]
        else:
            self.state = random.choice([[65,40,0], [80,60,200]])#random.choice([[65,40,0], [80,60,200], [50,30,201], [85,70,500], [40,20,501],[90,80,800]])#random.choice([[81, 61, 201],[-50,0,0],[120,100,2000],[70,50,100 ]])#[81, 61, 201] #
        self.equilibrium_cycles = self.metadata['equilibrium_cycles']

        info = self._get_info()
        #print('reset: ',self.state, ' : ', self.equilibrium_cycles)
        return self.state

    def step(self, action):
        #print('>>>>',action) 
        ap_scaled =self.action_space_meta[action] #[1 if e == 1 else -1 for e in action]  -- 0 (off) => -1 and 1 (on) => 1

        actionPrime = [a * b for a, b in zip(ap_scaled, self.metadata['weight_vector'])]

        ##actionAlgo = [(actionPrime[a] - actionPrime[a + 3]) for a in range(len(actionPrime) // 2)]
        
        #print('ap_scaled: ', ap_scaled, 'actionPrime: ', actionPrime,'actionAlgo: ', actionAlgo)
       
        ##actionAlgo.append(actionPrime[len(actionPrime) // 2])

        #print('***',actionAlgo, self.state)

        #abs_diff = [ abs(ps-s) for ps , s in zip(self.pre_state, self.state) ]

        #actionAlgo = [ a * b for a,b in zip(actionAlgo, abs_diff)]

        self.state = [round(a + b, 1) for a, b in zip(actionPrime, self.state) ]

        #self.pre_state[::] = self.state[::]

        #print('&&&', actionAlgo, self.state)

        #reduce tria simulation length by 1 second
        self.equilibrium_cycles -= 1

        #reward = [self.metadata['reward3'] if e >= self.metadata['range_dict'][i][0] and e<= self.metadata['range_dict'][i][1] else self.metadata['reward2'] if e >= self.metadata['range_dict'][i][2] and e<= self.metadata['range_dict'][i][3] else self.metadata['reward1'] if e >= self.metadata['range_dict'][i][4] and e <= self.metadata['range_dict'][i][5] else (((self.mean[i] + abs(self.state[i])) * 0.5 * -1) if self.state[i] < self.mean[i] else ((self.mean[i] - self.state[i]) * 0.5)) for i, e in enumerate(self.state)]
        #reward = [self.metadata['reward3'] if e >= self.metadata['range_dict'][i][0] and e<= self.metadata['range_dict'][i][1] else self.metadata['reward2'] if e >= self.metadata['range_dict'][i][2] and e<= self.metadata['range_dict'][i][3] else self.metadata['reward1'] if e >= self.metadata['range_dict'][i][4] and e <= self.metadata['range_dict'][i][5] else self.metadata['nreward'] for i, e in enumerate(self.state)]
        reward = [ self.c_reward(a,b) for a,b in zip(self.metadata['reward_calc_range'], self.state)]
        #reward = [r3 if e >= d1[i][0] and e <= d1[i][1] else nr3  for i, e in enumerate(self.state)]

        #add some abbrations remove it to make it more deterministic
        #st_random = np.random.uniform(self.metadata['stat_rand_min'],self.metadata['stat_rand_max'],3) 

        #st_random = np.random.randint(-1,1,size=3)
        
        #add some abbrations remove it to make it more deterministic
        #self.state += st_random

        #self.state = [(-1 + (2.0 * ((v - x[0]) /(x[1] - x[0])))) for x,v in zip(self.scale_range, self.state)]
        #print('reward:{} state:{} action: {} '.format(reward, self.state, actionPrime))

        reward = round(sum(reward), 1)
        

        if self.equilibrium_cycles <= 0:
            terminated = True
        else:
            terminated = False

        info = {}
        #print('reward:{} state:{} action:{} prime:{}'.format(reward, self.state, action, actionPrime))
        return self.state, reward, terminated,  info
    
    def scaleState(self):
        pass

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed
