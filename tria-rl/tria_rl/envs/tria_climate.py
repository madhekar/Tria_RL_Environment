import gym
from gym import spaces
import numpy as np
import random

class TriaClimateEnv(gym.Env):
    '''
      scale between -1 and 1 
    '''
    metadata = {
                't_ini': -39.0, 'h_ini': 0.0, 'a_ini': 1999.0,

                't_min':-40.0, 'h_min':0.0,   'a_min':0.0,
                't_max':110,   'h_max':100.0, 'a_max':2000.0,
                'stat_rand_min':-1.0, 'stat_rand_max':1.0, 'equilibrium_cycles':250,
                'reward1': -0.5, 'reward2': -0.25, 'reward3': 10.0, 'nreward': -10.0,
                'weight_vec': [1,1,1,1,1], 'action_states' : 2,
                'range_dict': {
                            0 : [65.0, 80.0, 50.0, 85.0, 40.0, 90.0],
                            1 : [40.0, 60.0, 30.0, 70.0, 20.0, 80.0],
                            2 : [0.0, 80.0, 81.0, 300.0, 301.0, 800.0]
                            }           
                }

    def __init__(self, render_mode=None):
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.pre_state = [self.metadata['t_ini'], self.metadata['h_ini'], self.metadata['a_ini']]

        self.scale_range = [(self.metadata['t_min'],self.metadata['t_max']), (self.metadata['h_min'],self.metadata['h_max']),(self.metadata['h_min'],self.metadata['h_max'])]
        
        low = np.array([self.metadata['t_min'], self.metadata['h_min'], self.metadata['a_min']]).astype(np.float32)
        high = np.array([self.metadata['t_max'], self.metadata['h_max'], self.metadata['a_max']]).astype(np.float32)

        self.observation_space = spaces.Box(low, high, shape=(3,))

        # We have 2 actions, corresponding to "on", "off"
        self.action_space =  spaces.MultiDiscrete(np.array([self.metadata['action_states'], self.metadata['action_states'],
                                                            self.metadata['action_states'], self.metadata['action_states'],
                                                            self.metadata['action_states']]))

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        self.state = [self.metadata['t_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
                      self.metadata['h_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
                      self.metadata['a_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max'])
                      ]

        self.equilibrium_cycles = self.metadata['equilibrium_cycles']

    def _get_obs(self):
        return {"state": self.state, "sample": self.observation_space.sample()}

    def _get_info(self):
        return {
          "triaClimateEnv"
        }

    def reset(self, seed=None, options=None):

        self.state = [self.metadata['t_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
                      self.metadata['h_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max']),
                      self.metadata['a_ini'] + random.uniform(self.metadata['stat_rand_min'], self.metadata['stat_rand_max'])
                      ]

        self.equilibrium_cycles = self.metadata['equilibrium_cycles']

        info = self._get_info()

        return self.state

    def step(self, action):
        #print(action) 
        ap_scaled = [1 if e == 1 else -1 for e in action]  # 0 (off) => -1 and 1 (on) => 1

        actionPrime = [a * b for a, b in zip(ap_scaled, self.metadata['weight_vec'])]



        actionAlgo = [(actionPrime[a] - actionPrime[a + 3]) for a in range(len(actionPrime) // 2)]
        
        #print('ap_scaled: ', ap_scaled, 'actionPrime: ', actionPrime,'actionAlgo: ', actionAlgo)
       
        actionAlgo.append(actionPrime[len(actionPrime) // 2])

        #print('***',actionAlgo, self.state)

        #abs_diff = [ abs(ps-s) for ps , s in zip(self.pre_state, self.state) ]

        #actionAlgo = [ a * b for a,b in zip(actionAlgo, abs_diff)]

        self.state = [ a + b for a, b in zip(actionAlgo, self.state) ]

        #self.pre_state[::] = self.state[::]

        #print('&&&', actionAlgo, self.state)

        #reduce tria simulation length by 1 second
        self.equilibrium_cycles -= 1

        reward = [self.metadata['reward3'] if e >= self.metadata['range_dict'][i][0] and e<= self.metadata['range_dict'][i][1] else self.metadata['reward2'] if e >= self.metadata['range_dict'][i][2] and e<= self.metadata['range_dict'][i][3] else self.metadata['reward1'] if e >= self.metadata['range_dict'][i][4] and e <= self.metadata['range_dict'][i][5] else self.metadata['nreward'] for i, e in enumerate(self.state)]
        #reward = [r3 if e >= d1[i][0] and e <= d1[i][1] else nr3  for i, e in enumerate(self.state)]

        #self.state = [(-1 + (2.0 * ((v - x[0]) /(x[1] - x[0])))) for x,v in zip(self.scale_range, self.state)]
        print('reward:{} state:{} action: {} '.format(reward, self.state, actionPrime))

        reward = sum(reward)
        

        if self.equilibrium_cycles <= 0:
            terminated = True
        else:
            terminated = False

        info = {}
        #print('reward:{} state:{}'.format(reward, self.state))
        return self.state, reward, terminated,  info
    
    def scaleState(self):
        pass

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass
