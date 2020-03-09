import gym
from gym import core, spaces
from gym.utils import seeding
from numpy import cos, sin, pi
import numpy as np
import math

class RobotArm2D(gym.Env):
    
    def __init__(self):
        self.goal = np.array([1,1])
        self.obs_high = np.array([1,1,2,2,2*pi, 2*pi]) #뒤에 두개 맞나?, state:x1,y1,x2,y2,r1,r21
        self.obs_low = -self.obs_high
        self.action_high = np.array([2*pi/180, 2*pi/180])    #최대 d_theta : 2도
        self.action_low = -self.action_high
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)
        self.state = np.empty(6)
        self.a = self.state
        self.l = np.empty(1)
        self.r1 = 0
        self.r21 = 0
        self.arm1 = 1.    #로봇팔 길이
        self.arm2 = 1.
        self.xd, self.yd = self.goal
        self.reward = 0
        self.gamma = 0.9
        self.step = 0
        
        self.seed()
        
    def step(self, action):    #action : d_theta로 받음
        #action을 받아서 state, reward, done을 출력해야함
        self.done = False
        action = np.clip(action, self.action_low, self.action_high)
        self.x1, self.y1, self.x2, self.y2, r1, r21 = self.state
        self.r1 += action[0]
        self.r21 += action[1]
        self.r2 = self.r21 + self.r1
        self.x1, self.y1 = cos(self.r1)*self.arm1, sin(self.r1)*self.arm1
        self.x2, self.y2 = self.x1+ cos(self.r2)*self.arm2, self.y1+ sin(self.r2)*self.arm2    #state구하기
        
        self.new_state = np.array([self.x1, self.y1, self.x2, self.y2, self.r1, self.r21])
        self.state = self.new_state
        
        self.l = -math.sqrt((self.xd-self.x2)**2+(self.yd-self.y2)**2)    #reward 구하기
        if -self.l < 0.01: # 목표점을 반지름 0.1인 원으로 설정
            self.reward = 1.0
            self.done = True    #****결과가안나오는것은 done이 False이기 때문에 그런 것이다 action이 어떻게 입력되는지 보자.
        else:
            self.reward = self.l    #reward 구하기
        self.step += 1
            
        return self.state, self.reward, self.done, {}
    def reset(self):
        return np.array([1,0,2,0,0,0])
    
    def train(self):
        pass
    
    def _get_reward(self, l):
        if l < 0.1: # 목표점을 반지름 0.1인 원으로 설정
            self.reward = 1.
            self.done = True    #done 구하기
        else:
            self.reward = l    #reward 구하기
    
    def seed(self, seed = None):     #seed가 정확히 뭐지?
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self):
        print('step : {}'.format(self.step))
        print('(x1,y1) = ({},{})'.format(self.x1.astype(str), self.y1.astype(str)))
        print('(x2,y2) = ({},{})'.format(self.x2.astype(str), self.y2.astype(str)))        