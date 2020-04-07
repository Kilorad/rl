"""
Jet table environment. Ice and table with 4 engines and 4 mini-locators. Obstacles on ice
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class jet_table_env(gym.Env):
    """
    Description:
        
    Source:
        
    Observation: 
        Type: Box(10)
        Num	Observation                 
        0	tm             
        1	vx             
        2	vy                 
        3	tx_rel     
        4	ty_rel   
        5	left_r
        6	right_r
        7	up_r
        8	down_r
        9	reward   
    Actions:
        Type: Discrete(4)
        Num	Action
        0	vx++
        1	vx--
        2	vy++
        3	vy--
    Reward:
        Reward = 0-range
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Episode length is greater than 250
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 110
    }

    def __init__(self):
        high = np.array([
            300,
            1,
            1,
            5,
            5,
            1,
            2,
            2,
            2,
            5
        ])
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        

        self.steps_beyond_done = None
        ########

        self.k_fr = 0.05
        self.dv = 0.05
        self.t = 0

        self.make_map()
        
    def make_map(self):
        self.table = {'x':16.,'y':16.,'vx':0.,'vy':0.}
        self.target = {'x':1.,'y':1.}
        self.obstacle_map = np.ones((18,18))
        self.obstacle_map[1:self.obstacle_map.shape[0]-1,1:self.obstacle_map.shape[1]-1]=0
        for i in range(7):
            orientation = np.random.rand()<0.5
            start = int(np.random.rand()*(self.obstacle_map.shape[0]-2))#1я координата
            length = int(np.random.rand()*(self.obstacle_map.shape[0]-2)*0.6)
            if start+length>=self.obstacle_map.shape[0]:
                length=self.obstacle_map.shape[0]-1-start
            start2 = int(np.random.rand()*(self.obstacle_map.shape[0]-2))#2я координата
            if (orientation):
                self.obstacle_map[start:start+length,start2:start2+1] = 1
            else:
                self.obstacle_map[start2:start2+1, start:start+length] = 1
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.reward=0
        
        #self.target['x']+=0.01
        #self.target['y']+=0.01
        
        
        rng1 = np.sqrt(((self.table['x']-self.target['x'])**2 + (self.table['y']-self.target['y'])**2 ))
        self.table['vx']*=1-self.k_fr
        self.table['vy']*=1-self.k_fr
        
        if action==0:
            self.table['vx']+=self.dv
        if action==1:
            self.table['vx']-=self.dv
        if action==2:
            self.table['vy']+=self.dv
        if action==3:
            self.table['vy']-=self.dv
        
        k_frac=10
        for i in range(k_frac):
            self.table['x']+=self.table['vx']/k_frac
            self.table['y']+=self.table['vy']/k_frac
            #координаты цели в сетке obstacle_map
            x_bmp = int(self.table['x'])
            y_bmp = int(self.table['y'])
            if self.obstacle_map[x_bmp,y_bmp]:
                hit=True
            else:
                hit=False
            if hit:
                self.table['x']-=self.table['vx']/k_frac
                self.table['y']-=self.table['vy']/k_frac
                self.table['vx']=0
                self.table['vy']=0
        
        x_bmp = int(self.table['x'])
        y_bmp = int(self.table['y'])
        #работа локатора
        l_where = []
        max_where = []
        l_where.append(np.where(self.obstacle_map[:x_bmp,y_bmp])[0])
        max_where.append(x_bmp)
        
        l_where.append(np.where(self.obstacle_map[x_bmp:,y_bmp])[0])
        max_where.append(self.obstacle_map.shape[0] - x_bmp)
        
        l_where.append(np.where(self.obstacle_map[x_bmp,:y_bmp])[0])
        max_where.append(y_bmp)
        
        l_where.append(np.where(self.obstacle_map[x_bmp,y_bmp:])[0])
        max_where.append(self.obstacle_map.shape[1] - y_bmp)
        
        radar = {}
        radar_names = ['left','right','down','up']
        for i in range(len(l_where)):
            l = l_where[i]
            max_where_curr = max_where[i]
            r_value=max_where_curr - l[-1]
            radar[radar_names[i]] = r_value
        ##
        rng2 = np.sqrt(((self.table['x']-self.target['x'])**2 + (self.table['y']-self.target['y'])**2 ))
        self.reward = rng1-rng2
        
        self.t += 1  
        self.state = (self.t,
                      self.table['vx'],
                      self.table['vy'],
                      self.target['x']-self.table['x'],
                      self.target['y']-self.table['y'],
                      radar['left'],
                      radar['right'],
                      radar['up'],
                      radar['down'],
                      self.reward)
        ############
        done = bool(self.t>=250)
        if not done:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            pass
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
        #Костыль. Чтобы распределение ревордов было примерно от -1 до 1
        return np.array(self.state), self.reward, done, {}

    def reset(self):
        if not self.viewer is None:
            self.viewer.close()
            del self.viewer
            self.viewer = None
            
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=(10,))
        self.steps_beyond_done = None
        ########

        self.k_fr = 0.05
        self.dv = 0.05
        self.t = 0

        self.make_map()
        self.close()
        
        return np.array(self.state)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        
        screen_width = 500
        screen_height = 500

        world_width = self.obstacle_map.shape[0]
        scale = screen_width/world_width

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.sq=[]
            #render map
            dx=screen_width/self.obstacle_map.shape[0]
            dy=screen_height/self.obstacle_map.shape[1]

            for x in range(self.obstacle_map.shape[0]):
                for y in range(self.obstacle_map.shape[1]):
                    if self.obstacle_map[x,y]:
                        x0=x*scale
                        x1=x*scale+dx
                        y0=y*scale
                        y1=y*scale+dy
                        sq = rendering.FilledPolygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
                        sq.set_color(.0,.0,.0)
                        self.sq.append(sq)
                        self.viewer.add_geom(sq)
            

            #render target
            dx=0.3*screen_width/self.obstacle_map.shape[0]
            dy=0.3*screen_height/self.obstacle_map.shape[1]
            x0=0
            x1=0+dx
            y0=0
            y1=0+dy
            sq_trg = rendering.FilledPolygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
            sq_trg.set_color(.0,.99,.0)
            self.sq_trg_trans = rendering.Transform()
            sq_trg.add_attr(self.sq_trg_trans)
            self.viewer.add_geom(sq_trg)
            
            #render table
            dx=0.3*screen_width/self.obstacle_map.shape[0]
            dy=0.3*screen_height/self.obstacle_map.shape[1]
            x0=0
            x1=0+dx
            y0=0
            y1=0+dy
            sq_table = rendering.FilledPolygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
            sq_table.set_color(.0,.0,.99)
            self.sq_table_trans = rendering.Transform()
            sq_table.add_attr(self.sq_table_trans)
            self.viewer.add_geom(sq_table)
            
        
        #clear_img = rendering.FilledPolygon([(0,0), (0,screen_height), (screen_width,screen_height), (screen_width,0)])
        #clear_img.set_color(1.,1.,1.)
        #self.viewer.add_geom(clear_img)   
        
        #обновить позицию цели
        dx=0.4*screen_width/self.obstacle_map.shape[0]
        dy=0.4*screen_height/self.obstacle_map.shape[1]
        x0=self.target["x"]*scale
        x1=self.target["x"]*scale+dx
        y0=self.target["y"]*scale
        y1=self.target["y"]*scale+dy
        self.sq_trg_trans.set_translation(x0, y0)
        
        #обновить позицию table
        dx=0.4*screen_width/self.obstacle_map.shape[0]
        dy=0.4*screen_height/self.obstacle_map.shape[1]
        x0=self.table["x"]*scale
        x1=self.table["x"]*scale+dx
        y0=self.table["y"]*scale
        y1=self.table["y"]*scale+dy
        self.sq_table_trans.set_translation(x0, y0)
        
        if self.state is None: return None
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if not self.viewer is None:
            self.viewer.close()
            del self.viewer
            self.viewer = None