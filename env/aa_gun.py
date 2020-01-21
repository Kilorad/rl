"""
Simple AA-gun environment. Constant plane speed, narrow gun-rotation space
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class planeClass(object):
    def __init__(self,x,y,vx,x_maneur,vy, dx_maneur):
        self.x=x
        self.y=y
        self.vx=vx
        self.x_maneur=x_maneur
        self.vy=vy
        self.dx_maneur=dx_maneur
    def act(self):
        self.x += self.vx
        if self.x>self.x_maneur and self.x<self.x_maneur+self.dx_maneur:
            self.y+=self.vy
            
class explosion(object):
    def __init__(self,x,y,env):
        self.x=x
        self.y=y
        self.t=0
        self.env=env
    def act(self):
        self.t += 1
        if self.t == 1:
            self.r=3
            self.color=(1,1,0)
        if self.t == 2:
            self.r=8
            self.color=(1,1,0)
        if self.t == 3:
            self.r=10
            self.color=(0.8,0.8,0)
        if self.t == 4:
            self.r=12
            self.color=(0.6,0.6,0)
        if self.t == 6:
            self.r=14
            self.color=(0.3,0.3,0)
        if self.t == 8:
            self.r=15
            self.color=(0.3,0.3,0.3)
        if self.t == 10:
            self.r=14
            self.color=(0.5,0.5,0.5)
        if self.t>=12:
            self.r=-1
        return
        
class projectile(object):
    def __init__(self,angle,start_speed,env):
        self.x=0
        self.y=0
        self.vy=start_speed*np.sin(angle*np.pi/180)
        self.vx=start_speed*np.cos(angle*np.pi/180)
        self.t=32
        self.env=env
        if env.long_projectiles:
            self.t*=2.5
    def act(self):
        #global reward
        #global hit
        #global g
        self.t -= 1
        self.vx *= 1 - self.env.k_fr
        self.vy *= 1 - self.env.k_fr
        self.vy -= self.env.g
        koef_fraction = 17
        self.r_previous=np.sqrt((self.env.plane.x - self.x)**2 + (self.env.plane.y - self.y)**2)
        for i in range(koef_fraction):
            self.x += self.vx/koef_fraction
            self.y += self.vy/koef_fraction
            if self.t<0:
                break
            if self.y<=0:
                self.t = 0
            r_curr=np.sqrt((self.env.plane.x - self.x)**2 + (self.env.plane.y - self.y)**2)
            reward=0
            if r_curr<10:
                self.env.expl_list.append(explosion(self.x,self.y,self.env))
                self.t = 0
                #точное попадание
                self.env.hit = 1
                print('hit')
                reward = 1
                self.env.reward += reward
                break
                    
            if r_curr>self.r_previous and r_curr<110:
                #удаляемся от цели, дистанционный подрыв
                self.env.expl_list.append(explosion(self.x,self.y,self.env))
                self.t = 0
                if r_curr<20:
                    reward = 0.5
                elif r_curr<35:
                    reward = 0.25
                elif r_curr<70:
                    reward = 0.125
                elif r_curr<110:
                    reward = 0.0625
                self.env.reward += reward
                break
            else:
                self.r_previous=r_curr

class cannon(object):
    def __init__(self,start_speed,env):
        self.cooldownmax=12
        self.cooldown=0
        self.angle=80#почти вертикально
        self.start_speed = start_speed
        self.env=env
    def wait(self):
        self.cooldown-=1
        if self.cooldown<=0:
            self.cooldown=0
    def shoot(self):
        if self.cooldown<=0:
            self.cooldown = self.cooldownmax
            self.env.projectiles_list.append(projectile(self.angle,self.start_speed,self.env))

class AA_gun_simple0_env(gym.Env):
    """
    Description:
        
    Source:
        
    Observation: 
        Type: Box(12)
        Num	Observation                 
        0	tm             
        1	proj_1_r             
        2	proj_1_dir                 
        3	proj_2_r     
        4   proj_2_dir
        5   cooldown
        6   angle
        7   plane_dir
        8   plane_r
        9   cw
        10  c_shoot
        11  reward   
    Actions:
        Type: Discrete(7)
        Num	Action
        0	turn + 0.5
        1	turn - 0.5
        2	turn + 2.5
        3	turn - 2.5
        4	turn + 7.5
        5	turn - 7.5
        6	shoot
    Reward:
        Reward 1 for 1-meter hit, 0.1 for 2-meter hit, 0.02 for 5-m hit
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Episode length is greater than 250
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 90
    }

    def __init__(self,fast_planes=False,long_projectiles=False,random_speed=False):
        high = np.array([
            250,
            500,
            180,
            500,
            180,
            10,
            180,
            180,
            500,
            10,
            1,
            2
        ])
        self.fast_planes=fast_planes
        self.random_speed=random_speed
        self.long_projectiles=long_projectiles
        
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.t = 0

        self.steps_beyond_done = None
        ########
        self.projectiles_list = []
        self.expl_list = []
        self.g = 0.1
        self.k_fr = 0.005
        self.cn_force = 15

        self.cn = cannon(self.cn_force,self)
        self.x_start = -170-np.random.rand()*100
        self.y_start = 5+np.random.rand()*130
        self.vx_start = 2
        if self.fast_planes:
            self.vx_start*=3.5
        self.x_maneur = self.x_start+np.random.rand()*220
        self.vy = 1.5
        if self.fast_planes:
            self.vy*=2
        self.dx_maneur = np.random.rand()*140
        
        self.random_speed=random_speed
        self.long_projectiles=long_projectiles
        
        if self.random_speed:
            mvx=1+(np.random.rand()-0.5)*0.4
            mvy=(np.random.rand()-0.5)*2.5
        else:
            mvx = 1
            mvy=(np.random.rand()-0.5)*2.5
        self.plane = planeClass(self.x_start,self.y_start,self.vx_start*mvx,self.x_maneur,self.vy*mvy,self.dx_maneur)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        cw = 0
        c_shoot = 0
        cw_lst=[0.5,-0.5,2.5,-2.5,7.5,-7.5,0]
        c_shoot_lst=[0,0,0,0,0,0,1]
        cw = cw_lst[action]
        c_shoot = c_shoot_lst[action]
        
        #self.state = (x,x_dot,theta,theta_dot)
        self.hit = 0
        self.reward=0
        #отходить всеми юнитами
        for i in range(len(self.projectiles_list)):
            self.projectiles_list[i].act()
        for i in range(len(self.expl_list)):
            self.expl_list[i].act()
        self.projectiles_list_new = []
        #отбрасываем сдохшие
        for i in range(len(self.projectiles_list)):
            if self.projectiles_list[i].t>0:
                self.projectiles_list_new.append(self.projectiles_list[i])
        self.projectiles_list = self.projectiles_list_new
        self.plane.act()
        self.cn.wait()
        #управление
        self.cn.angle+=cw
        if self.cn.angle>180:
            self.cn.angle-=180
        if self.cn.angle<-180:
            self.cn.angle+=180
        if c_shoot:
            self.cn.shoot()
            
        if self.plane.x>320 or self.hit==1:
            #перезапустить самолёт
            del self.plane           
            self.x_start = -170-np.random.rand()*100
            self.y_start = 5+np.random.rand()*330
            self.x_maneur = self.x_start+np.random.rand()*220
            self.dx_maneur = np.random.rand()*140
            if self.random_speed:
                mvx=1+(np.random.rand()-0.5)*0.4
                mvy=(np.random.rand()-0.5)*2.5
            else:
                mvx = 1
                mvy=(np.random.rand()-0.5)*2.5
            self.plane = planeClass(self.x_start,self.y_start,self.vx_start*mvx,self.x_maneur,self.vy*mvy,self.dx_maneur)
            #print('hit',hit)
        #фичи
        plane_dir = np.arctan2(self.plane.y,self.plane.x)*180/np.pi
        plane_r = np.sqrt(self.plane.y**2 + self.plane.x**2)
        dt = (self.cn.start_speed*np.cos(np.pi*self.cn.angle/180)-self.plane.vx)/self.plane.x#время долёта по x
        dx = dt*self.plane.vx
        dy = dt*self.plane.vy
        dy_gravity = dt*self.plane.vy + self.g*dt*dt/2 #с учётом смещения от гравитации. Самолёт как будто уносит гравитацией вверх - так мы компенсируем унос снаряда вниз
        dir_pred = np.arctan2((self.plane.y + dy_gravity),self.plane.x + dx)*180/np.pi
        cw_raw = dir_pred-self.cn.angle
        pred_angle = self.cn.angle + cw_raw
        if abs(pred_angle - dir_pred)<0.2:
            c_shoot_raw = 1
        else:
            c_shoot_raw = 0
        if len(self.projectiles_list)>=1:
            proj_1_dir = np.arctan2(self.projectiles_list[-1].y, self.projectiles_list[-1].x)*180/np.pi
            proj_1_r = np.sqrt(self.projectiles_list[-1].y**2 + self.projectiles_list[-1].x**2)
            proj_dist_1 = np.sqrt((self.projectiles_list[-1].y-self.plane.y)**2 + (self.projectiles_list[-1].x-self.plane.x)**2)
        else:
            proj_1_dir = 0
            proj_1_r = 0
            proj_dist_1 = 0

        if len(self.projectiles_list)>=2:
            proj_2_dir = np.arctan2(self.projectiles_list[-2].y, self.projectiles_list[-2].x)*180/np.pi
            proj_2_r = np.sqrt(self.projectiles_list[-2].y**2 + self.projectiles_list[-2].x**2)
            proj_dist_2 = np.sqrt((self.projectiles_list[-2].y-self.plane.y)**2 + (self.projectiles_list[-2].x-self.plane.x)**2)
        else:
            proj_2_dir = 0
            proj_2_r = 0
            proj_dist_2 = 0
        
        self.t += 1  
        self.state = (self.t,
                      proj_1_r-plane_r,
                      proj_1_dir-plane_dir,
                      proj_2_r-plane_r,
                      proj_2_dir-plane_dir,
                      self.cn.cooldown,
                      self.cn.angle,
                      plane_dir,
                      plane_r,
                      cw,
                      c_shoot,
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
        reward_scaled=self.reward*2-1
        reward_scaled=self.reward
        return np.array(self.state), reward_scaled, done, {}

    def reset(self):
        if not self.viewer is None:
            self.viewer.close()
            del self.viewer
            self.viewer = None
            
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=(12,))
        self.steps_beyond_done = None
        ########
        self.projectiles_list = []
        self.expl_list = []
        self.t = 0

        self.cn = cannon(self.cn_force,self)
        self.x_start = -170-np.random.rand()*100
        self.y_start = 5+np.random.rand()*200
        self.x_maneur = self.x_start+np.random.rand()*220
        self.dx_maneur = np.random.rand()*140
        if self.random_speed:
            mvx=1+(np.random.rand()-0.5)*0.4
            mvy=(np.random.rand()-0.5)*2.5
        else:
            mvx = 1
            mvy=(np.random.rand()-0.5)*2.5
        self.plane = planeClass(self.x_start,self.y_start,self.vx_start*mvx,self.x_maneur,self.vy*mvy,self.dx_maneur)
        
        self.close()
        
        return np.array(self.state)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        
        screen_width = 600
        screen_height = 400

        world_width = 400*2
        scale = screen_width/world_width

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
        clear_img = rendering.FilledPolygon([(0,0), (0,screen_height), (screen_width,screen_height), (screen_width,0)])
        clear_img.set_color(1.,1.,1.)
        self.viewer.add_geom(clear_img)  
        
        dx=7
        dy=3
        x0=self.plane.x*scale+dx+screen_width/2
        x1=self.plane.x*scale-dx+screen_width/2
        y0=self.plane.y*scale+dy
        y1=self.plane.y*scale-dy
        plane_img = rendering.FilledPolygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
        plane_img.set_color(.1,.0,.4)
        self.viewer.add_geom(plane_img)


        dx=2
        dy=2
        cannon_len=20
        x0=0+screen_width/2
        x1=cannon_len*np.cos(np.pi*self.cn.angle/180)+screen_width/2
        y0=0+dy
        y1=cannon_len*np.sin(np.pi*self.cn.angle/180)+dy
        cannon_img = rendering.FilledPolygon([(x0,y0), (x1,y1), (x1+dx,y1+dx), (x0+dx,y0+dx)])
        self.viewer.add_geom(cannon_img)
        for proj in self.projectiles_list:
            dx=3
            dy=3
            x0=proj.x*scale+dx+screen_width/2
            x1=proj.x*scale-dx+screen_width/2
            y0=proj.y*scale+dy
            y1=proj.y*scale-dy
            projectile_img = rendering.FilledPolygon([(x0,y0), (x1,y0), (x1,y1), (x0,y1)])
            self.viewer.add_geom(projectile_img)
        for expl in self.expl_list:
            if expl.r>0:
                expl_img=rendering.make_circle(expl.r)
                transformation=rendering.Transform(translation=(expl.x+screen_width/2, expl.y))
                expl_img.add_attr(transformation)
                expl_img.set_color(expl.color[0],expl.color[1],expl.color[2])
                self.viewer.add_geom(expl_img)
                #криво. Но удалять по-нормальному я их не буду
        if self.state is None: return None
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    def render2(self, mode='human'):
        screen_width = 600
        screen_height = 400
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
        #return None
        print(self.viewer)
        return self.viewer.render()

    def close(self):
        if not self.viewer is None:
            self.viewer.close()
            del self.viewer
            self.viewer = None