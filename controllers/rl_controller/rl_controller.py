import sys
from typing import List

from controller import Supervisor

import torch
import numpy as np
import gym
import gym.spaces
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
#from sac.main_sac import sac
import math
from typing import Callable

class myRLEnv(Supervisor, gym.Env):
    
    def __init__(self, reward_scale = 0.1, max_episode_steps=1000, torque=False):
        super().__init__()
        # Personal Preference
        self.use_torque = torque

        #Environment Specific
        self.__timestep = int(self.getBasicTimeStep())
        self.torque_multiplier = 1
        self.max_pos = 0
        reset_state = self.reset()
        
        # Open AI Gym generic
        self.action_space = self.create_action_space()
        self.observation_space = self.create_observation_space(reset_state)  
        self.reward_scale = reward_scale
        
        print('obs_shape',self.observation_space.shape)  # 29
        print(self.action_space.shape[0]) #8
        


    def reset(self):
        """
        1. Reset world
        2. Re_initialize Sensors
        3. return states
        """
        
        ######### Reset the world #########################
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)
        #################################################
        
        
        ##################SENSORS###########################
        # Get sensor names
        motor_names = [
            "front left shoulder abduction motor",  "front left shoulder rotation motor",  "front left elbow motor",
            "front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor",
            "rear left shoulder abduction motor",   "rear left shoulder rotation motor",   "rear left elbow motor",
            "rear right shoulder abduction motor",  "rear right shoulder rotation motor",  "rear right elbow motor"
            ]
        motor_names = [
            "front left shoulder rotation motor",  "front left elbow motor",
            "front right shoulder rotation motor", "front right elbow motor",
            "rear left shoulder rotation motor",   "rear left elbow motor",
            "rear right shoulder rotation motor",  "rear right elbow motor"
            ]
        gps_names = ["gps"]
        gyro_names = ["gyro"]
        inertial_names= ["inertial unit"]
        acc_name = ['accelerometer']
        touch_names = ['front left foot rubber touch', 'front right foot rubber touch', 'rear left foot rubber touch', 'rear right foot rubber touch']
        laser_names = ['distance sensor']
        
        # Initialize Sensors
        self.__motors = self.initialize_sensors(device_names=motor_names,
                                                enable_func_exist=False)
        self.gpss = self.initialize_sensors(gps_names)
        self.gyros = self.initialize_sensors(gyro_names)
        self.accelero = self.initialize_sensors(acc_name)
        self.inertials = self.initialize_sensors(inertial_names)
        self.pos_sensors = []
        self.touch_sensors = self.initialize_sensors(touch_names)
        self.laser_sensors = self.initialize_sensors(laser_names)
        
        self.max_motor_torques = []
        for i, motor in enumerate(self.__motors):
            if self.use_torque:
                motor.setTorque(0)
            else:
                motor.setPosition(float('inf'))
                motor.setVelocity(0)
                
            # if i in [1,3,5,7]:
            #     motor.enableTorqueFeedback(self.__timestep)
            self.max_motor_torques.append(motor.getAvailableTorque())
            pos_sensor = motor.getPositionSensor()
            pos_sensor.enable(self.__timestep)
            self.pos_sensors.append(pos_sensor)
        
        #############################################################
        # Open AI Gym generic
        # Return State
        state = self.get_states()
        self.prev_shaping = None

        #print('reset_state_len',len(state))
        return state
    
    
    def step(self,actions:np.ndarray):
        if len(actions) != self.action_space.shape[0]:
            actions = actions[0]
        
            
        # Execute the action
        for i, motor in enumerate(self.__motors):
            try:
                # vel = 0
                # force = motor.getMaxForce() * np.clip(actions[i].item(),-1,1)
                # motor.setForce(force)
                if self.use_torque:
                    tor = motor.getMaxTorque()* np.clip(actions[i].item(), -1, 1)
                    motor.setTorque(tor)
                #print(tor)
                else:
                    vel = motor.getMaxVelocity() * np.clip(actions[i].item(),-1,1)
                    motor.setVelocity(vel)
            except:
                print(f"Error found: {actions}")
                assert False, 'Hello'

        # Perform step
        super().step(self.__timestep)
        
        
        self.pos = self.get_pos()
        # Get observations , rewards, done
        self.state = self.get_states()
        self.done = self.get_done(self.state)
        self.reward = self.reward_function(self.state,actions,self.done)
        info = {}
        
        
        # TODO: state, reward, done, info
        return self.state, self.reward, self.done, info
    
    def create_action_space(self): 
        action_len = len(self.__motors)    

        action_space = gym.spaces.Box(
            low= np.array([-1.0 for motors in self.__motors]), 
            high = np.array([1.0 for motors in self.__motors]),
            shape=(action_len,),
            dtype=np.float32
        )
        return action_space
        
        
    def create_observation_space(self, reset_state):
        
        # spaces = {
        #     'position': gym.spaces.Box(low=np.inf ,high=np.inf, shape=(3,)),
        #     'velocity' : gym.spaces.Box(low=0, high= 2.0, shape=(3,)),
        #     'body_angle': gym.spaces.Box(low=-2.0, high= 2.0, shape=(3,)),
        #     'body_angular_vel' : gym.spaces.Box(low=-2.0, high= 2.0, shape=(3,))
        # }
        # dict_space = gym.spaces.Dict(spaces)
        
        body_ang_low = [-1.0 for i in range(3)]
        body_ang_high = [1.0 for i in range(3)]
        
        body_ang_vel_low = [-1.0 for i in range(3)]
        body_ang_vel_high = [1.0 for i in range(3)]
        
        vel_low = [-1.0 for i in range(3)]
        vel_high = [1.0 for i in range(3)]
        
        joint_angle_low = [-1.0  for i in self.pos_sensors]
        joint_angle_high = [1.0 for i in self.pos_sensors]
        
        joint_vel_low = [-1.0 for i in self.__motors]
        joint_vel_high = [1.0 for i in self.__motors]
        
        
        touch_sensors_low = [0.0 for i in self.touch_sensors]
        touch_sensors_high = [1.0 for i in self.touch_sensors]
        laser_sensor_low = [0.0 for i in self.laser_sensors]
        laser_sensor_high = [1.0 for i in self.laser_sensors]
        
        
        tmp_low  = [vel_low,  body_ang_low, body_ang_vel_low , joint_angle_low , joint_vel_low , touch_sensors_low , laser_sensor_low]
        tmp_high = [vel_high, body_ang_high,body_ang_vel_high, joint_angle_high, joint_vel_high, touch_sensors_high, laser_sensor_high]
        
        self.obs_space_low = np.array([item for subList in tmp_low for item in subList])
        self.obs_space_high = np.array([item for subList in tmp_high for item in subList])
        
        obs_space = gym.spaces.Box(
            low = self.obs_space_low,
            high = self.obs_space_high,
            dtype = np.float32,
            shape = reset_state.shape,
        )
        assert self.state_len == obs_space.shape[0], f"len of state isnt outputting {self.state_len }: len of state is {obs_space.shape[0]}"
        #print(obs_space)
        return obs_space
    
    def get_states(self):
        # 1. body velocity (3)
        # 2. Body angle (3)
        # 3. body angular vel (3)
        # 4. Joint angle (12)
        # 5. Joint angular vel(12)
        # 5  Contact on each leg True or false(4)
        
        #body_pos = self.get_pos()
        
        body_vel = self.get_vel()
        
        body_angle = self.get_body_angle()
        body_angular_vel = self.get_angular_Vel()
        
        joint_angle = self.get_joint_angle()
        joint_angular_vel,torque_feedback = self.get_motor_feedback()
        
        leg_contacts = self.get_contact_points()
        laser_dist = self.get_laser_distance()
        
        state = []
        
        state.extend(body_vel)
        state.extend(body_angle)
        state.extend(body_angular_vel)
        state.extend(joint_angle)
        state.extend(joint_angular_vel)
        state.extend(leg_contacts)
        state.extend(laser_dist)

        
        self.state_len = len(state)
        #print(state)
        return np.clip(np.array(state, dtype="float32"),-1,1)
    
    def get_done(self, state):
        def threshold(value,min_val,max_val):
            return value<min_val or value> max_val
        threshold_rotation = 0.5
        acc = 3
        #print(state[3], state[4], state[5])
        return bool(
            threshold(self.pos[0],-0.2, 15) or 
            threshold(self.pos[1],-0.35, 0.7) or
            threshold(self.pos[2], -1, 1) or
            threshold(state[3], -threshold_rotation, threshold_rotation) or
            # threshold(state[4], -threshold_rotation, threshold_rotation) or 
            threshold(state[5], 0, 1.0) 
            # threshold(state[6], -acc, acc) or
            # threshold(state[7], -acc, acc) or
            # threshold(state[8], -acc, acc) 
        )
    
    def reward_function(self, state, actions, done):
        shaping = float(3*self.pos[0])
        if self.prev_shaping is not None:
            forward_rew = shaping - self.prev_shaping
        else:
            forward_rew = shaping
        self.prev_shaping = shaping
        
        # Carrot
        forward_vel = float(15.0*state[0]) # Forward 
        timestep = float(self.__timestep/1000)
        
        # stick
        # Keep head parallel to the ground
        rotation_x = float(5.0* abs(state[3]))
        # Make sure robot goes in a straight line
        rotation_y = float(5.0* (abs(state[5] + 0.5)) ) 
        desired_height = float(5.0*abs(self.pos[1]))
        
        efforts = 0        
        for i,a in enumerate(actions):
            efforts += 0.06 * np.clip(np.abs(a), 0, 1)
        #print(efforts)
        #reward = 3*forward_vel + timestep - desired_height - actuator_effort
        reward = forward_vel + timestep - (rotation_x + rotation_y + desired_height + efforts) 

        if done:
            # if self.pos[0] >= self.max_pos:
            #     reward += 100
            # else:
            
            reward -= 200
            # Gift a hefty negative reward to speed up training. In reality what ended up was the robot ends up at a local optima after that time.
            #print(f'#######################\n##########################\
            #      \nforward {forward_rew:2f} \ntimestep {timestep:2f}\
            #      \nrotation_x {rotation_x:2f} \nrotation_y {rotation_y:2f} \
            #      \ndesired_height {desired_height:2f} \nefforts {efforts:2f}')
        return reward*self.reward_scale
        
    def initialize_sensors(self,device_names:list, enable_func_exist=True):
        sensors = []
        for i in range(len(device_names)):
            sensors.append(self.getDevice(device_names[i]))
            if enable_func_exist:
                sensors[i].enable(self.__timestep)
        return sensors
    
    
    def get_pos(self):
        # X Y Z
        return self.gpss[0].getValues()
    
    def get_vel(self):
        # Len = 6
        # [ vel  , angular]
        # [X Y Z , X Y Z  ]
        robot = self.getSelf()
        vel = robot.getVelocity()[0:3]
        vel_norm = [v/self.__timestep for v in vel]
        

        self.print_over_limit(vel_norm, 'vel_norm') 
        return vel_norm
    
    def get_body_angle(self):
        # Returns angles [x,y,z]
        angles = self.inertials[0].getRollPitchYaw()
        angles_norm = [angle/(math.pi) for angle in angles]
        
        
        self.print_over_limit(angles_norm, 'angles_norm') 
        return angles_norm
    
    def get_angular_Vel(self):
        # angular_velocity [x , y, z]
        angular_vel = self.gyros[0].getValues()
        angular_vel_norm = [rate/(math.pi*self.__timestep) for rate in angular_vel]
        
        
        self.print_over_limit(angular_vel_norm, 'angular_vel_norm') 
        return angular_vel_norm
    
    def get_joint_angle(self):
        # Returns Joint angle and Rate
        joint_angles_norm = []
        
        def del_addition(value,div):
            n = value//div
            return (value-(n*div))/div
        
        for pos_sensor in self.pos_sensors:
            joint_angles_norm.append( del_addition(pos_sensor.getValue(), math.pi) )
        
        
        self.print_over_limit(joint_angles_norm, 'joint_angles_norm')      
        return joint_angles_norm
    
    def get_motor_feedback(self):
        motor_vel= []
        torquefeedback = []
        #force_feedback = []
        for i,motor in enumerate(self.__motors):
            motor_vel.append(motor.getVelocity()/self.__timestep)
            # if i in [1,3,5,7]:
            #     torquefeedback.append(motor.getTorqueFeedback())

        self.print_over_limit(motor_vel, "motor_vel")
        return motor_vel, torquefeedback
    
    def get_contact_points(self):
        touchfeedback = []
        for touch in self.touch_sensors:
            touchfeedback.append(touch.getValue())
        self.print_over_limit(touchfeedback, 'touch_feedback')
        return touchfeedback
    
    def get_laser_distance(self):
        laser_dist = []
        for laser in self.laser_sensors:
            laser_dist.append(laser.getValue()/laser.getMaxValue())
        #print(laser_dist)
        self.print_over_limit(laser_dist,name='laser_dist', lower_limit=0, upper_limit=1.0)
        return laser_dist
    
    def print_over_limit(self, data:List, name:str, lower_limit = -1.0, upper_limit=1.0):
        if any([d<lower_limit or d>upper_limit for d in data]):
            #print(name)
            #assert False, f"{name},{data}"
            print(name)
    


def main():
    # SAC
    env = myRLEnv()
    check_env(env)
    
    # try high Tau > 0.5 , vs low tau < 0.1, High tau leads to instability
    # auto entropy coefficient: 1.0 means start with high randomness
    #
    
    # Custom actor architecture with two layers of 64 units each
    # Custom critic architecture with two layers of 400 and 300 units
    policy_kwargs = dict(net_arch=[256, 256, 256 ])
    model = SAC('MlpPolicy', env, verbose=1, device="cuda",
                batch_size=5120,buffer_size=510000, learning_starts=5000, 
                tau=0.001 ,gamma=0.99,learning_rate=0.0005,
                use_sde=True, sde_sample_freq=10,  ent_coef = 'auto',
                policy_kwargs = policy_kwargs,
                train_freq=1,target_update_interval=1
    )
    
    #model.collect_rollouts(replay_buffer=ReplayBuffer, learning_starts=10000)
    model.learn(total_timesteps=1_000_000, log_interval=4)
    model.save("sac_walking2")
    
    del model
    
    model = SAC.load("sac_walking2")
 
    obs = env.reset()
    for _ in range(100000):
        action = model.predict(obs, deterministic=False)
        obs,reward, done, info = env.step(action)
        if done:
            obs = env.reset()

from EnvRunner import EnvRunner   
            
def linear_schedule(initial_value: float,final_value = 0.00025) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, final_value)

    return func

from stable_baselines3.common.callbacks import CheckpointCallback
def main3():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = myRLEnv(reward_scale=1.0)
    env.seed(seed)
    save_name = 'spot_sac_vibrate_walking'
    
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    reward_scale = 0.01 
    nn_shape =  dict(net_arch = [1024,1024])
    
    
    model =  SAC(
        "MlpPolicy", env, policy_kwargs=nn_shape, verbose=1,
        learning_rate=linear_schedule(0.0001,0.00005),learning_starts=5000,
        batch_size = 256, buffer_size=1_000_000,
        action_noise=action_noise, 
        train_freq=1, target_update_interval=2,
        target_entropy=5e-3,
        tau = 0.005, gamma = 0.99, device = 'cuda',ent_coef=reward_scale
        )

    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/',
                                         name_prefix=save_name)
    
    model.learn(total_timesteps=2_000_000, log_interval= 100, callback=checkpoint_callback)
    model.save(save_name)
    del model
    
    
def eval(save_name:str):
    env = myRLEnv()
    model = SAC.load(save_name)
    print('loaded')
    obs = env.reset()
    while True:
        action, not_action = model.predict(obs, deterministic=False)
        obs,reward, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
            break



import sys
#import subprocess

#subprocess.run(['/full/path/to/venv/bin/python', 'path/to/script.py'])

if __name__ == '__main__':
    print(sys.executable)
    #main()
    #main3()
    eval('logs/successfulvibrate/spot_sac_vibrate_walking_1200000_steps')
    
    