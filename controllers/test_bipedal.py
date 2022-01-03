import gym
from stable_baselines3 import SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
#from stable_baselines3.her import her_replay_buffer
import numpy as np
from typing import Callable


def main():
    env = gym.make('BipedalWalker-v3')
    # eval_env = Monitor(env)
    # eval_callback = EvalCallback(eval_env, eval_freq=100,
    #                          deterministic=False, render=True)
    #vec_env = make_vec_env('BipedalWalker-v3',n_envs=20)
    action_noise= NormalActionNoise(np.array([0.1 for i in range(env.action_space.shape[0])]),
                                    np.array([0.1 for i in range(env.action_space.shape[0])]))
    policy_kwargs = {'net_arch': [256, 256]}
    model = SAC('MlpPolicy', env, verbose=1, device="cuda",
                batch_size=1024,buffer_size=5000000, learning_starts=5000, 
                tau=0.01 ,gamma=0.99,learning_rate=0.0003,
                use_sde=True, sde_sample_freq=10,  ent_coef = 'auto',
                policy_kwargs = policy_kwargs,
                target_update_interval=4
    )
    
    #model.collect_rollouts(replay_buffer=ReplayBuffer, learning_starts=10000)
    model.learn(total_timesteps=500000, log_interval=4)
    model.save("sac_walking")

    # del model

    model = SAC.load("sac_walking")

    obs = env.reset()
    for _ in range(100000):
        action, not_action = model.predict(obs, deterministic=False)
        obs,reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            
            
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(progress_remaining * initial_value, 0.00025)

    return func

def main2():
    model_save_name = 'bipedal_walker_ddpg'
    env = gym.make('BipedalWalker-v3')
    #vec_env = make_vec_env('BipedalWalker-v3',n_envs=20)
    #vec_env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3")])
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    
    nn_shape =  dict(net_arch = [128, 128, 128])
    model =  DDPG(
        "MlpPolicy", env, policy_kwargs=nn_shape, verbose=1,
        learning_rate=linear_schedule(0.001),learning_starts=5000,
        batch_size= 2560, buffer_size=5_000_000,
        action_noise=action_noise, train_freq=1,
        tau = 0.001, gamma = 0.99, device = 'cuda'
        )
    
    
    model.learn(total_timesteps=1_500_000,log_interval=100)
    model.save(model_save_name)

    del model
    model = DDPG.load(model_save_name)
    
    
    obs = env.reset()
    while True:
        action, not_action = model.predict(obs, deterministic=False)
        obs,reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            break
        
    
if __name__ == '__main__':
    main2()