import gym
from numpy import shape
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from test_obs import make_env

if __name__ == '__main__':

    vec_env = SubprocVecEnv([make_env for _ in range(1)])
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')
    # 加载预训练好的PPO模型（文件名为"ppo_mario.zip"），并绑定到当前环境
    model = PPO.load("best_model/best_model.zip", env=vec_env)

    # 重置环境，获取初始观测值（游戏初始画面的状态）
    obs = vec_env.reset()

    # 运行10000步游戏循环
    for i in range(10000):
        # 复制观测值（防止原观测数据被意外修改）
        obs = obs.copy()
        # 模型根据当前观测预测动作（deterministic=True表示使用确定性策略，输出最优动作）
        action, _ = model.predict(obs, deterministic=True)
        # 执行预测的动作，获取新的观测、奖励、是否结束标志、额外信息，返回下一帧
        obs, reward, done, info = vec_env.step(action)
        # 渲染游戏画面，实时显示游戏过程
        vec_env.render()
        # 如果游戏结束（如马里奥死亡或通关），重置环境重新开始
        if done:
          obs = vec_env.reset()