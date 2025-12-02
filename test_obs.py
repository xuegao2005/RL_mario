import gym
from numpy import shape
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from my_wrapper import SkipFrameWrapper

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
# 应用跳帧包装器（SkipFrameWrapper），每8帧执行一次动作并重复该动作，减少计算量，跳帧可以加快训练/运行速度，同时保留关键游戏状态变化
env = SkipFrameWrapper(env, skip=8)
# 压缩像素
env = ResizeObservation(env, shape(84, 84))

# 初始化游戏结束标志为True（表示需要重置环境）
done = True
# 循环执行4步动作（演示用）
for step in range(4):
    # 如果游戏结束（如马里奥死亡/通关），重置环境获取初始状态
    if done:
        state = env.reset()
    # 随机采样一个动作（env.action_space.sample()）并执行，获取新状态、奖励、结束标志、信息
    # 这里使用随机动作仅作演示，实际训练/测试时会用模型预测的动作
    state, reward, done, info = env.step(env.action_space.sample())
    # 使用matplotlib显示当前游戏帧（状态），以灰度图形式展示
    plt.imshow(state, cmap='gray')
    # 显示图像窗口
    plt.show()

# 关闭游戏环境，释放资源
env.close()