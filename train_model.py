# 导入OpenAI Gym库，用于创建强化学习环境
import gym
from gym.wrappers import GrayScaleObservation
# 从numpy导入shape（此处代码未实际使用，保留原导入）
from numpy import shape
# 从Stable Baselines3导入PPO算法（近端策略优化）
from stable_baselines3 import PPO
# 导入NES游戏手柄输入包装器，用于处理游戏动作空间
from nes_py.wrappers import JoypadSpace
# 导入超级马里奥兄弟的Gym环境
import gym_super_mario_bros
# 导入简化版的马里奥动作空间（减少动作数量，降低训练复杂度）
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from test_obs import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


def main():

    # 马里奥环境预处理
    env = make_env()

    # 多个子进程并行
    vec_env = SubprocVecEnv([make_env for _ in range(8)])

    # 帧堆叠
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')

    # 初始化PPO模型
    # 参数说明：
    # - CnnPolicy：使用卷积神经网络策略（适合处理图像类输入，如游戏画面）CNN
    # - env：绑定的训练环境
    # - verbose=1：打印训练过程中的日志信息
    # - tensorboard_log="logs"：将训练日志保存到logs目录，用于TensorBoard可视化
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs",
    n_steps = 2048,
    batch_size = 2048,
    )

    # 开始训练模型，总共训练1000个时间步（timesteps）
    model.learn(total_timesteps=10000000)

    # 保存训练好的模型到文件"ppo_mario"（会生成ppo_mario.zip等文件）
    model.save("ppo_mario")

if __name__ == "__main__":
    main()