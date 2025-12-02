# 导入OpenAI Gym库，用于创建强化学习环境
import gym
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

# 创建SuperMarioBros-v2版本的游戏环境
env = gym_super_mario_bros.make('SuperMarioBros-v2')
# 使用JoypadSpace包装环境，限制动作空间为SIMPLE_MOVEMENT（仅包含基本移动/跳跃动作）
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 初始化PPO模型
# 参数说明：
# - CnnPolicy：使用卷积神经网络策略（适合处理图像类输入，如游戏画面）
# - env：绑定的训练环境
# - verbose=1：打印训练过程中的日志信息
# - tensorboard_log="logs"：将训练日志保存到logs目录，用于TensorBoard可视化
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs")

# 开始训练模型，总共训练1000个时间步（timesteps）
model.learn(total_timesteps=1000)

# 保存训练好的模型到文件"ppo_mario"（会生成ppo_mario.zip等文件）
model.save("ppo_mario")