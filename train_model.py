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
from stable_baselines3.common.callbacks import EvalCallback
from test_obs import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


def main():

    # 马里奥环境预处理
    env = make_env()

    # 多个子进程并行
    vec_env = SubprocVecEnv([lambda: make_env() for _ in range(8)])

    # 帧堆叠
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')

    # 创建评估回调函数，用于训练过程中定期评估模型性能并保存最优模型
    eval_callback = EvalCallback(vec_env, best_model_save_path="./best_model/",
                                 log_path="./callback_logs/", eval_freq=10000//8)

    model_params = {

        'learning_rate' : 1e-4,  # 学习率：模型学得多快，太大容易学乱，太小学得慢
        'ent_coef' : 0.1,  # 鼓励模型多尝试新动作的程度，越大越爱瞎试
        'clip_range': 0.15,  # 模型更新时的“刹车”，防止步子迈太大导致学崩
        'target_kl' : 0.15,  # 限制模型每次更新别太离谱，超过这个值就停更
        'n_epochs': 10,  # 同样的数据反复学10遍，学透点

        'gamma' : 0.97,  # 模型看长远奖励的程度，越接近1越重视以后的奖励
        'n_steps': 2048,  # 每个环境攒2048步数据再更新模型，内存不够就调小这个数
        'batch_size': 2048,  # 每次更新模型时一次喂多少数据，得是（环境数×n_steps）的倍数，不然显存扛不住

        # log
        'tensorboard_log' : r'logs', # tensorboard_log="logs"：将训练日志保存到logs目录，用于TensorBoard可视化
        'verbose' : 1, # verbose=1：打印训练过程中的日志信息
        'policy' : "CnnPolicy", # CnnPolicy：使用卷积神经网络策略（适合处理图像类输入，如游戏画面）CNN
    }



    # 初始化PPO模型
    # - env：绑定的训练环境
    # model = PPO(env = vec_env, **model_params)

    # 读取已有的权重，规定参数，第二轮训练，需要把上面一行注释才能运行！
    model = PPO.load('./best_model/best_model.zip', env = vec_env, **model_params)



    # 开始训练模型，总共训练1000个时间步（timesteps）
    model.learn(total_timesteps=10000000, callback=eval_callback)

    # 保存训练好的模型到文件"ppo_mario"（会生成ppo_mario.zip等文件）
    model.save("ppo_mario")

if __name__ == "__main__":
    main()