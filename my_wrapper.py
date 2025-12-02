import gym

# 跳帧包装器
class SkipFrameWrapper(gym.Wrapper):

    def __init__(self, env, skip):

        # 调用父类gym.Wrapper的初始化方法，绑定原始环境
        super().__init__(env)

        # 保存需要跳过的帧数
        self.skip = skip

    # 跳帧
    def step(self, action):

        # 初始化变量：最终返回的观测、总奖励、结束标志、信息
        obs, reward_total, done, info = None, 0, False, None
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            if done:
                break

        return obs, reward_total, done, info