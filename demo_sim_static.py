# -*- coding: utf-8 -*-


# 1.环境实例化
from path_plan_env import StaticPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(StaticPathPlanning())


# 2.策略加载
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_static.onnx")


# 3.仿真LOOP
from copy import deepcopy

MAX_EPISODE = 20
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    obs = env.reset()
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        env.render()
        # 决策
        obs = obs.reshape(1, *obs.shape)                      # (*shape, ) -> (1, *shape, )
        act = policy.run(['action'], {'observation': obs})[0] # return [action, ...]
        act = act.flatten()
        print(act)# (1, dim, ) -> (dim, )
        # 仿真
        next_obs, _, _, info = env.step(act)
        # 回合结束
        if info["terminal"]:
            print('回合: ', episode,'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
#end for




#             ⠰⢷⢿⠄
#         ⠀⠀⠀⠀⠀⣼⣷⣄
#         ⠀⠀⣤⣿⣇⣿⣿⣧⣿⡄
#         ⢴⠾⠋⠀⠀⠻⣿⣷⣿⣿⡀
#         🏀   ⢀⣿⣿⡿⢿⠈⣿
#          ⠀⠀⢠⣿⡿⠁⢠⣿⡊⠀⠙
#          ⠀⠀⢿⣿⠀⠀⠹⣿
#           ⠀⠀⠹⣷⡀⠀⣿⡄
#            ⠀⣀⣼⣿⠀⢈⣧ 
#
#       你。。。干。。。嘛。。。
#       哈哈。。唉哟。。。