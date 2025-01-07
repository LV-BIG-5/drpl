# -*- coding: utf-8 -*-


# 1.环境实例化
from path_plan_env import DynamicPathPlanning
env = DynamicPathPlanning(800) # 动作空间本身就是 -1,1


# 2.策略加载
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_dynamic.onnx")


# 3.仿真LOOP
from copy import deepcopy

MAX_EPISODE = 5
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    obs = env.reset()
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        env.render()
        # 决策
        seq_points = obs['seq_points'].reshape(1, *obs['seq_points'].shape) # (1, seq_len, *points_shape, )
        seq_vector = obs['seq_vector'].reshape(1, *obs['seq_vector'].shape) # (1, seq_len, vector_dim, )
        act = policy.run(['action'], {'seq_points': seq_points, 'seq_vector': seq_vector})[0] # return [action, ...]
        act = act.flatten()                                                                   # (1, dim, ) -> (dim, )
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
