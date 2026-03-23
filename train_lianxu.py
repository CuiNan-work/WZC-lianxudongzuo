"""
增加更新频次，使其策略可以更新的更快

n_steps=1024      # 收集1024步后才更新一次
batch_size=128    # 每批处理128个样本
n_epochs=10       # 每次更新用同一批数据训练10个epoch

"""

from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

# 自动检测系统中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 环境参数设置
num_uavs = 3  # 无人机数量
num_users = 20  # GT数量
uav_H = 50  # 无人机高度(m)
ris_pos = np.array([0.0, 0.0, 30.0])  # RIS位置 (x, y, h)
ris_M = 64  # RIS反射单元数

lam = 0.1  # 波长 (m) (3GHz)
d_elem = lam / 2
PL0 = 1e-6  # 参考路径损耗
alpha = 2  # 路径损耗指数（增大以加强信道衰减，使通信时延更显著）

B_total = 20e6  # 总带宽 (Hz) 20MHz
P = 0.1  # GT发射功率(W)
P_n_dBm = -100  # 噪声功率 (dBm)
P_n_W = 10 ** ((P_n_dBm - 30) / 10.0)  # 将dBm转换为W

L_min = 0.7  # 最小计算任务量(Mbits)
L_max = 1.0  # 最大计算任务量(Mbits)
C = 800  # 每比特CPU周期数

F_max = 22000  # 无人机最大计算资源(MHz)
F_local = 1000  # GT计算能力(MHz)

P_uav = 0.2  # UAV下行发射功率(W)，用于回传结果
delta = 0.4  # 回传数据比例（计算结果大小 / 原始任务大小）

max_speed = 5  # 无人机最大飞行速度(m/step)
max_steps = 2048  # 每个episode最大步数
d_safe = 5.0  # 安全距离约束
penalty_const = 0.2  # 安全惩罚项

# 阻塞概率参数
blk_a = 0.05
blk_b = 0.005

# 奖励权重
w_time = 0.6
w_fair = 0.4


class UAVEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(UAVEnv, self).__init__()
        self.render_mode = render_mode

        # 无人机初始位置(固定)
        self.uav_positions = np.array([[200, 200], [-200, -200], [200, -200]])

        # GT初始位置(随机分布)
        self.user_positions = (np.random.rand(num_users, 2) - 0.5) * 400

        # GT计算任务
        self.user_tasks = np.random.uniform(L_min, L_max, num_users)

        # GT决策(0表示本地计算，[1,2,3]表示GT关联)
        self.user_decisions = np.random.randint(0, 4, num_users)

        # 无人机负载
        self.uav_L = [0] * num_uavs

        # UAV卸载速率
        self.uav_unload_rate = [0]

        # RIS 初始相移
        # self.ris_phase = np.random.uniform(0, 2 * np.pi, ris_M).astype(np.float32)
        self.ris_phase = np.zeros(ris_M, dtype=np.float32)  # 固定为全 0 相移

        # 通信时延
        self.users_comm_delay = [0] * num_users
        # 回传时延（UAV计算完成后将结果返回GT）
        self.users_return_delay = [0] * num_users
        # 计算时延(包括无人机和本地计算)
        self.users_comp_delay = [0] * num_users
        # 总时延
        self.total_time = 0

        # 无人机负负载均衡指标
        self.Jain_step = 0
        self.Jain_step_history = []  # step的Jain历史数据
        self.Jain_episode = 0
        self.Jain_episode_history = []  # episode的Jain历史数据

        # 预估归一化时延
        self.min_delay_theoretical = 0
        self.max_delay_theoretical = 0

        # 归一化时延
        self.normalized_delay = 0
        # 归一化Jain
        self.normalized_Jain = 0

        # 奖励
        self.reward = 0
        self.reward_history = []  # step奖励历史数据
        self.episode_reward = 0  # 每一步episode奖励
        self.episode_reward_history = []  # episode的奖励历史数据

        # 步数计数器
        self.step_count = 0

        # 无人机之间距离
        self.UAV_distance = []
        # UAV-GT之间距离
        self.UAV_GT = []
        # UAV-RIS之间距离
        self.UAV_RIS = []
        # RIS-GT之间距离
        self.RIS_GT = []

        # RIS 相移等级
        self.ris_phase_levels = 16

        # 动作空间：连续动作
        # 每架 UAV 有两个连续值 [speed_i, angle_i]（均在 [-1,1] 归一化）
        # 后接 num_users 个 GT 卸载决策（[-1,1] → 离散化为 0~3）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * num_uavs + num_users,),
            dtype=np.float32
        )

        # 状态空间
        # uav_positions(6) + user_positions(40) + user_tasks(20) + uav_loads(3) = 69
        # 所有特征均归一化：UAV/GT位置∈[-1,1]，任务量∈[-1,1]，负载∈[0,1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(69,), dtype=np.float32
        )

        # 预估最大最小时延
        self.compute_normalization_bounds()

    # 环境重置
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        super().reset(seed=seed)

        self.uav_positions = np.array([[200, 200], [-200, -200], [200, -200]])  # UAV位置
        self.user_positions = (np.random.rand(num_users, 2) - 0.5) * 400  # GT位置
        self.user_tasks = np.random.uniform(L_min, L_max, num_users)  # GT任务
        self.user_decisions = np.random.randint(0, 4, num_users)  # GT决策
        # self.ris_phase = np.random.uniform(0, 2 * np.pi, ris_M).astype(np.float32) # RIS相移
        self.ris_phase = np.zeros(ris_M, dtype=np.float32)

        self.uav_L = [0] * num_uavs  # 无人机负载

        self.users_comm_delay = [0] * num_users  # 通信时延
        self.users_return_delay = [0] * num_users  # 回传时延
        self.users_comp_delay = [0] * num_users  # 计算时延(包括无人机和本地计算)
        self.total_time = 0  # 总时延

        self.step_count = 0  # 重置步数计数器

        self.reward = 0  # 重置奖励
        self.episode_reward = 0  # 重置episode奖励

        self.Jain_step = 0  # 重置Jain
        self.Jain_episode = 0

        self.normalized_delay = 0  # 归一化时延
        self.normalized_Jain = 0  # 归一化Jain

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):

        self.user_tasks = np.random.uniform(L_min, L_max, num_users)  # GT任务

        idx = 0
        uav_raw = action[idx:idx + 2 * num_uavs]  # 每架UAV两个连续值 [speed, angle]
        idx += 2 * num_uavs
        gt_raw = action[idx:idx + num_users]  # GT卸载决策（连续值）

        # UAV速度：[-1,1] → [0, max_speed]
        uav_speeds = (uav_raw[0::2] + 1.0) / 2.0 * max_speed
        # UAV偏向角：[-1,1] → [-π, π]
        uav_angles = uav_raw[1::2] * np.pi

        # GT卸载决策：[-1,1] 均匀映射到 {0,1,2,3}，各占 0.5 宽度
        # 先将 gt_raw 限幅到 [−1, 1−ε) 避免 floor(4.0)=4 越界，再 clip 保险
        self.user_decisions = np.clip(
            np.floor((np.clip(gt_raw, -1.0, 1.0 - 1e-6) + 1.0) / 2.0 * 4).astype(int), 0, 3
        )

        # UAV移动——更新的UAV位置
        self.uav_move(uav_speeds, uav_angles)

        # 计算UAV负载
        self.compute_uav_load()

        # 计算卸载速率
        self.compute_unload_rate()

        # 计算卸载时延
        self.comm_delay()

        # 为每个 GT 记录实际使用的卸载速率
        user_unload_rates = np.zeros(num_users, dtype=np.float32)
        unload_rate_matrix = self.compute_unload_rate()
        for k in range(num_users):
            uav_id = self.user_decisions[k]
            if uav_id == 0:
                user_unload_rates[k] = 0.0
            else:
                user_unload_rates[k] = unload_rate_matrix[uav_id - 1, k]

        # 任务计算时延
        self.comp_delay()

        # 回传时延
        self.return_delay()

        # 计算总时延
        self.compute_total_delay()

        # 计算Jain
        self.compute_Jain()

        # 归一化
        self.normalize_delay()
        self.normalize_Jain()

        # 计算step奖励
        self.compute_step_reward()

        # 构造 observation
        obs = self._get_obs()

        self.step_count += 1

        # 判断是否结束
        done = self.step_count >= max_steps
        truncated = False  # 可选：提前终止条件

        # 累积 episode 奖励
        self.episode_reward += self.reward

        # 累积 Jain
        self.Jain_episode += self.Jain_step

        # 如果 episode 结束，记录
        if done:
            self.Jain_episode_history.append(self.Jain_episode / max_steps)
            self.episode_reward_history.append(self.episode_reward)

        # info 字典
        info = {
            "total_time": self.total_time,
            "Jain_step": self.Jain_step,
            "normalized_delay": self.normalized_delay,
            "normalized_Jain": self.normalized_Jain,
            "uav_load": self.uav_L.copy(),
            "step": self.step_count,
            "reward": self.reward,

            "composite_channel": self.compute_Composite_channel().copy(),
            "unload_rate": self.compute_unload_rate().copy(),
            "comm_delay": np.array(self.users_comm_delay).copy(),
            "return_delay": np.array(self.users_return_delay).copy(),
            "comp_delay": np.array(self.users_comp_delay).copy(),
            "user_decisions": self.user_decisions.copy(),
            "user_unload_rates": user_unload_rates.copy(),

        }

        return obs, self.reward, done, truncated, info

    # 归一化状态（AI）
    def _get_obs(self):
        # UAV 位置归一化到 [-1, 1]，范围 [-400, 400] → 除以 400
        uav_norm = self.uav_positions / 400.0

        # GT 位置归一化
        user_norm = self.user_positions / 400.0

        # 任务量归一化：L_min=0.7, L_max=1.5 → 映射到 [-1,1]
        task_norm = 2 * (self.user_tasks - L_min) / (L_max - L_min) - 1

        # UAV 负载归一化：最大可能负载 = L_max * num_users（所有用户卸载到一架无人机）
        max_uav_load = L_max * num_users
        load_norm = np.array(self.uav_L, dtype=np.float32) / max_uav_load

        # 拼接：(uav_x,y * 3) + (user_x,y * 20) + (tasks * 20) + (uav_loads * 3) = 6 + 40 + 20 + 3 = 69
        obs = np.concatenate([
            uav_norm.flatten(),
            user_norm.flatten(),
            task_norm,
            load_norm
        ]).astype(np.float32)

        return obs

    # 根据飞行速度和偏向角，更新UAV位置
    def uav_move(self, uav_speeds, uav_angles):
        for i in range(num_uavs):
            dx = uav_speeds[i] * np.cos(uav_angles[i])
            dy = uav_speeds[i] * np.sin(uav_angles[i])
            new_pos = self.uav_positions[i] + np.array([dx, dy])
            new_pos = np.clip(new_pos, -400, 400)  # 边界检查
            self.uav_positions[i] = new_pos

    # 计算UAV与RIS之间的距离
    def compute_UAV_RIS(self):
        # 将UAV位置转换为3维
        uav_positions_3d = np.array([[uav_pos[0], uav_pos[1], uav_H] for uav_pos in self.uav_positions])
        # 计算三维距离
        self.UAV_RIS = np.linalg.norm(uav_positions_3d - ris_pos, axis=1)

    # 计算UAV与GT之间的距离
    def compute_UAV_GT(self):
        uav_positions_3d = np.array([[uav_pos[0], uav_pos[1], uav_H] for uav_pos in self.uav_positions])
        user_positions_3d = np.array([[pos[0], pos[1], 0] for pos in self.user_positions])
        self.UAV_GT = np.linalg.norm(uav_positions_3d[:, None] - user_positions_3d, axis=2)

    # 计算RIS与GT之间的距离
    def compute_RIS_GT(self):
        user_positions_3d = np.array([[pos[0], pos[1], 0] for pos in self.user_positions])
        self.RIS_GT = np.linalg.norm(ris_pos - user_positions_3d, axis=1)


    # 复合信道——无RIS
    def compute_Composite_channel(self):
        # 复合信道增益 (实数)
        composite_channel = np.zeros((num_uavs, num_users), dtype=np.float32)

        # 计算UAV-GT之间的距离
        self.compute_UAV_GT()

        for m in range(num_uavs):
            for k in range(num_users):
                composite_channel[m, k] =  PL0 / self.UAV_GT[m, k] ** alpha

        return composite_channel

    # 卸载速率
    def compute_unload_rate(self):
        composite_channel = self.compute_Composite_channel()
        unload_rate = np.zeros((num_uavs, num_users), dtype=np.float32)

        B_K = B_total  # 总带宽
        P_K = P  # GT发射功率
        sigma_squared = P_n_W  # 噪声功率

        uav_user_count = np.zeros(num_uavs)
        for k in range(num_users):
            uav_id = self.user_decisions[k]
            if uav_id > 0:
                uav_user_count[uav_id - 1] += 1

        for m in range(num_uavs):

            N = uav_user_count[m]
            if N == 0:
                continue
            for k in range(num_users):
                # 卸载速率公式
                h_km_squared = composite_channel[m, k]

                top = P_K * h_km_squared
                bot = sigma_squared
                SNR = top / bot

                r = (B_K / N) * np.log2(1 + SNR)

                unload_rate[m, k] = r * 1e-6  # 转化为Mbps
        self.uav_unload_rate = unload_rate
        return unload_rate

    # 计算卸载时延(只有卸载给UAV的GT才需要计算卸载时延，即user_decision != 0)
    def comm_delay(self):

        unload_rate = self.compute_unload_rate()  # 卸载速率
        comm_delay = [0] * num_users
        for k in range(num_users):

            uav_id = self.user_decisions[k]

            if uav_id == 0:
                comm_delay[k] = 0
            else:
                comm_delay[k] = (self.user_tasks[k]) / unload_rate[uav_id - 1, k]  # 任务和卸载速率都是M

            self.users_comm_delay[k] = comm_delay[k]

    # 计算UAV负载
    def compute_uav_load(self):

        uav_load = [0] * num_uavs
        for k in range(num_users):
            uav_id = self.user_decisions[k]
            if uav_id > 0:
                uav_load[uav_id - 1] += self.user_tasks[k]

        self.uav_L = uav_load

    # 任务计算时延
    def comp_delay(self):

        comp_delay = [0] * num_users
        for k in range(num_users):

            uav_id = self.user_decisions[k]

            if uav_id == 0:
                comp_delay[k] = (self.user_tasks[k] * C) / F_local
            else:
                f_temp = F_max * (self.user_tasks[k] / self.uav_L[uav_id - 1])
                comp_delay[k] = (self.user_tasks[k] * C) / f_temp  # 任务和计算资源都是M

        self.users_comp_delay = comp_delay

    # 回传时延（UAV计算完成后将结果返回GT）
    def return_delay(self):
        composite_channel = self.compute_Composite_channel()
        ret_delay = [0] * num_users

        uav_user_count = np.zeros(num_uavs)
        for k in range(num_users):
            uav_id = self.user_decisions[k]
            if uav_id > 0:
                uav_user_count[uav_id - 1] += 1

        for k in range(num_users):
            uav_id = self.user_decisions[k]
            if uav_id == 0:
                ret_delay[k] = 0  # 本地计算无需回传
            else:
                m = uav_id - 1
                N = uav_user_count[m]
                if N == 0:
                    continue
                h_km_squared = composite_channel[m, k]
                SNR = P_uav * h_km_squared / P_n_W
                r_down = (B_total / N) * np.log2(1 + SNR) * 1e-6  # Mbps
                r_down = max(r_down, 1e-9)  # 防止除零
                ret_delay[k] = (delta * self.user_tasks[k]) / r_down

        self.users_return_delay = ret_delay

    # 计算总时延
    def compute_total_delay(self):
        self.total_time = np.sum(self.users_comp_delay) + np.sum(self.users_comm_delay) + np.sum(self.users_return_delay)

    # 计算每个step的jain指数
    def compute_Jain(self):
        top = np.sum(self.uav_L) ** 2
        bot = np.sum([L ** 2 for L in self.uav_L])

        if bot != 0:  # 全本地，设置jain指数为最小值
            self.Jain_step = top / (bot * num_uavs)
        else:
            self.Jain_step = 1 / num_uavs

        self.Jain_step_history.append(self.Jain_step)

    # 预估时延归一化范围
    def compute_normalization_bounds(self):

        print("\n" + "=" * 80)
        print("【归一化范围预计算】")
        print("=" * 80)

        # ========== 时延范围估计 ==========

        # 1. 最坏情况：所有用户本地计算，任务量最大
        max_task_per_user = L_max
        total_max_task = max_task_per_user * num_users
        max_delay_local = (total_max_task * C) / F_local

        # 2. 卸载最坏情况：所有用户卸载到单个UAV，距离最远
        max_distance_horizontal = np.sqrt(2 * 400 ** 2)  # 对角线距离 ≈ 565m
        max_distance_3d = np.sqrt(max_distance_horizontal ** 2 + uav_H ** 2)

        # 最差信道增益
        h_worst = PL0 / (max_distance_3d ** alpha)

        # 最差卸载速率（所有用户竞争带宽）
        SNR_worst = P * h_worst / P_n_W
        R_worst = (B_total / num_users) * np.log2(1 + SNR_worst)
        R_worst_Mbps = max(R_worst * 1e-6, 1e-6)  # 防止过小

        # 最坏通信延迟
        D_comm_worst = max_task_per_user / R_worst_Mbps

        # 最坏回传延迟
        SNR_worst_down = P_uav * h_worst / P_n_W
        R_worst_down = (B_total / num_users) * np.log2(1 + SNR_worst_down)
        R_worst_down_Mbps = max(R_worst_down * 1e-6, 1e-6)
        D_return_worst = (delta * max_task_per_user) / R_worst_down_Mbps

        # 最坏计算延迟（所有任务由单个UAV顺序执行）
        D_comp_worst = (total_max_task * C) / F_max

        # 总的最坏延迟
        max_delay_offload = D_comm_worst + D_return_worst + D_comp_worst

        # 取两种情况的最大值，加20%安全余量
        self.max_delay_theoretical = max(max_delay_local, max_delay_offload) * 1.2

        # ========== 最小延迟估计 ==========

        # 最好情况：完美负载均衡，最近距离
        ideal_distance_horizontal = 100  # 假设平均距离100m
        ideal_distance_3d = np.sqrt(ideal_distance_horizontal ** 2 + uav_H ** 2)

        h_best = PL0 / (ideal_distance_3d ** alpha)

        # 用户均匀分配
        users_per_uav = num_users / num_uavs
        B_per_user = B_total / users_per_uav

        SNR_best = P * h_best / P_n_W
        R_best = B_per_user * np.log2(1 + SNR_best)
        R_best_Mbps = R_best * 1e-6

        # 最小任务量
        min_task_per_user = L_min
        D_comm_best = min_task_per_user / R_best_Mbps

        # 最好回传延迟
        SNR_best_down = P_uav * h_best / P_n_W
        R_best_down = B_per_user * np.log2(1 + SNR_best_down)
        R_best_down_Mbps = R_best_down * 1e-6
        D_return_best = (delta * min_task_per_user) / R_best_down_Mbps

        # 完美负载均衡的计算延迟
        avg_load_per_uav = (np.mean([L_min, L_max]) * num_users / num_uavs)
        D_comp_best = (avg_load_per_uav * C) / F_max

        self.min_delay_theoretical = D_comm_best + D_return_best + D_comp_best

        # 打印信息
        print(f"  延迟范围:")
        print(f"  最小理论延迟: {self.min_delay_theoretical:.6f} s")
        print(f"  最大理论延迟:  {self.max_delay_theoretical:.6f} s")
        print(f"  范围宽度:     {self.max_delay_theoretical - self.min_delay_theoretical:.6f} s")

        print("=" * 80 + "\n")

    # 归一化时延
    def normalize_delay(self):
        # 使用预计算的固定范围进行线性映射
        delay_range = self.max_delay_theoretical - self.min_delay_theoretical

        if delay_range < 1e-8:  # 防止除零
            delay_range = 1.0

        # 线性归一化：将 [min_delay, max_delay] 映射到 [0, 1]
        normalized = (self.total_time - self.min_delay_theoretical) / delay_range

        # 裁剪到 [0, 1] 范围内
        self.normalized_delay = np.clip(normalized, 0.0, 1.0)

    # 归一化Jain
    def normalize_Jain(self):
        min_jain = 1.0 / num_uavs
        max_jain = 1.0
        jain_range = max_jain - min_jain

        self.normalized_Jain = (self.Jain_step - min_jain) / jain_range

    # 计算step奖励
    def compute_step_reward(self):

        self.reward = w_time * (1 - self.normalized_delay) + w_fair * self.normalized_Jain

        self.reward_history.append(self.reward)


class CustomPrintCallback(BaseCallback):
    def __init__(self, print_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode = 0
        self.episode_reward = 0.0
        self.episode_step = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        step = info["step"]
        reward = info["reward"]

        self.episode_reward += reward
        self.episode_step += 1

        if step % self.print_freq == 0:
            print(f"\n{'=' * 90}")
            print(f"Episode {self.episode:4d} | Step {self.episode_step:5d} | Reward {reward:9.4f}")
            print(f"  Total time : {info['total_time']:.4f} s")
            print(f"  Jain step  : {info['Jain_step']:.4f} "
                  f"(norm_delay={info['normalized_delay']:.4f}, norm_jain={info['normalized_Jain']:.4f})")
            print(f"  UAV load   : {info['uav_load']}")

            # 信道与速率
            ch = info["composite_channel"]  # (num_uavs, num_users)
            print(f"  Channel Gains (dB):")
            for m in range(num_uavs):
                gains_db = 10 * np.log10(ch[m] + 1e-12)  # 避免 log(0)
                row_str = " | ".join([f"GT{k}: {g:6.1f}" for k, g in enumerate(gains_db)])
                print(f"    UAV{m + 1} → {row_str}")

            # 每个用户的决策 + 时延 + 任务量（关键修复！）
            decisions = info["user_decisions"]
            comm_d = info["comm_delay"]
            return_d = info["return_delay"]
            comp_d = info["comp_delay"]
            rates = info["user_unload_rates"]

            # 正确获取环境实例
            current_env = self.training_env.envs[0]  # VecEnv 的第一个子环境

            print(f"  User Decisions & Delays:")
            for k in range(num_users):
                mode = "Local" if decisions[k] == 0 else f"UAV{decisions[k]}"
                task = current_env.user_tasks[k]
                print(
                    f"    GT{k:1d}: {mode:5s} | Comm: {comm_d[k]:6.4f}s | Return: {return_d[k]:6.4f}s | Comp: {comp_d[k]:6.4f}s | "f"Task: {task:.3f} Mbit | Rate: {rates[k]:6.2f} Mbps")

            print(f"{'=' * 90}\n")

        if done:
            print("\n" + "=" * 80)
            print(f"EPISODE {self.episode} FINISHED | Steps: {self.episode_step} | "
                  f"Total Reward: {self.episode_reward:9.4f}")
            print("=" * 80 + "\n")

            self.episode += 1
            self.episode_reward = 0.0
            self.episode_step = 0

        return True


# 绘制奖励曲线
def plot_reward_curves(env: UAVEnv, save_path="./reward_curves.png"):
    from scipy.ndimage import gaussian_filter1d

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100)

    # === 子图1：Step Reward ===
    steps = np.arange(len(env.reward_history))
    rewards = np.array(env.reward_history)

    # 原始曲线
    ax1.plot(steps, rewards, color='lightblue', alpha=0.6, label='Step Reward')

    # 平滑曲线（高斯滤波）
    if len(rewards) > 50:
        smoothed = gaussian_filter1d(rewards, sigma=5)
        ax1.plot(steps, smoothed, color='blue', linewidth=2, label='Smoothed')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward per Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === 子图2：Episode Average Reward ===
    episodes = np.arange(len(env.episode_reward_history))
    episodes_rewards = np.array(env.episode_reward_history)

    ax2.plot(episodes, episodes_rewards, 'g.-', markersize=4, linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title(' Reward per Episode')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"奖励曲线已保存至: {save_path}")


# 关闭交互模式 + 使用无头后端（关键！）
plt.switch_backend('Agg')  # 必须放在最前面！防止任何弹窗
import matplotlib.pyplot as plt


class SilentRealTimePlotCallback(BaseCallback):
    """
    完全静默运行，每 plot_freq 个 episode 更新并覆盖保存一张奖励图
    不弹窗、不交互、适合服务器训练
    """

    def __init__(self, plot_freq: int = 5, save_path: str = "./realtime_reward.png", verbose: int = 0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.save_path = save_path  # 实时更新的图片路径（会不断覆盖）

        self.episode_rewards = []  # 每个 episode 的总奖励
        self.current_reward = 0.0
        self.episode_count = 0

        # 创建文件夹（如果不存在）
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        print(f"[PlotCallback] 实时奖励图将每 {plot_freq} 个 episode 更新一次，保存至：{save_path}")

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = info["reward"]
        done = self.locals["dones"][0]

        self.current_reward += reward

        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)

            # 每 plot_freq 个 episode 更新一次图片
            if self.episode_count % self.plot_freq == 0:
                self._save_plot()

            # 重置
            self.current_reward = 0.0

        return True

    def _save_plot(self):
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(episodes, rewards, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(f'UAV-RIS Offloading Training Progress\n'
                     f'Latest Episode: {self.episode_count} | Reward: {rewards[-1]:.3f}',
                     fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        # 动态设置 y 范围
        y_min = rewards.min() - 5
        y_max = rewards.max() + 5
        ax.set_ylim(y_min, y_max)

        # 在图上显示最新值
        ax.annotate(f'{rewards[-1]:.2f}',
                    xy=(episodes[-1], rewards[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 必须关闭！防止内存泄漏

        print(f"Reward plot updated -> {self.save_path}  (Episode {self.episode_count})")

    def _on_training_end(self) -> None:
        # 训练结束时保存一份带时间戳的最终图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = f"./episode_reward_final_{timestamp}.png"
        self._save_plot()  # 先更新一次实时图
        os.system(f"cp {self.save_path} {final_path}")
        print(f"\n训练结束！最终奖励曲线已保存：{final_path}")


if __name__ == "__main__":
    # 创建环境
    env = lambda: UAVEnv()
    vec_env: VecEnv = make_vec_env(env, n_envs=1, seed=0)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    # 训练或加载模型
    train_model = True  # 设为True可以重新训练模型

    if train_model:
        # 初始化PPO模型
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh  # Tanh 对连续动作空间收敛更稳定
        )

        # PPO 模型
        model = PPO(
            "MlpPolicy",  # 关键：支持 Dict 动作/观测
            vec_env,
            verbose=1,
            tensorboard_log="./ppo_uav_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,       # 增大采样步数，减少梯度方差
            batch_size=256,     # 增大 batch 提高更新稳定性
            n_epochs=10,        # 每次更新更多 epoch，充分利用数据
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,      # 提高探索熵，适合连续动作空间
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cuda",
            policy_kwargs=policy_kwargs,
        )

        print_cb = CustomPrintCallback(print_freq=1)

        # 实时绘图：每5个episode更新一次
        plot_cb = SilentRealTimePlotCallback(
            plot_freq=1,
            save_path="./realtime_reward.png"
        )

        # 训练模型
        model.learn(total_timesteps=2048 * 10000, log_interval=10, callback=[print_cb, plot_cb], )

        # 保存模型
        model.save("ppo_uav")

        trained_env = vec_env.envs[0]  # n_envs=1

        plot_reward_curves(trained_env, save_path="./reward_curves.png")

