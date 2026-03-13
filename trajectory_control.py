"""Franka Emika Robot System model for MuJoCo with advanced trajectory control."""
import time
import mujoco
import mujoco.viewer
import numpy as np
import math
from enum import Enum

class ControlMode(Enum):
    POINT_TO_POINT = 1
    FIGURE_EIGHT = 2
    ELLIPSE = 3
    

class TrajectoryGenerator:
    """生成各种轨迹的类"""
    
    @staticmethod
    def generate_points(center, num_points=6, radius=0.1):
        """在空间中生成均匀分布的点"""
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            points.append(np.array([x, y, z]))
        return points
    
    @staticmethod
    def generate_figure_eight(center, t, scale_x=0.1, scale_y=0.05, freq=0.5):
        """生成8字形轨迹上的点"""
        x = center[0] + scale_x * np.sin(freq * t)
        y = center[1] + scale_y * np.sin(2 * freq * t)
        z = center[2]
        return np.array([x, y, z])
    
    @staticmethod
    def generate_ellipse(center, t, a=0.1, b=0.05, freq=0.5):
        """生成椭圆轨迹上的点"""
        x = center[0] + a * np.cos(freq * t)
        y = center[1] + b * np.sin(freq * t)
        z = center[2]
        return np.array([x, y, z])
    
    @staticmethod
    def interpolate(p1, p2, alpha):
        """在两点之间进行平滑插值（使用5次多项式）"""
        # 使用5次多项式插值，确保速度和加速度连续
        alpha_smooth = 10 * (alpha**3) - 15 * (alpha**4) + 6 * (alpha**5)
        return p1 + alpha_smooth * (p2 - p1)

class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("./world.xml")
        self.data = mujoco.MjData(self.model)
        self.run = True
        self.gripper(False)
        
        # 初始化机器人位置
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)
        
        # 控制相关变量
        self.xpos0 = None
        self.xquat0 = None
        self.xpos_d = None
        self.xquat_d = None
        
        # 轨迹控制相关变量
        self.control_mode = ControlMode.POINT_TO_POINT  # 默认为点位控制
        self.trajectory_time = 0  # 轨迹执行时间
        self.trajectory_points = []  # 点位控制的目标点
        self.current_point_index = 0  # 当前目标点索引
        self.transition_time = 2.0  # 在两点之间过渡的时间(秒)
        self.point_hold_time = 1.0  # 在每个点停留的时间(秒)
        self.transition_start_time = 0  # 过渡开始的时间
        self.point_reached = False  # 是否已到达当前目标点
        self.prev_position = None  # 前一个位置（用于插值）
        
        # 模式切换计时
        self.mode_switch_time = 25.0  # 每隔多少秒切换模式
        self.last_mode_switch = 0     # 上次切换的时间
        
    def gripper(self, open=True):
        self.data.actuator("pos_panda_finger_joint1").ctrl = (0.04, 0)[not open]
        self.data.actuator("pos_panda_finger_joint2").ctrl = (0.04, 0)[not open]
        
    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)
        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res
        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
                f"panda_joint{i}"
            ).qfrc_bias
            self.data.actuator(f"panda_joint{i}").ctrl += (
                J[:, dofadr].T @ np.diag(self.K) @ error
            )
            self.data.actuator(f"panda_joint{i}").ctrl -= (
                J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            )
            
    def init_control(self, mode=ControlMode.POINT_TO_POINT):
        """初始化控制模式"""
        self.control_mode = mode
        self.xpos0 = self.data.body("panda_hand").xpos.copy()
        self.xquat0 = self.data.body("panda_hand").xquat.copy()
        self.xpos_d = self.xpos0.copy()
        self.xquat_d = self.xquat0.copy()
        self.trajectory_time = 0
        self.last_mode_switch = 0
        
        if mode == ControlMode.POINT_TO_POINT:
            # 生成六个空间点
            self.trajectory_points = TrajectoryGenerator.generate_points(self.xpos0)
            self.current_point_index = 0
            self.transition_start_time = 0
            self.point_reached = False
            self.prev_position = self.xpos0.copy()
            print("模式: 点位控制 - 机械臂将在6个空间点之间运动")
        elif mode == ControlMode.FIGURE_EIGHT:
            print("模式: 8字形轨迹 - 机械臂将执行8字形运动")
        elif mode == ControlMode.ELLIPSE:
            print("模式: 椭圆轨迹 - 机械臂将执行椭圆轨迹运动")
            
    def update_trajectory(self, dt):
        """更新轨迹目标位置"""
        self.trajectory_time += dt
        
        # 自动切换模式
        if (self.trajectory_time - self.last_mode_switch) > self.mode_switch_time:
            next_mode = (self.control_mode.value % 3) + 1
            self.switch_control_mode(ControlMode(next_mode))
            self.last_mode_switch = self.trajectory_time
        
        if self.control_mode == ControlMode.POINT_TO_POINT:
            # 点位控制
            self.update_point_to_point()
        elif self.control_mode == ControlMode.FIGURE_EIGHT:
            # 8字形轨迹
            self.xpos_d = TrajectoryGenerator.generate_figure_eight(
                self.xpos0, self.trajectory_time)
        elif self.control_mode == ControlMode.ELLIPSE:
            # 椭圆轨迹
            self.xpos_d = TrajectoryGenerator.generate_ellipse(
                self.xpos0, self.trajectory_time)
    
    def update_point_to_point(self):
        """更新点到点的控制"""
        current_target = self.trajectory_points[self.current_point_index]
        
        if not self.point_reached:
            # 计算从当前位置到目标点的过渡
            elapsed_time = self.trajectory_time - self.transition_start_time
            if elapsed_time < self.transition_time:
                # 平滑过渡到目标点
                alpha = elapsed_time / self.transition_time
                self.xpos_d = TrajectoryGenerator.interpolate(
                    self.prev_position, current_target, alpha)
            else:
                # 到达目标点
                self.xpos_d = current_target
                self.point_reached = True
                self.transition_start_time = self.trajectory_time
        else:
            # 在目标点停留一段时间
            elapsed_hold_time = self.trajectory_time - self.transition_start_time
            if elapsed_hold_time > self.point_hold_time:
                # 前往下一个点
                self.prev_position = current_target.copy()
                self.current_point_index = (self.current_point_index + 1) % len(self.trajectory_points)
                self.point_reached = False
                self.transition_start_time = self.trajectory_time
        
    def switch_control_mode(self, mode):
        """切换控制模式"""
        if self.control_mode != mode:
            self.init_control(mode)
            
    def start(self) -> None:
        """启动模拟"""
        print("机械臂轨迹控制演示")
        print("程序将自动在三种控制模式间切换（每15秒）:")
        print("- 点位控制: 在6个预定义点之间移动")
        print("- 8字形轨迹: 执行8字形运动")
        print("- 椭圆轨迹: 执行椭圆轨迹运动")
        
        self.init_control(self.control_mode)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last_update_time = time.time()
            
            while viewer.is_running() and self.run:
                step_start = time.time()
                
                # 计算时间增量
                current_time = time.time()
                dt = current_time - last_update_time
                last_update_time = current_time
                
                # 更新轨迹
                self.update_trajectory(dt)
                
                # 应用控制
                self.control(self.xpos_d, self.xquat_d)
                
                # 物理步进
                mujoco.mj_step(self.model, self.data)
                
                # 与查看器同步
                viewer.sync()
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    Demo().start()