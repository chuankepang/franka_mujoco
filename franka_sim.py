import time
import mujoco
import mujoco.viewer
import numpy as np
from enum import Enum, auto

# ==========================================
# 1. 基础配置与枚举
# ==========================================
class TaskStage(Enum):
    IDLE = auto()
    PLACE_CERAMIC = auto()    # 阶段一：放置陶瓷片
    PLACE_FAN_BLADE = auto()  # 阶段二：放置风扇叶片
    DONE = auto()

class RobotState(Enum):
    MOVING = auto()           # 正在移动
    WAITING = auto()          # 到达目标点停顿

# ==========================================
# 2. 轨迹规划模块 (Trajectory Planner)
# ==========================================
class MotionPlanner:
    @staticmethod
    def quintic_interp(p0, pf, t, T):
        """五次多项式插值，保证速度和加速度在起终点为0"""
        if t >= T: return pf
        alpha = t / T
        # 5次多项式系数: 10t^3 - 15t^4 + 6t^5
        s = 10 * (alpha**3) - 15 * (alpha**4) + 6 * (alpha**5)
        return p0 + s * (pf - p0)

# ==========================================
# 3. 核心仿真类 (FrankaSimulation)
# ==========================================
class FrankaTaskController:
    def __init__(self, xml_path="./world.xml"):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 控制参数 (阻抗控制增益)
        self.K_pos = np.array([600.0, 600.0, 600.0])
        self.K_ori = np.array([30.0, 30.0, 30.0])
        self.D_pos = 2 * np.sqrt(self.K_pos)
        self.D_ori = 2 * np.sqrt(self.K_ori)

        # 任务流定义
        self.current_stage = TaskStage.PLACE_CERAMIC
        self.robot_state = RobotState.MOVING
        
        # 预设关键位姿 (需根据你的 XML 调整坐标)
        self.poses = {
            "home": np.array([0.4, 0.0, 0.5]),
            "ceramic_pick": np.array([0.5, -0.2, 0.2]),
            "ceramic_place": np.array([0.5, 0.2, 0.1]),
            "fan_pick": np.array([0.6, -0.3, 0.2]),
            "fan_place": np.array([0.6, 0.3, 0.15])
        }
        
        # 任务序列：每个阶段由一系列目标点组成
        self.task_sequences = {
            TaskStage.PLACE_CERAMIC: ["home", "ceramic_pick", "ceramic_place", "home"],
            TaskStage.PLACE_FAN_BLADE: ["home", "fan_pick", "fan_place", "home"]
        }
        
        self.seq_idx = 0
        self.start_pos = None
        self.target_pos = None
        self.move_duration = 3.0  # 每个动作耗时3秒
        self.stage_start_time = 0.0

    def get_ee_pose(self):
        """获取末端执行器当前位姿"""
        body = self.data.body("panda_hand")
        return body.xpos.copy(), body.xquat.copy()

    def set_gripper(self, width):
        """夹持器控制 (0.04=开, 0.0=关)"""
        self.data.actuator("pos_panda_finger_joint1").ctrl = width
        self.data.actuator("pos_panda_finger_joint2").ctrl = width

    def osc_control(self, target_pos, target_quat):
        """操作空间控制 (Operational Space Control)"""
        curr_pos, curr_quat = self.get_ee_pose()
        
        # 1. 计算雅可比矩阵
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        body_id = self.model.body("panda_hand").id
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        J = np.vstack([jacp, jacr])
        
        # 2. 计算位姿误差
        err = np.zeros(6)
        err[:3] = target_pos - curr_pos
        
        # 姿态误差 (Quaternion Error)
        quat_err = np.zeros(3)
        mujoco.mju_subQuat(quat_err, curr_quat, target_quat)
        # 将误差转换到世界坐标系
        # mujoco.mju_rotVecQuat(quat_err, quat_err, curr_quat) # 根据需要开启
        err[3:] = -quat_err 

        # 3. 动力学补偿 + 反馈控制
        v = J @ self.data.qvel
        # 映射到关节力矩
        forces = np.hstack([self.K_pos * err[:3] + self.D_pos * (-v[:3]),
                           self.K_ori * err[3:] + self.D_ori * (-v[3:])])
        
        tau = J.T @ forces
        
        # 4. 应用重力补偿并输出
        for i in range(7):
            joint_name = f"panda_joint{i+1}"
            self.data.actuator(joint_name).ctrl = self.data.joint(joint_name).qfrc_bias + tau[self.model.joint(joint_name).dofadr]

    def update(self, sim_time):
        """主逻辑更新"""
        if self.current_stage == TaskStage.DONE:
            return

        # 获取当前任务序列
        sequence = self.task_sequences[self.current_stage]
        if self.seq_idx >= len(sequence):
            # 阶段完成，切换下一个阶段
            print(f"完成阶段: {self.current_stage}")
            if self.current_stage == TaskStage.PLACE_CERAMIC:
                self.current_stage = TaskStage.PLACE_FAN_BLADE
                self.seq_idx = 0
            else:
                self.current_stage = TaskStage.DONE
            self.stage_start_time = sim_time
            return

        # 初始化当前动作的起点和终点
        if self.start_pos is None:
            self.start_pos, _ = self.get_ee_pose()
            self.target_pos = self.poses[sequence[self.seq_idx]]
            self.stage_start_time = sim_time

        # 计算插值位置
        elapsed = sim_time - self.stage_start_time
        curr_target = MotionPlanner.quintic_interp(self.start_pos, self.target_pos, elapsed, self.move_duration)
        
        # 执行控制 (保持初始姿态不变)
        fixed_quat = np.array([0, 0.707, 0.707, 0]) # 俯视姿态
        self.osc_control(curr_target, fixed_quat)

        # 动作切换逻辑
        if elapsed > self.move_duration + 0.5: # 动作完成后停留0.5s
            self.seq_idx += 1
            self.start_pos = None 
            
        # 夹持器简单的动作触发逻辑 (根据索引示例)
        if "pick" in sequence[self.seq_idx-1]: self.set_gripper(0.0)
        if "place" in sequence[self.seq_idx-1]: self.set_gripper(0.04)

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 初始位置设置
            q_init = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            for i, q in enumerate(q_init):
                self.data.joint(f"panda_joint{i+1}").qpos = q
            
            while viewer.is_running():
                step_start = time.time()
                
                self.update(self.data.time)
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 保持仿真频率
                elapsed = time.time() - step_start
                if elapsed < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - elapsed)

if __name__ == "__main__":
    sim = FrankaTaskController()
    sim.run_simulation()