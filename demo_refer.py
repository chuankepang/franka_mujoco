"""Demonstrates the Franka Emika Robot System model for MuJoCo."""
import time
import mujoco
import mujoco.viewer
import numpy as np
import random
from threading import Thread

class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("./world.xml")
        self.data = mujoco.MjData(self.model)
        self.run = True
        self.gripper(True)
        
        # 初始化机器人位置
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)
        
        # 控制相关变量
        self.xpos0 = None
        self.xquat0 = None
        self.xpos_d = None
        self.down = None
        self.up = None
        self.state = None

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
            
    def init_step_sequence(self):
        self.xpos0 = self.data.body("panda_hand").xpos.copy()
        self.xpos_d = self.xpos0.copy()
        self.xquat0 = self.data.body("panda_hand").xquat.copy()
        self.down = list(np.linspace(-1, 0, 2000))
        self.up = list(np.linspace(0, -1, 2000))
        self.state = "down"
        
    def step_sequence(self):
        if self.state == "down":
            if len(self.down):
                self.xpos_d = self.xpos0 + [0, 0, self.down.pop()]
            else:
                self.state = "grasp"
        elif self.state == "grasp":
            self.gripper(False)
            self.state = "up"
        elif self.state == "up":
            if len(self.up):
                self.xpos_d = self.xpos0 + [0, 0, self.up.pop()]
       

    def start(self) -> None:
        self.init_step_sequence()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            self.last_cube_time = start
            
            while viewer.is_running() and self.run:
                step_start = time.time()
                
                

                # 控制
                self.step_sequence()
                self.control(self.xpos_d, self.xquat0)
                
                # 物理步进
                mujoco.mj_step(self.model, self.data)
                
                # 与查看器同步
                viewer.sync()
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    Demo().start()