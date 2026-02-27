from __future__ import annotations
import time
from typing import Optional 

import numpy as np
import mujoco
import mujoco.viewer

from rti_qp import RTITrackerQP, RTIConfig, RTIWeights
from interface import RobotInterface, mpcTask

class MPCController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, iface: RobotInterface, N: int = 20, timestep: Optional[float] = None, K: float = 50.0, e_th: float = 5e-5,  M: int = 50, trigger_view: bool = True):
        self.model: mujoco.MjModel = model
        self.data: mujoco.MjData = data
        self.iface: RobotInterface = iface
        self.n: int = model.nq
        self.h: float = model.opt.timestep if timestep is None else timestep
        self.K: float = 5.0
        self.e_th: float = e_th 
        self.stable_count: int = 0 
        self.M = M

        self.kp = model.actuator_gainprm[:model.nu, 0].copy()
        self.lo = model.actuator_ctrlrange[:self.model.nu, 0]
        self.hi = model.actuator_ctrlrange[:self.model.nu, 1]

        cfg = RTIConfig(
            N=N, n=self.n, h=self.h,
            u_min=-5.0*np.ones(self.n),
            u_max=+5.0*np.ones(self.n),
        )
        self.rti = RTITrackerQP(cfg, RTIWeights.default())

        self.view = trigger_view
        self.viewer = None

    def step_all(self, steps=4000):
        if self.view:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        dt = self.K * self.h
        for i in range(steps):
            q = self.data.qpos[:self.n].copy()
            self.rti.warm_start_shift(q)
            # self.rti.cfg.u_min = 0.1*(self.lo - q - self.data.qfrc_bias / self.kp) / dt 
            # self.rti.cfg.u_max = 0.1*(self.hi - q - self.data.qfrc_bias / self.kp) / dt 
            u0 = self.rti.solve_rti(self.model, self.data, self.iface, q)

            self.data.qpos[:self.n] = q
            mujoco.mj_forward(self.model, self.data)

            q_cmd = np.clip(q + dt * u0 + self.data.qfrc_bias / self.kp, self.lo, self.hi)
            self.data.ctrl[:self.n] = q_cmd
            mujoco.mj_step(self.model, self.data)

            t = mpcTask.task_k(self.model, self.data, self.iface)
            norm_ek = float(np.linalg.norm(t.ek))
            if norm_ek < self.e_th: 
                self.stable_count += 1
            else: 
                self.stable_count = 0 

            if i % 200 == 0:
                print(i, "||e||", norm_ek, "||u0||", float(np.linalg.norm(u0)))
                print(i, "dmin", self.rti.last_dmin)
            
            if self.viewer is not None: 
                self.viewer.sync()

            if self.stable_count > self.M:
                q_hold = self.data.qpos[:self.n].copy()
                self.data.ctrl[:self.n] = q_hold
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None: 
                    self.viewer.close()
                break