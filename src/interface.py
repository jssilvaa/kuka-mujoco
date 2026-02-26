from __future__ import annotations

from dataclasses import dataclass 
import mujoco 
import numpy as np 

from lie_math import Jl_inv, vee, Log
from ik import compute_err, compute_err_mat, target_orientation

@dataclass 
class RobotInterface:
  model: mujoco.MjModel
  target_mat: np.ndarray  # desired rpy (3,3)
  target_id: int          # target object index
  ee_id: int              # end effector site index 

  @staticmethod 
  def handle(model: mujoco.MjModel, target_name: str, ee_name: str, zd: np.ndarray, up: np.ndarray) -> RobotInterface:
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
    if zd.size != 3 or up.size != 3: 
      raise ValueError(f"Incorrect sizes specified for vectors zd and/or up. Expected size for zd: 3. Got {zd.size}. Expected size for up: 3. Got {up.size}")
    else: 
      target_mat = target_orientation(zd, up)
      return RobotInterface(model=model, target_mat=target_mat, target_id=target_id, ee_id=ee_id)

@dataclass 
class mpcTask: 
  Jk: np.ndarray
  ek: np.ndarray

  @staticmethod 
  def task_k(model: mujoco.MjModel, data: mujoco.MjData, interface: RobotInterface) -> mpcTask:
    target_mat = interface.target_mat
    target_id = interface.target_id
    ee_id = interface.ee_id
    nq = model.nq 
    nv = model.nv 

    Jp = np.zeros((3,nv))
    Jr = np.zeros((3,nv))

    mujoco.mj_jacSite(model, data, Jp, Jr, ee_id)
    err_lin, phi = compute_err(data, target_id, ee_id), compute_err_mat(data, target_mat, ee_id)
    Jfull = np.vstack((Jp, Jl_inv(phi) @ Jr))
    J = Jfull[:, :nq]
    err = np.concatenate((err_lin, phi))

    return mpcTask(Jk=J, ek=err)