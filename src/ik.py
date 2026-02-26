from __future__ import annotations
import time

from lie_math import (
   hat, vee, Log, Jl_inv
)

import mujoco
import mujoco.viewer
import numpy as np 


def compute_err(data: mujoco.MjData, target_idx: int, ee_idx: int) -> np.ndarray:
  """ computes linear err w.r.t. world frame. outputs: `err_lin`: (3, )"""
  target_pos = data.xpos[target_idx].copy()
  ee_pos = data.site_xpos[ee_idx].copy()
  return target_pos - ee_pos 


def compute_err_mat(data: mujoco.MjData, target_mat: np.ndarray, ee_idx: int) -> np.ndarray: 
  """ computes rot. err vector w.r.t. world in axis-vector representation. outputs: `err_rot`: (3,)"""
  assert np.allclose(target_mat.T @ target_mat, np.eye(3), atol=1e-6)
  ee_mat = data.site_xmat[ee_idx].reshape(3,3).copy()
  return vee(Log(target_mat @ ee_mat.T))


def target_orientation(zd: np.ndarray, up: np.ndarray) -> np.ndarray:
  """ compute target desired rotation orientation from `zd`: (3,) and `up`: (3,)"""
  xd = np.cross(up, zd); xd /= np.linalg.norm(xd) 
  yd = np.cross(zd, xd); yd /= np.linalg.norm(yd)
  return np.column_stack((xd,yd,zd))


def solve_ik(
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    target_id: int, 
    ee_id: int, 
    target_mat: np.ndarray,
    dt: float=0.002,
    lam: float=1e-3,
    alpha: float=0.2,
    tol: float=1e-6,
    max_iter: int=1000,
    ) -> np.ndarray: 
  """ 
    solves: 
      dq = Je.T @ (Je @ Je.T + lam * I6)^-1 @ e 
      let y = Je.T^-1 dq, then Ay = e, solve this 
      where A = Je @ Je.T + lam * I6
      return dq = Je.T @ y
  """
  nq = model.nq
  nv = model.nv 
  Jp = np.zeros((3,nv))
  Jr = np.zeros((3,nv))
  I6 = np.eye(6)
  for k in range(max_iter): 
    mujoco.mj_forward(model, data)
    mujoco.mj_jacSite(model, data, Jp, Jr, ee_id)

    err_lin, phi = compute_err(data, target_id, ee_id), compute_err_mat(data, target_mat, ee_id)
    err = np.concatenate((err_lin, phi))
    Je = np.vstack((Jp, Jl_inv(phi) @ Jr))

    A = Je @ Je.T + lam * I6
    y = np.linalg.solve(A, err)
    dq = Je.T @ y

    data.qpos[:nq] += alpha * dq[:nq]
    if np.linalg.norm(err) < tol: 
      print(f"IK Converged at iteration {k} with ||err||: ", np.linalg.norm(err))
      break 
  return data.qpos.copy() 