from __future__ import annotations 

from dataclasses import dataclass 
import numpy as np 
import mujoco 
import scipy.sparse as sp 
import osqp 

from interface import RobotInterface, mpcTask 
from collision import min_distance, min_distance_and_grad, find_coating_geoms, find_sphere_obstacles


@dataclass
class RTIWeights: 
  We: np.ndarray    # (6,6)
  Wv: np.ndarray    # (7,7)
  Wa: np.ndarray    # (7,7)
  WeN: np.ndarray   # (6,6)
  WvN: np.ndarray   # (7,7)
  eps_dq: float     # (1,)
  Ws: float         # scalar (e.g. active set of min_distance m=1)
  rho_s: float 

  @staticmethod
  def default(): 
    return RTIWeights(
        We=np.diag([200, 200, 200, 20, 20, 20]),
        Wv=np.diag([1, 1, 1, 1, 1, 1, 1]),
        Wa=np.diag([1e-2]*7),
        WeN=np.diag([800, 800, 800, 40, 40, 40]),
        WvN=np.diag([2, 2, 2, 2, 2, 2, 2]),
        eps_dq=float(1e-3),
        Ws=float(1e3), 
        rho_s=float(10)
    )
  

@dataclass
class RTIConfig: 
  N: int              # horizon length 
  n: int              # no. of DOFs (assuming nq == nv)
  h: float            # discrete timestep in seconds 
  u_max: np.ndarray   # (n,)
  u_min: np.ndarray   # (n,)


class RTITrackerQP: 
  def __init__(self, model: mujoco.MjModel, cfg: RTIConfig, weights: RTIWeights): 
    self.cfg = cfg 
    self.w = weights
    self.d_safe = 0.05 # safety distance [m]
    self.clearance_active = False 

    self.capsule_gids = find_coating_geoms(model)
    self.sphere_gids = find_sphere_obstacles(model)

    # nominal trajectories - containers for warm start 
    self.qbar = np.zeros((cfg.N + 1, cfg.n))
    self.ubar = np.zeros((cfg.N, cfg.n))
    self.last_dmin = None 
    self.last_smax = None 

    self._solver = None 
    self._last_solution = None 

  def warm_start_shift(self, q_meas: np.ndarray):
      self.qbar[:-1] = self.qbar[1:]
      self.qbar[-1]  = self.qbar[-2]
      self.ubar[:-1] = self.ubar[1:]
      self.ubar[-1]  = self.ubar[-2]

      self.qbar[0] = q_meas

      for k in range(self.cfg.N):
          self.qbar[k+1] = self.qbar[k] + self.cfg.h * self.ubar[k]

  def solve_rti(self, model: mujoco.MjModel, data: mujoco.MjData, iface: RobotInterface, q_meas: np.ndarray): 
    N, n = self.cfg.N, self.cfg.n 
    nq = n
    nu = n 

    qpos0 = data.qpos[:n].copy()
    qvel0 = data.qvel[:n].copy()
        
    tasks = []
    for k in range(N+1):
        data.qpos[:n] = self.qbar[k]
        mujoco.mj_forward(model, data)
        tasks.append(mpcTask.task_k(model, data, iface))

    if self.clearance_active:
      dbar = np.zeros((N+1,), dtype=float)
      Dbar = np.zeros((N+1,n), dtype=float)

      for k in range(N+1): 
        res, grad = min_distance_and_grad(
            model, data,
            self.qbar[k],
            self.capsule_gids, self.sphere_gids,
            eps=1e-4
        )
        dbar[k] = res.dmin 
        Dbar[k, :] = grad

    # index terms in the qp 
    n_dq = (N+1)*nq
    n_du = N*nu 
    n_s = (N+1)*1
    nz = n_dq + n_du + n_s

    def idx_dq(k): return slice(k*nq, (k+1)*nq)
    def idx_du(k): return slice(n_dq + k*nu, n_dq + (k+1)*nu)
    def idx_s(k): return slice(n_dq + n_du + k*1, n_dq + n_du + (k+1)*1)

    # build H and f 
    H = np.zeros((nz,nz))
    f = np.zeros((nz,))

    for k in range(N):
      Jk = tasks[k].Jk
      ek = tasks[k].ek

      Qqq = Jk.T @ self.w.We @ Jk
      gq  = - Jk.T @ self.w.We @ ek

      H[idx_dq(k), idx_dq(k)] += 2.0 * Qqq + self.w.eps_dq * np.eye(n)
      f[idx_dq(k)] += 2.0 * gq
      
      H[idx_du(k), idx_du(k)] += 2.0 * self.w.Wa
      
      if self.clearance_active:
        H[idx_s(k), idx_s(k)] += 2.0 * self.w.Ws 
        f[idx_s(k)] += self.w.rho_s

    JN = tasks[N].Jk
    eN = tasks[N].ek
    QqqN = JN.T @ self.w.WeN @ JN
    gqN  = - JN.T @ self.w.WeN @ eN
    
    H[idx_dq(N), idx_dq(N)] += 2.0 * QqqN
    f[idx_dq(N)] += 2.0 * gqN
    if self.clearance_active:
      H[idx_s(N), idx_s(N)] += 2.0 * self.w.Ws
      f[idx_s(N)] += self.w.rho_s 

    n_eq = nq + N*nq
    Aeq = np.zeros((n_eq, nz))
    beq = np.zeros((n_eq,))
    row = 0

    # dq0 = q_meas - qbar0
    Aeq[row:row+nq, idx_dq(0)] = np.eye(nq)
    beq[row:row+nq] = q_meas - self.qbar[0]
    row += nq

    for k in range(N):
        Aeq[row:row+nq, idx_dq(k+1)] = np.eye(nq)
        Aeq[row:row+nq, idx_dq(k)]  += -np.eye(nq)
        Aeq[row:row+nq, idx_du(k)]  += -(self.cfg.h * np.eye(nu))
        row += nq

    A_list = [Aeq]
    l_list = [beq]
    u_list = [beq]

    # du bounds: u_min - ubar <= du <= u_max - ubar
    A_du = np.zeros((n_du, nz))
    l_du = np.zeros((n_du,))
    u_du = np.zeros((n_du,))
    for k in range(N):
        A_du[k*nu:(k+1)*nu, idx_du(k)] = np.eye(nu)
        l_du[k*nu:(k+1)*nu] = self.cfg.u_min - self.ubar[k]
        u_du[k*nu:(k+1)*nu] = self.cfg.u_max - self.ubar[k]

    A_list.append(A_du)
    l_list.append(l_du)
    u_list.append(u_du)

    if self.clearance_active: 
      A_clear = np.zeros((N+1, nz))
      l_clear = -np.inf * np.ones((N+1,))
      u_clear = np.zeros((N+1,))
      for k in range(N+1): 
        # -D_k * dq_k - 1 * s_k <= dbar_k - d_safe
        A_clear[k, idx_dq(k)] = - Dbar[k, :]
        A_clear[k, idx_s(k)] = -1.0 
        u_clear[k] = dbar[k] - self.d_safe 

      A_list.append(A_clear)
      l_list.append(l_clear)
      u_list.append(u_clear)

      A_spos = np.zeros((N+1, nz))
      l_spos = - np.inf * np.ones((N+1,))
      u_spos = np.zeros((N+1,))
      for k in range(N+1): 
        A_spos[k, idx_s(k)] = -1.0
        u_spos[k] = 0.0 

      A_list.append(A_spos)
      l_list.append(l_spos)
      u_list.append(u_spos)
    else: 
      A_szero = np.zeros((N+1, nz))
      l_szero = np.zeros((N+1,))
      u_szero = np.zeros((N+1,))
      for k in range(N+1):
          A_szero[k, idx_s(k)] = 1.0   # s_k = 0
      A_list.append(A_szero)
      l_list.append(l_szero)
      u_list.append(u_szero)

    A = np.vstack(A_list)
    l = np.concatenate(l_list)
    u = np.concatenate(u_list)

    # solve qp 
    P = sp.csc_matrix(0.5*(H + H.T))  # ensure symmetric
    q = f
    A_sp = sp.csc_matrix(A)

    self._solver = osqp.OSQP()
    self._solver.setup(
      P=P, q=q, A=A_sp, l=l, u=u,
      verbose=False,
      polish=True,
      max_iter=20000,
      eps_abs=1e-4,
      eps_rel=1e-4,
      adaptive_rho=True,
      scaling=10,
    )
    
    if self._last_solution is not None:
        self._solver.warm_start(x=self._last_solution)
    res = self._solver.solve()

    if res.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {res.info.status}")

    z = res.x
    self._last_solution = z
    if self.clearance_active:
      self.last_dmin = dbar[0]
    else: 
      self.last_dmin = min_distance(model, data, self.qbar[0], self.capsule_gids, self.sphere_gids).dmin 

    dq_all = z[:n_dq].reshape((N+1, nq))          # (N+1,n)
    du_all = z[n_dq:n_dq+n_du].reshape((N, nu))   # (N,n)
    if self.clearance_active:
      s_all  = z[n_dq+n_du:]                        # (N+1,)
      self.last_smax = float(np.max(s_all)) 

    self.qbar += dq_all
    self.ubar += du_all

    data.qpos[:n] = qpos0
    data.qvel[:n] = qvel0
    mujoco.mj_forward(model, data)

    return self.ubar[0].copy()