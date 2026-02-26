from __future__ import annotations 

from dataclasses import dataclass 
import numpy as np 
import mujoco 
import scipy.sparse as sp 
import osqp 

from interface import RobotInterface, mpcTask 


@dataclass
class RTIWeights: 
  We: np.ndarray    # (6,6)
  Wv: np.ndarray    # (7,7)
  Wa: np.ndarray    # (7,7)
  WeN: np.ndarray   # (6,6)
  WvN: np.ndarray   # (7,7)
  eps_dq: float     # (1,)

  @staticmethod
  def default(): 
    return RTIWeights(
        We=np.diag([200, 200, 200, 20, 20, 20]),
        Wv=np.diag([1, 1, 1, 1, 1, 1, 1]),
        Wa=np.diag([1e-2]*7),
        WeN=np.diag([400, 400, 400, 40, 40, 40]),
        WvN=np.diag([2, 2, 2, 2, 2, 2, 2]),
        eps_dq=float(1e-3),
    )
  

@dataclass
class RTIConfig: 
  N: int              # horizon length 
  n: int              # no. of DOFs (assuming nq == nv)
  h: float            # discrete timestep in seconds 
  u_max: np.ndarray   # (n,)
  u_min: np.ndarray   # (n,)


class RTITrackerQP: 
  def __init__(self, cfg: RTIConfig, weights: RTIWeights): 
    self.cfg = cfg 
    self.w = weights

    # nominal trajectories - containers for warm start 
    self.qbar = np.zeros((cfg.N + 1, cfg.n))
    self.ubar = np.zeros((cfg.N, cfg.n))

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

  def solve_rti(self, model, data, iface: RobotInterface, q_meas: np.ndarray): 
    N, n = self.cfg.N, self.cfg.n 
    nx = n
    nu = n 
    
    tasks = []
    for k in range(N+1):
        data.qpos[:n] = self.qbar[k]
        mujoco.mj_forward(model, data)
        tasks.append(mpcTask.task_k(model, data, iface))

    # index terms in the qp 
    n_dx = (N+1)*nx
    n_du = N*nu 
    nz = n_dx + n_du 

    def idx_dq(k): return slice(k*nx, (k+1)*nx)
    def idx_du(k): return slice(n_dx + k*nu, n_dx + (k+1)*nu)

    # build H and f 
    H = np.zeros((nz,nz))
    f = np.zeros((nz,))

    for k in range(N):
      Jk = tasks[k].Jk
      ek = tasks[k].ek

      Qqq = Jk.T @ self.w.We @ Jk
      gq  = - Jk.T @ self.w.We @ ek

      H[idx_dq(k), idx_dq(k)] += 2.0 * Qqq + self.w.eps_dq * np.eye(n)
      H[idx_du(k), idx_du(k)] += 2.0 * self.w.Wa
      f[idx_dq(k)] += 2.0 * gq


    JN = tasks[N].Jk
    eN = tasks[N].ek
    QqqN = JN.T @ self.w.WeN @ JN
    gqN  = - JN.T @ self.w.WeN @ eN
    
    H[idx_dq(N), idx_dq(N)] += 2.0 * QqqN
    f[idx_dq(N)] += 2.0 * gqN

    n_eq = nx + N*nx
    Aeq = np.zeros((n_eq, nz))
    beq = np.zeros((n_eq,))
    row = 0

    # dq0 = q_meas - qbar0
    Aeq[row:row+nx, idx_dq(0)] = np.eye(nx)
    beq[row:row+nx] = q_meas - self.qbar[0]
    row += nx

    for k in range(N):
        Aeq[row:row+nx, idx_dq(k+1)] = np.eye(nx)
        Aeq[row:row+nx, idx_dq(k)]  += -np.eye(nx)
        Aeq[row:row+nx, idx_du(k)]  += -(self.cfg.h * np.eye(nu))
        row += nx

    A_list = [Aeq]
    l_list = [beq]
    u_list = [beq]

    # du bounds: u_min - ubar <= du <= u_max - ubar
    A_du = np.zeros((n_du, nz))
    l_du = np.zeros((n_du,))
    u_du = np.zeros((n_du,))
    for k in range(N):
        sl = idx_du(k)
        A_du[k*nu:(k+1)*nu, sl] = np.eye(nu)
        l_du[k*nu:(k+1)*nu] = self.cfg.u_min - self.ubar[k]
        u_du[k*nu:(k+1)*nu] = self.cfg.u_max - self.ubar[k]

    A_list.append(A_du)
    l_list.append(l_du)
    u_list.append(u_du)

    A = np.vstack(A_list)
    l = np.concatenate(l_list)
    u = np.concatenate(u_list)

    # solve qp 
    P = sp.csc_matrix(0.5*(H + H.T))  # ensure symmetric
    q = f
    A_sp = sp.csc_matrix(A)

    self._solver = osqp.OSQP()
    self._solver.setup(P=P, q=q, A=A_sp, l=l, u=u, verbose=False, polish=True)
    
    if self._last_solution is not None:
        self._solver.warm_start(x=self._last_solution)
    res = self._solver.solve()

    if res.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {res.info.status}")

    z = res.x
    self._last_solution = z

    dq_all = z[:n_dx].reshape((N+1, nx))          # (N+1, n)
    du_all = z[n_dx:].reshape((N, nu))            # (N, n)

    self.qbar += dq_all
    self.ubar += du_all
    return self.ubar[0].copy()