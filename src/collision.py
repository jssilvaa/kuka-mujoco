from __future__ import annotations 
from dataclasses import dataclass 
from typing import List, Tuple

import mujoco 
import numpy as np 


@dataclass(frozen=True)
class CapsuleSegment: 
  a: np.ndarray     # (3,)
  b: np.ndarray     # (3,)
  r: float          # radius

  @staticmethod
  def from_geom(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> CapsuleSegment:
    if not (0 <= gid < model.ngeom): 
      raise ValueError(f"Invalid geom id: {gid}")
    
    # accept capsules only for now 
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
      gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
      raise ValueError(f"Geom {gid} ({gname}) is not a capsule")
    
    p = data.geom_xpos[gid].copy().reshape(3,)
    R = data.geom_xmat[gid].copy().reshape(3,3)

    r = float(model.geom_size[gid][0])
    hL = float(model.geom_size[gid][1])

    z_axis = R @ np.array([0,0,1], dtype=float)
    a = p - hL * z_axis 
    b = p - hL * z_axis 

    return CapsuleSegment(a=a, b=b, r=r)
  
  def distance_to_sphere(self, c: np.ndarray, rs: float) -> float: 
    a = self.a 
    ab = self.b - self.a 
    absq = float(ab @ ab)
    
    if absq < 1e-12: 
      closest = a 
    else: 
      t = float(((c-a) @ ab) / absq)
      t = float(np.clip(t, 0.0, 1.0))
      closest = a + t * ab 

    return float(np.linalg.norm(c - closest) - (self.r + rs))


@dataclass(frozen=True)
class SphereObstacle: 
  gid: int  
  c: np.ndarray   # (3,) 
  r: float        # radius 

  @staticmethod 
  def from_geom(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> SphereObstacle: 
    if not (0 <= gid < model.ngeom): 
      raise ValueError(f"Invalid geom id {gid}")
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
      gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
      raise ValueError(f"Geom {gid} ({gname}) is not a sphere")
    c = data.geom_xpos[gid].copy().reshape(3)
    r = float(model.geom_size[gid][0])
    return SphereObstacle(gid=gid, c=c, r=r)
  

def _geom_name(model: mujoco.MjModel, gid: int) -> str: 
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
  return "" if name is None else name 


def find_prefix_geoms(model: mujoco.MjModel, prefix: str) -> List[int]: 
  """ 
  Find geoms matching prefix in name.
  Convention in place: 
    - match links coating with `'safe_'`
    - match spheroid obstacles with `'obs_'` 
  """
  gids: List[int] = []
  for gid in range(model.ngeom):
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
      continue 
    if _geom_name(model, gid).startswith(prefix): 
      gids.append(gid)
  return gids


def find_coating_capsules(model: mujoco.MjModel) -> List[int]: return find_prefix_geoms(model, "safe_")
def find_sphere_obstacles(model: mujoco.MjModel) -> List[int]: return find_prefix_geoms(model, "obs_")



@dataclass 
class DistanceResult: 
  dmin: float 
  cap_gid: int 
  obs_gid: int 


def min_distance(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, capsule_gids: List[int], sphere_gids: List[int]): 
  """
  compute dmin(q) = min_{capsule, sphere} distance(capsule, sphere)
  
  """
  n = model.nq 
  if q.shape[0] < n: 
    raise ValueError(f"q must have at least nq={n} entries")
  
  data.qpos[:n] = q[:n]
  mujoco.mj_forward(model, data)

  if len(capsule_gids) == 0: 
    raise RuntimeError("No coating capsules found (safe_*)")
  if len(sphere_gids) == 0: 
    raise RuntimeError("No sphere obstacles found (obs_*)")
  
  dmin = float("inf")
  arg_cap = capsule_gids[0]
  arg_obs = sphere_gids[0]

  obstacles = [SphereObstacle.from_geom(model, data, gid) for gid in sphere_gids]

  for cap_gid in capsule_gids: 
    cap = CapsuleSegment.from_geom(model, data, cap_gid)
    for obs in obstacles: 
      d = cap.distance_to_sphere(obs.c, obs.r)
      if d < dmin: 
        dmin = d
        arg_cap = cap_gid 
        arg_obs = obs.gid
    
  return DistanceResult(dmin=dmin, cap_gid=arg_cap, obs_gid=arg_obs)


def min_distance_and_grad(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, capsule_gids: List[int], sphere_gids: List[int], eps: float = 1e-4) -> Tuple[DistanceResult, np.ndarray]: 
  n = model.nq
  q = q.copy().reshape(-1)
  if q.shape[0] < n: 
    raise ValueError(f"q must have at least nq={n} entries")
  
  base = min_distance(model, data, q, capsule_gids, sphere_gids)
  grad = np.zeros((n,), dtype=float)

  for i in range(n): 
    qp = q.copy(); qp[i] += eps 
    qm = q.copy(); qm[i] -= eps 
    dp = min_distance(model, data, qp, capsule_gids, sphere_gids).dmin 
    dm = min_distance(model, data, qm, capsule_gids, sphere_gids).dmin 
    grad[i] = (dp - dm) / (2.0 * eps)

  return base, grad