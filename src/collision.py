from __future__ import annotations 
from dataclasses import dataclass 
from typing import List, Tuple

import mujoco 
import numpy as np 

SUPPORTED_COATING_TYPES = {
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_SPHERE,
}


@dataclass(frozen=True)
class CapsuleSegment: 
  a: np.ndarray     # (3,)
  b: np.ndarray     # (3,)
  r: float          # radius

  @staticmethod
  def from_geom(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> CapsuleSegment:
    if not (0 <= gid < model.ngeom): 
      raise ValueError(f"Invalid geom id: {gid}")

    gtype = model.geom_type[gid]    
    p = data.geom_xpos[gid].copy().reshape(3,)
    R = data.geom_xmat[gid].copy().reshape(3,3)
    size = model.geom_size[gid]

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE: 
      r = float(size[0])
      return CapsuleSegment(a=p,b=p,r=r)
    
    elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
      r = float(size[0])
      hL = float(size[1])
      z = R @ np.array([0.0, 0.0, 1.0], dtype=float)
      a = p - hL * z
      b = p + hL * z 
      return CapsuleSegment(a=a,b=b,r=r)

    else: 
      raise ValueError(f"Unsupported coating geom type for gid={gid}: {gtype}")


  def distance_to_sphere(self, c: np.ndarray, rs: float) -> float: 
    ab = self.b - self.a 
    absq = float(ab @ ab)
    if absq < 1e-12: 
      closest = self.a 
    else: 
      t = float(((c-self.a) @ ab) / absq)
      t = float(np.clip(t, 0.0, 1.0))
      closest = self.a + t * ab 
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
  

def distance_pair(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, coat_gid: int, obs_gid: int) -> float: 
  """ Computes the distance between a (coating, sphere) pair """
  n = model.nq
  qpos0 = data.qpos[:n].copy()
  qvel0 = data.qvel[:n].copy()
  data.qpos[:n] = q 
  data.qvel[:n] = 0.0 
  mujoco.mj_forward(model, data)
  
  cap = CapsuleSegment.from_geom(model, data, coat_gid)
  obs = SphereObstacle.from_geom(model, data, obs_gid)
  d = cap.distance_to_sphere(obs.c, obs.r)
  
  data.qpos[:n] = qpos0 
  data.qvel[:n] = qvel0
  mujoco.mj_forward(model, data)
  
  return float(d)


def _geom_name(model: mujoco.MjModel, gid: int) -> str: 
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
  return "" if name is None else name 


def find_prefix_geoms(model: mujoco.MjModel, prefix: str, geom_type: mujoco.mjtGeom) -> List[int]:
    gids: List[int] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != geom_type:
            continue
        if _geom_name(model, gid).startswith(prefix):
            gids.append(gid)
    return gids

def find_sphere_obstacles(model: mujoco.MjModel) -> List[int]:
    return find_prefix_geoms(model, "obs_", mujoco.mjtGeom.mjGEOM_SPHERE)


def find_coating_geoms(
    model: mujoco.MjModel,
    group_id: int = 3,
    chosen_links: set[str] | None = None,
) -> List[int]:
    if chosen_links is None:
        chosen_links = {"link3","link4","link5","link6","link7"}

    gids: List[int] = []
    for gid in range(model.ngeom):
        if int(model.geom_group[gid]) != group_id:
            continue
        if model.geom_type[gid] not in SUPPORTED_COATING_TYPES:
            continue

        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname in chosen_links:
            gids.append(gid)

    return reduce_coatings(model, gids) 


def reduce_coatings(model: mujoco.MjModel, coat_gids: List[int], per_link: int = 2) -> List[int]:
    buckets = {}
    for gid in coat_gids:
        bid = int(model.geom_bodyid[gid])
        buckets.setdefault(bid, []).append(gid)

    out = []
    for bid, gids in buckets.items():
        gids_sorted = sorted(gids, key=lambda g: float(model.geom_size[g][0]), reverse=True)
        out.extend(gids_sorted[:per_link])
    return out


@dataclass 
class DistanceResult: 
  dmin: float 
  coat_gid: int 
  obs_gid: int 


def min_distance(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, coat_gids: List[int], sphere_gids: List[int]): 
  """
  Computes the minimal distance between a coating object (i.e. object collision mesh, modelled as a geom primitive) and environment obstacles (spheres). 
    `dmin(q)` = `min_{coating, sphere} distance(coating, sphere)`
  Returns: 
    `dmin`: minimum distance between collision mesh and environment obstacle  
    `coat_gid, obs_gid`: argmin pair 
  """
  n = model.nq 
  if q.shape[0] < n: 
    raise ValueError(f"q must have at least nq={n} entries")
  
  qpos0 = data.qpos[:n].copy()
  qvel0 = data.qvel[:n].copy()

  data.qpos[:n] = q[:n]
  data.qvel[:n] = 0.0 
  mujoco.mj_forward(model, data)

  if len(coat_gids) == 0: 
    raise RuntimeError("No coating geoms found in collision group=3")
  if len(sphere_gids) == 0: 
    raise RuntimeError("No sphere obstacles found (obs_*)")
  
  dmin = float("inf")
  arg_coat = coat_gids[0]
  arg_obs = sphere_gids[0]

  obstacles = [SphereObstacle.from_geom(model, data, gid) for gid in sphere_gids]

  for coat_gid in coat_gids: 
    cap = CapsuleSegment.from_geom(model, data, coat_gid)
    for obs in obstacles: 
      d = cap.distance_to_sphere(obs.c, obs.r)
      if d < dmin: 
        dmin = d
        arg_coat = coat_gid 
        arg_obs = obs.gid

  data.qpos[:n] = qpos0[:n]
  data.qvel[:n] = qvel0[:n]
  mujoco.mj_forward(model, data)
    
  return DistanceResult(dmin=dmin, coat_gid=arg_coat, obs_gid=arg_obs)


def min_distance_and_grad(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, capsule_gids: List[int], sphere_gids: List[int], eps: float = 1e-4) -> Tuple[DistanceResult, np.ndarray]: 
  n = model.nq
  q = q.copy().reshape(-1)
  if q.shape[0] < n: 
    raise ValueError(f"q must have at least nq={n} entries")
  
  base: DistanceResult = min_distance(model, data, q, capsule_gids, sphere_gids)
  coat_gid, obs_gid = base.coat_gid, base.obs_gid
  grad: np.ndarray = np.zeros((n,), dtype=float)

  for i in range(n): 
    qp = q.copy(); qp[i] += eps 
    qm = q.copy(); qm[i] -= eps 
    dp = distance_pair(model, data, qp, coat_gid, obs_gid) 
    dm = distance_pair(model, data, qm, coat_gid, obs_gid)
    grad[i] = (dp - dm) / (2.0 * eps)
  
  # prevent spikes when changing obstacles
  gmax = 50.0  
  ngrad = float(np.linalg.norm(grad))
  if ngrad > gmax:
      grad *= (gmax / ngrad)

  return base, grad


if __name__ == "__main__": 
  from main import _load_model
  model, _ = _load_model("data/kuka_iiwa_14/scene.xml")
  print("ncoat", len(find_coating_geoms(model)), "nobs", len(find_sphere_obstacles(model)))