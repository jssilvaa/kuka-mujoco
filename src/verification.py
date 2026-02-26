from __future__ import annotations

import mujoco
import numpy as np

def debug_metadata(model: mujoco.MjModel, data: mujoco.MjData): 
  """ print metadata at the end of controller run """
  nq = model.nq

  # qfrc metadata 
  print("qacc norm:", np.linalg.norm(data.qacc[:nq]))
  print("qfrc_constraint norm:", np.linalg.norm(data.qfrc_constraint[:nq]))
  print("qfrc_passive norm:", np.linalg.norm(data.qfrc_passive[:nq]))
  print("qfrc_bias:", data.qfrc_bias[:nq])
  print("qfrc_actuator:", data.qfrc_actuator[:nq])

  # distance to joint limits 
  q = data.qpos[:nq]
  lo = model.jnt_range[:nq,0]
  hi = model.jnt_range[:nq,1]
  print("min dist to lo:", np.min(q-lo))
  print("min dist to hi:", np.min(hi-q))

  # contacts info 
  print("ncon:", data.ncon)
  for i in range(data.ncon):
      c = data.contact[i]
      g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
      g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
      print(i, g1, "<->", g2, "dist:", c.dist)