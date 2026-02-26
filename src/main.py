from __future__ import annotations
from typing import Tuple

import numpy as np 
import mujoco

from ik import solve_ik, target_orientation
from interface import RobotInterface
from mpc_controller import MPCController
from verification import debug_metadata


def _load_model(path: str) -> Tuple[mujoco.MjModel, mujoco.MjData]: 
  """ loads scene/model from xml path """
  model = mujoco.MjModel.from_xml_path(path)
  data  = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _main(): 
  # load model, data
  model, data = _load_model("data/kuka_iiwa_14/scene.xml")
  iface = RobotInterface.handle(
      model=model,
      target_name="target_box",
      ee_name="attachment_site",
      zd=np.array([0,0,-1], dtype=float),
      up=np.array([0,1,0], dtype=float),
  )

  mpc = MPCController(model, data, iface, N=15, trigger_view=True)
  mpc.step_all(steps=4000)


if __name__ == "__main__": 
  _main()