# kuka-mujoco

Numerical inverse kinematics (IK) for the KUKA LBR iiwa 14 in MuJoCo, using a damped least-squares (DLS) objective on both position and orientation error.

## Menagerie model

- MuJoCo model: [google-deepmind/mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14)

## KUKA iiwa 14 in this project

- 7 revolute joints (`joint1`..`joint7`)
- End-effector site: `attachment_site` (`data/kuka_iiwa_14/iiwa14.xml`)
- Position target body: `target_box` (`data/kuka_iiwa_14/scene.xml`)


## DLS on position + orientation

Minimize:

- Position error: `e_pos = p_target - p_ee`
- Orientation error: `e_rot = Log(R_target R_ee^T)^vee`
- Stacked error: `e = [e_pos; e_rot]`

MuJoCo Jacobians:

- `Jp`: translational Jacobian of the end-effector site
- `Jr`: rotational Jacobian of the end-effector site
- Orientation linearization: `Jl(e_rot)^(-1) Jr`

The update is:

- `Je = [Jp; Jl(e_rot)^(-1) Jr]`
- `dq = Je^T (Je Je^T + lambda^2 I)^(-1) e`
- `q <- clip(q + alpha * dq, joint_limits)`


## Setup

Create a conda environment and install dependencies:

```bash
conda create -n kukaconda python=3.11
conda activate kukaconda
python -m pip install -r requirements
```

Run IK:

```bash
conda run -n kukaconda python src/ik.py
```

## References

- KUKA iiwa model package (MuJoCo Menagerie): [README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/kuka_iiwa_14/README.md)
