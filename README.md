# kuka-mujoco

KUKA LBR iiwa 14 in MuJoCo with:

-   A simple task-based inverse kinematics via DLS. 
-   A RHC loop running MPC on the end-effector, aiming to track a simple target within the robot's reachable workspace.

![iiwa_14](data/kuka_iiwa_14/iiwa_14.png)

---
## Model source (MuJoCo Menagerie)

All credits for the KUKA iiwa 14 model and assets go to:

-   MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14

This repository includes a local copy of the model under
`data/kuka_iiwa_14/`.

The scene is modified to include: 
- target bodies for end-effector
tracking 
- obstacles (for future clearance constraints)

--- 

## Robot model in this project

-   7 revolute joints (`joint1` ... `joint7`)
-   End-effector site: `attachment_site`
-   Position target body: `target_box`

Relevant files: - Robot XML: `data/kuka_iiwa_14/iiwa14.xml` - Scene XML:
`data/kuka_iiwa_14/scene.xml`

---

# MPC Formulation 

The main entry point is: `src/main.py`

This runs a receding-horizon control (RHC) loop that repeatedly solves a convex QP approximation (SQP) of the nonlinear tracking problem.

### Problem
State: `q_k` (joint angles at stage k)

Control: `u_k` (joint angle increments)

Discrete dynamics: `q_{k+1} = q_k + h * u_k`

At each time step, the nonlinear task residual is linearized: `e_k(q_k + Δq_k) ≈ e_k + J_k Δq_k`

The QP minimizes:

-   Task tracking error (quadratic in `Δq`)
-   Control effort

Subject to:

-   Linearized dynamics constraints
-   Box constraints on control and actuator limits

The QP is assembled and solved in `src/rti_qp.py` via OSQP. 

# Setup

Create environment:

``` bash
conda create -n kukaconda python=3.11
conda activate kukaconda
python -m pip install -r requirements.txt
```

Run:

``` bash
conda run -n kukaconda python src/main.py
```

On macOS:

``` bash
conda run -n kukaconda mjpython src/main.py
```

---

# References

-   MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

-   Grüne & Pannek: *Nonlinear Model Predictive Control*

-   Rawlings, Mayne, Diehl: *Model Predictive Control: Theory, Computation, and Design*

-   Lynch & Park: *Modern Robotics*
