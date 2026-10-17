import numpy as np
import mujoco


class DifferentialIK:
    """ Implement a simple differential inverse kinematics solver.

        The solver uses the Jacobian matrix to convert XYZ position targets
        into joint angles for a robotic manipulator.
    """

    def __init__(self, model, data, body_name="wrist_3_link", num_joints=6, damping=1e-2):
        self.model = model
        self.data = data
        self.body_name = body_name
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.num_joints = num_joints
        self.damping = damping

        # buffor for translational Jacobian (3 x num_joints)
        self._jac_buf = np.zeros((3, model.nv))

    def get_jacobian(self) -> np.ndarray:
        """Returns the translational Jacobian matrix (3 x num_joints)."""
        mujoco.mj_jacBody(self.model, self.data, self._jac_buf, None, self.body_id)
        return self._jac_buf[:, :self.num_joints]

    def damped_pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
        """Returns the damped pseudo-inverse of the Jacobian matrix."""
        reg = self.damping * np.eye(J.shape[0])
        return J.T @ np.linalg.inv(J @ J.T + reg)

    def compute_joint_velocities(
        self,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
        q_current: np.ndarray,
        q_home: np.ndarray,
        kp_task: float = 5.0,
        kp_null: float = 1.0,
        max_dq: float = 1.5,
    ) -> np.ndarray:
        """
        Computes the resulting joint velocities (dq):
        - dq_task: motion towards the XYZ target
        - dq_null: projection onto the null space (maintaining the frame near q_home)
        """
        J = self.get_jacobian()
        J_inv = self.damped_pseudo_inverse(J)

        # 1. Prędkość w przestrzeni zadania (XYZ)
        error_xyz = target_pos - current_pos
        dq_task = J_inv @ (kp_task * error_xyz)

        # 2. Rzutnik przestrzeni zerowej (Null-space projection)
        I = np.eye(self.num_joints)
        null_space_proj = I - (J_inv @ J)
        dq_null = null_space_proj @ (kp_null * (q_home - q_current))

        # 3. Suma i ograniczenie wartości (clamping)
        dq = dq_task + dq_null
        return np.clip(dq, -max_dq, max_dq)