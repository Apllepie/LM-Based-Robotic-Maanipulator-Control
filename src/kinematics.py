import numpy as np
import mujoco


class DifferentialIK:
    """IK różniczkowe wyliczające macierz Jakobiego dla punktu TCP (site)."""

    def __init__(self, model, data, site_name="pinch", num_joints=6, damping=1e-2):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.num_joints = num_joints
        self.damping = damping

        # Bufor na macierz Jakobiego translacji (3 x model.nv)
        self._jac_buf = np.zeros((3, model.nv))

    def get_jacobian(self) -> np.ndarray:
        """Zwraca macierz Jakobiego dla punktu pinch dociętą do 6 przegubów ramienia."""
        mujoco.mj_jacSite(self.model, self.data, self._jac_buf, None, self.site_id)
        return self._jac_buf[:, :self.num_joints]

    def damped_pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
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
        J = self.get_jacobian()
        J_inv = self.damped_pseudo_inverse(J)

        # 1. Błąd w przestrzeni XYZ
        error_xyz = target_pos - current_pos
        dq_task = J_inv @ (kp_task * error_xyz)

        # 2. Rzutnik przestrzeni zerowej (operujący tylko na 6 przegubach ramienia)
        I = np.eye(self.num_joints)
        null_space_proj = I - (J_inv @ J)

        q_curr_arm = q_current[:self.num_joints]
        q_home_arm = q_home[:self.num_joints]

        dq_null = null_space_proj @ (kp_null * (q_home_arm - q_curr_arm))

        # 3. Suma i ograniczenie prędkości
        dq = dq_task + dq_null
        return np.clip(dq, -max_dq, max_dq)