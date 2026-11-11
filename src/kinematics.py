import numpy as np
import mujoco


class DifferentialIK:
    """IK różniczkowe 6D (translacja + rotacja) dla punktu TCP robota."""

    def __init__(self, model, data, site_name="gripper_pinch", num_joints=6, damping=0.02):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.num_joints = num_joints
        self.damping = damping

        # Weryfikacja i pobranie ID site'a (obsługa prefiksu gripper_)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # Bufory na macierze Jakobiego translacji (jacp) i rotacji (jacr)
        self._jacp_buf = np.zeros((3, model.nv))
        self._jacr_buf = np.zeros((3, model.nv))

    def get_jacobian_6d(self, data_ref=None) -> np.ndarray:
        """Zwraca macierz Jakobiego 6x6 (translacja + rotacja) dla ramienia."""
        d = data_ref if data_ref is not None else self.data
        mujoco.mj_jacSite(self.model, d, self._jacp_buf, self._jacr_buf, self.site_id)
        J6 = np.vstack([self._jacp_buf[:, :self.num_joints], self._jacr_buf[:, :self.num_joints]])
        return J6

    def damped_pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
        """Tłumiony odwrotny Jakobian (Levenberg-Marquardt)."""
        m, _ = J.shape
        reg = (self.damping ** 2) * np.eye(m)
        return J.T @ np.linalg.inv(J @ J.T + reg)

    def solve_ik_6d(self, target_pos: np.ndarray, target_rot: np.ndarray, q_init: np.ndarray, max_steps=100) -> tuple[np.ndarray, bool]:
        """Iteracyjne rozwiązywanie IK 6D z zachowaniem orientacji TCP."""
        data_temp = mujoco.MjData(self.model)
        q_calc = q_init.copy()

        for step in range(max_steps):
            data_temp.qpos[:self.num_joints] = q_calc
            mujoco.mj_forward(self.model, data_temp)

            curr_pos = data_temp.site_xpos[self.site_id].copy()
            curr_rot = data_temp.site_xmat[self.site_id].reshape(3, 3).copy()

            # Błąd translacji (XYZ)
            err_pos = target_pos - curr_pos

            # Błąd orientacji (rotacja macierzowa -> macierz antysymetryczna)
            R_err = 0.5 * (target_rot @ curr_rot.T - curr_rot @ target_rot.T)
            err_rot = np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])

            err_6d = np.hstack([err_pos, err_rot])

            # Warunek zbieżności: błąd translacji < 1mm oraz rotacji < 0.01 rad
            if np.linalg.norm(err_pos) < 1e-3 and np.linalg.norm(err_rot) < 1e-2:
                return q_calc, True

            J = self.get_jacobian_6d(data_temp)
            J_inv = self.damped_pseudo_inverse(J)
            
            # Krok poprawki kątów
            q_calc += J_inv @ (0.5 * err_6d)

        return q_calc, False