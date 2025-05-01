import numpy as np
from gravity_g import GravityMatrix
from inertia_m import InertiaMatrix
from coriolis_c import CoriolisMatrix

class RobotDynamics:
    def __init__(self, json_path="input/robot_parameters.json"):
        """Khởi tạo class RobotDynamics với các tham số từ file JSON."""
        self.gravity = GravityMatrix(json_path)
        self.inertia = InertiaMatrix(json_path)
        self.coriolis = CoriolisMatrix(json_path)

    def compute_torque(self, q, dq, ddq):
        """Tính toán mô-men xoắn tau dựa trên q, dq, ddq."""
        if len(q) != 7 or len(dq) != 7 or len(ddq) != 7:
            raise ValueError("Vector q, dq, ddq phải có đúng 7 phần tử.")

        # Tính ma trận quán tính M
        M = self.inertia.compute(q)
        # print(f"Shape of M: {M.shape}")

        # Tính vector lực hấp dẫn gq
        gq = self.gravity.compute(q)
        # print(f"Shape of gq: {gq.shape}")

        # Tính C(q, dq) * dq
        C = self.coriolis.compute(q, dq)
        # print(f"Shape of C: {C.shape}")

        dq_reshaped = dq.reshape(7, 1)
        # print(f"Shape of dq_reshaped: {dq_reshaped.shape}")

        C_dq = C @ dq_reshaped  # Matrix multiplication
        # print(f"Shape of C_dq: {C_dq.shape}")

        # Tính mô-men xoắn: tau = M * ddq + C(q, dq) * dq + g(q)
        ddq_reshaped = ddq.reshape(7, 1)
        # print(f"Shape of ddq_reshaped: {ddq_reshaped.shape}")

        tau = M @ ddq_reshaped + C_dq + gq
        # print(f"Shape of tau before return: {tau.shape}")

        # Ensure tau is a 1D array
        tau = tau.flatten()

        # print(f"Shape of tau after flattening: {tau.shape}")
        return tau