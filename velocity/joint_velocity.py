import json
import numpy as np
import os
from jacobi_t import JacobiT
from jacobi_r import JacobiR
from joint_limit_optimizer import JointLimitOptimizer

class JointVelocity:
    def __init__(self, json_path="input/robot_parameters.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} không tồn tại.")
        
        with open(json_path, 'r') as f:
            params = json.load(f)

        self.d1 = params.get('d1', 0.0)
        self.d2 = params.get('d2', 0.0)
        self.d3 = params.get('d3', 0.0)
        self.d4 = params.get('d4', 0.0)
        self.d5 = params.get('d5', 0.0)
        self.d6 = params.get('d6', 0.0)
        self.d7 = params.get('d7', 0.0)
        self.a6 = params.get('a6', 0.0)
        self.g = params.get('g', 9.81)
        self.dt = 0.01
        self.K = 100 * np.eye(6) 
        self.err = np.zeros(6) 

    def compute(self, inputs):
        if len(inputs) != 33:
            raise ValueError("Vector inputs phải có đúng 33 phần tử.")
        rEvE = inputs[0:6]  
        oriang = inputs[6:12]  
        q = inputs[12:19]  
        if len(q) != 7:
            raise ValueError("Vector q phải có đúng 7 phần tử.")
        eta1 = rEvE[0:3]  
        vE = rEvE[3:6]  
        psi, theta, phi = oriang[0:3]  
        ome = oriang[3:6] 
        Q = np.array([
            [0.0, -np.sin(psi), np.cos(psi) * np.cos(theta)],
            [0.0, np.cos(psi), np.sin(psi) * np.cos(theta)],
            [1.0, 0.0, -np.sin(theta)]
        ])
        iQ = np.array([
            [np.cos(psi) * np.tan(theta), np.sin(psi) * np.tan(theta), 1.0],
            [-np.sin(psi), np.cos(psi), 0.0],
            [np.cos(psi) / np.cos(theta), np.sin(psi) / np.cos(theta), 0.0]
        ])
        eta1dot = vE 
        eta2dot = iQ @ ome  
        eta_dot = np.concatenate([eta1dot, eta2dot])  
        jacobi_t = JacobiT()
        jacobi_r = JacobiR()
        JT = jacobi_t.compute(q) 
        JR = jacobi_r.compute(q)  
        J = np.vstack([JT[:3, :], iQ @ JR])  
        JJT = J @ J.T  
        try:
            J_plus = J.T @ np.linalg.inv(JJT) 
        except np.linalg.LinAlgError:
            J_plus = J.T @ np.linalg.pinv(JJT)
        detJ = np.linalg.det(JJT)
        print(f"det(J*J'): {detJ}")  
        z0_optimizer = JointLimitOptimizer()
        z0 = z0_optimizer.compute_z0(q)
        En = np.eye(7) 
        qdot = J_plus @ (eta_dot + self.K @ self.err) + 10 * (En - J_plus @ J) @ z0
        return qdot

if __name__ == "__main__":
    try:
        joint_velocity = JointVelocity()
        q0 = np.array([2.0635, 4.4715, -0.1331, 2.6666, 0.1328, 4.4819, 2.7866])
        rEvE = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])  
        oriang = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]) 
        inputs = np.concatenate([rEvE, oriang, q0, np.zeros(14)]) 
        qdot = joint_velocity.compute(inputs)
        print("Vector inputs:")
        print(inputs)
        print("\nVận tốc khớp qdot (rad/s):")
        print(qdot)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")