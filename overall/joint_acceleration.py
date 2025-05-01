import numpy as np
import json
import os
from coriolis_c import CoriolisMatrix
from gravity_g import GravityMatrix
from inertia_m import InertiaMatrix

class JointAcceleration:
    def __init__(self, json_path="input/robot_parameters.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} không tồn tại.")
        
        with open(json_path, 'r') as f:
            params = json.load(f)
        self.g = float(params.get('g', 9.81))
        self.inertia = InertiaMatrix(json_path)
        self.coriolis = CoriolisMatrix(json_path)
        self.gravity = GravityMatrix(json_path)

    def compute(self, ins):
        """
        Tính gia tốc khớp q2dot (qddot) từ inputs.
        Đầu vào: ins - vector 21 phần tử [qd(7), q(7), u(7)]
        Đầu ra: q2dot - vector gia tốc khớp (7x1)
        """
        if len(ins) != 21:
            raise ValueError("Vector ins phải có đúng 21 phần tử.")

        # Trích xuất inputs
        qd = np.array(ins[0:7])  # Vận tốc khớp (7x1)
        q = np.array(ins[7:14])  # Góc khớp (7x1)
        u = np.array(ins[14:21])  # Lực điều khiển (7x1)

        # Tính các ma trận và vector
        M_mass = self.inertia.compute(q)  # Ma trận quán tính (7x7)
        C = self.coriolis.compute(q, qd)  # Ma trận Coriolis (7x7)
        gq = self.gravity.compute(q)  # Vector lực hấp dẫn (7x1)

        # Ma trận đơn vị
        E = np.eye(7)

        # Tính gia tốc khớp: q2dot = M^-1 * (u - C*qd - gq)
        try:
            M_inv = np.linalg.inv(M_mass)
        except np.linalg.LinAlgError:
            raise ValueError("Ma trận quán tính M không khả nghịch (singular).")

        q2dot = M_inv @ (E @ u - C @ qd - gq)

        return q2dot

if __name__ == "__main__":
    try:
        # Tạo instance của JointAcceleration
        ja = JointAcceleration()

        # Vector ins mẫu
        ins = np.ones(21)  # Tương tự MATLAB: ins = ones(1, 21)
        q2dot = ja.compute(ins)

        # In kết quả
        print("Vector ins:")
        print(ins)
        print("\nGia tốc khớp q2dot (rad/s^2):")
        print(q2dot)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except ImportError as e:
        print(f"Lỗi import: {e}. Vui lòng kiểm tra các file inertia_matrix.py, coriolis_matrix.py, gravity_matrix.py.")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")