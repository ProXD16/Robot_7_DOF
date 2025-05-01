import json
import numpy as np
import os
from inertia_m import InertiaMatrix
from coriolis_c import CoriolisMatrix
from gravity_g import GravityMatrix

class ControlInput:
    def __init__(self, json_path="input/robot_parameters.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} không tồn tại.")
        
        with open(json_path, 'r') as f:
            params = json.load(f)

        self.g = float(params.get('g', 9.81))  # Gia tốc trọng trường, mặc định 9.81 m/s²
        self.inertia = InertiaMatrix(json_path)
        self.coriolis = CoriolisMatrix(json_path)
        self.gravity = GravityMatrix(json_path)

    def compute(self, inputs):
        """
        Tính vector lực điều khiển u (mô-men xoắn) từ inputs.
        Đầu vào: inputs - vector 26 phần tử [y(7), q(7), qdot(7), zeros(5)]
        Đầu ra: u - vector mô-men xoắn (7x1), đơn vị Nm
        """
        if len(inputs) != 26:
            raise ValueError("Vector inputs phải có đúng 26 phần tử.")
        
        # Trích xuất inputs
        y = np.array(inputs[0:7])    # Gia tốc khớp (7x1, rad/s²)
        q = np.array(inputs[7:14])   # Góc khớp (7x1, rad)
        qdot = np.array(inputs[14:21])  # Vận tốc khớp (7x1, rad/s)

        if len(q) != 7 or len(qdot) != 7 or len(y) != 7:
            raise ValueError("Vector y, q, qdot phải có đúng 7 phần tử.")

        # Tính các ma trận và vector
        M = self.inertia.compute(q)      # Ma trận quán tính (7x7)
        C = self.coriolis.compute(q, qdot)  # Ma trận Coriolis (7x7)
        G = self.gravity.compute(q)      # Vector lực hấp dẫn (7x1)

        # Tính lực điều khiển: u = M*y + C*qdot + G
        u = M @ y + C @ qdot + G

        return u

if __name__ == "__main__":
    try:
        # Tạo instance của ControlInput
        control_input = ControlInput()

        # Vector inputs mẫu (tương tự ins = ones(1,26) trong MATLAB)
        inputs = np.ones(26)
        u = control_input.compute(inputs)

        # In kết quả
        print("Vector inputs:")
        print(inputs)
        print("\nLực điều khiển u (Nm):")
        print(u)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except ImportError as e:
        print(f"Lỗi import: {e}. Vui lòng kiểm tra các file inertia_m.py, coriolis_c.py, gravity_g.py.")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")