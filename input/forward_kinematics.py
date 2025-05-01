import numpy as np
import json
import os
from orientation_r7 import OrientationMatrix
from matrixangle_rpy import RPYAngles

class ForwardKinematics:
    def __init__(self, json_path="input/robot_parameters.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} không tồn tại.")
        
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        # Tham số khoảng cách
        self.d1 = float(params.get('d1', 0.0))
        self.d2 = float(params.get('d2', 0.0))
        self.d3 = float(params.get('d3', 0.0))
        self.d4 = float(params.get('d4', 0.0))
        self.d5 = float(params.get('d5', 0.0))
        self.d6 = float(params.get('d6', 0.0))
        self.d7 = float(params.get('d7', 0.0))
        self.a6 = float(params.get('a6', 0.0))

        # Khởi tạo các đối tượng từ các file khác
        self.orientation_matrix = OrientationMatrix(json_path)
        self.rpy_angles = RPYAngles()

    def orientation_mt7(self, q):
        """
        Tính ma trận định hướng R7 bằng cách sử dụng OrientationMatrix từ orientation_r7.py
        """
        return self.orientation_matrix.compute(q)

    def rotmatrix_to_rpy_angles(self, R):
        """
        Chuyển ma trận R thành các góc RPY (yaw, pitch, roll) bằng cách sử dụng RPYAngles từ matrixangle_rpy.py
        """
        return self.rpy_angles.compute(R)

    def compute(self, q):
        if len(q) != 7:
            raise ValueError("Vector q phải có đúng 7 phần tử.")
        
        q1, q2, q3, q4, q5, q6, q7 = map(float, q)
        s1, c1 = np.sin(q1), np.cos(q1)
        s2, c2 = np.sin(q2), np.cos(q2)
        s3, c3 = np.sin(q3), np.cos(q3)
        s4, c4 = np.sin(q4), np.cos(q4)
        s5, c5 = np.sin(q5), np.cos(q5)
        s6, c6 = np.sin(q6), np.cos(q6)
        s7, c7 = np.sin(q7), np.cos(q7)

        # Tính vị trí xE, yE, zE
        xe = (self.d5 * s4 * s1 * s3 - self.d5 * c1 * s2 * c4 + self.d5 * s4 * c1 * c2 * c3 +
              self.d4 * c1 * c2 * s3 - self.d4 * s1 * c3 + s1 * self.d2 + c1 * s2 * self.d3 +
              self.d7 * s6 * c5 * c4 * c1 * c2 * c3 + self.d7 * s6 * c5 * c4 * s1 * s3 +
              self.d7 * s6 * c5 * c1 * s2 * s4 + self.d7 * s6 * s5 * c1 * c2 * s3 -
              self.d7 * s6 * s5 * s1 * c3 - self.d7 * c6 * s4 * c1 * c2 * c3 -
              self.d7 * c6 * s4 * s1 * s3 + self.d7 * c6 * c1 * s2 * c4 +
              c6 * self.a6 * c5 * c4 * c1 * c2 * c3 + c6 * self.a6 * c5 * c4 * s1 * s3 +
              c6 * self.a6 * c5 * c1 * s2 * s4 + c6 * self.a6 * s5 * c1 * c2 * s3 -
              c6 * self.a6 * s5 * s1 * c3 + s6 * self.a6 * s4 * c1 * c2 * c3 +
              s6 * self.a6 * s4 * s1 * s3 - s6 * self.a6 * c1 * s2 * c4 +
              self.d6 * s5 * c4 * c1 * c2 * c3 + self.d6 * s5 * c4 * s1 * s3 +
              self.d6 * s5 * c1 * s2 * s4 - self.d6 * c5 * c1 * c2 * s3 + self.d6 * c5 * s1 * c3)

        ye = (self.d4 * c1 * c3 - self.d5 * s4 * c1 * s3 - self.d5 * s1 * s2 * c4 +
              self.d5 * s4 * s1 * c2 * c3 + self.d4 * s1 * c2 * s3 - c1 * self.d2 +
              c6 * self.a6 * c5 * c4 * s1 * c2 * c3 - c6 * self.a6 * c5 * c4 * c1 * s3 +
              c6 * self.a6 * c5 * s1 * s2 * s4 + c6 * self.a6 * s5 * s1 * c2 * s3 +
              c6 * self.a6 * s5 * c1 * c3 + s6 * self.a6 * s4 * s1 * c2 * c3 -
              s6 * self.a6 * s4 * c1 * s3 - s6 * self.a6 * s1 * s2 * c4 +
              self.d6 * s5 * c4 * s1 * c2 * c3 - self.d6 * s5 * c4 * c1 * s3 +
              self.d6 * s5 * s1 * s2 * s4 - self.d6 * c5 * s1 * c2 * s3 +
              s1 * s2 * self.d3 + self.d7 * s6 * c5 * c4 * s1 * c2 * c3 -
              self.d7 * s6 * c5 * c4 * c1 * s3 + self.d7 * s6 * c5 * s1 * s2 * s4 +
              self.d7 * s6 * s5 * s1 * c2 * s3 + self.d7 * s6 * s5 * c1 * c3 -
              self.d7 * c6 * s4 * s1 * c2 * c3 + self.d7 * c6 * s4 * c1 * s3 +
              self.d7 * c6 * s1 * s2 * c4 - self.d6 * c5 * c1 * c3)

        ze = (self.d7 * s6 * c5 * s2 * c3 * c4 - self.d7 * s6 * c5 * c2 * s4 +
              self.d7 * s6 * s2 * s3 * s5 - self.d7 * c6 * s2 * c3 * s4 - self.d7 * c6 * c2 * c4 +
              c6 * self.a6 * c5 * s2 * c3 * c4 - c6 * self.a6 * c5 * c2 * s4 +
              c6 * self.a6 * s2 * s3 * s5 + s6 * self.a6 * s2 * c3 * s4 + s6 * self.a6 * c2 * c4 +
              self.d6 * s5 * s2 * c3 * c4 - self.d6 * s5 * c2 * s4 - self.d6 * s2 * s3 * c5 +
              self.d5 * s2 * c3 * s4 + self.d5 * c2 * c4 + s2 * s3 * self.d4 - c2 * self.d3 + self.d1)

        R7 = self.orientation_mt7(q)
        x2 = self.rotmatrix_to_rpy_angles(R7)
        x1 = np.array([xe, ye, ze])
        x = np.concatenate([x1, x2])
        return x

if __name__ == "__main__":
    try:
        fk = ForwardKinematics()
        q = np.array([-90, 120, 20, -45, 45, 120, 90]) * np.pi / 180
        x = fk.compute(q)
        print("Vector q (radian):")
        print(q)
        print("\nKết quả động học thuận x = [xE, yE, zE, yaw, pitch, roll]:")
        print(x)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")