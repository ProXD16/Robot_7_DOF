import json
import numpy as np
import os

class OrientationMatrix:
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

    def compute(self, q):
        """Tính ma trận cosin hướng R7 dựa trên q (góc khớp)."""
        if len(q) != 7:
            raise ValueError("Vector q phải có đúng 7 phần tử.")
        R7 = np.zeros((3, 3))
        q1, q2, q3, q4, q5, q6, q7 = map(float, q)
        s1, c1 = np.sin(q1), np.cos(q1)
        s2, c2 = np.sin(q2), np.cos(q2)
        s3, c3 = np.sin(q3), np.cos(q3)
        s4, c4 = np.sin(q4), np.cos(q4)
        s5, c5 = np.sin(q5), np.cos(q5)
        s6, c6 = np.sin(q6), np.cos(q6)
        s7, c7 = np.sin(q7), np.cos(q7)

        R7[0,0] = (
            c7 * c6 * c5 * c4 * c1 * c2 * c3 + c7 * c6 * c5 * c4 * s1 * s3 +
            c7 * c6 * c5 * c1 * s2 * s4 + c7 * c6 * s5 * c1 * c2 * s3 -
            c7 * c6 * s5 * s1 * c3 + c7 * s6 * s4 * c1 * c2 * c3 +
            c7 * s6 * s4 * s1 * s3 - c7 * s6 * c1 * s2 * c4 +
            s7 * s5 * c4 * c1 * c2 * c3 + s7 * s5 * c4 * s1 * s3 +
            s7 * s5 * c1 * s2 * s4 - s7 * c5 * c1 * c2 * s3 +
            s7 * c5 * s1 * c3 - s7 * c6 * c5 * c4 * c1 * c2 * c3 -
            s7 * c6 * c5 * c4 * s1 * s3 - s7 * c6 * c5 * c1 * s2 * s4 -
            s7 * c6 * s5 * c1 * c2 * s3 + s7 * c6 * s5 * s1 * c3 -
            s7 * s6 * s4 * c1 * c2 * c3 - s7 * s6 * s4 * s1 * s3 +
            s7 * s6 * c1 * s2 * c4 + c7 * s5 * c4 * c1 * c2 * c3 +
            c7 * s5 * c4 * s1 * s3 + c7 * s5 * c1 * s2 * s4 -
            c7 * c5 * c1 * c2 * s3 + c7 * c5 * s1 * c3
        )
        R7[0,1] = (
            s6 * c5 * c4 * c1 * c2 * c3 + s6 * c5 * c4 * s1 * s3 +
            s6 * c5 * c1 * s2 * s4 + s6 * s5 * c1 * c2 * s3 -
            s6 * s5 * s1 * c3 - c6 * s4 * c1 * c2 * c3 -
            c6 * s4 * s1 * s3 + c6 * c1 * s2 * c4
        )
        R7[0,2] = (
            c7 * c6 * c5 * s2 * c3 * c4 - c7 * c6 * c5 * c2 * s4 +
            c7 * c6 * s2 * s3 * s5 + c7 * s6 * s2 * c3 * s4 +
            c7 * s6 * c2 * c4 + s7 * s5 * s2 * c3 * c4 -
            s7 * s5 * c2 * s4 - s7 * s2 * s3 * c5 -
            s7 * c6 * c5 * s2 * c3 * c4 + s7 * c6 * c5 * c2 * s4 -
            s7 * c6 * s2 * s3 * s5 - s7 * s6 * s2 * c3 * s4 -
            s7 * s6 * c2 * c4 + c7 * s5 * s2 * c3 * c4 -
            c7 * s5 * c2 * s4 - c7 * s2 * s3 * c5
        )
        R7[1,0] = (
            c7 * c6 * c5 * c4 * s1 * c2 * c3 - c7 * c6 * c5 * c4 * c1 * s3 +
            c7 * c6 * c5 * s1 * s2 * s4 + c7 * c6 * s5 * s1 * c2 * s3 +
            c7 * c6 * s5 * c1 * c3 + c7 * s6 * s4 * s1 * c2 * c3 -
            c7 * s6 * s4 * c1 * s3 - c7 * s6 * s1 * s2 * c4 +
            s7 * s5 * c4 * s1 * c2 * c3 - s7 * s5 * c4 * c1 * s3 +
            s7 * s5 * s1 * s2 * s4 - s7 * c5 * s1 * c2 * s3 -
            s7 * c5 * c1 * c3 - s7 * c6 * c5 * c4 * s1 * c2 * c3 +
            s7 * c6 * c5 * c4 * c1 * s3 - s7 * c6 * c5 * s1 * s2 * s4 -
            s7 * c6 * s5 * s1 * c2 * s3 - s7 * c6 * s5 * c1 * c3 -
            s7 * s6 * s4 * s1 * c2 * c3 + s7 * s6 * s4 * c1 * s3 +
            s7 * s6 * s1 * s2 * c4 + c7 * s5 * c4 * s1 * c2 * c3 -
            c7 * s5 * c4 * c1 * s3 + c7 * s5 * s1 * s2 * s4 -
            c7 * c5 * s1 * c2 * s3 - c7 * c5 * c1 * c3
        )
        R7[1,1] = (
            s6 * c5 * c4 * s1 * c2 * c3 - s6 * c5 * c4 * c1 * s3 +
            s6 * c5 * s1 * s2 * s4 + s6 * s5 * s1 * c2 * s3 +
            s6 * s5 * c1 * c3 - c6 * s4 * s1 * c2 * c3 +
            c6 * s4 * c1 * s3 + c6 * s1 * s2 * c4
        )
        R7[1,2] = (
            c7 * c6 * c5 * s2 * c3 * c4 - c7 * c6 * c5 * c2 * s4 +
            c7 * c6 * s2 * s3 * s5 + c7 * s6 * s2 * c3 * s4 +
            c7 * s6 * c2 * c4 + s7 * s5 * s2 * c3 * c4 -
            s7 * s5 * c2 * s4 - s7 * s2 * s3 * c5 -
            s7 * c6 * c5 * s2 * c3 * c4 + s7 * c6 * c5 * c2 * s4 -
            s7 * c6 * s2 * s3 * s5 - s7 * s6 * s2 * c3 * s4 -
            s7 * s6 * c2 * c4 + c7 * s5 * s2 * c3 * c4 -
            c7 * s5 * c2 * s4 - c7 * s2 * s3 * c5
        )
        R7[2,0] = (
            s6 * c5 * s2 * c3 * c4 - s6 * c5 * c2 * s4 +
            s6 * s2 * s3 * s5 - c6 * s2 * c3 * s4 - c6 * c2 * c4
        )
        R7[2,1] = (
            s6 * c5 * s2 * c3 * c4 - s6 * c5 * c2 * s4 +
            s6 * s2 * s3 * s5 - c6 * s2 * c3 * s4 - c6 * c2 * c4
        )
        R7[2,2] = (
            c7 * c6 * c5 * s2 * c3 * c4 - c7 * c6 * c5 * c2 * s4 +
            c7 * c6 * s2 * s3 * s5 + c7 * s6 * s2 * c3 * s4 +
            c7 * s6 * c2 * c4 + s7 * s5 * s2 * c3 * c4 -
            s7 * s5 * c2 * s4 - s7 * s2 * s3 * c5 -
            s7 * c6 * c5 * s2 * c3 * c4 + s7 * c6 * c5 * c2 * s4 -
            s7 * c6 * s2 * s3 * s5 - s7 * s6 * s2 * c3 * s4 -
            s7 * s6 * c2 * c4 + c7 * s5 * s2 * c3 * c4 -
            c7 * s5 * c2 * s4 - c7 * s2 * s3 * c5
        )
        return R7

if __name__ == "__main__":
    try:
        orientation_matrix = OrientationMatrix()
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        R7 = orientation_matrix.compute(q)
        print("Ma trận cosin hướng R7:")
        print(R7)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")