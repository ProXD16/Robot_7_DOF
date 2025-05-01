import numpy as np
from orientation_r7 import OrientationMatrix

class RPYAngles:
    def __init__(self):
        pass

    def compute(self, R):
        if R.shape != (3, 3):
            raise ValueError("Ma trận R phải có kích thước 3x3.")
        rz, ry, rx = 0.0, 0.0, 0.0
        if abs(R[2, 0]) != 1:
            stheta = -R[2, 0]
            ctheta = np.sqrt(1 - stheta**2)
            ry = np.arctan2(stheta, ctheta)  
            rz = np.arctan2(R[1, 0] / ctheta, R[0, 0] / ctheta) 
            rx = np.arctan2(R[2, 1] / ctheta, R[2, 2] / ctheta)  #
        elif R[2, 0] == -1:
            ry = np.pi / 2  
            rz = 0.0  
            rx = np.arctan2(R[0, 1], R[1, 1]) + rz 
        else:  
            ry = -np.pi / 2  
            rz = 0.0 
            rx = np.arctan2(-R[0, 1], R[1, 1]) - rz  
        return np.array([rz, ry, rx])

if __name__ == "__main__":
    try:
        orientation_matrix = OrientationMatrix()
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        R7 = orientation_matrix.compute(q)
        rpy_angles = RPYAngles()
        rpy = rpy_angles.compute(R7)

        print("Ma trận xoay R7:")
        print(R7)
        print("\nCác góc RPY [rz, ry, rx] (radian):")
        print(rpy)
        print("\nCác góc RPY [rz, ry, rx] (độ):")
        print(np.degrees(rpy))

        print("\nKiểm tra orthogonal (R7 @ R7.T):")
        print(np.round(R7 @ R7.T, 6))
        print("Determinant của R7:", np.round(np.linalg.det(R7), 6))

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")