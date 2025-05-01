import numpy as np
from matrixangle_rpy import RPYAngles

class RotmatrixRPY:
    def __init__(self):
        pass

    def compute(self, rpy_angle):
        if len(rpy_angle) != 3:
            raise ValueError("Vector rpy_angle phải có đúng 3 phần tử.")
        psi, theta, phi = map(float, rpy_angle)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        Rz0 = np.array([
            [c_psi, -s_psi, 0.0],
            [s_psi, c_psi, 0.0],
            [0.0, 0.0, 1.0]
        ])
        Ry1 = np.array([
            [c_theta, 0.0, s_theta],
            [0.0, 1.0, 0.0],
            [-s_theta, 0.0, c_theta]
        ])
        Rx2 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi, c_phi]
        ])
        R = Rz0 @ Ry1 @ Rx2

        return R

if __name__ == "__main__":
    try:
        rotmatrix_rpy = RotmatrixRPY()
        rpy_angle = np.array([3.1, -2.2, 4.3])
        R = rotmatrix_rpy.compute(rpy_angle)
        rpy_angles = RPYAngles()
        rpy_converted = rpy_angles.compute(R)

        print("Vector RPY đầu vào [psi, theta, phi] (radian):")
        print(rpy_angle)
        print("\nMa trận xoay R:")
        print(R)
        print("\nVector RPY chuyển ngược từ R [rz, ry, rx] (radian):")
        print(rpy_converted)
        print("\nVector RPY chuyển ngược từ R [rz, ry, rx] (độ):")
        print(np.degrees(rpy_converted))
        print("\nKiểm tra orthogonal (R @ R.T):")
        print(np.round(R @ R.T, 6))
        print("Determinant của R:", np.round(np.linalg.det(R), 6))
        print("\nSai số giữa RPY đầu vào và RPY chuyển ngược (radian):")
        print(rpy_angle - rpy_converted)

    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")