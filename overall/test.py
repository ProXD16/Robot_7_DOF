import numpy as np
import json
import os
from trajectory_rectilinear_xt import RectilinearTrajectory
from trajectory_Orientation_Rt import OrientationTrajectory
from trajectory_curvilinear_xt import CurvilinearTrajectory
from joint_velocity import JointVelocity

class InputController:
    def __init__(self, json_path="input/robot_parameters.json"):
        """
        Khởi tạo bộ điều khiển với tham số từ file JSON
        
        Args:
            json_path (str): Đường dẫn đến file tham số robot
        """
        # Kiểm tra và đọc file cấu hình
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} không tồn tại.")
        
        with open(json_path, 'r') as f:
            params = json.load(f)

        # Thiết lập tham số robot
        self.d1 = params.get('d1', 0.0)
        self.d2 = params.get('d2', 0.0)
        self.d3 = params.get('d3', 0.0)
        self.d4 = params.get('d4', 0.0)
        self.d5 = params.get('d5', 0.0)
        self.d6 = params.get('d6', 0.0)
        self.d7 = params.get('d7', 0.0)
        self.a6 = params.get('a6', 0.0)
        self.g = params.get('g', 9.81)
        
        # Khởi tạo góc khớp ban đầu từ JSON
        self.q = np.array([
            params.get('q1', 0.0),
            params.get('q2', 0.0),
            params.get('q3', 0.0),
            params.get('q4', 0.0),
            params.get('q5', 0.0),
            params.get('q6', 0.0),
            params.get('q7', 0.0)
        ])
        
        # Tham số điều khiển
        self.T = params.get('T', 15.0)   # Thời gian mô phỏng
        self.dt = params.get('dt', 0.01) # Bước thời gian
        self.K = 100 * np.eye(6)         # Ma trận độ lợi
        self.err = np.zeros(6)           # Lỗi điều khiển
        self.time = 0.0                  # Thời gian hiện tại
        
        # Khởi tạo JointVelocity
        self.joint_velocity = JointVelocity(json_path)
        
        # Dữ liệu quỹ đạo
        self.input_log = None            # Ma trận lưu quỹ đạo
        self.column_names = [
            'time',                      # 0
            'x', 'y', 'z',               # 1-3: Vị trí
            'vx', 'vy', 'vz',           # 4-6: Vận tốc
            'psi', 'theta', 'phi',       # 7-9: Góc Euler
            'omega_x', 'omega_y', 'omega_z', # 10-12: Vận tốc góc
            'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', # 13-19: Góc khớp
            'qdot1', 'qdot2', 'qdot3', 'qdot4', 'qdot5', 'qdot6', 'qdot7', # 20-26: Vận tốc khớp
            'qddot1', 'qddot2', 'qddot3', 'qddot4', 'qddot5', 'qddot6', 'qddot7', # 27-33: Gia tốc khớp
            'r1', 'r2', 'r3',           # 34-36: Ma trận quay (hàng 1)
            'r4', 'r5', 'r6',           # 37-39: Ma trận quay (hàng 2)
            'r7', 'r8', 'r9'            # 40-42: Ma trận quay (hàng 3)
        ]

    def calculate_trajectory(self):
        """
        Tính toán quỹ đạo chuyển động đầy đủ (vị trí, định hướng, góc khớp, vận tốc khớp, gia tốc khớp)
        Lưu kết quả vào self.input_log với cấu trúc đã định nghĩa
        """
        # Khởi tạo các đối tượng quỹ đạo
        traj_rect = RectilinearTrajectory(json_file='input/robot_parameters.json')
        traj_orient = OrientationTrajectory(json_file='input/robot_parameters.json')
        
        # Tạo mảng thời gian
        t_values = np.arange(0.0, self.T + 1e-9, self.dt)
        num_steps = len(t_values)
        
        # Khởi tạo mảng lưu kết quả
        self.input_log = np.zeros((num_steps, 43))  # 43 cột: 36 (cũ) + 7 (qddot)
        
        # Khởi tạo góc khớp và vận tốc khớp trước đó
        q_current = self.q.copy()
        qdot_prev = np.zeros(7)  # Vận tốc khớp trước đó (khởi tạo bằng 0)
        
        # Tính toán quỹ đạo
        for i, t in enumerate(t_values):
            # Tính vị trí và vận tốc
            rEvE = traj_rect.compute_trajectory(t)  
            
            # Tính định hướng và vận tốc góc
            RPY, R, oriang = traj_orient.compute_trajectory(t)  
            
            # Tính vận tốc khớp
            inputs = np.zeros(33)
            inputs[0:6] = rEvE
            inputs[6:12] = oriang
            inputs[12:19] = q_current
            qdot = self.joint_velocity.compute(inputs)
            
            # Tính gia tốc khớp: qddot = (qdot - qdot_prev) / dt
            if i == 0:
                qddot = np.zeros(7)  # Gia tốc tại t=0 là 0
            else:
                qddot = (qdot - qdot_prev) / self.dt
            
            # Cập nhật góc khớp
            q_current += (qdot - qdot_prev) * self.dt
            
            # Lưu dữ liệu
            self.input_log[i, 0] = t                     # Thời gian
            self.input_log[i, 1:7] = rEvE                # Vị trí và vận tốc
            self.input_log[i, 7:13] = oriang.flatten()   # Góc và vận tốc góc
            self.input_log[i, 13:20] = q_current         # Góc khớp
            self.input_log[i, 20:27] = qdot              # Vận tốc khớp
            self.input_log[i, 27:34] = qddot             # Gia tốc khớp
            self.input_log[i, 34:43] = R.flatten()       # Ma trận quay (đầy đủ 9 phần tử)
            
            # Cập nhật qdot_prev cho bước tiếp theo
            qdot_prev = qdot.copy()

    def save_to_csv(self, filename='output/trajectory_log.csv'):
        """
        Lưu dữ liệu quỹ đạo ra file CSV
        
        Args:
            filename (str): Đường dẫn file output
        """
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Tạo header
        header = ",".join(self.column_names)
        
        # Lưu file
        np.savetxt(filename, self.input_log, 
                  delimiter=',', 
                  fmt='%.6f',
                  header=header,
                  comments='')

    def plot_trajectory(self):
        """
        Vẽ đồ thị quỹ đạo (bao gồm góc khớp, vận tốc khớp và gia tốc khớp)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Tạo figure
            fig = plt.figure(figsize=(15, 20))
            
            # Quỹ đạo không gian 3D
            ax1 = fig.add_subplot(4, 2, 1, projection='3d')
            ax1.plot(self.input_log[:,1], self.input_log[:,2], self.input_log[:,3])
            ax1.set_title('Quỹ đạo không gian')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            
            # Vị trí theo thời gian
            ax2 = fig.add_subplot(4, 2, 2)
            ax2.plot(self.input_log[:,0], self.input_log[:,1], label='X')
            ax2.plot(self.input_log[:,0], self.input_log[:,2], label='Y')
            ax2.plot(self.input_log[:,0], self.input_log[:,3], label='Z')
            ax2.set_title('Vị trí theo thời gian')
            ax2.set_xlabel('Thời gian (s)')
            ax2.set_ylabel('Vị trí (m)')
            ax2.legend()
            
            # Góc Euler
            ax3 = fig.add_subplot(4, 2, 3)
            ax3.plot(self.input_log[:,0], np.degrees(self.input_log[:,7]), label='Psi')
            ax3.plot(self.input_log[:,0], np.degrees(self.input_log[:,8]), label='Theta')
            ax3.plot(self.input_log[:,0], np.degrees(self.input_log[:,9]), label='Phi')
            ax3.set_title('Góc Euler theo thời gian')
            ax3.set_xlabel('Thời gian (s)')
            ax3.set_ylabel('Góc (độ)')
            ax3.legend()
            
            # Vận tốc góc
            ax4 = fig.add_subplot(4, 2, 4)
            ax4.plot(self.input_log[:,0], self.input_log[:,10], label='Omega X')
            ax4.plot(self.input_log[:,0], self.input_log[:,11], label='Omega Y')
            ax4.plot(self.input_log[:,0], self.input_log[:,12], label='Omega Z')
            ax4.set_title('Vận tốc góc theo thời gian')
            ax4.set_xlabel('Thời gian (s)')
            ax4.set_ylabel('Vận tốc góc (rad/s)')
            ax4.legend()
            
            # Góc khớp
            ax5 = fig.add_subplot(4, 2, 5)
            for j in range(7):
                ax5.plot(self.input_log[:,0], np.rad2deg(self.input_log[:,13+j]), label=f'q{j+1}')
            ax5.set_title('Góc khớp theo thời gian')
            ax5.set_xlabel('Thời gian (s)')
            ax5.set_ylabel('Góc (độ)')
            ax5.legend()
            
            # Vận tốc khớp
            ax6 = fig.add_subplot(4, 2, 6)
            for j in range(7):
                ax6.plot(self.input_log[:,0], self.input_log[:,20+j], label=f'qdot{j+1}')
            ax6.set_title('Vận tốc khớp theo thời gian')
            ax6.set_xlabel('Thời gian (s)')
            ax6.set_ylabel('Vận tốc khớp (rad/s)')
            ax6.legend()
            
            # Gia tốc khớp
            ax7 = fig.add_subplot(4, 2, 7)
            for j in range(7):
                ax7.plot(self.input_log[:,0], self.input_log[:,27+j], label=f'qddot{j+1}')
            ax7.set_title('Gia tốc khớp theo thời gian')
            ax7.set_xlabel('Thời gian (s)')
            ax7.set_ylabel('Gia tốc khớp (rad/s²)')
            ax7.legend()
            
            plt.tight_layout()
            plt.savefig('output/trajectory_plot.png')
            plt.show()
            
        except ImportError:
            print("Warning: Matplotlib not installed. Skipping plotting.")

if __name__ == "__main__":
    try:
        print("=== Chương trình tính toán quỹ đạo robot ===")
        
        # Khởi tạo bộ điều khiển
        controller = InputController()
        print("Đã khởi tạo bộ điều khiển với các tham số:")
        print(f"- Thời gian mô phỏng: {controller.T}s")
        print(f"- Bước thời gian: {controller.dt}s")
        print(f"- Góc khớp ban đầu: {controller.q}")
        
        # Tính toán quỹ đạo
        print("\nĐang tính toán quỹ đạo, vận tốc khớp và gia tốc khớp...")
        controller.calculate_trajectory()
        print("Hoàn thành tính toán quỹ đạo, vận tốc khớp và gia tốc khớp!")
        
        # Hiển thị thông tin
        print("\n=== Thông tin quỹ đạo ===")
        print(f"Tổng số bước: {len(controller.input_log)}")
        print(f"Kích thước dữ liệu: {controller.input_log.shape}")
        
        # Lưu kết quả
        print("\nĐang lưu kết quả ra file...")
        controller.save_to_csv()
        print(f"Đã lưu kết quả vào: output/trajectory_log.csv")
        
        # Vẽ đồ thị
        print("\nĐang vẽ đồ thị quỹ đạo...")
        controller.plot_trajectory()
        
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()