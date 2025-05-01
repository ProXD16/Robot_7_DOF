import numpy as np
import json
import os
from trajectory_rectilinear_xt import RectilinearTrajectory
from trajectory_Orientation_Rt import OrientationTrajectory
from trajectory_curvilinear_xt import CurvilinearTrajectory
from joint_velocity import JointVelocity
from dynamics import RobotDynamics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

class InputController:
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
        self.q = np.array([
            params.get('q1', 0.0),
            params.get('q2', 0.0),
            params.get('q3', 0.0),
            params.get('q4', 0.0),
            params.get('q5', 0.0),
            params.get('q6', 0.0),
            params.get('q7', 0.0)
        ])
        self.T = params.get('T', 15.0)  
        self.dt = params.get('dt', 0.01) 
        self.K = 100 * np.eye(6)       
        self.err = np.zeros(6)          
        self.time = 0.0                
        self.joint_velocity = JointVelocity(json_path)
        self.dynamics = RobotDynamics(json_path)
        self.input_log = None           
        self.column_names = [
            'time',                    
            'x', 'y', 'z',              
            'vx', 'vy', 'vz',         
            'psi', 'theta', 'phi',       
            'omega_x', 'omega_y', 'omega_z', 
            'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7',
            'qdot1', 'qdot2', 'qdot3', 'qdot4', 'qdot5', 'qdot6', 'qdot7', 
            'qddot1', 'qddot2', 'qddot3', 'qddot4', 'qddot5', 'qddot6', 'qddot7', 
            'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6', 'tau7', 
            'r1', 'r2', 'r3',          
            'r4', 'r5', 'r6',        
            'r7', 'r8', 'r9'            
        ]

    def calculate_trajectory(self):
        traj_rect = RectilinearTrajectory(json_file='input/robot_parameters.json')
        traj_orient = OrientationTrajectory(json_file='input/robot_parameters.json')
        t_values = np.arange(0.0, self.T + 1e-9, self.dt)
        num_steps = len(t_values)
        self.input_log = np.zeros((num_steps, 50)) 
        q_current = self.q.copy()
        qdot_prev = np.zeros(7)
        for i, t in enumerate(t_values):
            rEvE = traj_rect.compute_trajectory(t)  
            RPY, R, oriang = traj_orient.compute_trajectory(t)  
            inputs = np.zeros(33)
            inputs[0:6] = rEvE
            inputs[6:12] = oriang
            inputs[12:19] = q_current
            qdot = self.joint_velocity.compute(inputs)
            if i == 0:
                qdot = np.zeros(7)
                qddot = np.zeros(7) 
            else:
                qdot = self.joint_velocity.compute(inputs)
                qddot = (qdot - qdot_prev) / self.dt
            tau = self.dynamics.compute_torque(q_current, qdot, qddot)
            if tau.ndim > 1:
                tau = tau.flatten()
            if tau.shape != (7,):
                if tau.size == 7:
                    tau = tau.reshape((7,))
                    print("Warning: Reshaping tau to (7,).  The original shape was not as expected.")
                else:
                    raise ValueError(f"Shape of tau is {tau.shape}, but should be (7,).")  
            q_current += qdot * self.dt
            self.input_log[i, 0] = t                 
            self.input_log[i, 1:7] = rEvE             
            self.input_log[i, 7:13] = oriang.flatten()  
            self.input_log[i, 13:20] = q_current      
            self.input_log[i, 20:27] = qdot      
            self.input_log[i, 27:34] = qddot 
            self.input_log[i, 34:41] = tau 
            self.input_log[i, 41:50] = R.flatten()
            qdot_prev = qdot.copy()

    def save_to_csv(self, filename='output/trajectory_log.csv'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = ",".join(self.column_names)
        np.savetxt(filename, self.input_log, 
                  delimiter=',', 
                  fmt='%.6f',
                  header=header,
                  comments='')

    def plot_trajectory(self):
        try:
            root = tk.Tk()
            root.title("Biểu đồ quỹ đạo, khớp và mô-men xoắn robot")
            notebook = ttk.Notebook(root)
            notebook.pack(fill='both', expand=True)

            def create_plot_tab(name, plot_func):
                frame = ttk.Frame(notebook)
                notebook.add(frame, text=name)
                fig = plt.Figure(figsize=(8, 6))
                ax = plot_func(fig)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
                return fig

            def plot_3d_trajectory(fig):
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(self.input_log[:,1], self.input_log[:,2], self.input_log[:,3])
                ax.set_title('Quỹ đạo không gian')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                return ax

            def plot_position(fig):
                ax = fig.add_subplot(111)
                ax.plot(self.input_log[:,0], self.input_log[:,1], label='X')
                ax.plot(self.input_log[:,0], self.input_log[:,2], label='Y')
                ax.plot(self.input_log[:,0], self.input_log[:,3], label='Z')
                ax.set_title('Vị trí theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Vị trí (m)')
                ax.legend()
                return ax

            def plot_euler_angles(fig):
                ax = fig.add_subplot(111)
                ax.plot(self.input_log[:,0], (self.input_log[:,7]), label='Psi')
                ax.plot(self.input_log[:,0], (self.input_log[:,8]), label='Theta')
                ax.plot(self.input_log[:,0], (self.input_log[:,9]), label='Phi')
                ax.set_title('Góc Euler theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Góc (độ)')
                ax.legend()
                return ax

            def plot_angular_velocity(fig):
                ax = fig.add_subplot(111)
                ax.plot(self.input_log[:,0], self.input_log[:,10], label='Omega X')
                ax.plot(self.input_log[:,0], self.input_log[:,11], label='Omega Y')
                ax.plot(self.input_log[:,0], self.input_log[:,12], label='Omega Z')
                ax.set_title('Vận tốc góc theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Vận tốc góc (rad/s)')
                ax.legend()
                return ax

            def plot_joint_angles(fig):
                ax = fig.add_subplot(111)
                for j in range(7):
                    ax.plot(self.input_log[:,0], np.rad2deg(self.input_log[:,13+j]), label=f'q{j+1}')
                ax.set_title('Góc khớp theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Góc (độ)')
                ax.legend()
                return ax

            def plot_joint_velocities(fig):
                ax = fig.add_subplot(111)
                for j in range(7):
                    ax.plot(self.input_log[:,0], self.input_log[:,20+j], label=f'qdot{j+1}')
                ax.set_title('Vận tốc khớp theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Vận tốc khớp (rad/s)')
                # ax.set_ylim(-0.8, 0.4)
                # ax.set_xlim(4,10)
                ax.legend()
                return ax

            def plot_joint_accelerations(fig):
                ax = fig.add_subplot(111)
                for j in range(7):
                    ax.plot(self.input_log[:,0], self.input_log[:,27+j], label=f'qddot{j+1}')
                ax.set_title('Gia tốc khớp theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Gia tốc khớp (rad/s²)')
                # ax.set_ylim(-0.5, 0.5)
                # ax.set_xlim(4,10)
                ax.legend()
                return ax

            def plot_torques(fig):
                ax = fig.add_subplot(111)
                for j in range(7):
                    ax.plot(self.input_log[:,0], self.input_log[:,34+j], label=f'tau{j+1}')
                ax.set_title('Mô-men xoắn theo thời gian')
                ax.set_xlabel('Thời gian (s)')
                ax.set_ylabel('Mô-men xoắn (N·m)')
                # ax.set_ylim(-60, 100)
                # ax.set_xlim(4,10)
                ax.legend()
                return ax

            tabs = [
                ("Quỹ đạo 3D", plot_3d_trajectory),
                ("Vị trí", plot_position),
                ("Góc Euler", plot_euler_angles),
                ("Vận tốc góc", plot_angular_velocity),
                ("Góc khớp", plot_joint_angles),
                ("Vận tốc khớp", plot_joint_velocities),
                ("Gia tốc khớp", plot_joint_accelerations),
                ("Mô-men xoắn", plot_torques)
            ]
            figs = [create_plot_tab(name, func) for name, func in tabs]
            for i in range(8):
                figs[i].savefig(f'output/trajectory_plot_{i}.png')
            root.mainloop()
            
        except ImportError as e:
            print(f"Warning: Required module not installed ({e}). Skipping plotting.")

if __name__ == "__main__":
    try:
        print("=== Chương trình tính toán quỹ đạo robot ===")
        controller = InputController()
        print("Đã khởi tạo bộ điều khiển với các tham số:")
        print(f"- Thời gian mô phỏng: {controller.T}s")
        print(f"- Bước thời gian: {controller.dt}s")
        print(f"- Góc khớp ban đầu: {controller.q}")

        print("\nĐang tính toán quỹ đạo, vận tốc khớp, gia tốc khớp và mô-men xoắn...")
        controller.calculate_trajectory()
        print("Hoàn thành tính toán quỹ đạo, vận tốc khớp, gia tốc khớp và mô-men xoắn!")

        print("\n=== Thông tin quỹ đạo ===")
        print(f"Tổng số bước: {len(controller.input_log)}")
        print(f"Kích thước dữ liệu: {controller.input_log.shape}")

        print("\nĐang lưu kết quả ra file...")
        controller.save_to_csv()
        print(f"Đã lưu kết quả vào: output/trajectory_log.csv")
 
        print("\nĐang vẽ đồ thị quỹ đạo, khớp và mô-men xoắn...")
        controller.plot_trajectory()
        
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()