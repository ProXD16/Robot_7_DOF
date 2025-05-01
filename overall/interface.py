import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import os
from joint_velocity import JointVelocity
from joint_acceleration import JointAcceleration
from trajectory_rectilinear_xt import RectilinearTrajectory

class RobotDynamicsGUI:
    def __init__(self, root, json_path="input/robot_parameters.json"):
        self.root = root
        self.root.title("Robot Dynamics Control Interface")
        self.json_path = json_path

        # Đọc thông số từ file JSON
        try:
            self.params = self.load_json_params()
        except (FileNotFoundError, KeyError, ValueError) as e:
            messagebox.showerror("Error", f"Error loading JSON: {e}")
            self.root.destroy()
            return

        # Khởi tạo các lớp tính toán
        try:
            self.joint_velocity = JointVelocity(json_path=self.json_path)
            self.joint_acceleration = JointAcceleration(json_path=self.json_path)
            self.trajectory = RectilinearTrajectory(json_file=self.json_path)
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"JSON file not found: {e}")
            self.root.destroy()
            return
        except ValueError as e:
            messagebox.showerror("Error", f"Trajectory error: {e}")
            self.root.destroy()
            return

        # Tạo notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Manual Input
        self.tab_manual = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_manual, text="Manual Input")
        self.setup_manual_tab()

        # Tab 2: Trajectory Simulation
        self.tab_trajectory = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trajectory, text="Trajectory Simulation")
        self.setup_trajectory_tab()

        # Biến để kiểm soát mô phỏng
        self.simulation_running = False
        self.current_time = 0.0
        self.time_step = 0.1  # Bước thời gian (s)

    def load_json_params(self):
        """Đọc thông số từ file JSON và kiểm tra các khóa bắt buộc."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"File {self.json_path} không tồn tại.")
        with open(self.json_path, 'r') as f:
            params = json.load(f)

        # Danh sách các khóa bắt buộc
        required_keys = ["rA_x", "rA_y", "rA_z", "rpyAng0_x", "rpyAng0_y", "rpyAng0_z"]
        for key in required_keys:
            if key not in params:
                params[key] = 0.0  # Gán giá trị mặc định nếu thiếu
                print(f"Warning: Missing key '{key}' in JSON. Using default value 0.0.")
            if not isinstance(params[key], (int, float)):
                raise ValueError(f"Giá trị của '{key}' trong JSON phải là số, nhưng nhận được: {params[key]}")

        # Gán các giá trị x, y, z, psi, theta, phi từ rA và rpyAng0
        params["x"] = params["rA_x"]
        params["y"] = params["rA_y"]
        params["z"] = params["rA_z"]
        params["psi"] = params["rpyAngf_x"]
        params["theta"] = params["rpyAngf_y"]
        params["phi"] = params["rpyAngf_z"]

        # Kiểm tra các khóa q1 đến q7, qd1 đến qd7, u1 đến u7 (tùy chọn, có giá trị mặc định)
        for i in range(1, 8):
            if f"q{i}" not in params:
                params[f"q{i}"] = 0.0
            if f"qd{i}" not in params:
                params[f"qd{i}"] = 0.0
            if f"u{i}" not in params:
                params[f"u{i}"] = 0.0

        return params

    def setup_manual_tab(self):
        """Thiết lập tab nhập thủ công."""
        main_frame = ttk.Frame(self.tab_manual, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Khung nhập vị trí và hướng đầu cuối
        end_effector_frame = ttk.LabelFrame(main_frame, text="End-Effector Position and Orientation", padding="5")
        end_effector_frame.pack(fill=tk.X, pady=5)

        # Vị trí (x, y, z)
        ttk.Label(end_effector_frame, text="x (m):").grid(row=0, column=0, padx=5, pady=2)
        self.x_entry = ttk.Entry(end_effector_frame, width=10)
        self.x_entry.grid(row=0, column=1, padx=5, pady=2)
        self.x_entry.insert(0, str(self.params["x"]))  # Lấy từ JSON

        ttk.Label(end_effector_frame, text="y (m):").grid(row=0, column=2, padx=5, pady=2)
        self.y_entry = ttk.Entry(end_effector_frame, width=10)
        self.y_entry.grid(row=0, column=3, padx=5, pady=2)
        self.y_entry.insert(0, str(self.params["y"]))  # Lấy từ JSON

        ttk.Label(end_effector_frame, text="z (m):").grid(row=0, column=4, padx=5, pady=2)
        self.z_entry = ttk.Entry(end_effector_frame, width=10)
        self.z_entry.grid(row=0, column=5, padx=5, pady=2)
        self.z_entry.insert(0, str(self.params["z"]))  # Lấy từ JSON

        # Hướng (psi, theta, phi)
        ttk.Label(end_effector_frame, text="ψ (rad):").grid(row=1, column=0, padx=5, pady=2)
        self.psi_entry = ttk.Entry(end_effector_frame, width=10)
        self.psi_entry.grid(row=1, column=1, padx=5, pady=2)
        self.psi_entry.insert(0, str(self.params["psi"]))  # Lấy từ JSON

        ttk.Label(end_effector_frame, text="θ (rad):").grid(row=1, column=2, padx=5, pady=2)
        self.theta_entry = ttk.Entry(end_effector_frame, width=10)
        self.theta_entry.grid(row=1, column=3, padx=5, pady=2)
        self.theta_entry.insert(0, str(self.params["theta"]))  # Lấy từ JSON

        ttk.Label(end_effector_frame, text="φ (rad):").grid(row=1, column=4, padx=5, pady=2)
        self.phi_entry = ttk.Entry(end_effector_frame, width=10)
        self.phi_entry.grid(row=1, column=5, padx=5, pady=2)
        self.phi_entry.insert(0, str(self.params["phi"]))  # Lấy từ JSON

        # Khung nhập trạng thái ban đầu (q, qdot, u)
        state_frame = ttk.LabelFrame(main_frame, text="Initial Joint State", padding="5")
        state_frame.pack(fill=tk.X, pady=5)

        # Góc khớp (q1 đến q7)
        self.joint_q_entries = []
        for i in range(7):
            ttk.Label(state_frame, text=f"q{i+1} (rad):").grid(row=0, column=i*2, padx=5, pady=2)
            entry = ttk.Entry(state_frame, width=10)
            entry.grid(row=0, column=i*2+1, padx=5, pady=2)
            q_val = self.params.get(f"q{i+1}", 0.0)
            entry.insert(0, f"{q_val:.6f}")
            self.joint_q_entries.append(entry)

        # Vận tốc khớp (qd1 đến qd7)
        self.joint_qd_entries = []
        for i in range(7):
            ttk.Label(state_frame, text=f"qd{i+1} (rad/s):").grid(row=1, column=i*2, padx=5, pady=2)
            entry = ttk.Entry(state_frame, width=10)
            entry.grid(row=1, column=i*2+1, padx=5, pady=2)
            qd_val = self.params.get(f"qd{i+1}", 0.0)
            entry.insert(0, f"{qd_val:.6f}")
            self.joint_qd_entries.append(entry)

        # Mô-men xoắn (u1 đến u7)
        ttk.Label(state_frame, text="Control Torques (u, Nm):").grid(row=2, column=0, columnspan=14, pady=2)
        self.control_entries = []
        for i in range(7):
            ttk.Label(state_frame, text=f"u{i+1}:").grid(row=3, column=i*2, padx=5, pady=2)
            entry = ttk.Entry(state_frame, width=10)
            entry.grid(row=3, column=i*2+1, padx=5, pady=2)
            u_val = self.params.get(f"u{i+1}", 0.0)
            entry.insert(0, f"{u_val:.6f}")
            self.control_entries.append(entry)

        # Khung hiển thị kết quả
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        result_frame.pack(fill=tk.X, pady=5)

        # Góc khớp (q1 đến q7)
        ttk.Label(result_frame, text="Joint Positions (q, rad):").grid(row=0, column=0, columnspan=14, pady=2)
        self.joint_q_labels = []
        for i in range(7):
            ttk.Label(result_frame, text=f"q{i+1}:").grid(row=1, column=i*2, padx=5, pady=2)
            label = ttk.Label(result_frame, text="0.000000", width=10)
            label.grid(row=1, column=i*2+1, padx=5, pady=2)
            self.joint_q_labels.append(label)

        # Vận tốc khớp (qd1 đến qd7)
        ttk.Label(result_frame, text="Joint Velocities (qd, rad/s):").grid(row=2, column=0, columnspan=14, pady=2)
        self.joint_qd_labels = []
        for i in range(7):
            ttk.Label(result_frame, text=f"qd{i+1}:").grid(row=3, column=i*2, padx=5, pady=2)
            label = ttk.Label(result_frame, text="0.000000", width=10)
            label.grid(row=3, column=i*2+1, padx=5, pady=2)
            self.joint_qd_labels.append(label)

        # Gia tốc khớp (qdd1 đến qdd7)
        ttk.Label(result_frame, text="Joint Accelerations (qdd, rad/s²):").grid(row=4, column=0, columnspan=14, pady=2)
        self.joint_qdd_labels = []
        for i in range(7):
            ttk.Label(result_frame, text=f"qdd{i+1}:").grid(row=5, column=i*2, padx=5, pady=2)
            label = ttk.Label(result_frame, text="0.000000", width=10)
            label.grid(row=5, column=i*2+1, padx=5, pady=2)
            self.joint_qdd_labels.append(label)

        # Nút tính toán
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        calc_button = ttk.Button(button_frame, text="Calculate Dynamics", command=self.calculate_dynamics_manual)
        calc_button.pack()

    def setup_trajectory_tab(self):
        """Thiết lập tab mô phỏng quỹ đạo."""
        main_frame = ttk.Frame(self.tab_trajectory, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Khung nhập thời gian
        time_frame = ttk.LabelFrame(main_frame, text="Simulation Time", padding="5")
        time_frame.pack(fill=tk.X, pady=5)

        ttk.Label(time_frame, text="Current Time (s):").grid(row=0, column=0, padx=5, pady=2)
        self.time_label = ttk.Label(time_frame, text="0.000", width=10)
        self.time_label.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(time_frame, text="Total Time (T, s):").grid(row=0, column=2, padx=5, pady=2)
        self.total_time_label = ttk.Label(time_frame, text=f"{self.trajectory.T:.3f}", width=10)
        self.total_time_label.grid(row=0, column=3, padx=5, pady=2)

        # Khung trạng thái ban đầu (q, qdot, u)
        state_frame = ttk.LabelFrame(main_frame, text="Initial Joint State", padding="5")
        state_frame.pack(fill=tk.X, pady=5)

        # Góc khớp (q1 đến q7)
        self.traj_joint_q_entries = []
        for i in range(7):
            ttk.Label(state_frame, text=f"q{i+1} (rad):").grid(row=0, column=i*2, padx=5, pady=2)
            entry = ttk.Entry(state_frame, width=10)
            entry.grid(row=0, column=i*2+1, padx=5, pady=2)
            q_val = self.params.get(f"q{i+1}", 0.0)
            entry.insert(0, f"{q_val:.6f}")
            self.traj_joint_q_entries.append(entry)

        # Vận tốc khớp (qd1 đến qd7)
        self.traj_joint_qd_entries = []
        for i in range(7):
            ttk.Label(state_frame, text=f"qd{i+1} (rad/s):").grid(row=1, column=i*2, padx=5, pady=2)
            entry = ttk.Entry(state_frame, width=10)
            entry.grid(row=1, column=i*2+1, padx=5, pady=2)
            qd_val = self.params.get(f"qd{i+1}", 0.0)
            entry.insert(0, f"{qd_val:.6f}")
            self.traj_joint_qd_entries.append(entry)

        # Mô-men xoắn (u1 đến u7)
        ttk.Label(state_frame, text="Control Torques (u, Nm):").grid(row=2, column=0, columnspan=14, pady=2)
        self.traj_control_entries = []
        for i in range(7):
            ttk.Label(state_frame, text=f"u{i+1}:").grid(row=3, column=i*2, padx=5, pady=2)
            entry = ttk.Entry(state_frame, width=10)
            entry.grid(row=3, column=i*2+1, padx=5, pady=2)
            u_val = self.params.get(f"u{i+1}", 0.0)
            entry.insert(0, f"{u_val:.6f}")
            self.traj_control_entries.append(entry)

        # Khung hiển thị kết quả
        result_frame = ttk.LabelFrame(main_frame, text="Trajectory Results", padding="5")
        result_frame.pack(fill=tk.X, pady=5)

        # Vị trí đầu cuối (x, y, z)
        ttk.Label(result_frame, text="End-Effector Position (m):").grid(row=0, column=0, columnspan=6, pady=2)
        ttk.Label(result_frame, text="x:").grid(row=1, column=0, padx=5, pady=2)
        self.traj_x_label = ttk.Label(result_frame, text="0.000000", width=10)
        self.traj_x_label.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(result_frame, text="y:").grid(row=1, column=2, padx=5, pady=2)
        self.traj_y_label = ttk.Label(result_frame, text="0.000000", width=10)
        self.traj_y_label.grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(result_frame, text="z:").grid(row=1, column=4, padx=5, pady=2)
        self.traj_z_label = ttk.Label(result_frame, text="0.000000", width=10)
        self.traj_z_label.grid(row=1, column=5, padx=5, pady=2)

        # Góc khớp (q1 đến q7)
        ttk.Label(result_frame, text="Joint Positions (q, rad):").grid(row=2, column=0, columnspan=14, pady=2)
        self.traj_joint_q_labels = []
        for i in range(7):
            ttk.Label(result_frame, text=f"q{i+1}:").grid(row=3, column=i*2, padx=5, pady=2)
            label = ttk.Label(result_frame, text="0.000000", width=10)
            label.grid(row=3, column=i*2+1, padx=5, pady=2)
            self.traj_joint_q_labels.append(label)

        # Vận tốc khớp (qd1 đến qd7)
        ttk.Label(result_frame, text="Joint Velocities (qd, rad/s):").grid(row=4, column=0, columnspan=14, pady=2)
        self.traj_joint_qd_labels = []
        for i in range(7):
            ttk.Label(result_frame, text=f"qd{i+1}:").grid(row=5, column=i*2, padx=5, pady=2)
            label = ttk.Label(result_frame, text="0.000000", width=10)
            label.grid(row=5, column=i*2+1, padx=5, pady=2)
            self.traj_joint_qd_labels.append(label)

        # Gia tốc khớp (qdd1 đến qdd7)
        ttk.Label(result_frame, text="Joint Accelerations (qdd, rad/s²):").grid(row=6, column=0, columnspan=14, pady=2)
        self.traj_joint_qdd_labels = []
        for i in range(7):
            ttk.Label(result_frame, text=f"qdd{i+1}:").grid(row=7, column=i*2, padx=5, pady=2)
            label = ttk.Label(result_frame, text="0.000000", width=10)
            label.grid(row=7, column=i*2+1, padx=5, pady=2)
            self.traj_joint_qdd_labels.append(label)

        # Nút mô phỏng
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        self.start_button = ttk.Button(button_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop Simulation", command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def calculate_dynamics_manual(self):
        """Tính toán động lực học trong tab nhập thủ công."""
        try:
            # Lấy vị trí và hướng đầu cuối từ ô nhập
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            z = float(self.z_entry.get())
            psi = float(self.psi_entry.get())
            theta = float(self.theta_entry.get())
            phi = float(self.phi_entry.get())

            # Tạo vector rEvE và oriang
            rEvE = np.array([x, y, z, 0.0, 0.0, 0.0])  # Giả định vận tốc đầu cuối = 0
            oriang = np.array([psi, theta, phi, 0.0, 0.0, 0.0])  # Giả định vận tốc góc = 0

            # Lấy trạng thái ban đầu từ ô nhập
            q = np.array([float(entry.get()) for entry in self.joint_q_entries])
            qdot = np.array([float(entry.get()) for entry in self.joint_qd_entries])
            u = np.array([float(entry.get()) for entry in self.control_entries])

            # Bước 1: Tính vận tốc khớp (qdot) từ JointVelocity
            inputs_velocity = np.concatenate([rEvE, oriang, q, np.zeros(14)])
            inputs_velocity_list = inputs_velocity.tolist()  # Chuyển thành danh sách Python
            qdot_new = self.joint_velocity.compute(inputs_velocity_list)
            qdot_new = np.array(qdot_new)  # Chuyển lại thành numpy array nếu cần

            # Cập nhật qdot trong GUI
            for i, label in enumerate(self.joint_qd_labels):
                label.config(text=f"{qdot_new[i]:.6f}")
            for i, entry in enumerate(self.joint_qd_entries):
                entry.delete(0, tk.END)
                entry.insert(0, f"{qdot_new[i]:.6f}")

            # Cập nhật q trong GUI
            for i, label in enumerate(self.joint_q_labels):
                label.config(text=f"{q[i]:.6f}")

            # Bước 2: Tính gia tốc khớp (q2dot) từ JointAcceleration
            inputs_acceleration = np.concatenate([qdot_new, q, u])
            inputs_acceleration_list = inputs_acceleration.tolist()  # Chuyển thành danh sách Python
            q2dot_computed = self.joint_acceleration.compute(inputs_acceleration_list)
            q2dot_computed = np.array(q2dot_computed)  # Chuyển lại thành numpy array nếu cần

            # Cập nhật q2dot trong GUI
            for i, label in enumerate(self.joint_qdd_labels):
                label.config(text=f"{q2dot_computed[i]:.6f}")

        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def start_simulation(self):
        """Bắt đầu mô phỏng quỹ đạo."""
        if self.simulation_running:
            return

        self.simulation_running = True
        self.current_time = 0.0
        self.update_simulation()

    def stop_simulation(self):
        """Dừng mô phỏng quỹ đạo."""
        self.simulation_running = False

    def update_simulation(self):
        """Cập nhật mô phỏng quỹ đạo theo thời gian thực."""
        if not self.simulation_running:
            return

        try:
            # Lấy thời gian hiện tại
            t = self.current_time
            if t > self.trajectory.T:
                self.stop_simulation()
                return

            # Cập nhật thời gian trên GUI
            self.time_label.config(text=f"{t:.3f}")

            # Lấy trạng thái ban đầu từ ô nhập
            q = np.array([float(entry.get()) for entry in self.traj_joint_q_entries])
            qdot = np.array([float(entry.get()) for entry in self.traj_joint_qd_entries])
            u = np.array([float(entry.get()) for entry in self.traj_control_entries])

            # Bước 1: Tính quỹ đạo tại thời điểm t
            rEvE = self.trajectory.compute_trajectory(t)
            x, y, z = rEvE[0], rEvE[1], rEvE[2]

            # Cập nhật vị trí đầu cuối trên GUI
            self.traj_x_label.config(text=f"{x:.6f}")
            self.traj_y_label.config(text=f"{y:.6f}")
            self.traj_z_label.config(text=f"{z:.6f}")

            # Bước 2: Tính vận tốc khớp (qdot) từ JointVelocity
            psi = float(self.params.get("psi", 0.0))
            theta = float(self.params.get("theta", 0.0))
            phi = float(self.params.get("phi", 0.0))
            oriang = np.array([psi, theta, phi, 0.0, 0.0, 0.0])  # Angular velocities assumed zero
            inputs_velocity = np.concatenate([rEvE, oriang, q, np.zeros(14)])
            inputs_velocity_list = inputs_velocity.tolist()  # Chuyển thành danh sách Python
            qdot_new = self.joint_velocity.compute(inputs_velocity_list)
            # qdot_new = np.array(qdot_new)  # Chuyển lại thành numpy array nếu cần

            # Cập nhật qdot trong GUI
            for i, label in enumerate(self.traj_joint_qd_labels):
                label.config(text=f"{qdot_new[i]:.6f}")
            for i, entry in enumerate(self.traj_joint_qd_entries):
                entry.delete(0, tk.END)
                entry.insert(0, f"{qdot_new[i]:.6f}")

            # Cập nhật q trong GUI
            for i, label in enumerate(self.traj_joint_q_labels):
                label.config(text=f"{q[i]:.6f}")

            # Bước 3: Tính gia tốc khớp (q2dot) từ JointAcceleration
            inputs_acceleration = np.concatenate([qdot_new, q, u])
            inputs_acceleration_list = inputs_acceleration.tolist()  # Chuyển thành danh sách Python
            q2dot_computed = self.joint_acceleration.compute(inputs_acceleration_list)
            # q2dot_computed = np.array(q2dot_computed)  # Chuyển lại thành numpy array nếu cần

            # Cập nhật q2dot trong GUI
            for i, label in enumerate(self.traj_joint_qdd_labels):
                label.config(text=f"{q2dot_computed[i]:.6f}")

            # Bước 4: Cập nhật trạng thái (tích phân số để tính q mới)
            q_new = q + qdot_new * self.time_step
            for i, entry in enumerate(self.traj_joint_q_entries):
                entry.delete(0, tk.END)
                entry.insert(0, f"{q_new[i]:.6f}")

            # Tăng thời gian
            self.current_time += self.time_step

            # Lên lịch cập nhật tiếp theo
            self.root.after(100, self.update_simulation)  # Cập nhật sau 100ms

        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {ve}")
            self.stop_simulation()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.stop_simulation()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotDynamicsGUI(root)
    root.mainloop()