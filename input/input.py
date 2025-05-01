import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from forward_kinematics import ForwardKinematics
import numpy as np

class RobotInputGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Parameters Input")

        # Initialize dictionaries to store entries
        self.entries = {}

        # Initialize ForwardKinematics
        try:
            self.fk = ForwardKinematics()
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Cannot initialize ForwardKinematics: {e}")
            self.root.destroy()
            return

        # Save button (placed at the top)
        save_button = ttk.Button(root, text="Save Parameters", command=self.save_parameters)
        save_button.pack(pady=10)

        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True, fill="both")

        # Create tabs
        self.create_joint_params_tab()
        self.create_trajectory_tab()

        # Load parameters from JSON file
        self.load_parameters()

    def save_parameters(self):
        data = {}
        for key, entry in self.entries.items():
            try:
                value = float(entry.get())
                data[key] = round(value, 10)  # Round to 10 decimal places
            except ValueError:
                messagebox.showwarning("Warning", f"Invalid value for {key}, defaulting to 0.0")
                data[key] = 0.0

        # Ensure input directory exists
        os.makedirs('input', exist_ok=True)

        # Save all parameters to JSON file
        with open('input/robot_parameters.json', 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)

        messagebox.showinfo("Success", "All parameters saved to robot_parameters.json")

    def load_parameters(self):
        # Check if JSON file exists
        json_file = 'input/robot_parameters.json'
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    params = json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON file. Please provide a valid robot_parameters.json.")
                self.root.destroy()
                return
        else:
            messagebox.showerror("Error", "JSON file not found. Please create input/robot_parameters.json.")
            self.root.destroy()
            return

        # Update entries with loaded parameters, formatted to 10 decimal places
        for key, value in params.items():
            if key in self.entries:
                try:
                    formatted_value = f"{float(value):.10f}"  # Format to 10 decimal places
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, formatted_value)
                except (ValueError, TypeError):
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, "0.0000000000") # 10 zeros

    def create_joint_params_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Joint Parameters")

        # Create a canvas to hold the scrollbar and the content
        canvas = tk.Canvas(frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure the canvas to work with the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Create a frame inside the canvas
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        # Joint parameters
        joint_params = [
            ('m', 'Mass (kg)', 7),
            ('d', 'Length (m)', 7),
            ('xC', 'Center of Mass X (m)', 7),
            ('yC', 'Center of Mass Y (m)', 7),
            ('zC', 'Center of Mass Z (m)', 7),
            ('Ix', 'Moment of Inertia X (kg·m²)', 7),
            ('Iy', 'Moment of Inertia Y (kg·m²)', 7),
            ('Iz', 'Moment of Inertia Z (kg·m²)', 7),
            ('Ixy', 'Product of Inertia XY (kg·m²)', 7),
            ('Ixz', 'Product of Inertia XZ (kg·m²)', 7),
            ('Iyz', 'Product of Inertia YZ (kg·m²)', 7)
        ]

        # Additional parameters for joint 6
        a6_frame = ttk.LabelFrame(inner_frame, text="Joint 6 Additional Parameter")
        a6_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        ttk.Label(a6_frame, text="a6 (m):").grid(row=0, column=0, padx=5, pady=2)
        self.entries['a6'] = ttk.Entry(a6_frame)
        self.entries['a6'].grid(row=0, column=1, padx=5, pady=2)

        # Create frames for each joint
        joint_frames = []
        for joint in range(1, 8):
            joint_frame = ttk.LabelFrame(inner_frame, text=f"Joint {joint}")
            joint_frames.append(joint_frame)

        # Grid layout for joint frames
        for i, joint_frame in enumerate(joint_frames[:6]):
            joint_frame.grid(row=1 + (i // 3), column=(i % 3), padx=5, pady=5, sticky="nw")

        # Special handling for the last joint
        if len(joint_frames) == 7:
            joint_frames[6].grid(row=3, column=0, padx=5, pady=5, sticky="nw")

        for joint in range(1, 8):
            if joint <= len(joint_frames):
                joint_frame = joint_frames[joint-1]

            for i, (prefix, label, count) in enumerate(joint_params):
                if prefix == 'm' and count == 7:
                    ttk.Label(joint_frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=2)
                    self.entries[f'{prefix}{joint}'] = ttk.Entry(joint_frame)
                    self.entries[f'{prefix}{joint}'].grid(row=i, column=1, padx=5, pady=2)
                elif count == 7 or (prefix.startswith('I') and joint < 7):
                    ttk.Label(joint_frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=2)
                    self.entries[f'{prefix}{joint}'] = ttk.Entry(joint_frame)
                    self.entries[f'{prefix}{joint}'].grid(row=i, column=1, padx=5, pady=2)

        # Gravity
        gravity_frame = ttk.LabelFrame(inner_frame, text="Gravity")
        gravity_frame.grid(row=5, column=0, padx=5, pady=5, sticky="nw")  # Adjusted row
        ttk.Label(gravity_frame, text="g (m/s²):").grid(row=0, column=0, padx=5, pady=2)
        self.entries['g'] = ttk.Entry(gravity_frame)
        self.entries['g'].grid(row=0, column=1, padx=5, pady=2)

        # Update scrollregion when inner frame's size changes
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def create_trajectory_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Trajectory Parameters")
        
        # Joint angle limits
        angle_frame = ttk.LabelFrame(frame, text="Joint Angle Limits")
        angle_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        
        for joint in range(1, 8):
            ttk.Label(angle_frame, text=f"Joint {joint}").grid(row=0, column=joint, padx=5, pady=2)
            ttk.Label(angle_frame, text="Max (rad):").grid(row=1, column=0, padx=5, pady=2)
            ttk.Label(angle_frame, text="Min (rad):").grid(row=2, column=0, padx=5, pady=2)
            ttk.Label(angle_frame, text="Mean (rad):").grid(row=3, column=0, padx=5, pady=2)
            
            self.entries[f'q{joint}M'] = ttk.Entry(angle_frame)
            self.entries[f'q{joint}m'] = ttk.Entry(angle_frame)
            self.entries[f'q{joint}'] = ttk.Entry(angle_frame)
            
            self.entries[f'q{joint}M'].grid(row=1, column=joint, padx=5, pady=2)
            self.entries[f'q{joint}m'].grid(row=2, column=joint, padx=5, pady=2)
            self.entries[f'q{joint}'].grid(row=3, column=joint, padx=5, pady=2)
        
        # Trajectory parameters
        traj_frame = ttk.LabelFrame(frame, text="Trajectory")
        traj_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        
        params = [
            ('T', 'Time (s)'),
            ('rA_x', 'Start Point X (m)'),
            ('rA_y', 'Start Point Y (m)'),
            ('rA_z', 'Start Point Z (m)'),
            ('rB_x', 'End Point X (m)'),
            ('rB_y', 'End Point Y (m)'),
            ('rB_z', 'End Point Z (m)'),
            ('rpyAng0_x', 'Start Roll (rad)'),
            ('rpyAng0_y', 'Start Pitch (rad)'),
            ('rpyAng0_z', 'Start Yaw (rad)'),
            ('rpyAngf_x', 'End Roll (rad)'),
            ('rpyAngf_y', 'End Pitch (rad)'),
            ('rpyAngf_z', 'End Yaw (rad)')
        ]
        
        for i, (key, label) in enumerate(params):
            ttk.Label(traj_frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=2)
            self.entries[key] = ttk.Entry(traj_frame)
            self.entries[key].grid(row=i, column=1, padx=5, pady=2)

        # Nút để mở cửa sổ nhập góc khớp và tính tọa độ
        calc_button = ttk.Button(traj_frame, text="Calculate rA, rB from Joint Angles", command=self.open_joint_angle_window)
        calc_button.grid(row=len(params), column=0, columnspan=2, pady=10)

    def open_joint_angle_window(self):
        # Tạo cửa sổ mới
        joint_window = tk.Toplevel(self.root)
        joint_window.title("Enter Joint Angles to Calculate rA and rB")
        joint_window.geometry("800x800")

        # Frame cho góc khớp hiện tại (q_current) với thanh kéo ngang
        current_frame = ttk.LabelFrame(joint_window, text="Current Joint Angles (do)")
        current_frame.pack(padx=10, pady=5, fill="x")

        # Tạo canvas và thanh kéo ngang cho current_frame
        current_canvas = tk.Canvas(current_frame)
        current_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        current_h_scrollbar = ttk.Scrollbar(current_frame, orient="horizontal", command=current_canvas.xview)
        current_h_scrollbar.pack(side=tk.BOTTOM, fill="x")

        current_canvas.configure(xscrollcommand=current_h_scrollbar.set)
        current_canvas.bind('<Configure>', lambda e: current_canvas.configure(scrollregion=current_canvas.bbox("all")))

        current_inner_frame = ttk.Frame(current_canvas)
        current_canvas.create_window((0, 0), window=current_inner_frame, anchor="nw")

        self.current_entries = {}
        for joint in range(1, 8):
            ttk.Label(current_inner_frame, text=f"q{joint}:").grid(row=0, column=joint-1, padx=5, pady=2)
            self.current_entries[f'q{joint}_current'] = ttk.Entry(current_inner_frame)
            self.current_entries[f'q{joint}_current'].grid(row=1, column=joint-1, padx=5, pady=2)

        current_inner_frame.bind("<Configure>", lambda e: current_canvas.configure(scrollregion=current_canvas.bbox("all")))

        # Frame cho góc khớp mong muốn (q_desired) với thanh kéo ngang
        desired_frame = ttk.LabelFrame(joint_window, text="Desired Joint Angles (do)")
        desired_frame.pack(padx=10, pady=5, fill="x")

        # Tạo canvas và thanh kéo ngang cho desired_frame
        desired_canvas = tk.Canvas(desired_frame)
        desired_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        desired_h_scrollbar = ttk.Scrollbar(desired_frame, orient="horizontal", command=desired_canvas.xview)
        desired_h_scrollbar.pack(side=tk.BOTTOM, fill="x")

        desired_canvas.configure(xscrollcommand=desired_h_scrollbar.set)

        desired_inner_frame = ttk.Frame(desired_canvas)
        desired_canvas.create_window((0, 0), window=desired_inner_frame, anchor="nw")

        self.desired_entries = {}
        for joint in range(1, 8):
            ttk.Label(desired_inner_frame, text=f"q{joint}:").grid(row=0, column=joint-1, padx=5, pady=2)
            self.desired_entries[f'q{joint}_desired'] = ttk.Entry(desired_inner_frame)
            self.desired_entries[f'q{joint}_desired'].grid(row=1, column=joint-1, padx=5, pady=2)

        # Cập nhật vùng cuộn cho desired_canvas
        desired_inner_frame.bind("<Configure>", lambda e: desired_canvas.configure(scrollregion=desired_canvas.bbox("all")))
        desired_canvas.bind("<Configure>", lambda e: desired_canvas.configure(scrollregion=desired_canvas.bbox("all")))
        
        # Đảm bảo vùng cuộn được cập nhật ngay sau khi tạo
        desired_inner_frame.update_idletasks()
        desired_canvas.configure(scrollregion=desired_canvas.bbox("all"))

        # Nút tính toán tọa độ
        calc_button = ttk.Button(joint_window, text="Calculate Positions", command=self.calculate_positions_from_joints)
        calc_button.pack(pady=10)

    def calculate_positions_from_joints(self):
        try:
            # Lấy góc khớp hiện tại (q_current)
            q_current = np.zeros(7)
            for joint in range(1, 8):
                value = self.current_entries[f'q{joint}_current'].get() 
                q_current[joint-1] = float(value) / 180 * np.pi

            # Lấy góc khớp mong muốn (q_desired)
            q_desired = np.zeros(7)
            for joint in range(1, 8):
                value = self.desired_entries[f'q{joint}_desired'].get() 
                q_desired[joint-1] = float(value) / 180 * np.pi

            # Tính tọa độ rA từ q_current
            x_current = self.fk.compute(q_current)
            rA_x, rA_y, rA_z = x_current[0:3]  # xE, yE, zE

            # Tính tọa độ rB từ q_desired
            x_desired = self.fk.compute(q_desired)
            rB_x, rB_y, rB_z = x_desired[0:3]  # xE, yE, zE

            # Cập nhật các ô nhập liệu trong giao diện chính
            self.entries['rA_x'].delete(0, tk.END)
            self.entries['rA_x'].insert(0, f"{rA_x:.10f}")
            self.entries['rA_y'].delete(0, tk.END)
            self.entries['rA_y'].insert(0, f"{rA_y:.10f}")
            self.entries['rA_z'].delete(0, tk.END)
            self.entries['rA_z'].insert(0, f"{rA_z:.10f}")

            self.entries['rB_x'].delete(0, tk.END)
            self.entries['rB_x'].insert(0, f"{rB_x:.10f}")
            self.entries['rB_y'].delete(0, tk.END)
            self.entries['rB_y'].insert(0, f"{rB_y:.10f}")
            self.entries['rB_z'].delete(0, tk.END)
            self.entries['rB_z'].insert(0, f"{rB_z:.10f}")

            # Tự động lưu vào file JSON
            self.save_parameters()

            messagebox.showinfo("Success", "rA and rB have been calculated and saved successfully!")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during calculation: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotInputGUI(root)
    root.mainloop()