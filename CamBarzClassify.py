#Will Use
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
from pymcprotocol import Type3E
import numpy as np
import os
from datetime import datetime

class ImageClassifier:  
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Camera Classifier")
        
        # Set the desired display size
        self.display_width = 320
        self.display_height = 240

        # Initialize the YOLO model
        self.model = YOLO('C:/Users/Tanarat/Desktop/IMP JOB/IMP PROJECT/Camera Brazing/runs/classify/train21/weights/last.pt')

        # Initialize the PLC client
        self.plc = Type3E()
        self.plc_config = {
            'host': '10.11.28.100',
            'port': 5001
        }

        self.setup_ui()
        self.initialize_variables()

    def setup_ui(self):
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg='black')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video frames
        self.video_frame = tk.Frame(self.main_frame, bg='black')
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Main video
        self.setup_video_container("Main Video Feed", self.video_frame, tk.LEFT)

        # Secondary video
        self.setup_video_container("Secondary Video Feed", self.video_frame, tk.RIGHT)

        # Capture frames
        self.capture_frame = tk.Frame(self.main_frame, bg='black')
        self.capture_frame.pack(side=tk.TOP, padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Captured rectangle
        self.setup_video_container("Captured ROI", self.capture_frame, tk.LEFT)

        # Captured full image
        self.setup_video_container("Captured Full Image", self.capture_frame, tk.RIGHT)

        # Bottom frame
        self.bottom_frame = tk.Frame(self.main_frame, bg='black')
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Status bar
        self.setup_status_bar()

        # Control buttons
        self.setup_control_buttons()

        # Status box
        self.status_box = tk.Text(self.bottom_frame, height=5, state='disabled', bg='black', fg='white')
        self.status_box.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

    def setup_video_container(self, title, parent_frame, side):
        container = tk.Frame(parent_frame, width=self.display_width, height=self.display_height + 30, bg='black')
        container.pack(side=side, padx=5, pady=5)
        container.pack_propagate(False)
        
        label = tk.Label(container, text=title, font=("Arial", 10, "bold"), bg='black', fg='white')
        label.pack(side=tk.TOP)
        
        video_label = tk.Label(container, bg='black')
        video_label.pack(expand=True, fill=tk.BOTH)
        
        setattr(self, f"{title.lower().replace(' ', '_')}_label", video_label)

    def setup_status_bar(self):
        self.status_bar = tk.Frame(self.bottom_frame, relief=tk.SUNKEN, borderwidth=1, bg='black')
        self.status_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5, 0))

        self.status_labels = {}
        for label in ['M100', 'M101', 'M102']:
            self.status_labels[label] = tk.Label(self.status_bar, text=f"{label}: ⚪", width=10, bg='black', fg='white')
            self.status_labels[label].pack(side=tk.LEFT, padx=5)

        self.ok_counter_label = tk.Label(self.status_bar, text="OK: 0", width=10, bg='black', fg='white')
        self.ok_counter_label.pack(side=tk.LEFT, padx=5)

        self.ng_counter_label = tk.Label(self.status_bar, text="NG: 0", width=10, bg='black', fg='white')
        self.ng_counter_label.pack(side=tk.LEFT, padx=5)

    def setup_control_buttons(self):
        self.capture_button = tk.Button(self.status_bar, text="CAPTURE", command=self.toggle_capture, width=10)
        self.capture_button.pack(side=tk.LEFT, padx=5)

        self.camera_button = tk.Button(self.status_bar, text="Start Camera", command=self.toggle_camera, width=15)
        self.camera_button.pack(side=tk.LEFT, padx=5, pady=5)

    def initialize_variables(self):
        self.camera_active = False
        self.current_frame = None
        self.previous_coil_state = None
        self.current_class_label = None
        self.rect_coords = None
        self.ok_count = 0
        self.ng_count = 0
        self.plc_states = {
            'M100': False, 'M101': False, 'M102': False, 'M0': False
        }

    def toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.camera_button.config(text="Start Camera")
            self.disconnect_plc()
            self.clear_video_labels()
        else:
            self.connect_plc()
            self.camera_active = True
            self.camera_button.config(text="Stop Camera")
            threading.Thread(target=self.process_camera, daemon=True).start()
            threading.Thread(target=self.monitor_plc, daemon=True).start()

    def clear_video_labels(self):
        for label_name in ['main_video_feed', 'secondary_video_feed', 'captured_roi', 'captured_full_image']:
            label = getattr(self, f"{label_name}_label")
            label.config(image='')
            label.image = None

    def process_camera(self):
        cap = cv2.VideoCapture(0)
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            self.current_frame = processed_frame
            self.display_frame(processed_frame, self.main_video_feed_label)

            secondary_frame, self.rect_coords = self.draw_grid_and_rectangle(processed_frame.copy())
            self.display_frame(secondary_frame, self.secondary_video_feed_label)

        cap.release()

    def process_frame(self, frame):
        results = self.model(frame)
        name_dict = self.model.names
        top1_index = results[0].probs.top1
        top1_confidence = results[0].probs.top1conf.item()
        top1_class_name = name_dict[top1_index]
        self.current_class_label = top1_class_name

        text = f'{top1_class_name}: {top1_confidence:.2f}'
        color = (0, 0, 255) if top1_class_name == "NG" else (0, 255, 0)
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, lineType=cv2.LINE_AA)

        return frame

    def draw_grid_and_rectangle(self, frame):
        height, width = frame.shape[:2]
        num_lines = 10

        cell_width = width // num_lines
        cell_height = height // num_lines

        for i in range(1, num_lines):
            x = i * width // num_lines
            cv2.line(frame, (x, 0), (x, height), color=(255, 255, 255), thickness=1)

        for i in range(1, num_lines):
            y = i * height // num_lines
            cv2.line(frame, (0, y), (width, y), color=(255, 255, 255), thickness=1)

        rect_width = 4 * cell_width
        rect_height = 6 * cell_height

        top_left_x = (width - rect_width) // 2
        top_left_y = (height - rect_height) // 2
        bottom_right_x = top_left_x + rect_width
        bottom_right_y = top_left_y + rect_height

        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color=(0, 255, 0), thickness=2)

        return frame, (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    def display_frame(self, frame, label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.display_width, self.display_height))
        im_pil = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        label.config(image=imgtk)
        label.image = imgtk

    def toggle_capture(self):
        self.plc_states['M0'] = not self.plc_states['M0']
        try:
            self.plc.batchwrite_bitunits(headdevice="M0", values=[int(self.plc_states['M0'])])
            self.update_capture_button()
            self.update_status_box(f"Wrote M0 status to PLC: {'ON' if self.plc_states['M0'] else 'OFF'}")
        except Exception as e:
            self.update_status_box(f"Error writing M0 to PLC: {e}")

    def capture_images(self):
        if self.current_frame is not None and self.rect_coords is not None:
            x1, y1, x2, y2 = self.rect_coords
            captured_rect = self.current_frame[y1:y2, x1:x2]
            captured_full = self.current_frame.copy()

            self.display_frame(captured_rect, self.captured_roi_label)
            self.display_frame(captured_full, self.captured_full_image_label)

            save_dir = 'saved_images'
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rect_filename = f"roi_{timestamp}.jpg"
            full_filename = f"full_{timestamp}.jpg"
            cv2.imwrite(os.path.join(save_dir, rect_filename), captured_rect)
            cv2.imwrite(os.path.join(save_dir, full_filename), captured_full)

            if self.current_class_label == "OK":
                self.ok_count += 1
            elif self.current_class_label == "NG":
                self.ng_count += 1

            try:
                if self.current_class_label == "OK":
                    self.plc.batchwrite_bitunits(headdevice="M101", values=[1])
                    self.plc_states['M101'] = True
                elif self.current_class_label == "NG":
                    self.plc.batchwrite_bitunits(headdevice="M102", values=[1])
                    self.plc_states['M102'] = True
                self.update_status_box(f"Wrote status to PLC: {self.current_class_label}")
                self.update_status()
            except Exception as e:
                self.update_status_box(f"Error writing to PLC: {e}")

            self.update_status_box(f"Images captured: {rect_filename}, {full_filename}")
            self.update_status_box(f"Classification: {self.current_class_label}")

    def connect_plc(self):
        try:
            self.plc.connect(self.plc_config['host'], self.plc_config['port'])
            self.update_status_box("Connected to PLC")
        except Exception as e:
            self.update_status_box(f"Error connecting to PLC: {e}")

    def disconnect_plc(self):
        try:
            self.plc.close()
            self.update_status_box("Disconnected from PLC")
        except Exception as e:
            self.update_status_box(f"Error disconnecting from PLC: {e}")

    def monitor_plc(self):
        while self.camera_active:
            try:
                for device in ['M100', 'M101', 'M102', 'M0']:
                    value = self.plc.batchread_bitunits(headdevice=device, readsize=1)
                    self.plc_states[device] = bool(value[0])

                if self.previous_coil_state is not None:
                    if self.previous_coil_state and not self.plc_states['M100']:
                        self.capture_images()

                self.previous_coil_state = self.plc_states['M100']
                self.update_status()

            except Exception as e:
                self.update_status_box(f"Error reading PLC coil: {e}")

    def update_status(self):
        for label, state in self.plc_states.items():
            if label in self.status_labels:
                self.status_labels[label].config(text=f"{label}: {'🟢' if state else '🔴'}")
        self.ok_counter_label.config(text=f"OK: {self.ok_count}")
        self.ng_counter_label.config(text=f"NG: {self.ng_count}")
        self.update_capture_button()

    def update_capture_button(self):
        self.capture_button.config(text="CAPTURE", 
                                   bg='green' if self.plc_states['M0'] else 'red',
                                   fg='white')

    def update_status_box(self, message):
        self.status_box.config(state='normal')
        self.status_box.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        self.status_box.see(tk.END)
        self.status_box.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='black')
    app = ImageClassifier(root)
    root.mainloop()