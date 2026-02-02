import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import time
import threading
import queue
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image, ImageDraw, ImageFont, ImageTk 

# ==============================================================================
# ⚙️ CONFIG & GLOBAL VARS
# ==============================================================================
def get_real_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_real_path()

# --- 📁 PATH SETUP ---
DIR_CONFIG = os.path.join(BASE_PATH, "config")
DIR_JSON   = os.path.join(BASE_PATH, "json")
DIR_IMG    = os.path.join(BASE_PATH, "img")
DIR_MODEL  = os.path.join(BASE_PATH, "model")
DIR_TEST   = os.path.join(BASE_PATH, "test")

CONFIG_FILE = os.path.join(DIR_JSON, "config_lpr_v6.json")
CREDENTIALS_FILE = os.path.join(DIR_JSON, "credentials.json")
SAVE_FOLDER = DIR_IMG

for folder in [DIR_JSON, DIR_IMG, DIR_MODEL, DIR_CONFIG, DIR_TEST]:
    if not os.path.exists(folder): os.makedirs(folder)

SHEET_NAME = "Thai_LPR"

sys.path.append(DIR_CONFIG)
try:
    from config.th_dict import th_dict
except ImportError:
    print("⚠️ Warning: th_dict.py not found.")
    th_dict = {}

DEFAULT_ZONES = {
    "single": [[400, 200], [880, 200], [980, 650], [300, 650]],
    "entry":  [[100, 200], [450, 200], [400, 650], [50, 650]],
    "exit":   [[800, 200], [1150, 200], [1200, 650], [850, 650]]
}

# ==============================================================================
# 📱 MAIN APP CLASS
# ==============================================================================
class LPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LPR System V6.5 - Select Source via GUI")
        self.root.geometry("1450x900") 
        self.root.configure(bg="#202020")

        # --- State Variables ---
        default_plate = os.path.join(DIR_MODEL, "best_plate.pt")
        default_char = os.path.join(DIR_MODEL, "best_char.pt")

        self.app_state = {
            "zones": {k: np.array(v, np.int32) for k, v in DEFAULT_ZONES.items()},
            "preset_name": "Default",
            "is_dual_mode": False,
            "model_plate_path": default_plate,
            "model_char_path": default_char,
            "speed": 20,
            "source_type": "webcam", # webcam or video
            "source_path": "0"       # "0" or "C:/path/to/video.mp4"
        }
        self.presets = {"Default": DEFAULT_ZONES}
        self.is_running = True
        self.cap = None
        self.current_edit_target = "single"
        self.selected_point_idx = -1
        
        self.latest_read_results = {} 
        self.plate_queue = queue.Queue()
        self.id_cooldown = {}

        # --- Load Config ---
        self.load_config()

        # --- AI Models ---
        self.model_plate = None
        self.model_char = None
        self.load_models()

        # --- GUI Setup ---
        self.setup_ui()

        # --- Threads ---
        threading.Thread(target=self.ocr_worker, daemon=True).start()
        
        # --- Start Camera ---
        self.change_source(init=True)
        self.video_loop() 

    def setup_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#202020")
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 🎥 Video Frame
        self.frm_video = tk.Frame(main_paned, bg="black")
        main_paned.add(self.frm_video, stretch="always")
        self.lbl_video = tk.Label(self.frm_video, bg="black", text="No Video Source", fg="white")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # 🎛️ Control Panel
        self.frm_ctrl = tk.Frame(main_paned, bg="#f0f0f0", width=400)
        main_paned.add(self.frm_ctrl, stretch="never")

        tk.Label(self.frm_ctrl, text="🛠️ Control Panel", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        # --- 1. Camera Source Selection (NEW) ---
        frm_src = tk.LabelFrame(self.frm_ctrl, text="🎥 Camera Source", bg="#e8e8e8", padx=5, pady=5)
        frm_src.pack(fill="x", padx=10, pady=5)
        
        self.src_type_var = tk.StringVar(value=self.app_state.get('source_type', 'webcam'))
        
        # Radio Buttons
        f_radio = tk.Frame(frm_src, bg="#e8e8e8")
        f_radio.pack(fill="x")
        tk.Radiobutton(f_radio, text="Webcam (Index)", variable=self.src_type_var, value="webcam", bg="#e8e8e8").pack(side="left")
        tk.Radiobutton(f_radio, text="Video File", variable=self.src_type_var, value="video", bg="#e8e8e8").pack(side="left", padx=10)
        
        # Input & Browse
        self.src_path_var = tk.StringVar(value=self.app_state.get('source_path', '0'))
        f_inp = tk.Frame(frm_src, bg="#e8e8e8"); f_inp.pack(fill="x", pady=2)
        
        self.ent_src = tk.Entry(f_inp, textvariable=self.src_path_var)
        self.ent_src.pack(side="left", fill="x", expand=True)
        tk.Button(f_inp, text="📂", width=3, command=self.browse_video_source).pack(side="right", padx=2)
        
        tk.Button(frm_src, text="▶ LOAD / SWITCH SOURCE", bg="#4CAF50", fg="white", command=lambda: self.change_source(init=False)).pack(fill="x", pady=5)

        # --- 2. AI Models ---
        frm_models = tk.LabelFrame(self.frm_ctrl, text="🤖 AI Models", bg="#e8e8e8", padx=5, pady=5)
        frm_models.pack(fill="x", padx=10, pady=5)
        
        self.plate_path_var = tk.StringVar(value=self.app_state['model_plate_path'])
        tk.Entry(frm_models, textvariable=self.plate_path_var).pack(fill="x", pady=2)
        tk.Button(frm_models, text="Browse Plate Model", command=lambda: self.browse_file(self.plate_path_var, "model")).pack(fill="x")
        
        self.char_path_var = tk.StringVar(value=self.app_state['model_char_path'])
        tk.Entry(frm_models, textvariable=self.char_path_var).pack(fill="x", pady=2)
        tk.Button(frm_models, text="Browse Char Model", command=lambda: self.browse_file(self.char_path_var, "model")).pack(fill="x")
        
        tk.Button(frm_models, text="↻ Reload Models", bg="#FF9800", fg="white", command=self.reload_models_action).pack(fill="x", pady=5)

        # --- 3. Settings & Preset ---
        frm_mode = tk.LabelFrame(self.frm_ctrl, text="⚙️ Settings", bg="#f0f0f0", padx=5, pady=5)
        frm_mode.pack(fill="x", padx=10, pady=5)
        
        # Mode Selection
        self.mode_var = tk.StringVar(value="dual" if self.app_state['is_dual_mode'] else "single")
        tk.Radiobutton(frm_mode, text="Single Zone", variable=self.mode_var, value="single", command=self.toggle_mode).pack(anchor="w")
        tk.Radiobutton(frm_mode, text="Dual Zone (In/Out)", variable=self.mode_var, value="dual", command=self.toggle_mode).pack(anchor="w")

        tk.Label(frm_mode, text="Select Preset:", bg="#f0f0f0", fg="gray").pack(anchor="w", pady=(5,0))
        
        # Preset Dropdown
        self.preset_var = tk.StringVar(value=self.app_state['preset_name'])
        self.combo_preset = ttk.Combobox(frm_mode, textvariable=self.preset_var, values=list(self.presets.keys()), state="readonly")
        self.combo_preset.pack(fill="x", pady=2)
        self.combo_preset.bind("<<ComboboxSelected>>", self.on_preset_change)
        
        # ช่องพิมพ์ชื่อ Preset ใหม่
        tk.Label(frm_mode, text="Or Save as New Name:", bg="#f0f0f0", fg="gray").pack(anchor="w", pady=(5,0))
        self.new_preset_var = tk.StringVar()
        tk.Entry(frm_mode, textvariable=self.new_preset_var).pack(fill="x", pady=2)

        # ปุ่ม Save
        tk.Button(frm_mode, text="💾 Save Config", bg="#2196F3", fg="white", command=self.save_config_action).pack(fill="x", pady=5)

        # --- 4. Zone Editor ---
        frm_edit = tk.LabelFrame(self.frm_ctrl, text="🎮 Zone Editor", bg="#f0f0f0", padx=5, pady=5)
        frm_edit.pack(fill="x", padx=10, pady=5)
        
        self.frm_zone_btns = tk.Frame(frm_edit, bg="#f0f0f0")
        self.frm_zone_btns.pack(fill="x", pady=5)

        self.btn_s = tk.Button(self.frm_zone_btns, text="Edit Single", command=lambda: self.set_target("single"))
        self.btn_in = tk.Button(self.frm_zone_btns, text="Edit Entry", command=lambda: self.set_target("entry"))
        self.btn_out = tk.Button(self.frm_zone_btns, text="Edit Exit", command=lambda: self.set_target("exit"))
        
        tk.Label(frm_edit, text="Move Speed:", bg="#f0f0f0").pack(anchor="w", pady=(5,0))
        self.speed_var = tk.IntVar(value=self.app_state['speed'])
        tk.Scale(frm_edit, from_=1, to=100, orient="horizontal", variable=self.speed_var).pack(fill="x")

        f_arr = tk.Frame(frm_edit, bg="#f0f0f0"); f_arr.pack(pady=5)
        btn_cfg = {"width": 5, "bg": "white"}
        tk.Button(f_arr, text="▲", command=lambda: self.move_zone(0,-1), **btn_cfg).grid(row=0, column=1)
        tk.Button(f_arr, text="◀", command=lambda: self.move_zone(-1,0), **btn_cfg).grid(row=1, column=0)
        tk.Button(f_arr, text="●", command=lambda: self.reset_selection(), **btn_cfg).grid(row=1, column=1) 
        tk.Button(f_arr, text="▶", command=lambda: self.move_zone(1,0), **btn_cfg).grid(row=1, column=2)
        tk.Button(f_arr, text="▼", command=lambda: self.move_zone(0,1), **btn_cfg).grid(row=2, column=1)

        f_crn = tk.Frame(frm_edit, bg="#f0f0f0"); f_crn.pack()
        for i, c in enumerate(["TL", "TR", "BR", "BL"]):
            tk.Button(f_crn, text=c, width=4, command=lambda x=i: self.select_corner(x)).pack(side="left", padx=2)

        self.lbl_status = tk.Label(frm_edit, text="Ready", fg="blue", bg="#f0f0f0")
        self.lbl_status.pack(pady=5)
        
        # --- 5. Logs ---
        tk.Label(self.frm_ctrl, text="📜 Last Reads", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(10,0))
        self.txt_log = tk.Text(self.frm_ctrl, height=10, width=30, state='disabled')
        self.txt_log.pack(padx=5, pady=5, fill="both", expand=True)

        self.refresh_mode_buttons() 

    # ==========================
    # 🧠 Logic Functions
    # ==========================
    def browse_file(self, var, ftype):
        init_dir = DIR_MODEL if ftype=="model" else BASE_PATH
        ext = "*.pt"
        f = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("File", ext)])
        if f: var.set(f)

    def browse_video_source(self):
        # เปิด Dialog เลือกไฟล์วิดีโอ (เริ่มที่ folder test)
        init_dir = DIR_TEST if os.path.exists(DIR_TEST) else BASE_PATH
        f = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")])
        if f:
            self.src_path_var.set(f)
            self.src_type_var.set("video")

    def change_source(self, init=False):
        """ ฟังก์ชันสลับ Source (เรียกตอนเริ่ม หรือตอนกดปุ่ม) """
        source_type = self.src_type_var.get()
        source_val = self.src_path_var.get()

        # Update State
        self.app_state['source_type'] = source_type
        self.app_state['source_path'] = source_val
        if not init: self.save_config()

        # Determine Input
        final_source = 0
        if source_type == "webcam":
            try: final_source = int(source_val) # ต้องเป็น int สำหรับ webcam
            except: final_source = 0
        else:
            final_source = source_val # เป็น path string
        
        print(f"🔄 Switching Source to: {final_source}")
        
        # Release Old
        if self.cap is not None:
            self.cap.release()
        
        # Open New
        self.cap = cv2.VideoCapture(final_source)
        
        # Settings
        if source_type == "webcam":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) 
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open source: {final_source}")
        else:
            if not init: messagebox.showinfo("Success", "Camera Source Loaded!")

    def load_models(self):
        print("⏳ Loading Models...")
        try:
            self.model_plate = YOLO(self.app_state['model_plate_path'])
            self.model_char = YOLO(self.app_state['model_char_path'])
            print("✅ Models Loaded")
        except Exception as e:
            print(f"❌ Model Error: {e}")
            messagebox.showerror("Error", f"Could not load models.\nCheck path.\nError: {e}")

    def draw_thai(self, img, text, pos, color=(0, 255, 255), size=40):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def reload_models_action(self):
        self.app_state['model_plate_path'] = self.plate_path_var.get()
        self.app_state['model_char_path'] = self.char_path_var.get()
        self.save_config()
        self.load_models()
        messagebox.showinfo("Success", "Models Reloaded!")

    def video_loop(self):
        if not self.is_running: return
        
        if self.cap is None or not self.cap.isOpened():
            self.root.after(100, self.video_loop)
            return

        success, frame = self.cap.read()
        if success:
            frame_h, frame_w = frame.shape[:2]
            clean_frame = frame.copy()
            current_time = time.time()

            active_zones = []
            if self.app_state['is_dual_mode']:
                active_zones = [("entry", (0, 255, 0), "ENTRY"), ("exit", (0, 0, 255), "EXIT")]
            else:
                active_zones = [("single", (255, 165, 0), "Check Point")]

            # 1. วาดเส้นโซน (Zone Lines)
            for z_key, z_color, z_label in active_zones:
                points = self.app_state['zones'][z_key]
                is_edit = (z_key == self.current_edit_target)
                
                # วาดเส้นกรอบโซน
                cv2.polylines(frame, [points], isClosed=True, color=z_color, thickness=2)
                
                # วาดจุดมุมเฉพาะตอนแก้ไข (Edit Mode)
                if is_edit:
                    cv2.polylines(frame, [points], isClosed=True, color=(255, 255, 255), thickness=4) # Highlight
                    for i, pt in enumerate(points):
                        col = (0, 0, 255) if i == self.selected_point_idx else z_color
                        cv2.circle(frame, tuple(pt), 8, col, -1)
                
                # ชื่อโซน
                cv2.putText(frame, z_label, (points[0][0], points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, z_color, 2)

            # 2. AI Process & Tracking
            if self.model_plate:
                # ลดความละเอียดตอน Detect ลงนิดนึงเพื่อความเร็ว (imgsz=640 ก็พอสำหรับ Realtime)
                results = self.model_plate.track(frame, persist=True, verbose=False, conf=0.20, imgsz=1280)
                
                if results and results[0].boxes and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    boxes = results[0].boxes.xyxy.cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        px1, py1, px2, py2 = map(int, box)
                        cx, cy = int((px1+px2)/2), int((py1+py2)/2)
                        
                        # --- ส่วนที่แสดงผล (Display Logic) ---
                        
                        # A. ตรวจสอบว่าอ่านทะเบียนได้หรือยัง?
                        plate_text = ""
                        if track_id in self.latest_read_results:
                            plate_text = self.latest_read_results[track_id]['text']

                        # B. วาดกรอบรอบป้าย (สีขาวถ้าปกติ สีเขียวถ้าอ่านได้แล้ว)
                        box_color = (0, 255, 0) if plate_text else (200, 200, 200)
                        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

                        # C. แสดงเลขทะเบียน (แบบคลีนๆ)
                        if plate_text:
                            # คำนวณตำแหน่งไม่ให้ตกขอบจอ
                            text_y = py1 - 15 if py1 - 15 > 30 else py2 + 40
                            
                            # วาดพื้นหลังดำรองตัวหนังสือ (จะได้อ่านง่ายๆ ไม่จมกับภาพ)
                            # (กะขนาดคร่าวๆ)
                            bg_w = len(plate_text) * 20 
                            cv2.rectangle(frame, (px1, text_y - 35), (px1 + bg_w, text_y + 5), (0, 0, 0), -1)
                            
                            # วาดตัวหนังสือภาษาไทย
                            frame = self.draw_thai(frame, plate_text, (px1, text_y - 30), color=(0, 255, 255), size=40)

                        # --- จบส่วนแสดงผล (ลบ Loop วาดตัวอักษรยิบย่อยออกแล้ว) ---

                        # D. Logic การจับภาพ (Capture Logic) - เหมือนเดิม
                        captured = False
                        for z_key, z_color, z_label in active_zones:
                            points = self.app_state['zones'][z_key]
                            in_zone = cv2.pointPolygonTest(points, (cx, cy), False) >= 0
                            cd_key = (track_id, z_key)
                            is_cd = (cd_key in self.id_cooldown) and (current_time - self.id_cooldown[cd_key] < 5.0)

                            if in_zone:
                                if is_cd:
                                    # ขึ้นสถานะ WAIT เล็กๆ มุมขวากรอบ
                                    cv2.putText(frame, "WAIT", (px2-60, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                                    captured = True
                                else:
                                    pad = 10
                                    try:
                                        snap = clean_frame[max(0,py1-pad):min(frame_h,py2+pad), max(0,px1-pad):min(frame_w,px2+pad)].copy()
                                        self.plate_queue.put((track_id, snap, z_label))
                                        self.id_cooldown[cd_key] = current_time
                                        # Flash Effect (กรอบหนา)
                                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 5)
                                        captured = True
                                    except: pass

            # Display Setup
            display_h = 750 
            aspect_ratio = frame_w / frame_h
            display_w = int(display_h * aspect_ratio)
            frame_resized = cv2.resize(frame, (display_w, display_h))
            
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.lbl_video.configure(image=img_tk)
            self.lbl_video.image = img_tk 
        else:
            if self.src_type_var.get() == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.root.after(10, self.video_loop)

    def ocr_worker(self):
        while self.is_running:
            try:
                item = self.plate_queue.get(timeout=1)
                track_id, plate_img, zone_label = item
                
                h, w = plate_img.shape[:2]
                scale = 2
                resized = cv2.resize(plate_img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
                kernel = np.array([[0, -0.5, 0], [-0.5, 3,-0.5], [0, -0.5, 0]])
                sharpened = cv2.filter2D(resized, -1, kernel)

                if self.model_char:
                    res = self.model_char(sharpened, verbose=False, conf=0.20, imgsz=640)
                    
                    char_data = [] 
                    screen_details = []
                    
                    for r in res:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            raw = self.model_char.names[cls_id]
                            real = th_dict.get(raw, raw)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx = (x1+x2)/2
                            cy = (y1+y2)/2
                            char_data.append((cx, cy, real))
                            screen_details.append({"box": (x1, y1, x2, y2), "char": real})

                    if char_data:
                        char_data.sort(key=lambda x: x[1])
                        lines = []
                        current_line = [char_data[0]]
                        for i in range(1, len(char_data)):
                            cx, cy, ch = char_data[i]
                            last_cx, last_cy, last_ch = current_line[-1]
                            if abs(cy - last_cy) < 20: 
                                current_line.append(char_data[i])
                            else:
                                lines.append(current_line)
                                current_line = [char_data[i]]
                        lines.append(current_line)

                        final_text_parts = []
                        for line in lines:
                            line.sort(key=lambda x: x[0])
                            text_line = "".join([x[2] for x in line])
                            final_text_parts.append(text_line)

                        text = "".join(final_text_parts)
                        
                        if len(text) > 2:
                            print(f"Read: {text}")
                            self.latest_read_results[track_id] = {
                                "text": text,
                                "chars": screen_details,
                                "timestamp": time.time()
                            }
                            self.update_log(f"[{zone_label}] {text}")
                            
                            dt = datetime.datetime.now()
                            dt_str = dt.strftime('%d_%m_%Y')
                            
                            if zone_label == "SINGLE":
                                fname = f"{text}_{dt_str}.jpg"
                            elif zone_label == "ENTRY":
                                fname = f"{text}_{dt_str}_ขาเข้า.jpg"
                            elif zone_label == "EXIT":
                                fname = f"{text}_{dt_str}_ขาออก.jpg"
                            else:
                                fname = f"{text}_{dt_str}_{zone_label}.jpg"
                            
                            cv2.imwrite(os.path.join(SAVE_FOLDER, fname), sharpened)
                            
                            status = "ขาเข้า" if zone_label == "ENTRY" else ("ขาออก" if zone_label == "EXIT" else "ทั่วไป")
                            self.log_to_sheet([dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M:%S"), text, status, fname])

                self.plate_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"OCR Error: {e}")

    # --- Helper Functions ---
    def update_log(self, msg):
        self.txt_log.config(state='normal')
        self.txt_log.insert('1.0', msg + "\n")
        self.txt_log.config(state='disabled')

    def toggle_mode(self):
        self.app_state['is_dual_mode'] = (self.mode_var.get() == "dual")
        self.refresh_mode_buttons()
        self.save_config()

    def refresh_mode_buttons(self):
        self.btn_s.pack_forget(); self.btn_in.pack_forget(); self.btn_out.pack_forget()
        if self.app_state['is_dual_mode']:
            self.btn_in.pack(side="left", fill="x", expand=True, padx=2)
            self.btn_out.pack(side="right", fill="x", expand=True, padx=2)
            if self.current_edit_target == "single": self.set_target("entry")
            else: self.set_target(self.current_edit_target)
        else:
            self.btn_s.pack(side="top", fill="x", expand=True, padx=2)
            self.set_target("single")

    def set_target(self, t):
        self.current_edit_target = t
        self.selected_point_idx = -1
        for k, b in [("single", self.btn_s), ("entry", self.btn_in), ("exit", self.btn_out)]:
            b.config(bg="#4CAF50" if k==t else "#ddd", fg="white" if k==t else "black")
        self.lbl_status.config(text=f"Editing: {t.upper()}")

    def move_zone(self, dx, dy):
        self.app_state['speed'] = self.speed_var.get()
        step = self.app_state['speed']
        pts = self.app_state['zones'][self.current_edit_target]
        shift = [dx*step, dy*step]
        if self.selected_point_idx == -1: pts += shift
        else: pts[self.selected_point_idx] += shift

    def select_corner(self, idx):
        self.selected_point_idx = idx
        self.lbl_status.config(text=f"Editing: Corner {idx}")

    def reset_selection(self):
        self.selected_point_idx = -1
        self.lbl_status.config(text="Editing: Whole Box")

    def on_preset_change(self, e):
        name = self.combo_preset.get()
        if name in self.presets:
            self.app_state['preset_name'] = name
            self.load_config()
            self.save_config()

    def save_config_action(self):
        # 1. เช็คว่ามีการพิมพ์ชื่อใหม่ไหม
        new_name = self.new_preset_var.get().strip()
        
        if new_name:
            # ถ้ามีชื่อใหม่ ให้ใช้ชื่อนี้เป็น Preset ปัจจุบัน
            self.app_state['preset_name'] = new_name
            # เพิ่มชื่อใหม่ลงในรายการ Presets (Copy โซนปัจจุบันไปใส่)
            # ต้อง copy() เพื่อไม่ให้ค่าทับกันมั่ว
            current_zones_copy = {k: v.tolist() for k, v in self.app_state['zones'].items()}
            self.presets[new_name] = current_zones_copy 
            
            # อัปเดต Dropdown
            self.combo_preset['values'] = list(self.presets.keys())
            self.combo_preset.set(new_name)
            
            # ล้างช่องพิมพ์ชื่อ
            self.new_preset_var.set("")
            print(f"✅ Created New Preset: {new_name}")

        # 2. บันทึกข้อมูลลงไฟล์ JSON
        success = self.save_config()
        
        if success:
            messagebox.showinfo("Saved", f"Configuration Saved!\nPreset: {self.app_state['preset_name']}")
        else:
            messagebox.showerror("Error", "Could not save config file.")
        
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.app_state.update(data.get('app_state', {}))
                    self.presets = data.get('presets', self.presets)
                    saved = self.presets.get(self.app_state['preset_name'], DEFAULT_ZONES)
                    for k in self.app_state['zones']:
                        if k in saved: self.app_state['zones'][k] = np.array(saved[k], np.int32)
            except: pass

    def save_config(self):
        zones_list = {k: v.tolist() for k, v in self.app_state['zones'].items()}
        self.presets[self.app_state['preset_name']] = zones_list
        # Save Source settings as well
        self.app_state['source_type'] = self.src_type_var.get()
        self.app_state['source_path'] = self.src_path_var.get()

        data = {
            "app_state": {
                "preset_name": self.app_state['preset_name'],
                "is_dual_mode": self.app_state['is_dual_mode'],
                "model_plate_path": self.app_state['model_plate_path'],
                "model_char_path": self.app_state['model_char_path'],
                "speed": self.app_state['speed'],
                "source_type": self.app_state['source_type'],
                "source_path": self.app_state['source_path']
            },
            "presets": self.presets
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except: return False

    def connect_sheet(self):
        if not os.path.exists(CREDENTIALS_FILE): return None
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
            client = gspread.authorize(creds)
            return client.open(SHEET_NAME).sheet1
        except: return None

    def log_to_sheet(self, row):
        def _w():
            s = self.connect_sheet()
            if s:
                try: s.append_row(row)
                except: pass
        threading.Thread(target=_w, daemon=True).start()

    def on_close(self):
        self.is_running = False
        if hasattr(self, 'cap') and self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LPRApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()