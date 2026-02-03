import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import time
import threading
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image, ImageDraw, ImageFont, ImageTk
from collections import Counter


# ==============================================================================
# ⚙️ CONFIG & GLOBAL VARS
# ==============================================================================
def get_real_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_real_path()

# --- 📁 PATH SETUP ---
DIR_CONFIG = os.path.join(BASE_PATH, "config")
DIR_JSON = os.path.join(BASE_PATH, "json")
DIR_IMG = os.path.join(BASE_PATH, "img")
DIR_MODEL = os.path.join(BASE_PATH, "model")
DIR_TEST = os.path.join(BASE_PATH, "test")

CONFIG_FILE = os.path.join(DIR_JSON, "config_lpr_v6.json")
CREDENTIALS_FILE = os.path.join(DIR_JSON, "credentials.json")
SAVE_FOLDER = DIR_IMG

for folder in [DIR_JSON, DIR_IMG, DIR_MODEL, DIR_CONFIG, DIR_TEST]:
    if not os.path.exists(folder):
        os.makedirs(folder)

SHEET_NAME = "Thai_LPR"

sys.path.append(DIR_CONFIG)
try:
    from config.th_dict import th_dict
except ImportError:
    print("⚠️ Warning: th_dict.py not found.")
    th_dict = {}

DEFAULT_ZONES = {
    "single": [[200, 200], [500, 200], [500, 500], [200, 500]],
    "entry": [[50, 200], [300, 200], [300, 500], [50, 500]],
    "exit": [[400, 200], [650, 200], [650, 500], [400, 500]],
}


# ==============================================================================
# 📱 MAIN APP CLASS
# ==============================================================================
class LPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LPR System V8.1 - Custom Sheet Columns")
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
            "speed": 10,
            "source_type": "webcam",
            "source_path": "0",
        }
        self.presets = {"Default": DEFAULT_ZONES}
        self.is_running = True
        self.cap = None
        self.current_edit_target = "single"
        self.selected_point_idx = -1

        # --- Session Tracking ---
        self.session = {
            "active": False,
            "first_data": {"img": None, "text": ""},
            "best_data": {"img": None, "text": "", "score": 0},
            "last_data": {"img": None, "text": ""},
            "reads": [],
            "direction": "",
            "last_seen_time": 0,
        }
        self.TIMEOUT_SEC = 2.5

        self.load_config()
        self.model_plate = None
        self.model_char = None
        self.load_models()
        self.setup_ui()
        threading.Thread(target=self.process_video_thread, daemon=True).start()

    def setup_ui(self):
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#202020")
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 1. Video Frame
        self.frm_video = tk.Frame(main_paned, bg="black")
        main_paned.add(self.frm_video, stretch="always")
        self.lbl_video = tk.Label(
            self.frm_video, bg="black", text="No Video Source", fg="white"
        )
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # 2. Control Panel
        container = tk.Frame(main_paned, bg="#f0f0f0", width=360)
        main_paned.add(container, stretch="never")

        canvas = tk.Canvas(container, bg="#f0f0f0")
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.frm_ctrl = tk.Frame(canvas, bg="#f0f0f0")

        self.ctrl_window_id = canvas.create_window(
            (0, 0), window=self.frm_ctrl, anchor="nw"
        )

        def on_canvas_configure(event):
            canvas.itemconfig(self.ctrl_window_id, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)
        self.frm_ctrl.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # --- Controls ---
        tk.Label(
            self.frm_ctrl,
            text="🛠️ Control Panel",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
        ).pack(pady=10)

        # Source
        frm_src = tk.LabelFrame(
            self.frm_ctrl, text="🎥 Camera Source", bg="#e8e8e8", padx=5, pady=5
        )
        frm_src.pack(fill="x", padx=10, pady=5)
        self.src_type_var = tk.StringVar(
            value=self.app_state.get("source_type", "webcam")
        )
        f_radio = tk.Frame(frm_src, bg="#e8e8e8")
        f_radio.pack(fill="x")
        tk.Radiobutton(
            f_radio,
            text="Webcam",
            variable=self.src_type_var,
            value="webcam",
            bg="#e8e8e8",
        ).pack(side="left")
        tk.Radiobutton(
            f_radio,
            text="Video File",
            variable=self.src_type_var,
            value="video",
            bg="#e8e8e8",
        ).pack(side="left", padx=10)

        self.src_path_var = tk.StringVar(value=self.app_state.get("source_path", "0"))
        f_inp = tk.Frame(frm_src, bg="#e8e8e8")
        f_inp.pack(fill="x", pady=2)
        self.ent_src = tk.Entry(f_inp, textvariable=self.src_path_var)
        self.ent_src.pack(side="left", fill="x", expand=True)
        tk.Button(f_inp, text="📂", width=3, command=self.browse_video_source).pack(
            side="right", padx=2
        )
        tk.Button(
            frm_src,
            text="▶ LOAD / SWITCH",
            bg="#4CAF50",
            fg="white",
            command=lambda: self.change_source(init=False),
        ).pack(fill="x", pady=5)

        # Models
        frm_models = tk.LabelFrame(
            self.frm_ctrl, text="🤖 AI Models", bg="#e8e8e8", padx=5, pady=5
        )
        frm_models.pack(fill="x", padx=10, pady=5)
        self.plate_path_var = tk.StringVar(value=self.app_state["model_plate_path"])
        tk.Entry(frm_models, textvariable=self.plate_path_var).pack(fill="x", pady=2)
        tk.Button(
            frm_models,
            text="Browse Plate",
            command=lambda: self.browse_file(self.plate_path_var, "model"),
        ).pack(fill="x")
        self.char_path_var = tk.StringVar(value=self.app_state["model_char_path"])
        tk.Entry(frm_models, textvariable=self.char_path_var).pack(fill="x", pady=2)
        tk.Button(
            frm_models,
            text="Browse Char",
            command=lambda: self.browse_file(self.char_path_var, "model"),
        ).pack(fill="x")
        tk.Button(
            frm_models,
            text="↻ Reload Models",
            bg="#FF9800",
            fg="white",
            command=self.reload_models_action,
        ).pack(fill="x", pady=5)

        # Settings
        frm_mode = tk.LabelFrame(
            self.frm_ctrl, text="⚙️ Settings", bg="#f0f0f0", padx=5, pady=5
        )
        frm_mode.pack(fill="x", padx=10, pady=5)
        self.mode_var = tk.StringVar(
            value="dual" if self.app_state["is_dual_mode"] else "single"
        )
        tk.Radiobutton(
            frm_mode,
            text="Single Zone",
            variable=self.mode_var,
            value="single",
            command=self.toggle_mode,
        ).pack(anchor="w")
        tk.Radiobutton(
            frm_mode,
            text="Dual Zone",
            variable=self.mode_var,
            value="dual",
            command=self.toggle_mode,
        ).pack(anchor="w")

        tk.Label(frm_mode, text="Select Preset:", bg="#f0f0f0", fg="gray").pack(
            anchor="w", pady=(5, 0)
        )
        self.preset_var = tk.StringVar(value=self.app_state["preset_name"])
        self.combo_preset = ttk.Combobox(
            frm_mode,
            textvariable=self.preset_var,
            values=list(self.presets.keys()),
            state="readonly",
        )
        self.combo_preset.pack(fill="x", pady=2)
        self.combo_preset.bind("<<ComboboxSelected>>", self.on_preset_change)

        tk.Label(frm_mode, text="Or Save New:", bg="#f0f0f0", fg="gray").pack(
            anchor="w", pady=(5, 0)
        )
        self.new_preset_var = tk.StringVar()
        tk.Entry(frm_mode, textvariable=self.new_preset_var).pack(fill="x", pady=2)
        tk.Button(
            frm_mode,
            text="💾 Save Config",
            bg="#2196F3",
            fg="white",
            command=self.save_config_action,
        ).pack(fill="x", pady=5)

        # Zone Editor
        frm_edit = tk.LabelFrame(
            self.frm_ctrl, text="🎮 Zone Editor", bg="#f0f0f0", padx=5, pady=5
        )
        frm_edit.pack(fill="x", padx=10, pady=5)
        self.frm_zone_btns = tk.Frame(frm_edit, bg="#f0f0f0")
        self.frm_zone_btns.pack(fill="x", pady=5)
        self.btn_s = tk.Button(
            self.frm_zone_btns,
            text="Edit Single",
            command=lambda: self.set_target("single"),
        )
        self.btn_in = tk.Button(
            self.frm_zone_btns,
            text="Edit Entry",
            command=lambda: self.set_target("entry"),
        )
        self.btn_out = tk.Button(
            self.frm_zone_btns,
            text="Edit Exit",
            command=lambda: self.set_target("exit"),
        )

        tk.Label(frm_edit, text="Speed/Size Step:", bg="#f0f0f0").pack(
            anchor="w", pady=(5, 0)
        )
        self.speed_var = tk.IntVar(value=self.app_state["speed"])
        tk.Scale(
            frm_edit, from_=1, to=100, orient="horizontal", variable=self.speed_var
        ).pack(fill="x")

        f_arr = tk.Frame(frm_edit, bg="#f0f0f0")
        f_arr.pack(pady=5)
        btn_cfg = {"width": 5, "bg": "white"}
        tk.Button(
            f_arr, text="▲", command=lambda: self.move_zone(0, -1), **btn_cfg
        ).grid(row=0, column=1)
        tk.Button(
            f_arr, text="◀", command=lambda: self.move_zone(-1, 0), **btn_cfg
        ).grid(row=1, column=0)
        tk.Button(
            f_arr, text="●", command=lambda: self.reset_selection(), **btn_cfg
        ).grid(row=1, column=1)
        tk.Button(
            f_arr, text="▶", command=lambda: self.move_zone(1, 0), **btn_cfg
        ).grid(row=1, column=2)
        tk.Button(
            f_arr, text="▼", command=lambda: self.move_zone(0, 1), **btn_cfg
        ).grid(row=2, column=1)

        f_crn = tk.Frame(frm_edit, bg="#f0f0f0")
        f_crn.pack()
        for i, c in enumerate(["TL", "TR", "BR", "BL"]):
            tk.Button(
                f_crn, text=c, width=4, command=lambda x=i: self.select_corner(x)
            ).pack(side="left", padx=2)

        self.lbl_status = tk.Label(frm_edit, text="Ready", fg="blue", bg="#f0f0f0")
        self.lbl_status.pack(pady=5)

        # Logs
        tk.Label(
            self.frm_ctrl,
            text="📜 Last Reads",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
        ).pack(pady=(10, 0))
        self.txt_log = tk.Text(self.frm_ctrl, height=8, width=30, state="disabled")
        self.txt_log.pack(padx=5, pady=5, fill="both", expand=True)

        self.refresh_mode_buttons()

    # ==========================
    # 🧠 Logic Functions
    # ==========================
    def browse_file(self, var, ftype):
        init_dir = DIR_MODEL if ftype == "model" else BASE_PATH
        ext = "*.pt"
        f = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("File", ext)])
        if f:
            var.set(f)

    def browse_video_source(self):
        init_dir = DIR_TEST if os.path.exists(DIR_TEST) else BASE_PATH
        f = filedialog.askopenfilename(
            initialdir=init_dir, filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")]
        )
        if f:
            self.src_path_var.set(f)
            self.src_type_var.set("video")

    def change_source(self, init=False):
        source_type = self.src_type_var.get()
        source_val = self.src_path_var.get()
        self.app_state["source_type"] = source_type
        self.app_state["source_path"] = source_val
        if not init:
            self.save_config()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print(f"🔄 Switching Source: {source_type} -> {source_val}")
        self.need_restart_source = True

    def load_models(self):
        print("⏳ Loading Models...")
        try:
            self.model_plate = YOLO(self.app_state["model_plate_path"])
            self.model_char = YOLO(self.app_state["model_char_path"])
            print("✅ Models Loaded")
        except Exception as e:
            print(f"❌ Model Error: {e}")
            messagebox.showerror(
                "Error", f"Could not load models.\nCheck path.\nError: {e}"
            )

    def reload_models_action(self):
        self.app_state["model_plate_path"] = self.plate_path_var.get()
        self.app_state["model_char_path"] = self.char_path_var.get()
        self.save_config()
        self.load_models()
        messagebox.showinfo("Success", "Models Reloaded!")

    def draw_thai(self, img, text, pos, color=(0, 255, 255), size=40):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", size)
        except:
            font = ImageFont.load_default()

        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def enhance_plate_image(self, img):
        if img is None:
            return None
        h, w = img.shape[:2]
        if h < 80:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        return img

    # ==========================
    # 📸 OCR Section
    # ==========================
    def read_plate_text(self, plate_img):
        if self.model_char is None or plate_img is None:
            return ""
        processed_img = self.enhance_plate_image(plate_img)

        results = self.model_char(processed_img, verbose=False, conf=0.50)

        char_data = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw = self.model_char.names[cls_id]
                real = th_dict.get(raw, raw)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                h_char = y2 - y1
                char_data.append((cx, cy, real, h_char))

        if not char_data:
            return ""

        avg_h = sum([c[3] for c in char_data]) / len(char_data)
        char_data.sort(key=lambda x: x[1])

        lines = []
        current_line = [char_data[0]]
        for i in range(1, len(char_data)):
            cx, cy, ch, h_char = char_data[i]
            last_cx, last_cy, last_ch, last_h = current_line[-1]
            if abs(cy - last_cy) < (avg_h * 0.6):
                current_line.append(char_data[i])
            else:
                lines.append(current_line)
                current_line = [char_data[i]]
        lines.append(current_line)

        final_text = ""
        for line in lines:
            line.sort(key=lambda x: x[0])
            final_text += "".join([x[2] for x in line])

        return final_text

    def process_video_thread(self):
        self.change_source(init=True)
        self.need_restart_source = False

        while self.is_running:
            if getattr(self, "need_restart_source", False):
                source_type = self.app_state["source_type"]
                source_val = self.app_state["source_path"]
                final_src = int(source_val) if source_type == "webcam" else source_val

                self.cap = cv2.VideoCapture(final_src)
                if source_type == "webcam":
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                self.need_restart_source = False

            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if self.app_state["source_type"] == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.reset_session()
                continue

            h, w = frame.shape[:2]
            target_h = 720
            scale = target_h / h
            target_w = int(w * scale)

            display_frame = cv2.resize(frame, (target_w, target_h))
            process_frame = display_frame.copy()
            clean_frame = frame.copy()

            current_time = time.time()
            found_plate_in_zone = False

            if self.model_plate:
                results = self.model_plate(
                    process_frame, verbose=False, conf=0.4, imgsz=1280
                )

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        zone_status = self.check_point_in_zones((cx, cy))

                        color = (0, 255, 0) if zone_status else (100, 100, 100)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                        scale_inv = 1 / scale
                        real_x1, real_y1 = int(x1 * scale_inv), int(y1 * scale_inv)
                        real_x2, real_y2 = int(x2 * scale_inv), int(y2 * scale_inv)

                        h_f, w_f, _ = clean_frame.shape
                        pad = 10
                        ny1, ny2 = max(0, real_y1 - pad), min(h_f, real_y2 + pad)
                        nx1, nx2 = max(0, real_x1 - pad), min(w_f, real_x2 + pad)

                        plate_crop = clean_frame[ny1:ny2, nx1:nx2]

                        # อ่าน OCR
                        current_text = self.read_plate_text(plate_crop)

                        # แสดงผลบนจอ
                        if len(current_text) > 1:
                            bg_w = len(current_text) * 15
                            bg_h = 30
                            cv2.rectangle(
                                display_frame,
                                (x1, y1 - bg_h),
                                (x1 + bg_w, y1),
                                (0, 0, 0),
                                -1,
                            )
                            display_frame = self.draw_thai(
                                display_frame,
                                current_text,
                                (x1, y1 - 28),
                                (0, 255, 255),
                                20,
                            )

                        if zone_status:
                            found_plate_in_zone = True

                            # 1. รูปแรก (First)
                            if not self.session["active"]:
                                self.session["active"] = True
                                self.session["direction"] = zone_status
                                self.session["first_data"] = {
                                    "img": plate_crop.copy(),
                                    "text": (
                                        current_text
                                        if len(current_text) > 1
                                        else "Unknown"
                                    ),
                                }

                            # 2. รูปสุดท้าย (Last)
                            self.session["last_seen_time"] = current_time
                            self.session["last_data"] = {
                                "img": plate_crop.copy(),
                                "text": (
                                    current_text if len(current_text) > 1 else "Unknown"
                                ),
                            }

                            # 3. รูปที่ดีที่สุด (Best)
                            score = len(current_text)
                            if len(current_text) > 4:
                                score += 10
                            if score > self.session["best_data"]["score"]:
                                self.session["best_data"] = {
                                    "img": plate_crop.copy(),
                                    "text": current_text,
                                    "score": score,
                                }

                            # เก็บประวัติ
                            if len(current_text) > 2:
                                self.session["reads"].append(current_text)

            if self.session["active"]:
                if not found_plate_in_zone and (
                    current_time - self.session["last_seen_time"] > self.TIMEOUT_SEC
                ):
                    self.finish_session_and_save()

            self.draw_zones(display_frame)
            self.update_ui_frame(display_frame)
            time.sleep(0.01)

    def reset_session(self):
        self.session = {
            "active": False,
            "first_data": {"img": None, "text": ""},
            "best_data": {"img": None, "text": "", "score": 0},
            "last_data": {"img": None, "text": ""},
            "reads": [],
            "direction": "",
            "last_seen_time": 0,
        }

    def finish_session_and_save(self):
        all_reads = self.session["reads"]

        if not all_reads:
            print("❌ Session Discarded (No Text Read)")
            self.reset_session()
            return

        # 1. หา Best Read สำหรับ Column กลาง
        most_common = Counter(all_reads).most_common(1)
        winner_text = most_common[0][0]

        # Smart Fill (ถ้า Winner สั้นไป ให้หาตัวยาวมาแทน)
        if len(winner_text) < 7:
            longest_candidate = max(all_reads, key=len)
            if len(longest_candidate) > len(winner_text):
                winner_text = longest_candidate

        print(f"✅ Saving Session: {winner_text}")

        # 🟢 เตรียมข้อมูลลง Sheet
        first_txt = self.session["first_data"]["text"]
        last_txt = self.session["last_data"]["text"]

        # 🟢 กำหนดทิศทาง (IN/OUT หรือ -)
        if self.app_state["is_dual_mode"]:
            dir_str = "IN" if self.session["direction"] == "entry" else "OUT"
        else:
            dir_str = "-"

        # 🟢 Format: [Date, Time, Best, Direction, First, Last]
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d"),
            datetime.datetime.now().strftime("%H:%M:%S"),
            winner_text,  # Best Read
            dir_str,  # IN/OUT/-
            first_txt,  # First Read
            last_txt,  # Last Read
        ]

        self.log_to_sheet(row)

        # 🟢 ส่วน Save Image (แยกชื่อใครชื่อมัน)
        images_to_save = [
            ("First", self.session["first_data"]["img"], first_txt),
            ("Best", self.session["best_data"]["img"], winner_text),
            ("Last", self.session["last_data"]["img"], last_txt),
        ]

        direction_folder = dir_str if dir_str != "-" else "Single"
        threading.Thread(
            target=self.save_log_image,
            args=(images_to_save, winner_text, direction_folder),
        ).start()

        self.root.after(0, lambda: self.update_log_ui(f"[{dir_str}] {winner_text}"))
        self.reset_session()

    def save_log_image(self, images_data, folder_text, direction=""):
        if not images_data:
            return

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        safe_folder_name = "".join(
            [c for c in folder_text if c.isalnum() or c in (" ", "-", "_")]
        ).replace(" ", "")
        if not safe_folder_name:
            safe_folder_name = "Unknown"

        if self.app_state["is_dual_mode"]:
            folder_name = f"{safe_folder_name}_{date_str}_{time_str}_{direction}"
        else:
            folder_name = f"{safe_folder_name}_{date_str}_{time_str}"

        full_folder_path = os.path.join(SAVE_FOLDER, date_str, folder_name)
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        for suffix, img, txt in images_data:
            if img is None:
                continue

            safe_fname_text = "".join(
                [c for c in txt if c.isalnum() or c in (" ", "-", "_")]
            ).replace(" ", "")
            if not safe_fname_text:
                safe_fname_text = "Unknown"

            fname = f"{safe_fname_text}_{suffix}_{date_str}_{time_str}.jpg"
            save_path = os.path.join(full_folder_path, fname)

            is_success, im_buf = cv2.imencode(".jpg", img)
            if is_success:
                with open(save_path, "wb") as f:
                    im_buf.tofile(f)

    def check_point_in_zones(self, point):
        zones = ["entry", "exit"] if self.app_state["is_dual_mode"] else ["single"]
        for z_key in zones:
            poly = self.app_state["zones"][z_key]
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return z_key
        return None

    def draw_zones(self, frame):
        zones = (
            [("entry", (0, 255, 0)), ("exit", (0, 0, 255))]
            if self.app_state["is_dual_mode"]
            else [("single", (255, 165, 0))]
        )
        for z_key, color in zones:
            points = self.app_state["zones"][z_key]
            cv2.polylines(frame, [points], True, color, 2)

            if z_key == self.current_edit_target:
                cv2.polylines(frame, [points], True, (255, 255, 255), 4)
                for i, pt in enumerate(points):
                    c = (0, 0, 255) if i == self.selected_point_idx else color
                    cv2.circle(frame, tuple(pt), 8, c, -1)

    def update_ui_frame(self, frame):
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.root.after(0, lambda: self.lbl_video.configure(image=img_tk))
            self.lbl_video.image = img_tk
        except:
            pass

    def update_log_ui(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("1.0", msg + "\n")
        self.txt_log.config(state="disabled")

    def toggle_mode(self):
        self.app_state["is_dual_mode"] = self.mode_var.get() == "dual"
        self.refresh_mode_buttons()
        self.save_config()

    def refresh_mode_buttons(self):
        self.btn_s.pack_forget()
        self.btn_in.pack_forget()
        self.btn_out.pack_forget()
        if self.app_state["is_dual_mode"]:
            self.btn_in.pack(side="left", fill="x", expand=True, padx=2)
            self.btn_out.pack(side="right", fill="x", expand=True, padx=2)
            if self.current_edit_target == "single":
                self.set_target("entry")
        else:
            self.btn_s.pack(side="top", fill="x", expand=True, padx=2)
            self.set_target("single")

    def set_target(self, t):
        self.current_edit_target = t
        self.selected_point_idx = -1
        self.lbl_status.config(text=f"Editing: {t.upper()}")
        for k, b in [
            ("single", self.btn_s),
            ("entry", self.btn_in),
            ("exit", self.btn_out),
        ]:
            b.config(
                bg="#4CAF50" if k == t else "#ddd", fg="white" if k == t else "black"
            )

    def move_zone(self, dx, dy):
        step = self.speed_var.get()
        pts = self.app_state["zones"][self.current_edit_target]
        shift = [dx * step, dy * step]
        if self.selected_point_idx == -1:
            pts += shift
        else:
            pts[self.selected_point_idx] += shift

    def select_corner(self, idx):
        self.selected_point_idx = idx
        self.lbl_status.config(text=f"Editing: Corner {idx}")

    def reset_selection(self):
        self.selected_point_idx = -1
        self.lbl_status.config(text="Editing: Whole Box")

    def on_preset_change(self, e):
        name = self.combo_preset.get()
        if name in self.presets:
            self.app_state["preset_name"] = name
            self.load_config()
            self.save_config()

    def save_config_action(self):
        new_name = self.new_preset_var.get().strip()
        if new_name:
            self.app_state["preset_name"] = new_name
            self.presets[new_name] = {
                k: v.tolist() for k, v in self.app_state["zones"].items()
            }
            self.combo_preset["values"] = list(self.presets.keys())
            self.combo_preset.set(new_name)
            self.new_preset_var.set("")
        if self.save_config():
            messagebox.showinfo(
                "Saved",
                f"Configuration Saved!\nPreset: {self.app_state['preset_name']}",
            )

    def save_config(self):
        self.app_state["source_type"] = self.src_type_var.get()
        self.app_state["source_path"] = self.src_path_var.get()
        zones_export = {k: v.tolist() for k, v in self.app_state["zones"].items()}
        self.presets[self.app_state["preset_name"]] = zones_export
        data = {
            "app_state": {
                "preset_name": self.app_state["preset_name"],
                "is_dual_mode": self.app_state["is_dual_mode"],
                "model_plate_path": self.app_state["model_plate_path"],
                "model_char_path": self.app_state["model_char_path"],
                "speed": self.app_state["speed"],
                "source_type": self.app_state["source_type"],
                "source_path": self.app_state["source_path"],
            },
            "presets": self.presets,
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except:
            return False

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.app_state.update(data.get("app_state", {}))
                    self.presets = data.get("presets", self.presets)
                    saved = self.presets.get(
                        self.app_state["preset_name"], DEFAULT_ZONES
                    )
                    for k in self.app_state["zones"]:
                        if k in saved:
                            self.app_state["zones"][k] = np.array(saved[k], np.int32)
            except:
                pass

    def connect_sheet(self):
        if not os.path.exists(CREDENTIALS_FILE):
            return None
        try:
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                CREDENTIALS_FILE, scope
            )
            client = gspread.authorize(creds)
            return client.open(SHEET_NAME).sheet1
        except:
            return None

    def log_to_sheet(self, row):
        threading.Thread(
            target=lambda: (
                self.connect_sheet().append_row(row) if self.connect_sheet() else None
            ),
            daemon=True,
        ).start()

    def on_close(self):
        self.is_running = False
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LPRApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
