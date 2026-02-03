from roboflow import Roboflow
from ultralytics import YOLO
import os

if __name__ == '__main__':
    
    # 1. โหลดข้อมูล (Roboflow จะเช็คเองว่ามีไฟล์หรือยัง ถ้ามีแล้วมันจะไม่โหลดซ้ำ)
    rf = Roboflow(api_key="6QOHxd1aWKehDcdO7lOC")
    project = rf.workspace("thanathorn-msskh").project("thai-license-plate-character-recognition-su5fk")
    version = project.version(2)
    dataset = version.download("yolov8")
                

    # rf = Roboflow(api_key="6QOHxd1aWKehDcdO7lOC")
    # project = rf.workspace("lru").project("lru-license-plate")
    # version = project.version(1)
    # dataset = version.download("yolov8")
                
                
    # 2. ระบุ Path ของไฟล์ last.pt จากการเทรนรอบที่แล้ว
    # เช็คให้ชัวร์ว่า Path นี้ถูกต้อง (เข้าไปดูในโฟลเดอร์ Train-License-Plate/run.../weights/)
    weights_path = r'Train-License-Plate/run/weights/last.pt' 

    if os.path.exists(weights_path):
        print(f"✅ พบไฟล์เก่าที่: {weights_path}")
        print("🔄 กำลัง Resume เทรนต่อจากจุดเดิม...")
        
        model = YOLO(weights_path)
        
        # YOLOv8 ใช้คำสั่งนี้เพื่อ Resume ได้เลย
        # มันจะโหลด Epoch ล่าสุด และ Optimizer state มาให้เอง
        model.train(resume=True) 
        
    else:
        print(f"⚠️ ไม่พบไฟล์เก่า เริ่มเทรนใหม่ทั้งหมด...")
        model = YOLO('yolov8n.pt')
        
        model.train(
            data=f'{dataset.location}/data.yaml',
            epochs=3000,
            patience=50,
            imgsz=640,
            batch=16,
            workers=0,
            device=0,
            project='Train-License-Plate',
            name='run',      # ชื่อนี้ต้องตรงกับ Path ด้านบน
            exist_ok=True,   # เขียนทับโฟลเดอร์เดิม (อันนี้โอเคแล้ว)
            resume=False
        )