import torch
print("GPU Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("ยังเป็น CPU อยู่! เช็ค Driver การ์ดจอด้วยนะ")