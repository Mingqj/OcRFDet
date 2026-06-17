import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/pretrain_model/model.pt").to(device)

model = VGGT()  # 先实例化模型
state_dict = torch.load("/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/pretrain_model/model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/examples/nuscenes/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg", "/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/examples/nuscenes/n008-2018-05-21-11-06-59-0400__CAM_BACK_LEFT__1526915243047295.jpg", "/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/examples/nuscenes/n008-2018-05-21-11-06-59-0400__CAM_BACK_RIGHT__1526915243027813.jpg", "/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/examples/nuscenes/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg", "/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/examples/nuscenes/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915243004917.jpg", "/opt/data/private/codeN2/DualBEV/mmdet3d/models/backbones/vggt/examples/nuscenes/n008-2018-05-21-11-06-59-0400__CAM_FRONT_RIGHT__1526915243019956.jpg"]  
images = load_and_preprocess_images(image_names).to(device)
print('Images shape:', images.shape)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        start_time = time.time()
        predictions = model(images)
        end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")