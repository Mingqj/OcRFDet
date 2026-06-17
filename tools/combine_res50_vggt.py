import torch

ckpt1 = torch.load(
    "path/to/resnet50.pth",
    map_location="cpu"
)

ckpt2 = torch.load(
    "path/to/vggt.pt",
    map_location="cpu"
)

state_dict = ckpt1["state_dict"]

for k, v in ckpt2.items():
    state_dict[k] = v

ckpt1["state_dict"] = state_dict

torch.save(
    ckpt1,
    "path/to/r50_vggt.pth"
)

print("Done")

