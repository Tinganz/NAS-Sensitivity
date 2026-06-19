import torch

path = "/home/tingan/NAS-Sensitivity/safety-nas/test-best-runs-tp0/2720f4/heading_error_arch8_trial195.pt"
path2 = "/home/tingan/NAS-Sensitivity/safety-nas/test-best-runs-tp0/2720f4/left_wall_dist_arch8_trial195.pt"
path3 = "/home/tingan/NAS-Sensitivity/safety-nas/test-best-runs-tp0/2720f4/track_width_arch8_trial195.pt"

model = torch.jit.load(path, map_location="cpu")
model2 = torch.jit.load(path2, map_location="cpu")
model3 = torch.jit.load(path3, map_location="cpu")
params = sum(p.numel() for p in model.parameters())
params2 = sum(p.numel() for p in model2.parameters())
params3 = sum(p.numel() for p in model3.parameters())

print("Parameters:", params)
print("Parameters:", params2)
print("Parameters:", params3)