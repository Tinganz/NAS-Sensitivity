import torch

path = "/nas/longleaf/home/tingan/NAS-Sensitivity/safety-nas/dnn-output/trial_artifacts/20260609T011726_3414423_6f5bfc_trial00114/left_wall_dist/models/left_wall_dist_arch8.pt"

model = torch.jit.load(path, map_location="cpu")
params = sum(p.numel() for p in model.parameters())

print("Parameters:", params)