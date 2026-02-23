import torch

checkpoint = torch.load("weights/best_model.pth", map_location="cpu")
print("Best model saved at epoch:", checkpoint["epoch"])


