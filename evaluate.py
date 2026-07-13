import sys

import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
# importing train builds its `datasets` list, so eval() runs on the same data
from train import eval, get_default_device

if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    device = get_default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NanoTabPFNModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(eval(NanoTabPFNClassifier(model, device)))
