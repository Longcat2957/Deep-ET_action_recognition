import torch
from vivit import ViViT

if __name__ == '__main__':
    vid = torch.randn((1, 16, 3, 224, 224))
    model = ViViT(224, 16, 100, 16)

    # export as onnx
    torch.onnx.export(
        model,
        vid,
        "vivit.onnx",
        verbose=True
    )




