import torch
import torchvision.models as models

model = models.resnet18()
input_tensor = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input_tensor, onnx_path, verbose=True)
net = cv2.dnn.readNetFromONNX(onnx_path)



model = models.mobilenet_v3_small(pretrained=True)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input_tensor, onnx_path, verbose=True)
net = cv2.dnn.readNetFromONNX(onnx_path)
