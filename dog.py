from torchvision import models
from PIL import Image
import torch
import sys

resnet = models.resnet18(pretrained=True)  # 创建一个resnet18的实例，并且下载训练好的权重

# 图像预处理
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # 图像缩放：256x256
        transforms.CenterCrop(224),  # 围绕中心图像裁切：224x224
        transforms.ToTensor(),  # 转换成张量
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    ]
)


img = Image.open(sys.argv[1])  # 载入图像
img_t = preprocess(img)  # 传递
batch_t = torch.unsqueeze(img_t, 0)


# 推理
resnet.eval()  # 进入eval模式
out = resnet(batch_t)


# 弄一个文件存储
with open("./ML/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

_, idx = torch.max(out, 1)  # max生成张量的最大值和其索引

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
_, indices = torch.sort(out, descending=True)
# print([(labels[idx], percentage[idx].item()) for idx in indices[0][:1]])
output = [(labels[idx], percentage[idx].item()) for idx in indices[0][:1]]
print(output[0][0])
