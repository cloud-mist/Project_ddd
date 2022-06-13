import os
import json
import sys

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


os.chdir(os.path.dirname(os.path.abspath(__file__)))
d_explain = {
    "AKIEC": "光化性角化病(AKIEC)",
    "BCC": "基底细胞癌(BCC)",
    "BKL": "良性角化病(BKL)",
    "DF": "皮肤纤维瘤(DF)",
    "MEL": "黑色素瘤(MEL)",
    "NV": "黑素细胞痣(NV)",
    "VASC": "血管病变(VASC)",
}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {
        "s": [300, 384],  # train_size, val_size
        "m": [384, 480],
        "l": [384, 480],
    }
    num_model = "s"

    data_transform = transforms.Compose(
        [
            transforms.Resize(img_size[num_model][1]),
            transforms.CenterCrop(img_size[num_model][1]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # load image
    img_path = sys.argv[1]
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=7).to(device)
    # load model weights
    model_weight_path = "./weights/model-69.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # print_res = "class: {}   prob: {:.3}".format(
    #     class_indict[str(predict_cla)], predict[predict_cla].numpy()
    # )
    print_res = class_indict[str(predict_cla)]
    print(d_explain[print_res.strip()])


if __name__ == "__main__":
    main()
