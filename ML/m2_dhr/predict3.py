import os
import sys
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import geffnet
import sys
import cv2

# ----------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

dic_positon = {
    "前躯干": "site_anterior_torso",
    "头/脖子": "site_head/neck",
    "侧躯干": "site_lateral_torso",
    "下肢": "site_lower_extremity",
    "手掌/脚底": "site_palms/soles",
    "后躯干": "site_posterior_torso",
    "上肢": "site_upper_extremity",
    "未知": "site_nan",
}

# ----------------------------------


image_size = 512
use_amp = False
batch_size = 2
num_workers = 0
init_lr = 3e-5
out_dim = 2
device = torch.device("cpu")
freeze_epo = 0
warmup_epo = 1
cosine_epo = 4
enet_type = "efficientnet-b0"
n_epochs = freeze_epo + warmup_epo + cosine_epo
kernel_type = "effnetb0_512_meta_9c_ext_5epo"
use_external = "_ext" in kernel_type
use_meta = "meta" in kernel_type


ls = sys.argv
temp_image_name = ls[1]
temp_sex = ls[2]
temp_age = ls[3]
temp_site = dic_positon[ls[4]]


c1 = temp_site.split("_")
temp_site = ""
i = 1
for strc in c1:
    if i != 1:
        temp_site = temp_site + " " + strc
    else:
        temp_site = temp_site + strc
        i = 2
list1 = [
    "image_name",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
    "target",
    "tfrecord",
    "width",
    "height",
]
list2 = [
    temp_image_name,
    -1,
    temp_sex,
    temp_age,
    temp_site,
    "NV",
    "benign",
    0,
    1,
    1022,
    767,
]
temp_pd = pd.DataFrame(columns=list1, data=[list2])
temp_pd["age_approx"] = temp_pd["age_approx"].astype(float)
if use_external:
    df_train2 = temp_pd
    df_train2 = df_train2[df_train2["tfrecord"] >= 0].reset_index(drop=True)
    df_train2["is_ext"] = 1
    df_train2["filepath"] = df_train2["image_name"].apply(lambda x: f"{x}")

    df_train2["diagnosis"] = df_train2["diagnosis"].apply(
        lambda x: x.replace("NV", "nevus")
    )
    df_train2["diagnosis"] = df_train2["diagnosis"].apply(
        lambda x: x.replace("MEL", "melanoma")
    )
    df_train = df_train2.reset_index(drop=True)
    df_train3 = pd.read_csv("train.csv")
    df_train3 = df_train3.reset_index(drop=True)
if use_meta:
    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat(
        [
            df_train["anatom_site_general_challenge"],
            df_train3["anatom_site_general_challenge"],
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix="site")
    df_train = pd.concat([df_train, dummies.iloc[: df_train.shape[0]]], axis=1)
    # Sex features
    df_train["sex"] = df_train["sex"].map({"男": 1, "女": 0})
    df_train["sex"] = df_train["sex"].fillna(-1)
    # Age features
    df_train["age_approx"] /= 90
    df_train["age_approx"] = df_train["age_approx"].fillna(0)
    df_train["patient_id"] = df_train["patient_id"].fillna(0)
    # n_image per user
    df_train["n_images"] = df_train.patient_id.map(
        df_train.groupby(["patient_id"]).image_name.count()
    )
    df_train.loc[df_train["patient_id"] == -1, "n_images"] = 1
    df_train["n_images"] = np.log1p(df_train["n_images"].values)
    # image size
    train_images = df_train["filepath"].values
    diagnosis2idx = {"melanoma": 0, "nevus": 1}
    df_train["target"] = df_train["diagnosis"].map(diagnosis2idx)
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(train_images):
        train_sizes[i] = os.path.getsize(img_path)
    df_train["image_size"] = np.log(train_sizes)
    meta_features = [
        "sex",
        "age_approx",
        "n_images",
        "image_size",
        "site_anterior torso",
        "site_head/neck",
        "site_lateral torso",
        "site_lower extremity",
        "site_palms/soles",
        "site_posterior torso",
        "site_upper extremity",
        "site_nan",
    ]

    n_meta_features = len(meta_features)
else:
    n_meta_features = 0


class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if use_meta:
            data = (
                torch.tensor(image).float(),
                torch.tensor(self.csv.iloc[index][meta_features]).float(),
            )
        else:
            data = torch.tensor(image).float()

        if self.mode == "test":
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


transforms_val = A.Compose([A.Resize(image_size, image_size), A.Normalize()])

from pylab import rcParams

sigmoid = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


swish = Swish.apply


class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)


swish_layer = Swish_module()


class enetv2(nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=True):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type.replace("-", "_"), pretrained=True)
        self.dropout = nn.Dropout(0.5)
        in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, 512),
                nn.BatchNorm1d(512),
                Swish_module(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                Swish_module(),
            )
            in_ch += 128
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            x = self.myfc(self.dropout(x))
        return x


criterion = nn.CrossEntropyLoss()


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, is_ext=None, n_test=1, get_output=False):
    model.eval()
    PROBS = []
    with torch.no_grad():
        for (data, target) in loader:

            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test
            PROBS.append(probs.detach().cpu())

    PROBS = torch.cat(PROBS).numpy()
    if get_output:
        return PROBS


def run(fold):
    i_fold = 1

    df_valid = df_train

    dataset_valid = SIIMISICDataset(df_valid, "train", "val", transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=num_workers
    )

    model = enetv2(enet_type, n_meta_features=n_meta_features, out_dim=2).to("cpu")
    model.load_state_dict(
        torch.load(
            "effnetb0_512_meta_9c_ext_5epo_model_fold1.pth",
            map_location=torch.device("cpu"),
        ),
        strict=True,
    )

    auc_max = 0.0

    for epoch in range(1, 2):
        prec = val_epoch(
            model, valid_loader, is_ext=df_valid["is_ext"].values, get_output=True
        )
        temp_c = prec.max(1)
        if temp_c == 0:
            print("NV")
        else:
            print("MEL")


scores = []
scores_20 = []
for fold in range(1):
    run(fold)
