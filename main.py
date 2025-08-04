from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import torchvision.transforms as T
import torch
import cv2
from skimage.transform import resize

def readData(path, class_part_index = 1, augment=False):
    labels: list = os.listdir(path)
    labels.remove('.DS_Store')

    img_dirs = []
    for label in labels:
        img_dirs.append(Path(path + "/" + label))

    paths = []
    for path in img_dirs:
        paths += random.sample(list(path.glob('*.jpg')), 30)
    random.shuffle(paths) # Randomly shuffle the list in place

    split_ratio = 0.8 # 80% for the first part
    split_index = int(len(paths) * split_ratio)

    train_paths = paths[:split_index]
    test_paths = paths[split_index:]

    def apply_transform(image) -> torch.Tensor:
        transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomApply([T.RandomResizedCrop(size=(200, 200), scale=(0.6, 0.9))], p=0.5),
            T.RandomRotation(degrees=(-25, 25)),
            # T.RandomApply([T.CenterCrop(size=random.randint(160, 200)), T.Resize(size=(256, 256))], p=0.5),
            # T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.65)
        ])
        return transform(image)

    X_train, y_train = [], []
    X_test, y_test = [], []
    for path in train_paths:
        if augment == True:
            for path in paths:
                img_pre_torch = cv2.resize(cv2.imread(path), (256, 256)) # image before being processed
                
                gray = cv2.cvtColor(img_pre_torch, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0)
                edges = cv2.Canny(blur, threshold1=40, threshold2=70)
                edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
                img_post_torch = np.transpose(edges_tensor.numpy(), (1, 2, 0))  # (256, 256, 1)
                X_train.append((img_post_torch).flatten())
                y_train.append(path.parts[class_part_index])
                for j in range(10):
                    img_torch: torch.Tensor = apply_transform(img_pre_torch)
                    img_post_torch = np.transpose(img_torch.numpy(), (1, 2, 0))

                    img_cv2 = (img_post_torch * 255).astype(np.uint8)  # convert to [0, 255]
                    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR) # convert RGB to BGR

                    gray = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0)
                    edges = cv2.Canny(blur, threshold1=40, threshold2=70)

                    edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
                    img_post_torch = np.transpose(edges_tensor.numpy(), (1, 2, 0))  # (256, 256, 1)

                    X_train.append(resize(img_post_torch, (256, 256)).flatten()) 
                    y_train.append(path.parts[class_part_index])
        else:
            for path in paths:
                img_pre_torch = cv2.resize(cv2.imread(path), (256, 256))

                gray = cv2.cvtColor(img_pre_torch, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0)
                edges = cv2.Canny(blur, threshold1=40, threshold2=70)
                edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
                img_post_torch = np.transpose(edges_tensor.numpy(), (1, 2, 0))  # (256, 256, 1)

                X_train.append(resize(img_post_torch, (256, 256)).flatten()) 
                y_train.append(path.parts[class_part_index])
    for path in test_paths:
        img_pre_torch = cv2.resize(cv2.imread(path), (256, 256))

        gray = cv2.cvtColor(img_pre_torch, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=0)
        edges = cv2.Canny(blur, threshold1=40, threshold2=70)
        edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
        img_post_torch = np.transpose(edges_tensor.numpy(), (1, 2, 0))  # (256, 256, 1)

        plt.imshow(img_post_torch)
        plt.show()
        X_test.append(resize(img_post_torch, (256, 256)).flatten()) 
        y_test.append(path.parts[class_part_index])
    
    imp = SimpleImputer(strategy='most_frequent')
    X_train = pd.DataFrame(X_train)
    X_train = pd.DataFrame(imp.fit_transform(X_train))
    y_train = pd.Series(y_train)

    X_test = pd.DataFrame(X_test)
    X_test = pd.DataFrame(imp.fit_transform(X_test))
    y_test = pd.Series(y_test)
    X, y = (X_train, X_test), (y_train, y_test)
    print(X_test)
    print(y_test)
    return X, y

readData("images", augment=True)
print('nice')