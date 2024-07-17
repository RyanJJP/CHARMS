from collections import defaultdict
import os
import json
import glob
from typing import Dict, List

from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


from torchvision import transforms
import tqdm


class PetFinderDiscreteDataset(Dataset):
    """
    """
    def __init__(self, table_path: str, image_path: str):
        """
        :param table_path: dataset/...
        :param image_path: dataset/.../images
        """
        self.table_path = table_path
        self.image_path = image_path

        self.table_df = pd.read_csv(self.table_path)
        self.data_indice = []

        table_y = self.table_df["AdoptionSpeed"]
        table_X = self.table_df.drop(columns=['AdoptionSpeed', 'RescuerID'])

        image_files = list(glob.glob(os.path.join(image_path, "*-*.jpg.pt")))

        image_id_dict: Dict[str, List[str]] = defaultdict(list)
        for image_file in image_files:
            image_id = os.path.basename(image_file).split("-")[0]
            image_id_dict[image_id].append(image_file)

        for index, row in tqdm.tqdm(table_X.iterrows(), total=len(table_X)):
            label = table_y[index]
            pet_id = row["PetID"]

            features = row.drop("PetID").to_numpy(dtype=float)
            pet_image_files = image_id_dict[pet_id]
            for image_file in pet_image_files:
                image_id = image_file.split("-")[-1][:-7]   # 图像后缀名是 .jpg.pt 所以取 -7
                self.data_indice.append((pet_id, image_id, features, label))
    
    def __getitem__(self, index):
        pet_id, image_id, features, label = self.data_indice[index]
        image_features_path = os.path.join(self.image_path, f"{pet_id}-{image_id}.jpg.pt")
        image_features = torch.load(image_features_path)
        table_features = torch.tensor(features, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return (table_features, image_features), label_tensor

    def __len__(self):
        return len(self.data_indice)


class PetFinderImageDataset(Dataset):
    """
    从 raw_dataset 中读取原始的图像数据 (.jpg)
    """
    def __init__(self, table_path: str, image_path: str):
        """
        :param table_path:
        :param image_path:
        """
        self.table_path = table_path
        self.image_path = image_path

        image_size = 224
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

        self.table_df = pd.read_csv(self.table_path)
        self.data_indice = []

        table_y = self.table_df["AdoptionSpeed"]
        table_X = self.table_df.drop(['AdoptionSpeed'], axis=1)

        image_files = list(glob.glob(os.path.join(image_path, "*-*.jpg")))

        image_id_dict: Dict[str, List[str]] = defaultdict(list)
        for image_file in image_files:
            image_id = os.path.basename(image_file).split("-")[0]
            image_id_dict[image_id].append(image_file)

        for index, row in tqdm.tqdm(table_X.iterrows(), total=len(table_X)):
            label = table_y[index]
            pet_id = row["PetID"]

            features = row.drop("PetID").to_numpy(dtype=int)
            pet_image_files = image_id_dict[pet_id]
            for image_file in pet_image_files:
                image_id = image_file.split("-")[-1][:-4]
                self.data_indice.append((pet_id, image_id, features, label))
    
    def __getitem__(self, index):
        pet_id, image_id, features, label = self.data_indice[index]
        image_path = os.path.join(self.image_path, f"{pet_id}-{image_id}.jpg")
        image = self.transform(Image.open(image_path).convert('RGB'))

        table_features = torch.tensor(features, dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return (table_features, image), label_tensor

    def __len__(self):
        return len(self.data_indice)


class PetFinderConCatImageDataset(Dataset):
    def __init__(self, table_path: str, image_path: str):
        self.table_path = table_path
        self.image_path = image_path
        
        image_size = 224
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.table_df = pd.read_csv(self.table_path)
        self.con_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'desc_length', 'average_word_length',
                         'magnitude', 'desc_words', 'score']
        self.cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                    'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                    'Sterilized', 'Health', 'State']
        for col in self.cat_cols:
            if self.table_df[col].max() > 100:
                self.table_df[col] = pd.cut(self.table_df[col], bins=20, labels=range(20))
        for col in self.con_cols:
            # if self.table_df[col].max() > 100:
            #     self.table_df[col] = pd.cut(self.table_df[col], bins=100, labels=range(100))
            # z-score normalization
            col_data = torch.tensor(self.table_df[col].values, dtype=torch.float)
            mean = torch.mean(col_data)
            std = torch.std(col_data)
            self.table_df[col] = torch.div(torch.sub(col_data, mean), std)
        self.cat_cardinalities = [(self.table_df[col].max() + 1) for col in self.cat_cols]
        print("self.cat_cardinalities:", self.cat_cardinalities)

        self.data_indice = []

        table_y = self.table_df["AdoptionSpeed"]
        table_X = self.table_df.drop(['AdoptionSpeed'], axis=1)

        image_files = list(glob.glob(os.path.join(image_path, "*-*.jpg")))

        image_id_dict: Dict[str, List[str]] = defaultdict(list)
        for image_file in image_files:
            image_id = os.path.basename(image_file).split("-")[0]
            image_id_dict[image_id].append(image_file)

        for index, row in tqdm.tqdm(table_X.iterrows(), total=len(table_X)):
            label = table_y[index]
            pet_id = row["PetID"]

            features = row.drop("PetID")
            features_con = features[self.con_cols].to_numpy(dtype=float)
            features_cat = features[self.cat_cols].to_numpy(dtype=int)

            pet_image_files = image_id_dict[pet_id]
            for image_file in pet_image_files:
                image_id = image_file.split("-")[-1][:-4]
                self.data_indice.append((pet_id, image_id, features_con, features_cat, label))

    def __getitem__(self, index):
        pet_id, image_id, features_con, features_cat, label = self.data_indice[index]
        image_path = os.path.join(self.image_path, f"{pet_id}-{image_id}.jpg")
        image = self.transform(Image.open(image_path).convert('RGB'))

        table_features_con = torch.tensor(features_con, dtype=torch.float)
        table_features_cat = torch.tensor(features_cat, dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return (table_features_con, table_features_cat, image), label_tensor

    def __len__(self):
        return len(self.data_indice)



class PetFinderCLIPDataset(Dataset):
    def __init__(self, table_path: str, image_path: str, preprocess:None):
        self.table_path = table_path
        self.image_path = image_path
        self.preprocess = preprocess

        image_size = 224
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.table_df = pd.read_csv(self.table_path)
        self.con_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'desc_length', 'average_word_length',
                         'magnitude', 'desc_words', 'score']
        self.cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                    'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                    'Sterilized', 'Health', 'State']
        for col in self.cat_cols:
            if self.table_df[col].max() > 100:
                self.table_df[col] = pd.cut(self.table_df[col], bins=20, labels=range(20))
        for col in self.con_cols:
            # if self.table_df[col].max() > 100:
            #     self.table_df[col] = pd.cut(self.table_df[col], bins=100, labels=range(100))
            # z-score normalization
            col_data = torch.tensor(self.table_df[col].values, dtype=torch.float)
            mean = torch.mean(col_data)
            std = torch.std(col_data)
            self.table_df[col] = torch.div(torch.sub(col_data, mean), std)

        self.data_indice = []

        table_y = self.table_df["AdoptionSpeed"]
        table_X = self.table_df.drop(['AdoptionSpeed'], axis=1)

        image_files = list(glob.glob(os.path.join(image_path, "*-*.jpg")))

        image_id_dict: Dict[str, List[str]] = defaultdict(list)
        for image_file in image_files:
            image_id = os.path.basename(image_file).split("-")[0]
            image_id_dict[image_id].append(image_file)

        for index, row in tqdm.tqdm(table_X.iterrows(), total=len(table_X)):
            label = table_y[index]
            pet_id = row["PetID"]

            features = row.drop("PetID")
            features_con = features[self.con_cols].to_numpy(dtype=float)
            features_cat = features[self.cat_cols].to_numpy(dtype=int)

            text = ''
            for idx, tmp in enumerate(features_con):
                text += self.con_cols[idx] + ":" + str("%.2f"%tmp)
            for idx, tmp in enumerate(features_cat):
                text += self.cat_cols[idx] + ":" + str(int(tmp))

            pet_image_files = image_id_dict[pet_id]
            for image_file in pet_image_files:
                image_id = image_file.split("-")[-1][:-4]
                self.data_indice.append((pet_id, image_id, text, label))

    def __getitem__(self, index):
        pet_id, image_id, text, label = self.data_indice[index]
        image_path = os.path.join(self.image_path, f"{pet_id}-{image_id}.jpg")
        # image = self.transform(Image.open(image_path).convert('RGB'))
        image = self.preprocess(Image.open(image_path).convert('RGB'))
        text = clip.tokenize(text, truncate=True).squeeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return (text, image), label_tensor

    def __len__(self):
        return len(self.data_indice)