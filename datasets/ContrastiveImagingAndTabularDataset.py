from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image

import glob
import os
from collections import defaultdict
from PIL import Image

class ContrastiveImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int, live_loading: bool) -> None:
            
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.generate_marginal_distributions(data_path_tabular)
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular
    
    # Classifier
    self.labels = torch.load(labels_path)
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data_df = pd.read_csv(data_path)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data_imaging[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255
    ims = [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(im))
    else:
      ims.append(self.default_transform(im))

    orig_im = self.default_transform(im)
    
    return ims, orig_im

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)



class ContrastiveImagingAndTabularDataset_PetFinder(ContrastiveImagingAndTabularDataset):

  def __init__(
          self,
          data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float,
          data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
          labels_path: str, img_size: int, live_loading: bool) -> None:

    # self.data_imaging = torch.load(data_path_imaging) # torch.load() 只能加载 .pt 文件
    self.data_path_imaging = data_path_imaging
    self.data_imaging = self.read_image_files(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      pass
      # for im in self.data_imaging:
      #   im[0, :, :] = 0

    self.default_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    # transforms.Compose([
    #   transforms.Resize(size=(img_size, img_size)),
    #   transforms.ToTensor(),
    #   # transforms.Lambda(lambda x: x.float())
    # ])

    # Tabular
    self.c = corruption_rate
    self.field_lengths_tabular = torch.tensor([2,1,20,20,3,7,7,6,4,3,3,3,3,3,1,1,14,1,1,1,1,1,1,1], dtype=torch.int).cpu()
    self.one_hot_tabular = one_hot_tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)

    # Classifier
    # print("self.data_tabular[(self.data_tabular.PhotoAmt == 0.0)]:", self.data_tabular[(self.data_tabular.PhotoAmt == 0.0)].PetID)
    # Delete tabular entries that have no corresponding photo. ()

    # WARNING: ↓ must be called after the tabular data setup has finished.
    self.labels = self.data_tabular['AdoptionSpeed']
    self.id = self.data_tabular['PetID']  # Series
    self.data_tabular.drop(columns=['AdoptionSpeed', 'PetID'], inplace=True)
    self.generate_marginal_distributions(data_path_tabular)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Randomly select one of the pictures corresponding to the specific PetID.
    WARNING: 存在没有对应的图像数据的表格数据
    """
    pet_id = self.id.iloc[index]
    if not len(self.data_imaging[pet_id]):
      print("pet_id: ", pet_id)
      raise Exception("This tabular entry has no corresponding images.")
    rand_img_idx = random.randint(1, len(self.data_imaging[pet_id])) - 1
    # print("self.data_imaging[id][rand_img_idx]}:", self.data_imaging[id][rand_img_idx])
    # im = os.path.join(self.data_path_imaging, f"{id}-{self.data_imaging[id][rand_img_idx]}.jpg")
    im = self.data_imaging[pet_id][rand_img_idx]
    # im = Image.open(im).convert('RGB')
    if self.live_loading:
      # im = read_image(im)
      im = Image.open(im).convert('RGB')
      # print(im)
      # im = im / 255
    ims = [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(im))
    else:
      ims.append(self.default_transform(im))

    orig_im = self.default_transform(im)
    # print("type(orig_im):", type(orig_im))
    # print("type(ims):", type(ims))

    return ims, orig_im

  def read_and_parse_csv(self, path_tabular: str) -> pd.DataFrame:
    """
    """
    data = pd.read_csv(path_tabular)
    data = data.drop(index=data[(data.PhotoAmt == 0.0)].index.tolist())
    con_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'desc_length', 'average_word_length',
                         'magnitude', 'desc_words', 'score']
    cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                'Sterilized', 'Health', 'State']
    for col in cat_cols:
        if data[col].max() > 100:
            data[col] = pd.cut(data[col], bins=20, labels=range(20))
    for col in con_cols:
        # if data[col].max() > 100:
        #     data[col] = pd.cut(data[col], bins=100, labels=range(100))
        # z-score normalization
        col_data = torch.tensor(data[col].values, dtype=torch.float)
        mean = torch.mean(col_data)
        std = torch.std(col_data)
        data[col] = torch.div(torch.sub(col_data, mean), std)
    return data

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    self.marginal_distributions = self.data_tabular.transpose().values.tolist()


  def read_image_files(self, img_path) -> dict:
    """
    :return: {PetID: corresponding_image_name}
    """
    image_files = list(glob.glob(os.path.join(img_path, "*-*.jpg")))
    image_id_dict: Dict[str, List[str]] = defaultdict(list)
    for image_file in image_files:
      image_id = os.path.basename(image_file).split("-")[0]
      image_id_dict[image_id].append(image_file)
    return image_id_dict

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    imaging_views, unaugmented_image = self.generate_imaging_views(index=index)
    # print("list(self.data_tabular.iloc[index]):\n", list(self.data_tabular.iloc[index]))
    tabular_views = [torch.tensor(self.data_tabular.iloc[index], dtype=torch.float),
                     torch.tensor(self.corrupt(list(self.data_tabular.iloc[index])), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)