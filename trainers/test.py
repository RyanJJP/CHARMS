from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.CHARMS_dataset import PetFinderConCatImageDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint


def test(hparams, wandb_logger=None, model=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  
  if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
    test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading)
    
    print(test_dataset.transform_val.__repr__())    
  elif hparams.datatype == 'tabular':
    test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.datatype == 'charms':
    if hparams.target == 'adoption':
      test_dataset = PetFinderConCatImageDataset(hparams.data_test_eval_tabular, hparams.data_test_eval_imaging) 
      hparams.input_size = test_dataset.__len__()
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  

  if hparams.datatype != 'charms':
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
      pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

    hparams.dataset_length = len(test_loader)
    model = Evaluator(hparams)

  else:
    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=hparams.batch_size, shuffle=False)
    model = model

  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
  print("test starts")
  print(test_dataset.__len__())
  trainer.test(model=model, dataloaders=test_loader, verbose=True)
  print("test ends")
