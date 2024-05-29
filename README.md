<<<<<<< HEAD
# Multimodal Contrastive Learning with Tabular and Imaging Data

Please cite our CVPR paper, [Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data](https://arxiv.org/abs/2303.14080), if this code was helpful.

```
@InProceedings{Hager_2023_CVPR,
    author    = {Hager, Paul and Menten, Martin J. and Rueckert, Daniel},
    title     = {Best of Both Worlds: Multimodal Contrastive Learning With Tabular and Imaging Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23924-23935}
}
```

If you want an overview of the paper checkout:
- [this podcast episode](https://www.linkedin.com/posts/harpreetsahota204_deeplearning-cvpr2023-computervision-activity-7078096344346738688-m7-C?utm_source=share&utm_medium=member_desktop)
- [the video abstract](https://www.youtube.com/watch?v=iHVPSMEM6WM)


## Instructions

Install environment using `conda env create --file environment.yaml`. 

To run, execute `python run.py`.

### Arguments - Command Line

If pretraining, pass `pretrain=True` and `datatype={imaging|multimodal|tabular}` for the desired pretraining type. `multimodal` uses our strategy from the paper, `tabular` uses SCARF, and `imaging` can be specified with the `loss` argument. Default is SimCLR, other options are byol, simsiam, and barlowtwins.

If you do not pass `pretrain=True`, the model will train fully supervised with the data modality specified in `datatype`, either `tabular` or `imaging`.

You can evaluate a model by passing the path to the final pretraining checkpoint with the argument `checkpoint={PATH_TO_CKPT}`. After pretraining, a model will be evaluated with the default settings (frozen eval, lr=1e-3).

### Arguments - Hydra

All argument defaults can be set in hydra yaml files found in the configs folder.

Most arguments are set to those in the paper and work well out of the box. Default model is ResNet50.

Code is integrated with weights and biases, so set `wandb_project` and `wandb_entity` in [config.yaml](configs/config.yaml).

Path to folder containing data is set through the `data_base` argument and then joined with filenames set in the dataset yamls. Best strategy is to take [dvm_all_server.yaml](configs/dataset/dvm_all_server.yaml) as a template and fill in the appropriate filenames. 
- For the images, provide a .pt with a list of your images or a list of the paths to your images.
  - If providing a list of paths, set `live_loading=True`.
- `delete_segmentation` deletes the first channel of a three channel image (historical reasons) and should typically be left to false.
- If `weights` is set, during finetuning a weighted sampled will be used instead of assuming the evaluation train data has been properly balanced
- `eval_metric` supports `acc` for accuracy (top-1) and `auc` (for unbalanced data)
- If doing multimodal pretraining or tabular pretraining (SCARF), the tabular data should be provided as *NOT* one-hot encoded so the sampling from the empirical marginal distribution works correctly. You must provide a file `field_lengths_tabular` which is an array that in the order of your tabular columns specifies how many options there are for that field. Continuous fields should thus be set to 1 (i.e. no one-hot encoding necessary), while categorical fields should specify how many columns should be created for the one_hot encoding  

### Data

The UKBB data is semi-private. You can apply for access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access).

The DVM cars dataset is open-access and can be found [here](https://deepvisualmarketing.github.io/).

Processing steps for the DVM dataset can be found [here](https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/data/create_dvm_dataset.ipynb).

The exact data splits used in the paper are saved in the [data folder](https://github.com/paulhager/MMCL-Tabular-Imaging/tree/main/data).
=======
# CHARMS
>>>>>>> df986bc9d2c482b632d67809d51d738b578a63de
