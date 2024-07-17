import os
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanSquaredError

import ot
import rtdl
import random

from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..") 
from datasets.CHARMS_dataset import PetFinderConCatImageDataset


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageClassifier(nn.Module):
    def __init__(self, img_reduction_dim: int, model_name: str = 'resnet',
                 out_dims: int = 5,
                 n_num_features: int = 0,
                 cat_cardinalities: list = [],
                 d_token: int = 8, ):
        super().__init__()
        # random.seed(42)
        self.model_name = model_name
        if model_name == "resnet":
            backbone = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.fc = Identity()
        elif model_name == "densenet":
            backbone = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)
            in_dims = backbone.classifier.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.classifier = Identity()
        elif model_name == "inception":
            backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.fc = Identity()
        elif model_name == "mobilenet":
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_dims = backbone.classifier[-1].in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.classifier[-1] = Identity()

        self.in_dims = in_dims
        self.img_reduction_dim = img_reduction_dim
        self.table_dim = n_num_features + len(cat_cardinalities)

        # con_fc
        linears = []
        for i in range(n_num_features):
            linears.append(nn.Linear(in_dims, 1))
        self.con_fc = nn.ModuleList(linears)
        self.con_fc_num = self.con_fc.__len__()

        # cat_fc
        linears = []
        for i in range(len(cat_cardinalities)):
            linears.append(nn.Linear(in_dims, cat_cardinalities[i]))
        self.cat_fc = nn.ModuleList(linears)
        self.cat_fc_num = self.cat_fc.__len__()

        self.tab_model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            attention_dropout=0.1,
            n_blocks=2,
            ffn_d_hidden=6,
            ffn_dropout=0.2,
            residual_dropout=0.0,
            # last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=out_dims,
        )

        print("self.tab_model: \n", self.tab_model)

        self.backbone = backbone
        self.img_fc = img_fc

        # self.mask = self.get_mask('res_tmp/cluster_res_CelebA_Updating.txt', 'res_tmp/OToutput40_CelebA_Updating.txt')
        self.mask = torch.ones((self.table_dim, in_dims), dtype=torch.long)

    def forward(self, img, tab_con, tab_cat):
        mask = self.mask.to(img.device)
        extracted_feats = self.backbone(img)
        # Try to freeze the backbone ?
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        img_out = self.img_fc(extracted_feats)

        con_out = []
        for i in range(self.con_fc_num):
            masked_feat = mask[i] * extracted_feats
            con_out.append(self.con_fc[i](masked_feat).squeeze(-1))

        cat_out = []
        for i in range(self.cat_fc_num):
            masked_feat = mask[self.con_fc_num + i] * extracted_feats
            cat_out.append(self.cat_fc[i](masked_feat))

        if self.con_fc_num == 0:
            tab_con = None
        if self.cat_fc_num == 0:
            tab_cat = None
        table_features_embed, table_embed_out = self.tab_model(tab_con, tab_cat)
        # print("table_features_embed.shape, table_embed_out.shape:", table_features_embed.shape, table_embed_out.shape)

        return img_out, con_out, cat_out, table_embed_out

    def compute_OT(self, dataset, device):
        test_table_feat, test_channel_feat = self.getTableChannelFeat(dataset, device)
        CostMatrix = self.getCostMatrix(test_table_feat, test_channel_feat)
        P, W = self.compute_coupling(test_table_feat, test_channel_feat, CostMatrix)

        OTOutFileName = 'res_tmp/OToutput_' + str(self.img_reduction_dim) + '_Adoption_Updating.txt'
        np.savetxt(OTOutFileName, P)

        return P

    def get_mask(self, cluster_path: str = None, OT_path: str = None):
        cluster_dict = {}
        with open(cluster_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = list(line[1:-2].split(","))
                cluster_dict[idx] = np.array(line, dtype=int)

        OT = np.loadtxt(OT_path)

        img_dim = self.in_dims
        mask = np.zeros((self.table_dim, img_dim))
        for i in range(self.table_dim):
            for idx_OT, j in enumerate(OT[i]):
                channel_id = cluster_dict[idx_OT]
                if j != 0:
                    mask[i, channel_id] = 1
                else:
                    mask[i, channel_id] = 0
        return torch.tensor(mask, dtype=torch.long)  # np.array (39, 2048)

    def getTableChannelFeat(self, dataset, device):
        resnet = self.backbone
        tab_model = self.tab_model

        test_channel_feat = []
        test_table_feat = []
        index = 0
        for index, row in enumerate(dataset):
            feats, _ = row
            table_features_con = feats[0]
            table_features_cat = feats[1]
            image = feats[2]

            table_features_con = table_features_con.unsqueeze(0).to(device)
            table_features_cat = table_features_cat.unsqueeze(0).to(device)
            image = image.unsqueeze(0).to(device)

            channel_feat = self.getChannelFeature(resnet, image)
            table_feat = self.getTableFeature(tab_model, table_features_con, table_features_cat)

            test_channel_feat.append(channel_feat.unsqueeze(1))
            test_table_feat.append(table_feat.unsqueeze(1))

        print("index: ", index)

        test_channel_feat = torch.cat(test_channel_feat, dim=1)
        test_table_feat = torch.cat(test_table_feat, dim=1)
        return test_table_feat, test_channel_feat

    def getChannelFeature(self, resnet, image=None):
        resnet.eval()
        if self.model_name == "mobilenet":
            new_resnet = nn.Sequential(*list(resnet.children())[:-1])
        else:
            new_resnet = nn.Sequential(*list(resnet.children())[:-2])
        channel_feat = new_resnet(image)  # [1, 2048, 7, 7]
        channel_feat = channel_feat.squeeze(0)
        channel_feat = channel_feat.reshape((self.in_dims, -1)).detach().cpu().numpy()  # (2048, 7 * 7)

        return torch.tensor(channel_feat, dtype=torch.float)  # (2048, 49)

    def getTableFeature(self, model, table_features_con, table_features_cat):
        model.eval()
        if self.con_fc_num == 0:
            table_features_con = None
        if self.cat_fc_num == 0:
            table_features_cat = None
        table_features_embed, _ = self.tab_model(table_features_con, table_features_cat)
        return table_features_embed.squeeze(0)

    def getCostMatrix(self, test_table_feat, test_channel_feat):
        # src_x.shape: (table_feat_num, num, table_embed)  tar_x.shape: (img_feat_num, num, img_embed)
        src_x, tar_x = test_table_feat.detach().cpu().numpy(), test_channel_feat.detach().cpu().numpy()
        img_embed = tar_x.shape[2]
        tar_x = tar_x.reshape((self.in_dims, -1))

        kmeans = KMeans(n_clusters=self.img_reduction_dim, random_state=0, n_init="auto").fit(tar_x)
        channel_feat_cluster = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
        tar_x = kmeans.cluster_centers_.reshape((self.img_reduction_dim, -1, img_embed))
        with open('res_tmp/cluster_centering_.txt', mode='w') as f:
            for i in range(self.img_reduction_dim):
                f.write(str(tar_x[i]) + '\n')

        labels = kmeans.labels_
        OutFileName = 'res_tmp/cluster_res_' + str(self.img_reduction_dim) + '_Adoption_Updating.txt'
        with open(OutFileName, 'w') as f:
            for i in range(self.img_reduction_dim):
                f.write(str(np.where(labels == i)[0].tolist()) + '\n')

        cost = np.zeros((src_x.shape[0], tar_x.shape[0]))
        for i in range(src_x.shape[0]):
            src_x_similarity_i = src_x[i] / np.linalg.norm(src_x[i])
            src_x_similarity_i = np.dot(src_x_similarity_i, src_x_similarity_i.transpose(1, 0))
            for j in range(tar_x.shape[0]):
                tar_x_similarity_j = tar_x[j] / np.linalg.norm(tar_x[j])
                tar_x_similarity_j = np.dot(tar_x_similarity_j, tar_x_similarity_j.transpose(1, 0))
                # print(src_x_similarity_i.shape, tar_x_similarity_j.shape)
                cost[i, j] = ((src_x_similarity_i - tar_x_similarity_j) ** 2).sum()
                # cost[i, j] = (np.abs(src_x_similarity_i - tar_x_similarity_j)).sum()
        return cost

    def compute_coupling(self, X_src, X_tar, Cost):
        # P = ot.bregman.sinkhorn(ot.unif(X_src.shape[0]), ot.unif(40), Cost, 0.001, numItermax=100000)
        P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(self.img_reduction_dim), Cost, numItermax=100000)
        # P = 0
        W = np.sum(P * np.array(Cost))

        return P, W


class ImageModelPetFinderWithRTDL(pl.LightningModule):
    def __init__(self, n_num_features, cat_cardinalities, reverse=False, img_reduction_dim=40):
        super().__init__()
        self.net_img_clf = ImageClassifier(model_name='resnet', n_num_features=n_num_features, cat_cardinalities=cat_cardinalities,
                                           img_reduction_dim=img_reduction_dim)
        self.test_acc = Accuracy(task="multiclass", num_classes=5)
        self.reverse = reverse
        self.img_reduction_dim = img_reduction_dim
        self.valid_loader = self.val_dataloader()
        self.loss_weight_dict = {'con_loss': 0.03, 'cat_loss': 0.03, 'tab_loss': 0.6, 'img_loss': 1}

    def val_dataloader(self):
        valid_dataset = PetFinderConCatImageDataset("/data/jiangjp/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_valid.csv",
                                                    "/data/jiangjp/PetFinder_datasets/raw_dataset/petfinder_adoptionprediction/train_images")
        valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=8, shuffle=False)
        return valid_loader

    def training_step(self, batch, batch_idx):
        (table_features_con, table_features_cat, image_features), label = batch
        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(image_features, table_features_con,
                                                                      table_features_cat)

        img_loss = F.cross_entropy(img_out, label)

        con_loss = []
        for idx, out_t in enumerate(con_out):
            con_loss.append(F.mse_loss(out_t, table_features_con[:, idx]))
        con_loss_mean = torch.stack(con_loss, dim=0).mean(dim=0) if table_features_con.shape[1] != 0 else 0

        cat_loss = []
        for idx, out_t in enumerate(cat_out):
            cat_loss.append(F.cross_entropy(out_t, table_features_cat[:, idx]))
        cat_loss_mean = torch.stack(cat_loss, dim=0).mean(dim=0) if table_features_cat.shape[1] != 0 else 0

        loss = self.loss_weight_dict['img_loss'] * img_loss + self.loss_weight_dict['con_loss'] * con_loss_mean \
               + self.loss_weight_dict['cat_loss'] * cat_loss_mean

        table_embed_loss = F.cross_entropy(table_embed_out, label)
        loss = loss + self.loss_weight_dict['tab_loss'] * table_embed_loss

        self.log("img_loss", img_loss)
        self.log("tab_con_loss", con_loss_mean)
        self.log("tab_cat_loss", cat_loss_mean)
        self.log("tab_embed_loss", table_embed_loss)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (table_features_con, table_features_cat, image_features), label = batch
        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(image_features, table_features_con,
                                                                      table_features_cat)

        img_loss = F.cross_entropy(img_out, label)

        con_loss = []
        for idx, out_t in enumerate(con_out):
            con_loss.append(F.mse_loss(out_t, table_features_con[:, idx]))
        con_loss_mean = torch.stack(con_loss, dim=0).mean(dim=0) if table_features_con.shape[1] != 0 else 0

        cat_loss = []
        for idx, out_t in enumerate(cat_out):
            cat_loss.append(F.cross_entropy(out_t, table_features_cat[:, idx]))
        cat_loss_mean = torch.stack(cat_loss, dim=0).mean(dim=0) if table_features_cat.shape[1] != 0 else 0

        loss = self.loss_weight_dict['img_loss'] * img_loss + self.loss_weight_dict['con_loss'] * con_loss_mean \
               + self.loss_weight_dict['cat_loss'] * cat_loss_mean

        table_embed_loss = F.cross_entropy(table_embed_out, label)
        loss = loss + self.loss_weight_dict['tab_loss'] * table_embed_loss

        preds = self.net_img_clf.backbone(image_features)
        preds = self.net_img_clf.img_fc(preds)
        val_acc = self.test_acc(preds, label).item()

        self.log("val_img_loss", img_loss)
        self.log("val_tab_con_loss", con_loss_mean)
        self.log("val_tab_cat_loss", cat_loss_mean)
        self.log("val_tab_embed_loss", table_embed_loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 5 != 0:
            return

        valid_dataset = self.valid_loader.dataset
        self.net_img_clf.compute_OT(valid_dataset, device=self.device)
        self.net_img_clf.mask = self.net_img_clf.get_mask('res_tmp/cluster_res_' + str(self.img_reduction_dim) +
                                                          '_Adoption_Updating.txt', 'res_tmp/OToutput_' +
                                                          str(self.img_reduction_dim) + '_Adoption_Updating.txt')
        if self.reverse:
            self.net_img_clf.mask = 1 - self.net_img_clf.mask
        return

    def test_step(self, batch, batch_idx):
        (_, _, image_features), label = batch
        preds = self.net_img_clf.backbone(image_features)
        preds = self.net_img_clf.img_fc(preds)
        loss = F.cross_entropy(preds, label)
        return {"loss": loss, "preds": preds.detach(), "y": label.detach()}

    def test_step_end(self, outputs):
        test_acc = self.test_acc(outputs['preds'], outputs['y']).item()
        self.log("Reduction dimension of image", self.img_reduction_dim, on_epoch=True, on_step=False)
        self.log("test_acc", test_acc, on_epoch=True, on_step=False)
        self.log("test_loss", outputs["loss"].mean(), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.net_img_clf.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
        optimizer_config = {
            "optimizer": optimizer,
        }
        print("optimizer_config:\n", optimizer_config)
        if scheduler:
            optimizer_config.update({
                "lr_scheduler": {
                    "name": 'MultiStep_LR_scheduler',
                    "scheduler": scheduler,
                }})
            print("scheduler_config:\n", scheduler.state_dict())
        return optimizer_config
