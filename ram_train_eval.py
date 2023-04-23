import os
import time
from datetime import timedelta
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import Config, ProgressBar
from transformers import AutoConfig, AutoModel


class RamDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train=True, num_relation_classes=56):
        super().__init__()
        self.num_relation_classes = num_relation_classes
        data = np.load(data_path, allow_pickle=True)
        self.samples = data['arr_0']
        sample_num = self.samples.size
        self.sample_idx_list = []
        for idx in range(sample_num):
            if self.samples[idx]['is_train'] == is_train:
                self.sample_idx_list.append(idx)

    def __getitem__(self, idx):
        sample = self.samples[self.sample_idx_list[idx]]
        object_num = sample['feat'].shape[0]
        embedding = torch.from_numpy(sample['feat'])
        gt_rels = sample['relations']
        rel_target = self._get_target(object_num, gt_rels)
        return embedding, rel_target, gt_rels

    def __len__(self):
        return len(self.sample_idx_list)

    def _get_target(self, object_num, gt_rels):
        rel_target = torch.zeros(
            [self.num_relation_classes, object_num, object_num])
        for ii, jj, cls_relationship in gt_rels:
            rel_target[cls_relationship, ii, jj] = 1
        return rel_target


class RamModel(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path,
                 load_pretrained_weights=True,
                 num_transformer_layer=2,
                 input_feature_size=256,
                 output_feature_size=768,
                 cls_feature_size=512,
                 num_relation_classes=56,
                 pred_type='attention',
                 loss_type='bce'):
        super().__init__()
        # 0. config
        self.cls_feature_size = cls_feature_size
        self.num_relation_classes = num_relation_classes
        self.pred_type = pred_type
        self.loss_type = loss_type

        # 1. fc input and output
        self.fc_input = nn.Sequential(
            nn.Linear(input_feature_size, output_feature_size),
            nn.LayerNorm(output_feature_size),)
        self.fc_output = nn.Sequential(
            nn.Linear(output_feature_size, output_feature_size),
            nn.LayerNorm(output_feature_size),)
        # 2. transformer model
        if load_pretrained_weights:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path)
            self.model = AutoModel.from_config(config)
        if num_transformer_layer != 'all' and isinstance(num_transformer_layer, int):
            self.model.encoder.layer = self.model.encoder.layer[:num_transformer_layer]
        # 3. predict head
        self.cls_sub = nn.Linear(output_feature_size,
                                 cls_feature_size * num_relation_classes)
        self.cls_obj = nn.Linear(output_feature_size,
                                 cls_feature_size * num_relation_classes)
        # 4. loss
        if self.loss_type == 'bce':
            self.bce_loss = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'multi_label_ce':
            print('Use Multi Label Cross Entropy Loss.')

    def forward(self, embeds, attention_mask=None):
        """
            embeds: (batch_size, token_num, feature_size)
            attention_mask: (batch_size, token_num)
        """
        # 1. fc input
        embeds = self.fc_input(embeds)
        # 2. transformer model
        position_ids = torch.ones([1, embeds.shape[1]]).to(
            embeds.device).to(torch.long)
        outputs = self.model.forward(inputs_embeds=embeds,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids)
        embeds = outputs['last_hidden_state']
        # 3. fc output
        embeds = self.fc_output(embeds)
        # 4. predict head
        batch_size, token_num, feature_size = embeds.shape
        sub_embeds = self.cls_sub(embeds).reshape(
            [batch_size, token_num, self.num_relation_classes, self.cls_feature_size]).permute([0, 2, 1, 3])
        obj_embeds = self.cls_obj(embeds).reshape(
            [batch_size, token_num, self.num_relation_classes, self.cls_feature_size]).permute([0, 2, 1, 3])
        if self.pred_type == 'attention':
            cls_pred = sub_embeds @ torch.transpose(obj_embeds, 2, 3) / self.cls_feature_size ** 0.5  # noqa
        elif self.pred_type == 'einsum':
            cls_pred = torch.einsum(
                'nrsc,nroc->nrso', sub_embeds, obj_embeds)
        return cls_pred

    def loss(self, pred, target, attention_mask):
        loss_dict = dict()
        batch_size, relation_num, _, _ = pred.shape

        mask = torch.zeros_like(pred).to(pred.device)
        for idx in range(batch_size):
            n = torch.sum(attention_mask[idx]).to(torch.int)
            mask[idx, :, :n, :n] = 1
        pred = pred * mask - 9999 * (1 - mask)

        if self.loss_type == 'bce':
            loss = self.bce_loss(pred, target)
        elif self.loss_type == 'multi_label_ce':
            input_tensor = torch.permute(pred, (1, 0, 2, 3))
            target_tensor = torch.permute(target, (1, 0, 2, 3))
            input_tensor = pred.reshape([relation_num, -1])
            target_tensor = target.reshape([relation_num, -1])
            loss = self.multilabel_categorical_crossentropy(
                target_tensor, input_tensor)
            weight = (loss / loss.max())
            loss = loss * weight
        loss = loss.mean()
        loss_dict['loss'] = loss

        # running metric
        recall_20 = get_recall_N(pred, target, object_num=20)
        loss_dict['recall@20'] = recall_20
        return loss_dict

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """
            https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss


def get_recall_N(y_pred, y_true, object_num=20):
    """
        y_pred: [batch_size, 56, object_num, object_num]
        y_true: [batch_size, 56, object_num, object_num]
    """

    device = y_pred.device
    recall_list = []

    for idx in range(len(y_true)):
        sample_y_true = []
        sample_y_pred = []

        # find topk
        _, topk_indices = torch.topk(
            y_true[idx:idx+1].reshape([-1, ]), k=object_num)
        for index in topk_indices:
            pred_cls = index // (y_true.shape[2] ** 2)
            index_subject_object = index % (y_true.shape[2] ** 2)
            pred_subject = index_subject_object // y_true.shape[2]
            pred_object = index_subject_object % y_true.shape[2]
            if y_true[idx, pred_cls, pred_subject, pred_object] == 0:
                continue
            sample_y_true.append([pred_subject, pred_object, pred_cls])

        # find topk
        _, topk_indices = torch.topk(
            y_pred[idx:idx+1].reshape([-1, ]), k=object_num)
        for index in topk_indices:
            pred_cls = index // (y_pred.shape[2] ** 2)
            index_subject_object = index % (y_pred.shape[2] ** 2)
            pred_subject = index_subject_object // y_pred.shape[2]
            pred_object = index_subject_object % y_pred.shape[2]
            sample_y_pred.append([pred_subject, pred_object, pred_cls])

        recall = len([x for x in sample_y_pred if x in sample_y_true]
                     ) / (len(sample_y_true) + 1e-8)
        recall_list.append(recall)

    recall = torch.tensor(recall_list).to(device).mean() * 100
    return recall


class RamTrainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._build_dataset()
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_lr_scheduler()

    def _build_dataset(self):
        self.dataset = RamDataset(**self.config.dataset)

    def _build_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True if self.config.dataset.is_train else False,)

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from))
        self.model.train()

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            eps=self.config.optim.eps,
            betas=self.config.optim.betas)

    def _build_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.optim.lr_scheduler.step,
            gamma=self.config.optim.lr_scheduler.gamma)

    def train(self):
        t_start = time.time()
        running_avg_loss = 0
        for epoch_idx in range(self.config.num_epoch):
            for batch_idx, batch_data in enumerate(self.dataloader):
                batch_embeds = batch_data[0].to(torch.float32).to(self.device)
                batch_target = batch_data[1].to(torch.float32).to(self.device)
                attention_mask = batch_embeds.new_ones(
                    (batch_embeds.shape[0], batch_embeds.shape[1]))
                batch_pred = self.model.forward(batch_embeds, attention_mask)
                loss_dict = self.model.loss(
                    batch_pred, batch_target, attention_mask)
                loss = loss_dict['loss']
                recall_20 = loss_dict['recall@20']
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.optim.max_norm,
                                               self.config.optim.norm_type)
                self.optimizer.step()
                running_avg_loss += loss.item()

                if batch_idx % 100 == 0:
                    t_current = time.time()
                    num_finished_step = epoch_idx * self.config.num_epoch * \
                        len(self.dataloader) + batch_idx + 1
                    num_to_do_step = (self.config.num_epoch - epoch_idx - 1) * \
                        len(self.dataloader) + \
                        (len(self.dataloader) - batch_idx - 1)
                    avg_speed = num_finished_step / (t_current - t_start)
                    eta = num_to_do_step / avg_speed
                    print('ETA={:0>8}, Epoch={}, Batch={}/{}, LR={}, Loss={:.4f}, RunningAvgLoss={:.4f}, Recall@20={:.2f}%'.format(
                        str(timedelta(seconds=int(eta))
                            ), epoch_idx + 1, batch_idx,
                        len(self.dataloader), self.lr_scheduler.get_last_lr()[
                            0], loss.item(),
                        running_avg_loss/num_finished_step, recall_20.item()))
            self.lr_scheduler.step()
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)
            save_path = os.path.join(
                self.config.output_dir, 'epoch_{}.pth'.format(epoch_idx + 1))
            print('Save epoch={} checkpoint to {}'.format(
                epoch_idx + 1, save_path))
            torch.save(self.model.state_dict(), save_path)
        return save_path


class RamPredictor(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._build_dataset()
        self._build_dataloader()
        self._build_model()

    def _build_dataset(self):
        self.dataset = RamDataset(**self.config.dataset)

    def _build_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False)

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from))
        self.model.eval()

    def predict(self, batch_embeds, pred_keep_num=100):
        """
        Parameters
        ----------
            batch_embeds: (batch_size=1, token_num, feature_size)
            pred_keep_num: int
        Returns
        -------
            batch_pred: (batch_size, relation_num, object_num, object_num)
            pred_rels: [[sub_id, obj_id, rel_id], ...]
        """
        if not isinstance(batch_embeds, torch.Tensor):
            batch_embeds = torch.asarray(batch_embeds)
        batch_embeds = batch_embeds.to(torch.float32).to(self.device)
        attention_mask = batch_embeds.new_ones(
            (batch_embeds.shape[0], batch_embeds.shape[1]))
        batch_pred = self.model.forward(batch_embeds, attention_mask)
        for idx_i in range(batch_pred.shape[2]):
            batch_pred[:, :, idx_i, idx_i] = -9999
        batch_pred = batch_pred.sigmoid()

        pred_rels = []
        for idx in range(batch_embeds.shape[0]):
            this_pred = batch_pred[idx]
            # find topk
            this_pred_rels = []
            _, topk_indices = torch.topk(
                this_pred.reshape([-1, ]), k=pred_keep_num)

            # subject, object, relation
            for index in topk_indices:
                pred_relation = index // (
                    this_pred.shape[1] ** 2)
                index_subject_object = index % (
                    this_pred.shape[1] ** 2)
                pred_subject = index_subject_object // this_pred.shape[1]
                pred_object = index_subject_object % this_pred.shape[1]
                pred = [pred_subject.item(),
                        pred_object.item(),
                        pred_relation.item()]
                this_pred_rels.append(pred)
            pred_rels.append(this_pred_rels)
        if batch_embeds.shape[0] == 1:
            pred_rels = pred_rels[0]
        return batch_pred, pred_rels

    def eval(self):
        sum_recall_20 = 0.
        sum_recall_50 = 0.
        sum_recall_100 = 0.
        prog_bar = ProgressBar(len(self.dataloader))
        for batch_idx, batch_data in enumerate(self.dataloader):
            batch_embeds = batch_data[0]
            batch_target = batch_data[1]
            gt_rels = batch_data[2]
            batch_pred, pred_rels = self.predict(batch_embeds)
            this_recall_20 = get_recall_N(
                batch_pred, batch_target, object_num=20)
            this_recall_50 = get_recall_N(
                batch_pred, batch_target, object_num=50)
            this_recall_100 = get_recall_N(
                batch_pred, batch_target, object_num=100)
            sum_recall_20 += this_recall_20.item()
            sum_recall_50 += this_recall_50.item()
            sum_recall_100 += this_recall_100.item()
            prog_bar.update()
        recall_20 = sum_recall_20 / len(self.dataloader)
        recall_50 = sum_recall_50 / len(self.dataloader)
        recall_100 = sum_recall_100 / len(self.dataloader)
        metric = {
            'recall_20': recall_20,
            'recall_50': recall_50,
            'recall_100': recall_100,
        }
        return metric


if __name__ == '__main__':
    # Config
    config = dict(
        dataset=dict(
            data_path='/scratch/grp/grv_shi/k21163430/data/ram/feat_0420.npz',
            is_train=True,
            num_relation_classes=56,
        ),
        dataloader=dict(
            batch_size=1,
        ),
        model=dict(
            pretrained_model_name_or_path='/scratch/grp/grv_shi/k21163430/work_dirs/checkpoints/bert-base-uncased',
            load_pretrained_weights=True,
            num_transformer_layer=2,
            input_feature_size=256,
            output_feature_size=768,
            cls_feature_size=512,
            num_relation_classes=56,
            pred_type='attention',
            loss_type='multi_label_ce',
        ),
        optim=dict(
            lr=1e-4,
            weight_decay=0.05,
            eps=1e-8,
            betas=(0.9, 0.999),
            max_norm=0.01,
            norm_type=2,
            lr_scheduler=dict(
                step=[6, 10],
                gamma=0.1,
            ),
        ),
        num_epoch=12,
        output_dir='/users/k21163430/workspace/RelateAnything/work_dirs/ram_v0_0',
        load_from=None,
    )

    # Train
    config = Config(config)
    trainer = RamTrainer(config)
    last_model_path = trainer.train()

    # Test/Eval
    config.dataset.is_train = False
    # config.load_from = '/users/k21163430/workspace/RelateAnything/work_dirs/ram_v0_0/epoch_12.pth'
    config.load_from = last_model_path
    predictor = RamPredictor(config)
    metric = predictor.eval()
    print(metric)
