# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import json
import numpy as np
import torch.nn.functional as F
from torchvision.ops import roi_align

class DSCL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, beta=6.0, num_classes=1000):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(DSCL, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self.contrastive_loss = DSCLLoss(temperature=T, base_temperature=None, K=K, weighted_beta=beta)
        self.distillation_loss = PBSDLoss(temperature=T, base_temperature=None)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # TODO: maybe should fill the queue at first
        self.register_buffer("main_label_queue", torch.randint(0, num_classes, (1, K)))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.layer4_feat = None
        self.layer4_feat_k = None
        self._register_hook()

    def reset_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient 

    def _get_feature_hook(self, module, input, output):
        self.layer4_feat = output
    
    def _get_feature_hook_k(self, module, input, output):
        self.layer4_feat_k = output

    def _register_hook(self):
        handle_q = self.encoder_q.layer4.register_forward_hook(self._get_feature_hook)
        handle_k = self.encoder_k.layer4.register_forward_hook(self._get_feature_hook_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, main_labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        main_labels = concat_all_gather(main_labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        self.main_label_queue[:, ptr:ptr + batch_size] = main_labels.unsqueeze(0)

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, im_q, im_k, small_im_qs, small_bbox_q, main_labels):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        roi_features = []
        for bb in small_bbox_q:
            re_org_bb = [bb[i].unsqueeze(0) for i in range(bb.size(0))]
            pooled_feature = roi_align(self.layer4_feat, re_org_bb, (2, 2), spatial_scale=(1 / 32))
            pooled_feature = torch.mean(pooled_feature, dim=(2, 3))
            pooled_feature = self.encoder_q.fc(pooled_feature)
            pooled_feature = nn.functional.normalize(pooled_feature, dim=1)
            roi_features.append(pooled_feature)

        q = nn.functional.normalize(q, dim=1)

        split_small_qs = []
        for small_im_q in small_im_qs:
            small_q = self.encoder_q(small_im_q)

            small_q = nn.functional.normalize(small_q, dim=1)
            split_small_qs.append(small_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        features = torch.cat((q, k, self.queue.T.clone().detach()), dim=0)
        target = torch.cat((main_labels, main_labels, self.main_label_queue.squeeze(0).clone().detach()), dim=0)

        contrastive_loss = self.contrastive_loss(features, target)
        distillation_loss = self.distillation_loss(roi_features, split_small_qs, k, self.queue, main_labels, self.main_label_queue)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, main_labels)

        return contrastive_loss, distillation_loss

    def forward_test(self, im_q):
        q = self.encoder_q(im_q)
        return self.layer4_feat

    def forward(self, im_q, im_k=None, small_im_qs=None, small_bbox=None, main_labels=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.training:
            return self.forward_train(im_q, im_k, small_im_qs, small_bbox, main_labels)
        else:
            return self.forward_test(im_q)


class DSCLLoss(nn.Module):
    def __init__(self, temperature=1.0, base_temperature=None, K=128, weighted_beta=8.0):
        super(DSCLLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.weighted_beta = weighted_beta

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K) // 2

        labels = labels.contiguous().view(-1, 1) # 2BKx1 ) 

        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # logits = anchor_dot_contrast
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        class_weighted = torch.ones_like(mask) / (mask.sum(dim=1, keepdim=True) - 1.0 + 1e-12) * self.weighted_beta
        class_weighted = class_weighted.scatter(1, torch.arange(batch_size).view(-1, 1).to(device) + batch_size, 1.0)

        # compute mean of log-likelihood over positive
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob * class_weighted).sum(1) / (mask * class_weighted).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * (mean_log_prob_pos)
        loss_contrastive = loss.mean()

        return loss_contrastive


class PBSDLoss(nn.Module):
    def __init__(self, temperature=1.0, base_temperature=None):
        super(PBSDLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature

    def forward(self, roi_features, small_qs, k, queue, main_labels, main_label_queue):
        for index, (q1, small_q) in enumerate(zip(roi_features, small_qs)):
            l_pos = torch.einsum('nc,nc->n', [q1, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q1, queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.temperature
            soft_target = F.softmax(logits, dim=1)

            mask = torch.eq(main_labels.unsqueeze(1), main_label_queue.clone().detach()).float() # NxK
            mask = torch.cat((torch.ones_like(main_labels.unsqueeze(1)).float(), mask), dim=1) # Nx(1+K)

            l_pos2 = torch.einsum('nc,nc->n', [small_q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg2 = torch.einsum('nc,ck->nk', [small_q, queue.clone().detach()])

            # logits: Nx(1+K)
            logits2 = torch.cat([l_pos2, l_neg2], dim=1)
            # apply temperature
            logits2 /= self.temperature
            prob2 = F.log_softmax(logits2, dim=1)

            if index == 0:
                loss = - (self.temperature / self.base_temperature) * (soft_target.detach() * prob2).sum(1)
            else:
                loss = loss + (- (self.temperature / self.base_temperature) * (soft_target.detach() * prob2).sum(1))

        loss_contrastive = loss.mean() / len(small_qs)

        return loss_contrastive


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
