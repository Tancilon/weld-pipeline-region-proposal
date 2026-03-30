import sys
import os
import argparse
import pickle
import time
import json
import numpy as np
import torch
import cv2
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ipdb import set_trace
from tqdm import tqdm


# from datasets.datasets_nocs import get_data_loaders_from_cfg, process_batch
from datasets.datasets_omni6dpose import get_data_loaders_from_cfg, process_batch, array_to_SymLabel
from networks.posenet_agent import PoseNet 
from configs.config import get_config
from utils.misc import exists_or_mkdir, get_pose_representation
from utils.genpose_utils import merge_results
from utils.misc import average_quaternion_batch, parallel_setup, parallel_cleanup
from utils.metrics import get_metrics, get_rot_matrix
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image
from utils.transforms import *
from cutoop.utils import draw_3d_bbox
from cutoop.transform import *
from cutoop.data_types import *
from cutoop.eval_utils import *

def train_score(cfg, train_loader, val_loader, test_loader, score_agent, teacher_model=None):
    """ Train score network or energe network without ranking
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        score_agent (torch.nn.Module): score network or energy network without ranking
    Returns:
    """
    
    for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
        ''' train '''
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        pbar = tqdm(train_loader)
        for i, batch_sample in enumerate(pbar):

            ''' warm up'''
            if score_agent.clock.step < cfg.warmup:
                score_agent.update_learning_rate()
                
            ''' load data '''
            batch_sample = process_batch(
                batch_sample = batch_sample, 
                device=cfg.device, 
                pose_mode=cfg.pose_mode, 
                PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
            )
            
            ''' train score or energe without feedback'''
            losses = score_agent.train_func(data=batch_sample, gf_mode='score', teacher_model=teacher_model)
            
            pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
            score_agent.clock.tick()
        
        ''' updata learning rate and clock '''
        # if epoch >= 50 and epoch % 50 == 0:
        score_agent.update_learning_rate()
        score_agent.clock.tock()

        ''' start eval '''
        if score_agent.clock.epoch % cfg.eval_freq == 0:   
            data_loaders = [train_loader, val_loader, test_loader]    
            data_modes = ['train', 'val', 'test']   
            for i in range(len(data_modes)):
                test_batch = next(iter(data_loaders[i]))
                data_mode = data_modes[i]
                test_batch = process_batch(
                    batch_sample=test_batch,
                    device=cfg.device,
                    pose_mode=cfg.pose_mode,
                )
                score_agent.eval_func(test_batch, data_mode)
                
            ''' save (ema) model '''
            score_agent.save_ckpt()


def train_energy(cfg, train_loader, val_loader, test_loader, energy_agent, score_agent=None, ranking=False, distillation=False):
    """ Train score network or energe network without ranking
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        energy_agent (torch.nn.Module): energy network with ranking
        score_agent (torch.nn.Module): score network
        ranking (bool): train energy network with ranking or not
    Returns:
    """
    if ranking is False:
        teacher_model = None if not distillation else score_agent.net
        train_score(cfg, train_loader, val_loader, test_loader, energy_agent, teacher_model)
    else:
        for epoch in range(energy_agent.clock.epoch, cfg.n_epochs):
            torch.cuda.empty_cache()
            pbar = tqdm(train_loader)
            for i, batch_sample in enumerate(pbar):
                
                ''' warm up '''
                if energy_agent.clock.step < cfg.warmup:
                    energy_agent.update_learning_rate()
                    
                ''' get data '''
                batch_sample = process_batch(
                    batch_sample = batch_sample, 
                    device=cfg.device, 
                    pose_mode=cfg.pose_mode, 
                    PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
                )
                
                ''' get pose samples from pretrained score network '''
                pred_pose = score_agent.pred_func(data=batch_sample, repeat_num=5, save_path=None)
                
                ''' train energy '''
                losses = energy_agent.train_func(data=batch_sample, pose_samples=pred_pose, gf_mode='energy')
                pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
                
                energy_agent.clock.tick()
            energy_agent.update_learning_rate()
            energy_agent.clock.tock()

            ''' start eval '''
            if energy_agent.clock.epoch % cfg.eval_freq == 0:   
                data_loaders = [train_loader, val_loader, test_loader]    
                data_modes = ['train', 'val', 'test']   
                for i in range(len(data_modes)):
                    test_batch = next(iter(data_loaders[i]))
                    data_mode = data_modes[i]
                    test_batch = process_batch(
                        batch_sample=test_batch,
                        device=cfg.device,
                        pose_mode=cfg.pose_mode,
                    )
                    
                    ''' get pose samples from pretrained score network '''
                    pred_pose = score_agent.pred_func(data=test_batch, repeat_num=5, save_path=None)
                    energy_agent.eval_func(test_batch, data_mode, None, 'score')
                    energy_agent.eval_func(test_batch, data_mode, pred_pose, 'energy')
                
                ''' save (ema) model '''
                energy_agent.save_ckpt()

def train_scale(cfg, train_loader, val_loader, test_loader, scale_agent, score_agent):
    """ Train scale network
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        scale_agent (torch.nn.Module): scale network
        score_agent (torch.nn.Module): score network
    Returns:
    """

    score_agent.eval()
    
    for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
        ''' train '''
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        pbar = tqdm(train_loader)
        for i, batch_sample in enumerate(pbar):
            
            ''' warm up'''
            if score_agent.clock.step < cfg.warmup:
                score_agent.update_learning_rate()
                
            ''' load data '''
            batch_sample = process_batch(
                batch_sample = batch_sample, 
                device=cfg.device, 
                pose_mode=cfg.pose_mode, 
                PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
            )
            
            ''' train scale'''
            with torch.no_grad():
                score_agent.encode_func(data=batch_sample)
            losses = scale_agent.train_func(data=batch_sample, gf_mode='scale')
            
            pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
            scale_agent.clock.tick()
        
        ''' updata learning rate and clock '''
        # if epoch >= 50 and epoch % 50 == 0:
        scale_agent.update_learning_rate()
        scale_agent.clock.tock()

        ''' start eval '''
        if scale_agent.clock.epoch % cfg.eval_freq == 0:   
            data_loaders = [train_loader, val_loader, test_loader]    
            data_modes = ['train', 'val', 'test']   
            for i in range(len(data_modes)):
                test_batch = next(iter(data_loaders[i]))
                data_mode = data_modes[i]
                test_batch = process_batch(
                    batch_sample=test_batch,
                    device=cfg.device,
                    pose_mode=cfg.pose_mode,
                )
                with torch.no_grad():
                    score_agent.encode_func(data=test_batch)
                scale_agent.eval_func(test_batch, data_mode, gf_mode='scale')
                
            ''' save (ema) model '''
            scale_agent.save_ckpt()

def train_classification(cfg, train_loader, val_loader, score_agent):
    """Train classification head only (Stage 1: freeze encoder, train MLP).

    Expects each batch from train_loader to contain 'class_label' with integer
    class indices (0-5 for the 6 workpiece categories).
    """
    import torch.nn.functional as F

    if not hasattr(score_agent.net, 'classification_head'):
        raise RuntimeError("classification_enabled must be True to train classification head")

    # Freeze everything except classification_head
    for name, param in score_agent.net.named_parameters():
        if 'classification_head' not in name:
            param.requires_grad = False

    cls_optimizer = torch.optim.Adam(
        score_agent.net.classification_head.parameters(), lr=cfg.lr
    )
    cls_scheduler = torch.optim.lr_scheduler.ExponentialLR(cls_optimizer, gamma=cfg.lr_decay)

    best_val_acc = 0.0
    for epoch in range(cfg.n_epochs):
        score_agent.net.train()
        torch.cuda.empty_cache()
        pbar = tqdm(train_loader)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, batch_sample in enumerate(pbar):
            batch_sample = process_batch(
                batch_sample=batch_sample,
                device=cfg.device,
                pose_mode=cfg.pose_mode,
                PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS,
            )

            with torch.no_grad():
                pts_feat = score_agent.net(batch_sample, mode='pts_feature')
                dino_global = score_agent.net.dino(batch_sample['roi_rgb'])
            cls_input = torch.cat([pts_feat, dino_global], dim=-1)
            class_logits = score_agent.net.classification_head(cls_input)
            loss = F.cross_entropy(class_logits, batch_sample['class_label'])

            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()

            pred_cls = torch.argmax(class_logits, dim=-1)
            total_correct += (pred_cls == batch_sample['class_label']).sum().item()
            total_samples += batch_sample['class_label'].shape[0]
            total_loss += loss.item()
            pbar.set_description(
                f"CLS_EPOCH_{epoch}[{i}/{len(pbar)}]"
                f"[loss: {loss.item():.4f}, acc: {total_correct/total_samples:.4f}]"
            )

        cls_scheduler.step()
        train_acc = total_correct / max(total_samples, 1)
        print(f"Epoch {epoch}: train_loss={total_loss/max(len(train_loader),1):.4f}, train_acc={train_acc:.4f}")

        # Validation
        if val_loader is not None and epoch % cfg.eval_freq == 0:
            score_agent.net.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_sample in val_loader:
                    batch_sample = process_batch(
                        batch_sample=batch_sample,
                        device=cfg.device,
                        pose_mode=cfg.pose_mode,
                    )
                    pts_feat = score_agent.net(batch_sample, mode='pts_feature')
                    dino_global = score_agent.net.dino(batch_sample['roi_rgb'])
                    cls_input = torch.cat([pts_feat, dino_global], dim=-1)
                    class_logits = score_agent.net.classification_head(cls_input)
                    pred_cls = torch.argmax(class_logits, dim=-1)
                    val_correct += (pred_cls == batch_sample['class_label']).sum().item()
                    val_total += batch_sample['class_label'].shape[0]
            val_acc = val_correct / max(val_total, 1)
            print(f"Epoch {epoch}: val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                score_agent.save_ckpt('best_cls')

    # Unfreeze all parameters for potential Stage 2 joint training
    for param in score_agent.net.parameters():
        param.requires_grad = True

    print(f"Classification training complete. Best val_acc={best_val_acc:.4f}")


def main():
    # load config
    cfg = get_config()
    
    ''' Init data loader '''
    if not (cfg.eval or cfg.pred):
        data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['train', 'val', 'test'])
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
        print('train_set: ', len(train_loader))
        print('val_set: ', len(val_loader))
        print('test_set: ', len(test_loader))
    else:
        data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['test'])
        test_loader = data_loaders['test_loader']   
        print('test_set: ', len(test_loader))
  
    
    ''' Init trianing agent and load checkpoints'''
    if cfg.agent_type == 'score':
        cfg.agent_type = 'score'
        score_agent = PoseNet(cfg)
        tr_agent = score_agent
        
    elif cfg.agent_type == 'energy':
        cfg.agent_type = 'energy'
        energy_agent = PoseNet(cfg)
        if cfg.pretrained_score_model_path is not None:
            energy_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
            energy_agent.net.pose_score_net.output_zero_initial()
        if cfg.distillation is True:
            cfg.agent_type = 'score'
            score_agent = PoseNet(cfg)
            score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
            cfg.agent_type = 'energy'
        tr_agent = energy_agent
        
    elif cfg.agent_type == 'energy_with_ranking':
        cfg.agent_type = 'score'
        score_agent = PoseNet(cfg)    
        cfg.agent_type = 'energy'
        energy_agent = PoseNet(cfg)
        score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        tr_agent = energy_agent
    
    elif cfg.agent_type == 'scale':
        cfg.agent_type = 'score'
        cfg.agent_type = 'score'
        score_agent = PoseNet(cfg)
        score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        cfg.agent_type = 'scale'
        scale_agent = PoseNet(cfg)
        tr_agent = scale_agent
    else:
        raise NotImplementedError
    
    ''' Load checkpoints '''
    if cfg.use_pretrain or cfg.eval or cfg.pred:
        tr_agent.load_ckpt(
            model_dir=(
                cfg.pretrained_score_model_path if cfg.agent_type == 'score' else (
                    cfg.pretrained_energy_model_path if cfg.agent_type in ['energy', 'energy_with_ranking']
                        else cfg.pretrained_scale_model_path
                )
            ), 
            model_path=True, 
            load_model_only=False
        )
                
        
    ''' Start training loop '''
    if cfg.agent_type == 'score':
        train_score(cfg, train_loader, val_loader, test_loader, tr_agent)
    elif cfg.agent_type == 'energy':
        if cfg.distillation:
            train_energy(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent, False, True)
        else:
            train_energy(cfg, train_loader, val_loader, test_loader, tr_agent)
    elif cfg.agent_type == 'energy_with_ranking':
        train_energy(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent, True)
    else:
        train_scale(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent)
if __name__ == '__main__':
    main()


