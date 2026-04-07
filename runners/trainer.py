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
from datasets.datasets_nuclear import get_nuclear_data_loaders, process_batch_seg
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

def validate_segmentation_pose_init(cfg):
    """Require a pose-init checkpoint for segmentation training."""
    pretrained_score_model_path = getattr(cfg, 'pretrained_score_model_path', None)
    if not pretrained_score_model_path:
        raise ValueError(
            "segmentation training requires cfg.pretrained_score_model_path"
        )
    return pretrained_score_model_path


def freeze_pose_params(agent):
    """Freeze all parameters except the segmentation head/tail modules."""
    net = agent.net.module if isinstance(agent.net, torch.nn.DataParallel) else agent.net
    for param in net.parameters():
        param.requires_grad = False

    def _unfreeze(module):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = True

    _unfreeze(getattr(net, 'eomt_head', None))
    dino_wrapper = getattr(net, 'dino_wrapper', None)
    if dino_wrapper is not None:
        _unfreeze(getattr(dino_wrapper, 'query_embed', None))
        _unfreeze(getattr(dino_wrapper, 'seg_blocks', None))
        _unfreeze(getattr(dino_wrapper, 'seg_norm', None))

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"[freeze_pose_params] Trainable: {trainable:,} / Total: {total:,}")


def train_segmentation(cfg, train_loader, val_loader, seg_agent):
    """Train EoMT segmentation/classification head.

    Args:
        cfg: Config namespace.
        train_loader: Training data loader (NuclearWorkpieceDataset).
        val_loader: Validation data loader.
        seg_agent: PoseNet agent with enable_segmentation=True.
    """
    for epoch in range(seg_agent.clock.epoch, cfg.n_epochs):
        torch.cuda.empty_cache()
        pbar = tqdm(train_loader)
        for i, batch_sample in enumerate(pbar):
            # Warm up
            if seg_agent.clock.step < cfg.warmup:
                seg_agent.update_learning_rate()

            batch_sample = process_batch_seg(batch_sample, cfg.device)
            losses = seg_agent.train_func(data=batch_sample, gf_mode='segmentation')

            loss_vals = [f"{k}:{v.item():.4f}" for k, v in losses.items()]
            pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][{', '.join(loss_vals)}]")
            seg_agent.clock.tick()

        seg_agent.update_learning_rate()
        seg_agent.clock.tock()

        # Evaluation
        if seg_agent.clock.epoch % cfg.eval_freq == 0:
            if val_loader is not None:
                val_batch = next(iter(val_loader))
                val_batch = process_batch_seg(val_batch, cfg.device)
                seg_agent.eval_func(val_batch, 'val', gf_mode='segmentation')
            seg_agent.save_ckpt()


def main():
    # load config
    cfg = get_config()
    
    ''' Init data loader '''
    if getattr(cfg, 'dataset_type', 'omni6dpose') == 'nuclear':
        # Nuclear workpiece dataset for segmentation training
        data_loaders = get_nuclear_data_loaders(cfg, data_type=['train', 'val'])
        train_loader = data_loaders.get('train_loader')
        val_loader = data_loaders.get('val_loader')
        test_loader = val_loader  # reuse val as test for nuclear dataset
        if train_loader:
            print('train_set: ', len(train_loader))
        if val_loader:
            print('val_set: ', len(val_loader))
    elif not (cfg.eval or cfg.pred):
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

    elif cfg.agent_type == 'segmentation':
        cfg.enable_segmentation = True
        validate_segmentation_pose_init(cfg)
        seg_agent = PoseNet(cfg)
        seg_agent.load_ckpt(
            model_dir=cfg.pretrained_score_model_path,
            model_path=True,
            load_model_only=True,
        )
        freeze_pose_params(seg_agent)
        tr_agent = seg_agent
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
    elif cfg.agent_type == 'segmentation':
        train_segmentation(cfg, train_loader, val_loader, tr_agent)
    else:
        train_scale(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent)
if __name__ == '__main__':
    main()

