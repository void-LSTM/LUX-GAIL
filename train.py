import numpy as np
import json
from pathlib import Path
import os
import sys
import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, List

from config import *
from agent_constants import *
from model.policy_network import PolicyNetwork,Discriminator,Critic
from model.teacher_network import PolicyNetwork_teacher
from model.feature import feature_net
from test import EvalEnv
import argparse
from load_data import create_filepath_dataset_from_json
from dataset import LuxDataset, DatasetOutput
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter  

from torchsummary import summary
def get_state_f(model,states, maskings):
    feature=model.get_feature(states, maskings)
    feature=feature.detach()
    return feature
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def take_target_loss(outs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor,
                     weight: Optional[torch.Tensor] = None):
    b, h, w, out_dim = outs.shape
    outs = outs.reshape(-1, out_dim)

    _, preds = torch.max(outs, dim=1)

    targets = targets.reshape(-1)
    num_targets = torch.sum(targets).item()
    if num_targets==0.0:
        num_targets+=1.0
    actions = actions.reshape(-1)
    ce_loss_batch = F.cross_entropy(outs, actions, weight=weight, reduce=False) * targets

    loss = torch.sum(ce_loss_batch) / num_targets
    acc = torch.sum((preds == actions.data) * targets) / num_targets
    return loss, acc, num_targets

def get_prob(actions,action_probs,targets):
    b,w,h=actions.shape
    actions=actions.reshape(b,w,h,1)
    action_probs = action_probs.gather(3, actions)
    action_probs=action_probs.reshape(b,-1)
    targets=targets.reshape(b,-1)
    action_probs=action_probs*targets
    action_probs=torch.sum(action_probs,1)
    action_probs=action_probs.reshape(b,-1)
    return action_probs
    

def collate_fn(batch: List[DatasetOutput]):
    state_arrays = np.array([b.state_array for b in batch])
    action_arrays = np.array([b.action_array for b in batch])
    target_arrays = np.array([b.target_array for b in batch])
    city_action_arrays = np.array([b.city_action_array for b in batch])
    city_target_arrays = np.array([b.city_target_array for b in batch])
    maskings = np.array([b.masking for b in batch])
    agent_labels = np.array([b.agent_label for b in batch])

    return state_arrays, action_arrays, target_arrays, city_action_arrays, city_target_arrays, maskings, agent_labels
def _get_advantages(deltas):
    advantages = []

    #反向遍历deltas
    s = 0.0
    for delta in deltas[::-1]:
        s = 0.98 * 0.95 * s + delta
        advantages.append(s)

    #逆序
    advantages.reverse()
    return advantages
def main():
    seed_everything(SEED)
    writer = SummaryWriter('D:\LuxAI-main-ppo\log\log')
    train_obses_list, train_actions_list, valid_obses_list, valid_actions_list = [], [], [], []
    train_labels_list, valid_labels_list = [], []
    for i, (submission_id, team_name) in enumerate(zip(SUBMISSION_ID_LIST, TEAM_NAME_LIST)):
        episode_dir = f'{DATASET_PATH}/{submission_id}/'
        train_obses, valid_obses = \
            create_filepath_dataset_from_json(episode_dir, team_name=team_name, val_ratio=VAL_RATIO)
        train_obses_list.extend(train_obses)
        valid_obses_list.extend(valid_obses)
        for _ in range(len(train_obses)):
            train_labels_list.append(i)
        for _ in range(len(valid_obses)):
            valid_labels_list.append(i)
        print(f'submission ID: {submission_id}, train data: {len(train_obses)}, valid data: {len(valid_obses)}')

    device = torch.device('cuda:0')
    train_dataset = LuxDataset(train_obses_list, train_labels_list)
    valid_dataset = LuxDataset(valid_obses_list, valid_labels_list)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=False)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=False)

    torch.backends.cudnn.benchmark = True
    model = PolicyNetwork(feature_size=FEATURE_SIZE,num_unit_actions=UNIT_ACTIONS, num_citytile_actions=CITYTILE_ACTIONS,map_size= MAP_SIZE).cuda()
    checkpoint_policy = torch.load('D:\LuxAI-main-ppo\ppo_checkpoint.pth')
    model.load_state_dict(checkpoint_policy['model'])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    critic=Critic(feature_size=FEATURE_SIZE).cuda()
    optimizer_td = torch.optim.Adam(critic.parameters(),
                                            lr=1e-2,weight_decay=WEIGHT_DECAY)
    discriminator = Discriminator(feature_size=FEATURE_SIZE,num_actions=UNIT_ACTIONS+CITYTILE_ACTIONS).cuda()
    checkpoint_discriminator = torch.load('D:\LuxAI-main-ppo\discriminator_checkpoint.pth')
    discriminator.load_state_dict(checkpoint_discriminator['model'])
    discriminator.train()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4,weight_decay=WEIGHT_DECAY)
    teacher=PolicyNetwork_teacher(in_channels=STATE_CHANNELS, feature_size=FEATURE_SIZE, layers=LAYERS,
                               num_unit_actions=UNIT_ACTIONS, num_citytile_actions=CITYTILE_ACTIONS).cuda()
    checkpoint = torch.load('D:\LuxAI-main\checkpoint.pth')
    teacher.load_state_dict(checkpoint['model'])
    teacher.eval() 
    bce_loss = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    parser = argparse.ArgumentParser(description='Lux AI Evaluation')
    parser.add_argument('--num_games', '-n', help='the number of games', default=1, type=int)
    parser.add_argument('--map_size', '-s', help='the size of map', default=12,type=int)
    parser.add_argument('--seed', '-r', help='the random seed', default=42, type=int)
    args = parser.parse_args()
    up_count=1
    for epoch in range(NUM_EPOCHS):
        train_modes = ['train', 'valid'] if VAL_RATIO > 0. else ['train']
        for mode in train_modes:
            if mode == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = valid_dataloader

            epoch_loss, epoch_acc, epoch_targets = 0.0, 0, 0
            epoch_citytile_loss, epoch_citytile_acc, epoch_citytile_targets = 0.0, 0, 0
            weight = torch.Tensor([1., 1., 1., 1., CENTER_WEIGHT, 1., 1., 1., 1., 1.]).to(device)
            citytile_weight = torch.Tensor([1., 1., 1.]).to(device)
            control=0
            count=0
            
            for states, actions, unit_targets, citytile_actions, citytile_targets, maskings, labels in tqdm(dataloader):
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                unit_targets = torch.FloatTensor(unit_targets).to(device)
                citytile_actions = torch.LongTensor(citytile_actions).to(device)
                citytile_targets = torch.FloatTensor(citytile_targets).to(device)
                maskings = torch.BoolTensor(maskings).to(device)
                labels = torch.LongTensor(labels).to(device)
                teacher_unit_outs, teacher_citytile_outs = teacher.forward(states, maskings)              
                teacher_all_actions=torch.cat([teacher_unit_outs,teacher_citytile_outs],dim=-1)
                feature=get_state_f(teacher,states, maskings)
                unit_outs, citytile_outs = model.forward(feature)
                all_actions=torch.cat([unit_outs,citytile_outs],dim=-1)
                prob_teacher = discriminator(teacher_all_actions,feature)
                prob_student = discriminator(all_actions,feature)
                #老师的用0表示,学生的用1表示,计算二分类loss
                loss_teacher_d = bce_loss(prob_teacher, torch.zeros_like(prob_teacher))
                loss_student_d = bce_loss(prob_student, torch.ones_like(prob_student))
                control_back=prob_student.sum()/len(prob_student)
                control_back2=prob_teacher.sum()/len(prob_teacher)
                loss_d = loss_teacher_d + loss_student_d
                #调整鉴别器的loss                
                if control_back<0.75 or control_back2>0.35:
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
                    if control_back<0.75:
                        up_count+=0.05
                with torch.autograd.set_detect_anomaly(True):
                    if epoch>=0:
                        
                        ##################################################
                        rewards = (-(prob_student).log()*up_count).detach()
                        #critic_value_old
                        if control==0:
                            critic_value_old=torch.zeros([BATCH_SIZE,1]).to(device)
                            critic_value_old.requires_grad=True
                            #critic_value_old=critic(get_state_f(teacher,states, maskings))
                            
                            control+=1
                        else:
                            critic_value_old=critic(feature)
                        #[b, 4] -> [b, 1]
                        critic_value=critic(feature)
                        targets = critic_value * 0.98
                        targets = rewards+targets
                        
                        #[b, 1]
                        deltas = (targets - critic_value_old).squeeze(dim=1).tolist()
                        
                        advantages = []
                        #反向遍历deltas
                        s = 0.0
                        for delta in deltas[::-1]:
                            s = 0.98 * 0.95 * s + delta
                            advantages.append(s)

                        #逆序
                        advantages.reverse()
                        advantages = torch.FloatTensor(advantages).reshape(-1, 1).to(device)
                        advantages.requires_grad=True

                        (actions2, action_probs, _), (citytile_actions2, citytile_action_probs, _) = model.act(feature,states[:,25,:,:])
                        unit_action_old_porb=get_prob(actions2,action_probs,unit_targets)
                        citytile_action_old_porb=get_prob(citytile_actions2,citytile_action_probs,citytile_targets)
                        old_porb=unit_action_old_porb+citytile_action_old_porb
                        old_porb=old_porb.detach()
                        #每批数据反复训练10次
                        for _ in range(1):
                            if epoch%2==0:
                                (actions2, action_probs, _), (citytile_actions2, citytile_action_probs, _) = model.act(feature,states[:,25,:,:])
                                unit_action_new_porb=get_prob(actions2,action_probs,unit_targets)
                                citytile_action_new_porb=get_prob(citytile_actions2,citytile_action_probs,citytile_targets)
                                
                                
                                new_prob=unit_action_new_porb+citytile_action_new_porb+1e-16
                                ratios = old_porb / new_prob
                                surr1 = ratios * advantages
                                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                                loss_p = (-torch.min(surr1, surr2)).mean()
                            else:
                                unit_outs, citytile_outs = model.forward(feature)
                                unit_loss, unit_acc, num_unit_targets = take_target_loss(unit_outs, actions, unit_targets, weight)
                                citytile_loss, citytile_acc, num_citytile_targets = take_target_loss(citytile_outs, citytile_actions,
                                                                                                citytile_targets, citytile_weight)
                                loss_p = unit_loss + citytile_loss
                            

                            critic_value=critic(feature)
                            targets = critic_value * 0.98
                            loss_critic = loss_fn(critic_value_old.detach(), targets)

                            
                            #更新参数
                            optimizer_td.zero_grad()
                            loss_critic.backward()
                            optimizer_td.step()

                            optimizer.zero_grad()
                            loss_p.backward()
                            optimizer.step()
                        ####################################################
                    #teacher 的loss
                
                unit_loss, unit_acc, num_unit_targets = take_target_loss(unit_outs, actions, unit_targets, weight)
                citytile_loss, citytile_acc, num_citytile_targets = take_target_loss(citytile_outs, citytile_actions,
                                                                                    citytile_targets, citytile_weight)
                loss_p = unit_loss + citytile_loss

                epoch_citytile_loss += citytile_loss.item() * num_citytile_targets
                epoch_citytile_acc += citytile_acc * num_citytile_targets
                epoch_citytile_targets += num_citytile_targets
                # if epoch<5:
                #     optimizer.zero_grad()
                #     loss_p.backward()
                #     optimizer.step()
               
                epoch_loss += unit_loss.item() * num_unit_targets
                epoch_acc += unit_acc * num_unit_targets
                epoch_targets += num_unit_targets
                count+=1               
            epoch_loss = epoch_loss / epoch_targets
            epoch_acc = epoch_acc / epoch_targets
            epoch_citytile_loss = epoch_citytile_loss / epoch_citytile_targets
            epoch_citytile_acc = epoch_citytile_acc / epoch_citytile_targets
            print(
                f'Epoch {epoch + 1}/{NUM_EPOCHS} | {mode} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | '
                f'CLoss: {epoch_citytile_loss:.4f} | CACC: {epoch_citytile_acc:.4f}')

        scheduler.step()
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch
        }, f'ppo_checkpoint.pth')
        torch.save({
            'model': teacher.state_dict(),
            'epoch': epoch
        }, f'teacher_checkpoint.pth')
        torch.save({
            'model': discriminator.state_dict(),
            'epoch': epoch
        }, f'discriminator_checkpoint.pth')
        torch.save({
            'model': critic.state_dict(),
            'epoch': epoch
        }, f'critic_checkpoint.pth')
        if epoch%1==0:
                eval_env = EvalEnv(args)
                eval_env.run()

    model_name = f'model.pth'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.cpu().policy_net.state_dict(), model_name)
    else:
        torch.save(model.cpu().policy_net.state_dict(), model_name)


if __name__ == '__main__':
    main()