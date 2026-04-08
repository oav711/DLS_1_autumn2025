#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Многократное обучение CE и ArcFace с разными seed.

Примеры вызова:
    python train_multiseed.py --model-type ce --seeds 123,8,1000 --epochs 40 --output-dir robust_ce
    python train_multiseed.py --model-type arcface --seeds 123,8,1000 --epochs 40 --output-dir robust_arcface
"""

import os
import sys
import math
import json
import random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')



# ToTensor() для NumPy 2.x
def safe_to_tensor(pic):
    arr = np.array(pic, dtype=np.uint8)
    if arr.ndim == 3: arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32) / 255.0

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Lambda(safe_to_tensor), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(safe_to_tensor), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class FRDataset(Dataset):
    def __init__(self, metadata_file, images_dir, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.images_dir = images_dir
        self.transform = transform
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_id'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, int(row['class_id']) 


def get_loaders(data_dir, metadata_dir, batch_size=32):
    train_ds = FRDataset(os.path.join(metadata_dir, 'train_metadata.csv'), os.path.join(data_dir, 'train'), train_transform)
    val_ds   = FRDataset(os.path.join(metadata_dir, 'val_metadata.csv'),   os.path.join(data_dir, 'val'),   val_transform)
    test_ds  = FRDataset(os.path.join(metadata_dir, 'test_metadata.csv'),  os.path.join(data_dir, 'test'),  val_transform)
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
            len(train_ds.metadata['class_id'].unique()))


class FRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, num_classes))
    def forward(self, x): return self.model(x)


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=40.0, m=0.57):
        super().__init__()
        self.in_features, self.out_features, self.s, self.m = in_features, out_features, s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.s


class FRModel_ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_size=1024):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding_layer = nn.Sequential(nn.Linear(512, embedding_size), nn.BatchNorm1d(embedding_size))
        self.arcface = ArcFaceLoss(embedding_size, num_classes)
    
    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        embeddings = self.embedding_layer(x)
        if labels is not None:
            return self.arcface(embeddings, labels), embeddings
        return embeddings



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                device='cpu', num_epochs=1, start_epoch=0, history=None,
                is_arcface=False, checkpoint_dir='checkpoints', model_name='model'):
    if history is None:
        best_val_acc = 0.0
        history = {
            'train_losses': [], 
            'val_losses': [], 
            'train_accs': [], 
            'val_accs': [], 
            'current_lr': [],
            'best_val_acc': 0.0 
        }
    else: best_val_acc = history.get('best_val_acc', 0.0)
    best_model_path = None
    os.makedirs(checkpoint_dir, exist_ok=True)
        
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_str = f"Эпоха {epoch+1}/{start_epoch + num_epochs}"
        
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        batch_count = len(train_loader)
        train_pbar = tqdm(enumerate(train_loader), total=batch_count, desc=f"{epoch_str} | Обучение", bar_format='{l_bar}{bar:30}{r_bar}')
        
        for _, (images, labels) in train_pbar:
            images, labels = images.to(device), labels.to(device)
            if is_arcface:
                outputs, _ = model(images, labels) 
            else:
                outputs = model(images)
                        
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(preds==labels).sum().item()/labels.size(0):.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
            
        history['train_losses'].append(total_loss / batch_count)
        history['train_accs'].append(correct / total)

        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        batch_count = len(val_loader)
        val_pbar = tqdm(enumerate(val_loader), total=batch_count, desc=f"{epoch_str} | Валидация", bar_format='{l_bar}{bar:30}{r_bar}')
        
        with torch.no_grad():
            for _, (images, labels) in val_pbar:
                images, labels = images.to(device), labels.to(device)
                if is_arcface:
                    embeddings = model(images)
                    outputs = F.linear(F.normalize(embeddings), F.normalize(model.arcface.weight)) * model.arcface.s
                else: outputs = model(images)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{(preds==labels).sum().item()/labels.size(0):.4f}'})
                
        val_loss, val_acc = total_loss / batch_count, correct / total
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['current_lr'].append(optimizer.param_groups[0]['lr'])

        if scheduler: scheduler.step(val_acc)
        
        if (val_acc > best_val_acc) and (val_acc > 0.7):
            best_val_acc = val_acc
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(checkpoint_dir, f'{model_name}_{ts}_epoch_{epoch+1}_val_{best_val_acc:.4f}.pth')
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc, 'model_name': model_name}, best_model_path)
            print(f"Лучшая модель сохранена: Val Acc={val_acc:.4f}")
            history['best_model_path'] = best_model_path
            history['best_val_acc'] = best_val_acc
            
    return history



def test_model(model, test_loader, device, is_arcface=False):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Тестирование"):
            images, labels = images.to(device), labels.to(device)
            if is_arcface:
                emb = F.normalize(model(images))
                weights = F.normalize(model.arcface.weight)
                outputs = torch.mm(emb, weights.t()) * model.arcface.s
            else: outputs = model(images)
            
            _, preds = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    return {'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)}







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['ce', 'arcface'])
    parser.add_argument('--seeds', type=str, default='42,123,2024')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data-dir', type=str, default='data_CelebA_mini/data_fr')
    parser.add_argument('--metadata-dir', type=str, default='data_CelebA_mini/metadata_fr')
    parser.add_argument('--output-dir', type=str, default='output_multiseed')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n🔹 Запуск seed={seed} | Модель: {args.model_type.upper()} | Устройство: {device}")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

        train_loader, val_loader, test_loader, num_classes = get_loaders(args.data_dir, args.metadata_dir)
        
        if args.model_type == 'ce':
            model = FRModel(num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            is_arcface = False; model_prefix = 'ce'
        else:
            model = FRModel_ArcFace(num_classes, embedding_size=1024).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(list(model.backbone.parameters()) + list(model.embedding_layer.parameters()) + list(model.arcface.parameters()), lr=1e-4, weight_decay=5e-4)
            is_arcface = True; model_prefix = 'arcface'

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        run_dir = os.path.join(args.output_dir, f"{model_prefix}_seed{seed}")
        
        history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                              device=device, num_epochs=args.epochs, start_epoch=0, history=None,
                              is_arcface=is_arcface, checkpoint_dir=run_dir, model_name=model_prefix)
                              
        best_ckpt = history.get('best_model_path')

        if best_ckpt:
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            test_res = test_model(model, test_loader, device, is_arcface)
            print(f"Тест: Acc={test_res['accuracy']:.4f}, F1={test_res['f1']:.4f}")
            
            all_results.append({
                'model_type': args.model_type,
                'seed': seed, 
                'val_acc': history['best_val_acc'], 
                **test_res, 
                'ckpt': best_ckpt
            })
        else: 
            print("Лучшая модель не найдена (val_acc < 0.7)")
            
    if all_results:
        csv_path = os.path.join(args.output_dir, 'summary.csv')
        df_new = pd.DataFrame(all_results)
        
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['model_type', 'seed'], keep='last')
            df_combined = df_combined.sort_values(['model_type', 'seed']).reset_index(drop=True)
        else:
            df_combined = df_new
            
        print("\nИТОГОВАЯ СВОДКА (все накопленные запуски):")
        print(df_combined.drop(columns=['ckpt'], errors='ignore').to_string(index=False))
        
        df_combined.to_csv(csv_path, index=False)
        print(f"\nРезультаты сохранены/обновлены в {csv_path}")


if __name__ == "__main__":
    main()