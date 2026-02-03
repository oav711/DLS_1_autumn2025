#
# Модуль с моделями для распознавания лиц
# для проекта Face Recognition, DLS, 1 семестр, осень 2025 г.
#


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import math




# набор классов для реализации класса StackedHourglass

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)


class Hourglass(nn.Module):
    def __init__(self, depth, f, increase=0):
        """
        depth: глубина hourglass (≥1)
        f: число каналов
        increase: увелечение числа каналов на глубине
        """
        super().__init__()
        self.depth = depth
        nf = f + increase

        self.up1 = ResidualBlock(f, f)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = ResidualBlock(f, nf)

        # Рекурсия
        if depth > 1:
            self.low2 = Hourglass(depth - 1, nf, increase)
        else:
            self.low2 = ResidualBlock(nf, nf)

        self.low3 = ResidualBlock(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        
        # Приводим up2 к размерам up1 если необходимо
        if up1.size(2) != up2.size(2) or up1.size(3) != up2.size(3):
            up2 = F.interpolate(up2, size=(up1.size(2), up1.size(3)), mode='nearest')
            
        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.conv = nn.Conv2d(x_dim, y_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(y_dim)

    def forward(self, x):
        return self.bn(self.conv(x))


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return ((pred - gt) ** 2).mean()



class StackedHourglass(nn.Module):
    def __init__(self, nstack=2, inp_dim=128, oup_dim=5, increase=0):
        super().__init__()
        """
        nstack: число HourGlass блоков
        inp_dim: число входных каналов
        oup_dim: число выходных каналов (число keypoints)
        increase: увелечение числа каналов на глубине
        """

        self.nstack = nstack
        self.increase = increase
        self.oup_dim = oup_dim
        
        # Предобработка - УПРОЩЕННАЯ версия как у студента
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, inp_dim)
        )

        # Hourglass стеки
        self.hgs = nn.ModuleList([
            Hourglass(depth=3, f=inp_dim, increase=increase)
            for _ in range(nstack)
        ])

        # Слои обработки признаков
        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(inp_dim, inp_dim),
                nn.Conv2d(inp_dim, inp_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(inp_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(nstack)
        ])

        # Слои предсказания heatmap'ов
        self.outs = nn.ModuleList([
            nn.Conv2d(inp_dim, oup_dim, kernel_size=1)
            for _ in range(nstack)
        ])
        
        # Слои объединения для skip-соединений
        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for _ in range(nstack-1)
        ])
        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for _ in range(nstack-1)
        ])
        
        # Loss функция - ПРОСТОЙ MSE как у успешного студента
        self.heatmap_loss = HeatmapLoss()

    def forward(self, x):
        x = self.pre(x)  # [batch, inp_dim, 28, 28]
        combined_hm_preds = []
        
        for i in range(self.nstack):
            hg = self.hgs[i](x)  # [batch, inp_dim, 28, 28]
            feature = self.features[i](hg)  # [batch, inp_dim, 28, 28]
            preds = self.outs[i](feature)  # [batch, oup_dim, 28, 28]
            combined_hm_preds.append(preds)
            
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        
        # Возвращаем стеки как [batch, nstack, oup_dim, 28, 28]
        return torch.stack(combined_hm_preds, dim=1)

    def calc_loss(self, outputs, targets):
        """
        outputs: [batch, nstack, 5, 28, 28]
        targets: [batch, 5, 28, 28]
        """
        total_loss = 0.0
        nstacks = outputs.size(1)
        
        for i in range(nstacks):
            pred = outputs[:, i]  # [batch, 5, 28, 28]
            total_loss += self.heatmap_loss(pred, targets)
            
        return total_loss / nstacks

    def predict_landmarks(self, heatmaps):
        """
        ИСПРАВЛЕННАЯ функция преобразования heatmap -> координаты
        
        Args:
            heatmaps: [batch, 5, 28, 28] - предсказанные heatmap'ы
            
        Returns:
            landmarks: [batch, 5, 2] - координаты в системе 112x112
        """
        batch_size, num_landmarks, h, w = heatmaps.shape
        landmarks = torch.zeros((batch_size, num_landmarks, 2), device=heatmaps.device)
        
        for b in range(batch_size):
            for i in range(num_landmarks):
                hm = heatmaps[b, i]
                
                # Находим координаты максимума (hard argmax) - более стабильно для начальной стадии
                max_idx = torch.argmax(hm)
                y_hm = max_idx // w
                x_hm = max_idx % w
                
                # ПЕРЕСЧЕТ КООРДИНАТ: heatmap 28x28 -> изображение 112x112
                x_img = x_hm * (112.0 / 28.0)
                y_img = y_hm * (112.0 / 28.0)
                
                landmarks[b, i, 0] = x_img
                landmarks[b, i, 1] = y_img
        
        return landmarks




# модель для классификации, CE Loss
class FRModel(nn.Module):
    """
    Модель распознавания лиц на базе ResNet18 с заменой последнего слоя
    """
    def __init__(self, num_classes):
        super(FRModel, self).__init__()

        # предобученная ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # заменяем последний fully-connected слой (in_features = 512 для ResNet18
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # защищаемся от переобучения
            nn.Linear(in_features=512, out_features=num_classes)  # выходной слой для классификации
        )
    
    def forward(self, x):
        return self.model(x)
    





class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss для распознавания лиц
    """
    
    def __init__(self, in_features, out_features, s=40.0, m=0.57): # параметры s и m подобраны экспериментально
        """
        Args:
            in_features: размерность эмбеддингов
            out_features: количество классов
            s: масштабирующий коэффициент
            m: угловой отступ (margin)
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # веса классификатора (центры классов)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # инициализация весов
        
        # предвычисленные значения для эффективности
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # порог для обрезки
        self.mm = math.sin(math.pi - m) * m  # корректировка для обрезки
    

    def forward(self, inputs, labels):
        """
        Args:
            inputs: нормированные эмбеддинги [batch_size, in_features]
            labels: метки классов [batch_size]
        """
        # нормализация весов классификатора
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        
        # получаем синус зная косинус
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # угловой отступ: cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # обрезка для стабильности обучения
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # one-hot кодирование меток
        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # отступ только к целевому классу
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # масштабирование
        
        return output
    



# модель для ArcFace Loss
class FRModel_ArcFace(nn.Module):
    """
    ResNet18 + ArcFace для распознавания лиц
    """
    
    def __init__(self, num_classes, embedding_size=512):
        super(FRModel_ArcFace, self).__init__()
        
        # базовая модель - ResNet18 без последнего слоя
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # удаляем avgpool и fc
        
        # создаем слой для получения эмбеддингов (нормированных)
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # ArcFace слой
        self.arcface = ArcFaceLoss(embedding_size, num_classes)
    

    def forward(self, x, labels=None):
        """
        Args:
            x: входные изображения [batch_size, 3, 224, 224]
            labels: метки классов [batch_size] (требуются для обучения)
        """
        # получаем признаки
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        
        # получаем эмбеддинги
        embeddings = self.embedding_layer(x)
        
        if labels is not None:
            # для обучения: возвращаем логиты с ArcFace
            return self.arcface(embeddings, labels), embeddings
        else:
            # для инференса: возвращаем только эмбеддинги
            return embeddings




class TripletLoss(nn.Module):
    """Triplet Loss для обучения эмбеддингов лиц"""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    

    def forward(self, anchor, positive, negative):
        """
        Вычисляет triplet loss
        
        Args:
            anchor: эмбеддинги anchor изображений [batch_size, embedding_dim]
            positive: эмбеддинги positive изображений [batch_size, embedding_dim] 
            negative: эмбеддинги negative изображений [batch_size, embedding_dim]
            
        Returns:
            loss: scalar значение лосса
        """
        # нормализация эмбеддингов для стабильности
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Косинусное сходство
        cos_ap = F.cosine_similarity(anchor, positive, dim=1)
        cos_an = F.cosine_similarity(anchor, negative, dim=1)
        
        # Косинусное расстояние = 1 - сходство
        d_ap = 1 - cos_ap
        d_an = 1 - cos_an
        
        # Triplet loss: max(d_ap - d_an + margin, 0)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()
            
        return loss.mean()
    


# модель для Triplet Loss
class EmbeddingModel(nn.Module):
    """
    Модель для генерации эмбеддингов лиц на базе ResNet18
    
    Особенности:
    - Использует pretrained ResNet18 как backbone
    - Включает BatchNorm без обучаемых параметров для нормализации эмбеддингов
    - Выходные эмбеддинги имеют фиксированную длину (нормализованы)
    """
    def __init__(self, embedding_size=512):
        """
        Args:
            embedding_size: размерность выходных эмбеддингов
        """
        super().__init__()
        # загружаем pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # удаляем avgpool и fc слои
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # замораживаем слои backbone для увеличения скорости обучения и снижения переобучения
        for param in self.backbone.parameters():
            param.requires_grad = False

        # добавляем слои
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_size)
        # BatchNorm без обучаемых параметров для нормализации эмбеддингов
        self.bn = nn.BatchNorm1d(embedding_size, affine=False)

    

    def forward(self, x):
        """
        Прямой проход модели
        
        Args:
            x: входные изображения [batch_size, 3, 224, 224]
        
        Returns:
            x: нормализованные эмбеддинги [batch_size, embedding_size]
        """
        x = self.backbone(x)           # [batch_size, 512, 7, 7]
        x = self.avgpool(x)            # [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)        # [batch_size, 512]
        x = self.fc(x)                 # [batch_size, embedding_size]
        x = self.bn(x)                 # Нормализация до длины ~1.0
        return x
    

    def forward_triplet(self, anchor, positive, negative):
        """
        Прямой проход модели сразу по тройке изображений
        
        Args:
            anchor, positive, negative: изображения для трех компонентов триплета
        
        Returns:
            anchor_emb, pos_emb, neg_emb: эмбеддинги для всех трех компонентов
        """
        anchor_emb = self.forward(anchor)
        pos_emb = self.forward(positive) 
        neg_emb = self.forward(negative)
        return anchor_emb, pos_emb, neg_emb