#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
модуль пайплайна распознавания лиц

использование как скрипта:
    python pipeline.py --image test.jpg --output results/

использование в ноутбуке:
    from pipeline import FaceRecognitionPipeline
    pipeline = FaceRecognitionPipeline(device='cuda')
    embeddings = pipeline.recognize('test.jpg')
"""


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse
import pickle
import random


# пути к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'utils'))


# устанавливаем ядра рандомов
RANDOM_SEED = 4242
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)



# импорт детектора лиц - будем использовать mtcnn - хороший баланс скорости и качества
try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("ошибка: не установлен facenet-pytorch")
    print("установка: pip install facenet-pytorch --no-deps")
    sys.exit(1)


try:
    # импортируем наши модели и датасеты
    from src.models import StackedHourglass, FRModel, FRModel_ArcFace, EmbeddingModel
    from src.utils import align_face, STANDARD_LANDMARKS
except ImportError as e:
    print(f"ошибка импорта модулей: {e}")
    print("Cтруктура проекта должна быть такой:")
    print("DLS_1_project/")
    print("  ├── src/")
    print("  │   ├── models/")
    print("  │   └── utils.py")
    print("  └── pipeline.py")
    sys.exit(1)




class FR_Pipeline:
    """Класс pipeline распознавания лиц: детекция -> лендмарки -> выравнивание -> эмбеддинги"""
    
    def __init__(self, hourglass_state_path=None, arcface_state_path=None, device='None'):
        """
        Инициализация pipeline
        
        Args:
            hourglass_state_path: путь к чекпоинту модели лендмарков
            arcface_state_path: путь к чекпоинту модели arcface
            device: устройство для вычислений ('cuda' или 'cpu')
        """
        # определяем устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Вычислительное устройство: {self.device}")
        
        # пути к чекпоинтам/весам лучших моделей
        self.hourglass_state_path = hourglass_state_path
        self.arcface_state_path = arcface_state_path
        
        # загрузка моделей
        self.face_detector = self._load_face_detector()
        self.hourglass = self._load_hourglass_model()
        self.arcface = self._load_arcface_model()
        
        # трансформации
        self.transform_hourglass = transforms.Compose([
            #transforms.Resize((112, 112)), # после детектора лиц mtcnn изображения уже 112х112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_arcface = transforms.Compose([
            transforms.Resize((224, 224)), # т.к. мы обучали ArcFace на 224х224 (ха-ха, я только учусь, переучивать сеть не хочется ;))
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    
    def _load_face_detector(self):
        """загрузка детектора лиц mtcnn"""
        # первый запуск может быть долгим, т.к. веса модели загружаются из интернета и кэшируются в ~/.facenet_pytorch/
        model = MTCNN(
            image_size=112, # размер выходного изображения
            margin=20, # отступы он детектированного bbox'а
            min_face_size=40, # минимальный размер лица
            thresholds=[0.6, 0.7, 0.7], # пороги для трех сетей каскада mtcnn - P-Net, R-Net, O-Net
            factor=0.709, # коэффициент масштабирования пирамиды изображений (эмпирически оптимальное значение из оригинальной статьи MTCNN)
            post_process=True, # применение нормализации к выходным изображениям
            device=self.device,
            keep_all=True # возвращать все обнаруженные лица с разной степенью уверенности (если False, то вернется только одно лицо, в котором детектор уверен больше всего)
        )
        print("Модель детектора mtcnn создана!")
        return model
    

    def _load_hourglass_model(self):
        """загрузка модели StackedHourglass для предсказания heatmaps (и landmarks, 5 ключевых точек)"""
        if not os.path.exists(self.hourglass_state_path):
            raise FileNotFoundError(f"Не найдены веса StackedHourglass модели: {self.hourglass_state_path}")
        
        model = StackedHourglass(nstack=2, inp_dim=128, oup_dim=5, increase=0) # модель создаем такую же как и обучали, иначе веса не загрузятся
        checkpoint = torch.load(self.hourglass_state_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device).eval()
        print(f"Модель StackedHourglass создана, веса загружены: {self.hourglass_state_path}")
        return model
    

    def _load_arcface_model(self):
        """загрузка модели ArcFace для вычисления эмбеддинга изображения"""
        if not os.path.exists(self.arcface_state_path):
            raise FileNotFoundError(f"Не найдены веса ArcFace модели: {self.arcface_state_path}")
        
        model = FRModel_ArcFace(num_classes=440, embedding_size=1024)
        checkpoint = torch.load(self.arcface_state_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device).eval()
        print(f"Модель ArcFace создана, веса загружены: {self.arcface_state_path}")
        return model


    def denormalize(self, img_tensor):
        """Преобразует нормализованное изображение [0, 1] в [0, 255]"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img_tensor.numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1) * 255
        return img.astype(np.uint8)
    

    def denormalize_MTCNN(self, img_tensor):
        """
        Денормализация для выхода MTCNN с post_process=True
        
        Вход: тензор [3, H, W] в диапазоне [-1, 1]
        Выход: numpy массив [H, W, 3] в диапазоне [0, 255], uint8
        """
        # Обратное преобразование: [-1, 1] → [0, 1]
        img_tensor = img_tensor * 0.5 + 0.5
        # Преобразование в [0, 255]
        img_tensor = img_tensor * 255
        # Ограничение диапазона и конвертация в uint8
        img_tensor = img_tensor.clamp(0, 255).byte()
        # Перестановка размерностей: [3, H, W] → [H, W, 3]
        return img_tensor.permute(1, 2, 0).numpy()


    def recognize(self, image_path, output_dir='output'):
        """
        Полный pipeline распознавания лиц
        
        Args:
            image_path: путь к входному изображению
            output_dir: папка для сохранения результатов
        
        Returns: 
            словарь с результатами:
                embeddings: список эмбеддингов [(512,), ...]
                aligned_faces: список выровненных изображений [np.array, ...]
                landmarks: список лендмарков [[5, 2], ...]
                face_filenames: список имен файлов кропнутых лиц ['obama_1.jpg', ...]
                num_faces: количество обнаруженных лиц
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        # загрузка исходного изображения
        img = Image.open(image_path).convert('RGB')
        # сохраняем имя для формирования имен файлов детектированных лиц
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # детекция лиц (выход: тензоры 112x112 благодаря image_size=112 в mtcnn)
        faces_cropped = self.face_detector(img)
        if faces_cropped is None:
            print(f"На изображении  {image_path}  лица не обнаружены!")
            return {
                'embeddings': [], 
                'aligned_faces': [], 
                'landmarks': [], 
                'face_filenames': [],
                'num_faces': 0
            }
        
        # наш детектор может возвращать лица в разном формате, поэтому нужно проверять тип возврата для дальнейшей работы
        if not isinstance(faces_cropped, list):
            # если это тензор с батчем лиц (4D)
            if faces_cropped.dim() == 4:
                # создаем список отдельных лиц
                faces_cropped = [faces_cropped[i] for i in range(faces_cropped.size(0))]
            # если же это тензор с одним лицом (3D)
            elif faces_cropped.dim() == 3:
                # создаем список с одним лицом
                faces_cropped = [faces_cropped]
            # в остальных случаях
            else:
                raise ValueError(f"Неожиданный размер тензора: {faces_cropped.shape}")

        print(f"На изображении  {image_path}   обнаружено {len(faces_cropped)} лиц.")
        
        embeddings = []
        aligned_faces = []
        landmarks_list = []
        face_filenames = []
                
        save_results = bool(output_dir.strip())
        if save_results:
            os.makedirs(output_dir.strip(), exist_ok=True)

        for i, face_tensor in enumerate(faces_cropped):
            # конвертируем тензор в PIL изображение
            face_pil = transforms.ToPILImage()(face_tensor.cpu())
            # сохраняем кропнутое лицо
            face_filename = f"{img_basename}_{i+1}.jpg"
            face_filenames.append(face_filename)            
            if save_results:
                face_path = os.path.join(output_dir, face_filename)
                Image.fromarray(self.denormalize_MTCNN(face_tensor.cpu())).save(face_path)  # сохраняем изображение в [0, 255]
            
            # трансформацию для StackedHourglass
            face_tensor_hg = self.transform_hourglass(face_pil).unsqueeze(0).to(self.device)
            
            # получаем heatmaps            
            with torch.no_grad():
                outputs = self.hourglass(face_tensor_hg)  # [1, 2, 5, 28, 28] при nstack=2
                # извлекаем хитмапы последнего стека StackedHourglass
                heatmaps = outputs[:, -1]  # [1, 5, 28, 28]
                # преобразуем в координаты landmarks через метод нашей модели
                landmarks = self.hourglass.predict_landmarks(heatmaps)  # [1, 5, 2]
            
            landmarks = landmarks[0].cpu().numpy()  # [5, 2]    переносим на cpu и в numpy так как сама модель на GPU и результат обработки тоже остается на GPU
            landmarks_list.append(landmarks)
            
            # выравниваем лицо по landmarks
            face_np = np.array(face_pil)
            aligned_face, _ = align_face(face_np, landmarks, output_size=(112, 112), ideal_landmarks=STANDARD_LANDMARKS)
            aligned_faces.append(aligned_face)
            
            # получаем эмбеддинг через ArcFace
            face_tensor_arc = self.transform_arcface(Image.fromarray(aligned_face)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.arcface(face_tensor_arc)
            
            # нормализуем эмбеддинг
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0])
            
            if save_results:
                print(f"\tлицо {i+1} сохранено в {face_filename};   эмбеддинг получен.")
        

        # сохраняем результаты
        if save_results and embeddings:
            embeddings_path = os.path.join(output_dir, f"{img_basename}_embeddings.pkl")
            with open(embeddings_path, 'wb') as f:
                pickle.dump({
                    'image_path': image_path,
                    'face_filenames': face_filenames,
                    'embeddings': embeddings,
                    'num_faces': len(embeddings)
                }, f)
            print(f"Эмбеддинги сохранены: {os.path.basename(embeddings_path)}")
        
        return {
            'embeddings': embeddings,
            'aligned_faces': aligned_faces,
            'landmarks': landmarks_list,
            'face_filenames': face_filenames,
            'num_faces': len(embeddings)
        }


def compare_faces(embeddings1, embeddings2, threshold=0.5):
    """
    Сравнивает два набора эмбеддингов изображений по косинусному сходству.
    
    Args:
        embeddings1: список нормализованных эмбеддингов лиц, найденных на первом изображении
        embeddings2: список нормализованных эмбеддингов лиц, найденных на втором изображении
        threshold: порог косинусного сходства для определения одного человека
    
    Returns:
        словарь с результатами:
            - 'similarity_matrix': матрица косинусных сходств [n1, n2]
            - 'decision_matrix': матрица решений (True = один человек)
            - 'best_pair': кортеж (i, j, similarity) для самой похожей пары
    """
    n1 = len(embeddings1)
    n2 = len(embeddings2)
    
    if n1 == 0 or n2 == 0:
        return {
            'similarity_matrix': np.zeros((n1, n2)),
            'decision_matrix': np.zeros((n1, n2), dtype=bool),
            'best_pair': None
        }
    
    # матрица косинусных сходств
    similarity_matrix = np.zeros((n1, n2))
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            # косинусное сходство для нормализованных векторов - это их скалярное произведение
            similarity = np.dot(emb1, emb2)  # = cos(theta)
            similarity_matrix[i, j] = similarity
    
    # принимаем решение на основе порога
    decision_matrix = similarity_matrix >= threshold
    
    # самая похожая пара лиц
    max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    best_pair = (max_idx[0], max_idx[1], similarity_matrix[max_idx])
    
    return {
        'similarity_matrix': similarity_matrix,
        'decision_matrix': decision_matrix,
        'best_pair': best_pair,
        'threshold': threshold
    }


def visualize_similarity_matrix(result1, result2, output_dir=None, show=True, threshold=0.5):
    """
    Визуализация матрицы сходства с изображениями лиц
    
    Args:
        result1, result2: результаты метода recognize() для двух изображений
        output_dir: директория для сохранения визуализации (опционально)
    """
    # сравниваем эмбеддинги
    comparison = compare_faces(
        result1['embeddings'], 
        result2['embeddings'], 
        threshold=threshold
    )

    sim_matrix = comparison['similarity_matrix']
    n1, n2 = sim_matrix.shape
    threshold = comparison['threshold']
    
    if n1 == 0 or n2 == 0:
        print("Невозможно визуализировать: одно или оба изображения не содержат лиц")
        return None
            
    
    # Выводим ранжированные пары
    print("\nРанжированные пары лиц по сходству:")
    pairs = []
    for i in range(n1):
        for j in range(n2):
            if result1 is result2 and i >= j: # если анализируются лица с одного изображения, то формируем только верхний угол матрицы без диагональных элементов
                continue
            pairs.append((i, j, sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True) # сортируем по убыванию сходства
    
    for rank, (i, j, sim) in enumerate(pairs[:min(10, len(pairs))], 1):
        if sim >= threshold: # считаем, что это один и тот же человек
            status = "один человек"  
        elif (sim < 0.6 * threshold) | (sim < 0.6):  # совсем не похожи друг на друга, гарантированно разные люди
            status = "разные люди"
        else: # скорее всего это разные люди, но очень похожи друг на друга
            status = "очень похожи"
        print(f"{rank}. {result1['face_filenames'][i]} <-> {result2['face_filenames'][j]}   \tСходство: {sim:.4f} -> {status}")

    
    # визуализируем лица и их сходство
    fig = plt.figure(figsize=(2 * n2 + 1, 2 * n1 + 1))
    gs = fig.add_gridspec(n1 + 1, n2 + 1, hspace=0.1, wspace=0.2)
    
    # изображения первого набора - слева
    for i in range(n1):
        ax = fig.add_subplot(gs[i+1, 0])
        img_path = os.path.join(output_dir or '.', result1['face_filenames'][i])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
        ax.set_title(f"{os.path.splitext(result1['face_filenames'][i])[0]}", fontsize=8)
        ax.axis('off')
    
    # изображения второго набора - сверху
    for j in range(n2):
        ax = fig.add_subplot(gs[0, j+1])
        img_path = os.path.join(output_dir or '.', result2['face_filenames'][j])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
        ax.set_title(f"{os.path.splitext(result2['face_filenames'][j])[0]}", fontsize=8)
        ax.axis('off')
    
    # матрица сходства
    for i in range(n1):
        for j in range(n2):
            ax = fig.add_subplot(gs[i+1, j+1])
            sim = sim_matrix[i, j]
            
            # цвет назначаем в зависимости от сходства
            if sim >= threshold: # считаем, что это один и тот же человек
                color = 'green'
                text_color = 'white'
            elif (sim < 0.6 * threshold) | (sim < 0.6): # совсем не похожи друг на друга, гарантированно разные люди
                color = 'red'
                text_color = 'white'
            else: # скорее всего это разные люди, но очень похожи друг на друга
                color = 'orange'
                text_color = 'black'
            
            ax.text(0.5, 0.5, f"{sim:.2f}", 
                   ha='center', va='center', fontsize=14, 
                   color=text_color,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
            ax.axis('off')
    
    plt.suptitle(f'Матрица сходства лиц (порог={threshold})', fontsize=16, y=0.91)
    
    if output_dir:
        os.makedirs(output_dir.strip(), exist_ok=True)
        viz_path = os.path.join(output_dir, 'similarity_matrix.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {viz_path}")
    
    if show:
        plt.show()
    else:
        plt.close()



def main():
    """Функция - точка входа для запуска модуля как программы из терминала"""
    
    parser = argparse.ArgumentParser(
        description='Пайплайн распознавания лиц: детекция -> лендмарки -> выравнивание -> эмбеддинги',
        epilog='''
Примеры использования:
  # Анализ лиц на одном изображении:
  python pipeline_oav.py --image image1.jpg --output output/compare_img1 --threshold 0.8
  
  # Сравнение лиц между двумя изображениями:
  python pipeline_oav.py --image image2.jpg --image2 image3.jpg --output output/compare_img2_img3 --threshold 0.8
  
  # Режим терминала (без графического вывода):
  python pipeline_oav.py --image image1.jpg --image2 image2.jpg --output output/compare_img1_img2 --no-show
  
  # Режим терминала без явного указания папки с результатами (генерируется случайно), а матрица соответсвия выводится в графическом окне:
  python pipeline_oav.py --image image2.jpg --threshold 0.8
''',
        formatter_class=argparse.RawTextHelpFormatter  # Сохраняет переносы строк в epilog
    )


    parser.add_argument('--image', 
                        type=str, 
                        required=True, 
                        help='Путь к входному изображению (для сравнения лиц на одном изображении)')
    parser.add_argument('--image2', 
                        type=str, 
                        help='Путь ко второму изображению (для сравнения двух изображений)')    
    parser.add_argument('--hourglass', 
                        type=str, 
                        default='checkpoints/StackedHourglass_best_model_nme_20260123_233506_epoch_64_nme_5.60.pth',
                        help='Файл с весами модели StackedHourglass')
    parser.add_argument('--arcface', 
                        type=str, 
                        default='checkpoints/arcface_model_20260126_163544_epoch_38_val_0.7848.pth',
                        help='Файл с весами модели StackedHourglass')
    parser.add_argument('--output', 
                        type=str, 
                        default=f'output{random.randint(1, 1e5)}', 
                        help='Папка для сохранения результатов (кропы, эмбеддинги, визуализация). Если не указана — ничего не сохраняется.')
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda', 
                        choices=['cuda', 'cpu'], help='Устройство для вычислений')
    parser.add_argument('--no-show', 
                        action='store_true',
                        help='Не отображать графики на экране (только сохранение в файл при --output)')
    parser.add_argument('--threshold', 
                        type=float, 
                        default=0.5,
                        help='порог косинусного сходства (0.0-1.0)')
    
    
    args = parser.parse_args()
    

    # валидация аргументов
    if not (args.image or (args.image and args.image2)):
        print("Ошибка: укажите либо --image (одно изображение), либо --image1 и --image2 (два изображения)")
        sys.exit(1)
    
    if not args.image:
        print("Ошибка: --image обязателен, для двух изображений укажите --image и --image2")
        sys.exit(1)
    
    # определяем режим работы - обрабатываем одно или два изображения
    single_image_mode = args.image2 is None # если нет второго изображения, то "простой" режим
    image1_path = args.image
    image2_path = None if single_image_mode else args.image2
    
    # создаем пайплайн
    try:
        pipeline = FR_Pipeline(
            hourglass_state_path=args.hourglass,
            arcface_state_path=args.arcface,
            device=args.device
        )
    except Exception as e:
        print(f"Ошибка инициализации пайплайна: {e}")
        sys.exit(1)
    

    # распознавание на первом изображении выполняется в любом случае
    try:
        print(f"Распознавание лиц на изображении: {image1_path}")
        result1 = pipeline.recognize(
            image1_path,
            output_dir=args.output
        )
        
        if result1['num_faces'] == 0:
            print("\tлица на изображении не обнаружены!")
            sys.exit(0)
        
        print(f"\tобнаружено {result1['num_faces']} лиц")
        
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        sys.exit(1)
    

    # распознавание на втором изображении (если режим двух изображений)
    if not single_image_mode:
        try:
            print(f"\nРаспознавание лиц на изображении: {image2_path}")
            result2 = pipeline.recognize(
                image2_path,
                output_dir=args.output
            )
            
            if result2['num_faces'] == 0:
                print("\tлица на изображении не обнаружены!")
                sys.exit(0)
            
            print(f"\tобнаружено {result2['num_faces']} лиц")
            
        except Exception as e:
            print(f"Ошибка обработки второго изображения: {e}")
            sys.exit(1)
    else:
        # если режим простой, то будем сравнивать два одинаковых изображения между собой
        result2 = result1
    

    # сравнение эмбеддингов
    comparison = compare_faces(result1['embeddings'], result2['embeddings'], threshold=args.threshold)
    
    if single_image_mode:
        print(f"\nСравнение лиц на одном изображении с порогом сходства={args.threshold}")
        
        # формируем попарные сравнения
        pairs = []
        for i in range(result1['num_faces']):
            for j in range(i+1, result1['num_faces']): # только уникальные пары (без диагонали), поэтому i+1
                sim = comparison['similarity_matrix'][i, j]
                pairs.append((i, j, sim))
        
        if not pairs:
            print("Недостаточно лиц для сравнения, нужно минимум 2.")
        else:
            pairs.sort(key=lambda x: x[2], reverse=True) # сортируем по убыванию сходства
            for rank, (i, j, sim) in enumerate(pairs[:min(5, len(pairs))], 1):
                status = "один человек" if sim >= args.threshold else "разные люди"
                print(f"\t{rank}. {result1['face_filenames'][i]} <-> {result1['face_filenames'][j]} \tсходство={sim:.4f} -> {status}")
    else:
        print("\nСравнение лиц между изображениями:")
        best = comparison['best_pair']
        if best:
            status = "один человек" if best[2] >= args.threshold else "разные люди"
            print(f"Самая похожая пара:")
            print(f"\t{result1['face_filenames'][best[0]]} ↔ {result2['face_filenames'][best[1]]} \tсходство: {best[2]:.4f} → {status}")
    
    # визуализация матрицы сходства
    visualize_similarity_matrix(result1, result2, output_dir=args.output, show=not args.no_show)
    
    print("\nОбработка завершена.")




if __name__ == "__main__":
    main()