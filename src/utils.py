import numpy as np
import cv2

# стандартные координаты для выравнивания в системе 112x112
STANDARD_LANDMARKS = np.array([
    [38.2946, 51.6963],  # левый глаз (слева на изображении)
    [73.5318, 51.5014],  # правый глаз
    [56.0252, 71.7366],  # нос
    [41.5493, 92.3655],  # левый угол рта
    [70.7291, 92.2041]   # правый угол рта
])


def align_face(cropped_face, predicted_landmarks, output_size=(112, 112), ideal_landmarks=STANDARD_LANDMARKS):
    """
    Выравнивание лица по 5 ключевым точкам
    
    Args:
        cropped_face: обрезанное изображение лица [H, W, 3] в RGB, ожидается 112x112
        predicted_landmarks: предсказанные координаты [5, 2] в пикселях исходного изображения
        output_size: размер выходного изображения
        ideal_landmarks: идеальные координаты для выравнивания [5, 2]
    
    Returns:
        aligned_face: выровненное изображение [H, W, 3]
        aligned_landmarks: выровненные координаты [5, 2]
    """
    h, w = cropped_face.shape[:2]
    assert h == 112 and w == 112, f"Ожидается изображение 112x112, получено {w}x{h}"
    
    # используем 3 точки для стабильного аффинного преобразования (глаза + нос)
    src_points = predicted_landmarks[:3].astype(np.float32)
    dst_points = ideal_landmarks[:3].astype(np.float32)
    
    M = cv2.estimateAffinePartial2D(src_points, dst_points, confidence=0.99)[0]
    
    if M is None:
        # fallback: возвращаем исходное изображение без преобразования
        return cropped_face.copy(), predicted_landmarks.copy()
    
    # применяем преобразование
    aligned_face = cv2.warpAffine(
        cropped_face,
        M,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # вычисляем выровненные лендмарки
    aligned_landmarks = np.zeros_like(ideal_landmarks)
    for j in range(len(ideal_landmarks)):
        x, y = ideal_landmarks[j]
        aligned_landmarks[j] = np.dot(M, np.array([x, y, 1]))
    
    return aligned_face, aligned_landmarks


