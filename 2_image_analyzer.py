# 2_image_analyzer.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

class ImageAnalyzer:
    """
    Класс для классификации изображений с использованием TensorFlow/Keras (MobileNetV2).
    """
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        
        print("[INFO] Загрузка модели MobileNetV2...")
        self.model = MobileNetV2(weights='imagenet')
        print("[INFO] Модель успешно загружена!")

    def analyze(self, image_path: str, top_k: int = 3) -> dict:
        """
        Метод принимает путь к картинке и возвращает топ-K предсказанных классов.
        """
        if not os.path.exists(image_path):
            return {"error": f"Файл не найден: {image_path}"}

        try:
            img = image.load_img(image_path, target_size=(224, 224))
            
            img_array = image.img_to_array(img)
            
            img_array_expanded = np.expand_dims(img_array, axis=0)
            
            #от -1 до 1
            preprocessed_img = preprocess_input(img_array_expanded)
            

            predictions = self.model.predict(preprocessed_img, verbose=0)
            

            decoded_preds = decode_predictions(predictions, top=top_k)[0]
            

            results = []
            for _, label, prob in decoded_preds:
                results.append({
                    "class": label.replace("_", " ").capitalize(),
                    "probability": round(prob * 100, 2)
                })
                
            return {
                "file": os.path.basename(image_path),
                "predictions": results
            }
            
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    analyzer = ImageAnalyzer()
    
    data_dir = "data/photo"
    if not os.path.exists(data_dir):
        print(f"Ошибка: Папка '{data_dir}' не найдена.")
    else:
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(data_dir, filename)
                
                print(f"\n--- Анализ изображения: {filename} ---")
                result = analyzer.analyze(img_path)
                
                if "error" in result:
                    print(f"Ошибка: {result['error']}")
                else:
                    for idx, pred in enumerate(result['predictions'], 1):
                        print(f"{idx}. Объект: {pred['class']} | Вероятность: {pred['probability']}%")