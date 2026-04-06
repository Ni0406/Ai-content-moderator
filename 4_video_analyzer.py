import os
import cv2
import torch
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

class VideoAnalyzer:
    """
    Класс для детекции объектов на видео с использованием PyTorch и YOLOv5.
    """
    def __init__(self):
        print("[INFO] Загрузка модели YOLOv5s из PyTorch Hub...")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[INFO] Используем устройство: {device}")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        print("[INFO] Модель видео успешно загружена!")

    def analyze(self, input_path: str, output_path: str) -> dict:
        """
        Метод читает видео, применяет YOLO к кадрам и сохраняет новое видео с рамками.
        """
        if not os.path.exists(input_path):
            return {"error": f"Файл не найден: {input_path}"}

        try:
            cap = cv2.VideoCapture(input_path)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"[INFO] Начинаем обработку видео ({total_frames} кадров)...")
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break 
                
                # OpenCV читает картинки задом наперед (Синий-Зеленый-Красный), А YOLOv5 адекватная
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                

                results = self.model(img_rgb)
                

                rendered_frame_rgb = results.render()[0]
                

                rendered_frame_bgr = cv2.cvtColor(rendered_frame_rgb, cv2.COLOR_RGB2BGR)
                
                out.write(rendered_frame_bgr)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Обраработано кадров: {frame_count} / {total_frames}")


            cap.release()
            out.release()
            
            return {
                "status": "success",
                "input_file": input_path,
                "output_file": output_path,
                "processed_frames": frame_count
            }
            
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    
    input_video = os.path.join("data", "input_video.mov")
    output_video = os.path.join("data", "output_video.mp4")
    
    print(f"\n--- Анализ видео: {input_video} ---")
    result = analyzer.analyze(input_video, output_video)
    
    if "error" in result:
        print(f"Ошибка: {result['error']}")
    else:
        print(f"Готово! Обработано кадров: {result['processed_frames']}.")
        print(f"Результат с нарисованными рамками сохранен в: {result['output_file']}")