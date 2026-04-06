import os
from transformers import pipeline

class AudioAnalyzer:
    """
    Класс для транскрибации аудио (Speech-to-Text) с использованием Hugging Face и Whisper.
    """
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        print(f"[INFO] Загрузка модели аудио {model_name}...")
        # pipeline 'automatic-speech-recognition' сам обрабатывает аудиофайлы, 
        # нарезает их на куски (если длинные) и переводит в текст.
        # Модель tiny весит около 150 МБ и работает очень быстро на CPU.
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            chunk_length_s=30, 
        )
        print("[INFO] Модель аудио успешно загружена!")

    def analyze(self, audio_path: str) -> dict:
        """
        Метод принимает путь к аудиофайлу и возвращает распознанный текст.
        """
        if not os.path.exists(audio_path):
            return {"error": f"Файл не найден: {audio_path}"}

        try:
            # Делаем предсказание. Whisper сам поймет, что язык русский (автоопределение)
            result = self.asr_pipeline(audio_path, generate_kwargs={"task": "transcribe"})
            
            return {
                "file": os.path.basename(audio_path),
                "transcription": result["text"].strip()
            }
            
        except Exception as e:
            return {"error": str(e)}

# Блок тестирования
if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    
    # Путь к тестовому аудиофайлу
    test_audio_path = os.path.join("data", "input_audio.mp3")
    
    print(f"\n--- Результат транскрибации аудио: {test_audio_path} ---")
    result = analyzer.analyze(test_audio_path)
    
    if "error" in result:
        print(f"Ошибка: {result['error']}")
        print("Пожалуйста, положи аудиофайл 'input_audio.m4a' в папку 'data'")
    else:
        print(f"Распознанный текст:\n>> {result['transcription']}")