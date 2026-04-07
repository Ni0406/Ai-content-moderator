from transformers import pipeline

class TextAnalyzer:
    """
    Класс для анализа тональности текста с использованием Hugging Face.
    """
    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-sentiment"):
        print(f"[INFO] Загрузка модели {model_name}...")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        print("[INFO] Модель успешно загружена!")

    def analyze(self, text: str) -> dict:
        """
        Метод принимает текст и возвращает его тональность и уверенность модели.
        """
        if not text.strip():
            return {"error": "Пустой текст"}

        result = self.sentiment_pipeline(text)[0]
        

        label = result['label']
        score = result['score']
        

        label_mapping = {
            "POSITIVE": "Позитивный",
            "NEGATIVE": "Негативный",
            "NEUTRAL": "Нейтральный"
        }
        
        return {
            "text": text,
            "sentiment": label_mapping.get(label, label), 
            # (value_1, value_2) value_1 - ключ; value_2 - затычка, если ключ после какой-нибудь обновы изменится
            "confidence": round(score * 100, 2) 
        }


if __name__ == "__main__":
    analyzer = TextAnalyzer()
    

    test_texts = [
        "Отличный сервис, мне всё очень понравилось! Буду рекомендовать друзьям.",
        "Ужасная доставка, курьер опоздал на 2 часа, а еда приехала холодной.",
        "Обычный продукт, ничего особенного, свою функцию выполняет."
    ]
    
    print("\n--- Результаты анализа текстов ---")
    for txt in test_texts:
        res = analyzer.analyze(txt)
        print(f"Текст: '{res['text']}'")
        print(f"Тональность: {res['sentiment']} (Уверенность: {res['confidence']}%)")
        print("-" * 30)