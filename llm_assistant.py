import ollama

class LLMAssistant:
    """
    Класс для работы с локальной LLM через приложение Ollama.
    Идеально оптимизировано для Apple Silicon (M1/M2/M3).
    """
    def __init__(self, model_name="qwen2.5:1.5b"):
        self.model_name = model_name
        print(f"[INFO] Подключение к локальной LLM (Ollama: {model_name})...")

    def summarize_text(self, text: str) -> dict:
        """
        Метод отправляет текст в локальную нейросеть для краткой выжимки.
        """
        if not text.strip():
            return {"error": "Текст пуст."}

        try:
            print("[INFO] LLM генерирует ответ (используется Metal API Mac)...")
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'system',
                    'content': 'Ты полезный модератор-ассистент. Отвечай кратко, на русском языке.'
                },
                {
                    'role': 'user',
                    'content': f'Сделай краткую выжимку этого текста (1-2 предложения):\n\n{text}',
                },
            ])
            
            return {
                "original_length": len(text),
                "summary": response['message']['content']
            }
        except Exception as e:
            return {"error": f"Ошибка Ollama: {str(e)}. Убедитесь, что приложение Ollama запущено."}

if __name__ == "__main__":
    assistant = LLMAssistant()
    
    sample_text = (
        "Всем привет! Вчера заказывал доставку продуктов на дом. Курьер приехал вовремя, "
        "всё было аккуратно упаковано. Однако, помидоры оказались немного помятыми, "
        "а молоко было со сроком годности, который истекает уже завтра. "
        "В целом, нормально, но хотелось бы больше внимания к свежести продуктов. "
        "Поддержка ответила быстро и дала промокод на следующий заказ, это плюс."
    )
    
    print("\n--- Исходный текст ---")
    print(sample_text)
    
    print("\n--- Выжимка от локальной LLM ---")
    result = assistant.summarize_text(sample_text)
    
    if "error" in result:
        print(result["error"])
    else:
        print(f">> {result['summary']}")