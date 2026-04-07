import os
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType, Tool
from text_analyzer_1 import TextAnalyzer
from llm_assistant import LLMAssistant

class ContentModerationAgent:
    """
    AI-агент на базе локальной LLM, который автоматически выбирает нужный инструмент
    для обработки текста пользователя.
    """
    def __init__(self, model_name="qwen2.5:1.5b"):
        print(f"[INFO] Инициализация AI-агента (Мозг: {model_name})...")
        
        # Подключаем локальную LLM через интеграцию LangChain с Ollama
        self.llm = Ollama(model=model_name, temperature=0.1)
        
        # Загружаем наши готовые инструменты из Части 1 и 3
        self.text_analyzer = TextAnalyzer()
        self.summarizer = LLMAssistant(model_name=model_name)
        
        # Регистрируем инструменты (Tools) для агента, чтобы он знал, что они умеют
        self.tools = [
            Tool(
                name="Sentiment_Analyzer",
                func=self._analyze_sentiment,
                description="Используй этот инструмент, когда нужно определить тональность (позитивный/негативный) или настроение текста."
            ),
            Tool(
                name="Text_Summarizer",
                func=self._summarize_text,
                description="Используй этот инструмент, когда нужно сделать краткую выжимку, пересказ или сократить длинный текст."
            )
        ]
        
        # Инициализируем агента типа ZERO_SHOT_REACT_DESCRIPTION
        # (Агент читает описание инструментов и решает, какой применить)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, # Включаем логирование, чтобы видеть мысли агента
            handle_parsing_errors=True # Защита от галлюцинаций парсинга
        )
        print("[INFO] Агент готов к работе!")

    def _analyze_sentiment(self, text: str) -> str:
        """Обертка для инструмента тональности"""
        result = self.text_analyzer.analyze(text)
        return f"Тональность текста: {result['sentiment']} (Уверенность: {result['confidence']}%)"

    def _summarize_text(self, text: str) -> str:
        """Обертка для инструмента суммаризации"""
        result = self.summarizer.summarize_text(text)
        return f"Краткая выжимка: {result.get('summary', 'Ошибка генерации')}"

    def run(self, user_query: str):
        """Запуск агента с промптом пользователя"""
        try:
            print(f"\n[USER]: {user_query}")
            response = self.agent.run(user_query)
            print(f"[AGENT]: {response}")
            return response
        except Exception as e:
            print(f"[ERROR] Ошибка агента: {e}")
            return str(e)


if __name__ == "__main__":
    agent = ContentModerationAgent()
    
    # Тест 1: Просим определить тональность (Агент должен выбрать Sentiment_Analyzer)
    agent.run("Определи тональность этого отзыва: 'Ужасный сервис, курьер опоздал на два часа и нагрубил мне!'")
    
    print("-" * 50)
    
    # Тест 2: Просим сделать выжимку (Агент должен выбрать Text_Summarizer)
    agent.run("Сделай краткую выжимку текста: 'Вчера я ходил в кино на новый фильм Марвел. Фильм шел три часа, спецэффекты были крутые, но сюжет немного затянут. В целом, мне понравилось, особенно игра главного актера. Советую сходить с друзьями на выходных.'")