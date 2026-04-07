import os
from langfuse.langchain import CallbackHandler
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import Tool
from text_analyzer_1 import TextAnalyzer
from llm_assistant import LLMAssistant

class ContentModerationAgent:
    """
    AI-агент на базе локальной LLM, который автоматически выбирает нужный инструмент
    для обработки текста пользователя.
    """
    def __init__(self, model_name="qwen2.5:1.5b"):
        print(f"[INFO] Инициализация AI-агента (Мозг: {model_name})...")
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-5fef7f1f-2384-435e-9061-366ead0c3df2"
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-93097d4c-ecb4-4843-8262-28edb5a0d152"
        os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
        
        self.langfuse_handler = CallbackHandler()

        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.text_analyzer = TextAnalyzer()
        self.summarizer = LLMAssistant(model_name=model_name)
        
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
        
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools
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

            response = self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Ты AI-агент модерации. "
                                "Отвечай только на русском языке. "
                                "Не показывай пользователю служебные сообщения, tool calls, "
                                "промежуточные шаги и внутреннюю трассировку. "
                                "Дай только итоговый ответ."
                            )
                        },
                        {
                            "role": "user",
                            "content": user_query
                        }
                    ]
                },
                config={"callbacks": [self.langfuse_handler]}  
            )

            messages = response["messages"]

            final_answer = "None"
            for msg in reversed(messages):
                if msg.__class__.__name__ == "AIMessage" and msg.content:
                    # msg.content: защищает нас от пустых системных сообщений:
                    final_answer = msg.content
                    break

            print(f"[AGENT]: {final_answer}")
            return final_answer

        except Exception as e:
            print(f"[ERROR] Ошибка агента: {e}")
            return str(e)


if __name__ == "__main__":
    agent = ContentModerationAgent()
    
    agent.run("Определи тональность этого отзыва: 'Ужасный сервис, курьер опоздал на два часа и нагрубил мне!'")
    print("-" * 50)
    agent.run("Сделай краткую выжимку текста: 'Вчера я ходил в кино на новый фильм Марвел. " \
            "Фильм шел три часа, спецэффекты были крутые, но сюжет немного затянут. " \
            "В целом, мне понравилось, особенно игра главного актера. Советую сходить с друзьями на выходных.'")