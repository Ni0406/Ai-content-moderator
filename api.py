from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from text_analyzer_1 import TextAnalyzer

# Инициализация приложения и модели
app = FastAPI(
    title="AI Content Moderator API",
    description="API для анализа тональности текста",
    version="1.0.0"
)

analyzer = TextAnalyzer()

# Схема входящих данных
class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(request: TextRequest):
    """Принимает текст и возвращает его тональность"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    result = analyzer.analyze(request.text)
    return result

@app.get("/")
async def root():
    """Тестовый маршрут проверки доступности"""
    return {"message": "API работает! Отправьте POST запрос на /analyze"}