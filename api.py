from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from text_analyzer_1 import TextAnalyzer
from llm_assistant import LLMAssistant 

app = FastAPI(
    title="AI Content Moderator API",
    description="API для анализа контента и суммаризации текста",
    version="1.0.0"
)

text_analyzer = TextAnalyzer()
llm_assistant = LLMAssistant()

#Проверка что text: str
class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(request: TextRequest):
    """Принимает текст и возвращает его тональность"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    return text_analyzer.analyze(request.text)

@app.post("/summarize")
async def summarize_text(request: TextRequest):
    """Принимает текст и возвращает краткую выжимку от LLM"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    result = llm_assistant.summarize_text(request.text)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

@app.get("/")
async def root():
    return {"message": "API работает! Отправьте POST запрос на /analyze или /summarize"}
