from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_read_root():
    """Проверяем доступность главной страницы"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API работает! Отправьте POST запрос на /analyze или /summarize"}

def test_analyze_positive_text():
    """Проверяем успешную обработку текста анализатором тональности"""
    response = client.post("/analyze", json={"text": "Я обожаю этот продукт, всё просто супер!"})
    
    assert response.status_code == 200
    
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_analyze_empty_text():
    """Проверяем обработку пустого текста"""
    response = client.post("/analyze", json={"text": "   "})
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Текст не может быть пустым"}


def test_summarize_empty_text():
    """Проверяем обработку пустого текста для суммаризации"""
    response = client.post("/summarize", json={"text": ""})
    
    assert response.status_code == 400
    assert response.json() == {"detail": "Текст не может быть пустым"}