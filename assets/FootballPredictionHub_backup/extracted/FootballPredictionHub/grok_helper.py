import os
import json
from openai import OpenAI

client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))

def analyze_widget_errors(error_data):
    """Widget hatalarını analiz eder ve çözüm önerileri sunar"""
    try:
        response = client.chat.completions.create(
            model="grok-2-1212",
            messages=[
                {
                    "role": "system",
                    "content": "API-Football widget hatalarını analiz et ve çözüm öner."
                },
                {
                    "role": "user",
                    "content": f"Bu widget hatası için çözüm öner: {json.dumps(error_data)}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata analizi yapılamadı: {str(e)}"

def analyze_match_data(match_data):
    """Maç verilerini analiz eder ve öngörüler sunar"""
    try:
        response = client.chat.completions.create(
            model="grok-2-1212",
            messages=[
                {
                    "role": "system",
                    "content": "Futbol maç verilerini analiz et ve öngörülerde bulun."
                },
                {
                    "role": "user",
                    "content": f"Bu maç verisi için analiz yap: {json.dumps(match_data)}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Maç analizi yapılamadı: {str(e)}"
