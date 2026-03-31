import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("Clé Mistral manquante dans le .env")

with Mistral(api_key=api_key) as client:
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Donne-moi une idée de sortie culturelle à Lille."
            }
        ]
    )

print(response.choices[0].message.content)