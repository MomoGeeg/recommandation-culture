import re
from app import app

client = app.test_client()
donnees = {"Nitrogen": 100, "Phosphorus": 82, "Potassium": 50, "Temperature": 27.4,
           "Humidity": 80.4, "pH_Value": 6.0, "Rainfall": 104.6}

reponse = client.post("/predict", data=donnees)
html = reponse.get_data(as_text=True)

print(reponse.status_code)
print(re.findall(r'class="erreur">(.*?)</div>', html, re.S))
print(re.findall(r'class="nom">(.*?)</span>\s*<span class="pourcent">(.*?)</span>', html, re.S))



