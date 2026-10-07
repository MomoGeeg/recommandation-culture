import joblib
import pandas as pd

modele = joblib.load("model/rf_model.joblib")
scaler = joblib.load("model/scaler.joblib")
encoder = joblib.load("model/encoder.joblib")

df = pd.read_csv("data/Crop_recommendation.csv")
df = df.rename(columns={"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
                        "temperature": "Temperature", "humidity": "Humidity",
                        "ph": "pH_Value", "rainfall": "Rainfall", "label": "Crop"})
print(list(df.columns))

colonnes = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature',
            'Humidity', 'pH_Value', 'Rainfall']

# Test 1 : cohérence des fichiers sur tout le dataset
X = scaler.transform(df[colonnes])
y = encoder.transform(df["Crop"])
print("Accuracy sur tout le dataset :", round(modele.score(X, y), 3))

# Test 2 : exemple Banana, calculé hors de Flask
exemple = pd.DataFrame([[100, 82, 50, 27.4, 80.4, 6.0, 104.6]], columns=colonnes)
probas = modele.predict_proba(scaler.transform(exemple))[0]
for i in probas.argsort()[::-1][:3]:
    print(encoder.inverse_transform([modele.classes_[i]])[0], round(probas[i] * 100, 1))

# Test 3 : confiance sur de vraies lignes Banana
idx = encoder.transform(["Banana"])[0]
banana = df[df["Crop"] == "Banana"].copy()
banana["confiance_%"] = (modele.predict_proba(scaler.transform(banana[colonnes]))[:, idx] * 100).round(1)
print("Confiance moyenne :", banana["confiance_%"].mean().round(1))
print(banana.sort_values("confiance_%", ascending=False).head(3)[colonnes + ["confiance_%"]].to_string(index=False))