from flask import Flask, render_template, request
import joblib as jb
import pandas as pd
import numpy as np

app = Flask(__name__)

modele = jb.load("model/rf_model.joblib")
scaler = jb.load("model/scaler.joblib")
encoder = jb.load("model/encoder.joblib")

COLONNES = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature',
            'Humidity', 'pH_Value', 'Rainfall']

# Plages observées dans le dataset d'entraînement
PLAGES = {
    "Nitrogen": (0, 140),
    "Phosphorus": (5, 145),
    "Potassium": (5, 205),
    "Temperature": (8.8, 43.7),
    "Humidity": (14.3, 100),
    "pH_Value": (3.5, 9.9),
    "Rainfall": (20.2, 298.6),
}

def form_page(erreur=None, saisies = None, resultats=None):
  return render_template("index.html", colonnes = COLONNES, plages = PLAGES, erreur= erreur, saisies=saisies or {}, resultats=resultats)


@app.route("/")
def accueil():
  return form_page()


@app.route("/predict", methods=["POST"])
def predict():
  valeurs = []

  for col in COLONNES:
    texte = request.form.get(col, "").replace(",", ".")

    # Conversion en nombre
    try:
      val = float(texte)
    except ValueError:
      return form_page(f"{col} : Veuillez saisir un nombre valide.", request.form)

    # Verification de la plage
    mini, maxi = PLAGES[col]
    if not (mini <= val <= maxi):
      return form_page(f"{col} : la valeur doit être prise entre {mini} et {maxi}.", request.form)

    valeurs.append(val)

  # Mise en form, dans l'ordre des colonnes, puis standardisation
  data = pd.DataFrame([valeurs], columns=COLONNES)
  data_scaled = scaler.transform(data)

  # Predictions
  prediction = modele.predict_proba(data_scaled)[0]
  top3 = np.argsort(prediction)[::-1][:3]
  noms = encoder.inverse_transform(modele.classes_[top3])


  resultats = [{"culture": noms[i], "proba":round(float(prediction[top3[i]]) * 100, 1)} for i in range(3)]

  return form_page(saisies=request.form, resultats=resultats)


if __name__ == "__main__":
  app.run(debug=True)