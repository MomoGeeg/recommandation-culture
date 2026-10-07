## Checklist du projet : Recommandation de culture

### Phase 0 : Préparation
- [ ] Créer le dossier du projet et un environnement virtuel
- [ ] Installer les bibliothèques (pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, flask)
- [ ] Télécharger `Crop_recommendation.csv`
- [ ] Définir le problème en une phrase et la variable cible (`label`)

### Phase 1 : Exploration des données
- [ ] Charger le fichier et afficher les premières lignes
- [ ] Dimensions, types de variables, `info()` et `describe()`
- [ ] Valeurs manquantes et doublons
- [ ] Équilibre des classes (nombre d'exemples par culture)
- [ ] Distribution de chaque variable (histogrammes)
- [ ] Valeurs aberrantes (boxplots)
- [ ] Corrélations entre variables (heatmap)
- [ ] Profil moyen de chaque culture (ex. quelle culture exige le plus de pluie ?)

### Phase 2 : Préparation des données
- [ ] Séparer les variables explicatives (X) et la cible (y)
- [ ] Encoder la cible (`LabelEncoder`)
- [ ] Découper en train/test (80/20, avec `stratify=y`)
- [ ] Décider si un scaling est nécessaire (oui pour KNN et régression logistique, non pour Random Forest)
- [ ] Construire un `Pipeline` pour que le prétraitement soit sauvegardé avec le modèle

### Phase 3 : Modélisation
- [ ] Entraîner un modèle de base (baseline)
- [ ] Entraîner 3 modèles à comparer (régression logistique, KNN, Random Forest)
- [ ] Validation croisée pour chaque modèle
- [ ] Tableau comparatif des scores

### Phase 4 : Évaluation
- [ ] Accuracy sur le jeu de test
- [ ] Rapport de classification (precision, recall, F1 par culture)
- [ ] Matrice de confusion
- [ ] Repérer les cultures les plus confondues et comprendre pourquoi
- [ ] Importance des variables (Random Forest)
- [ ] Choisir le modèle final et justifier le choix

### Phase 5 : Optimisation (optionnelle)
- [ ] `GridSearchCV` ou `RandomizedSearchCV` sur le modèle retenu
- [ ] Vérifier que le gain justifie la complexité

### Phase 6 : Sauvegarde
- [ ] Sauvegarder le pipeline avec `joblib` (`model.joblib`)
- [ ] Sauvegarder l'encodeur de la cible
- [ ] Tester le rechargement dans un notebook propre

### Phase 7 : Application Flask
- [ ] Créer la structure (`app.py`, `templates/`, `static/`)
- [ ] Route `/` : afficher le formulaire à 7 champs
- [ ] Route `/predict` : récupérer les valeurs, appeler le modèle, renvoyer le résultat
- [ ] Validation des entrées (plages réalistes, champs vides, texte à la place d'un nombre)
- [ ] Afficher les 3 meilleures cultures avec leur probabilité (`predict_proba`)
- [ ] Gérer les erreurs sans faire planter l'application
- [ ] Tester avec des cas connus (ex. des valeurs typiques du riz)

### Phase 8 : Finition
- [ ] Améliorer le design du formulaire (CSS)
- [ ] Rédiger le `requirements.txt`
- [ ] Rédiger le README (problème, données, méthode, résultats, limites, capture d'écran, lancement)
- [ ] Documenter la limite : cultures du dataset non sénégalaises
- [ ] Mettre le projet sur GitHub

### Phase 9 : Aller plus loin (bonus)
- [ ] Déploiement en ligne (Render, PythonAnywhere…)
- [ ] Route API JSON (`/api/predict`)
- [ ] Adapter à des cultures locales (mil, arachide, niébé…) avec des données collectées
