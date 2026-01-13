import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

print("1. Cargar datos de MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# Corrección del error previo: usamos int nativo de Python
X, y = mnist["data"], mnist["target"].astype(int)

print("2. Dividir datos (Train/Test)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Ejecutando Cross-Validation (K=3)...")
clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print(f"Precisión media en CV: {cv_scores.mean():.4f}")

print("4. Entrenando el modelo final...")
clf.fit(X_train, y_train)

print("5. Evaluando y generando métricas...")
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test) # Probabilidades para ROC

# --- RMSE ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# --- MATRIZ DE CONFUSIÓN ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, ax=ax[0])
ax[0].set_title("Matriz de Confusión")

# --- CURVA ROC MULTICLASE ---
# Binarizamos las etiquetas para calcular ROC por cada clase
y_test_bin = label_binarize(y_test, classes=np.arange(10))
n_classes = 10

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, label=f'Clase {i} (AUC = {roc_auc:.2f})')

ax[1].plot([0, 1], [0, 1], 'k--', lw=2) # Línea diagonal de referencia
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('Tasa de Falsos Positivos (FPR)')
ax[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
ax[1].set_title('Curvas ROC por Clase')
ax[1].legend(loc="lower right", fontsize='small')

plt.tight_layout()
plt.show()

# --- GUARDAR MODELO ---
joblib.dump(clf, 'mnist_model.pkl')
print("6. Modelo guardado con éxito como mnist_model.pkl")