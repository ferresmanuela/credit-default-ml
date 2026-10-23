# Predicción de Default en Tarjetas de Crédito

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ferresmanuela/credit-default-ml/blob/main/proyecto_final.ipynb)

Proyecto final — **Data Science II: Machine Learning para la Ciencia de Datos** (Coderhouse)
**Autor:** Pablo Ferres

## Objetivo
Predecir qué clientes de tarjetas de crédito incurrirán en default el mes siguiente, para que la entidad emisora pueda tomar acciones preventivas.

- **Tarea:** clasificación binaria supervisada
- **Variable objetivo:** `default` (1 = default el mes siguiente)
- **Métrica principal:** ROC AUC | **Métrica de negocio:** Recall de la clase positiva

## Dataset
[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — UCI Machine Learning Repository (id 350). 30.000 clientes, 23 variables, Taiwán 2005.

Los datos se descargan automáticamente desde UCI mediante la librería `ucimlrepo`; no es necesario descargar archivos manualmente.

## Pipeline
1. Formulación del problema e hipótesis
2. Análisis exploratorio (calidad de datos, distribuciones, correlaciones, outliers)
3. Limpieza (duplicados, categorías no documentadas)
4. Ingeniería de atributos (6 variables nuevas: utilización, ratio de pago, historial de atrasos, tendencia de deuda)
5. Preprocesamiento con `Pipeline` y `ColumnTransformer` (sin data leakage)
6. Modelos base: Regresión Logística, Árbol de Decisión, Random Forest, XGBoost
7. Optimización: `GridSearchCV` y `RandomizedSearchCV` con validación cruzada estratificada
8. Evaluación en test y selección del umbral de decisión
9. Explicabilidad con SHAP

## Resultados

| Modelo | AUC (CV) | AUC (test) |
|---|---|---|
| **XGBoost** ✅ | **0,789** | **0,780** |
| Random Forest | 0,788 | 0,776 |
| Regresión Logística | 0,764 | 0,757 |

- Con umbral 0,35, el modelo detecta el **82% de los clientes que incurren en default**.
- Las variables más importantes son las de **historial de atrasos**; las 6 variables creadas figuran entre las 15 más relevantes.

## Cómo reproducir

**Opción 1 — Google Colab (recomendada):** hacer clic en el botón *Open in Colab* (requiere cuenta de Google) y ejecutar todas las celdas (*Entorno de ejecución → Ejecutar todas*).

**Opción 2 — Local:**
```bash
git clone https://github.com/ferresmanuela/credit-default-ml.git
cd credit-default-ml
pip install -r requirements.txt
jupyter notebook proyecto_final.ipynb
```

Tiempo de ejecución aproximado: 20-25 minutos (la optimización de Random Forest es la etapa más lenta). Todas las semillas aleatorias están fijadas (`random_state=42`).

## Estructura
```
credit-default-ml/
├── README.md
├── requirements.txt
└── proyecto_final.ipynb
```
