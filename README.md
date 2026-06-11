# TFG - Predicción Direccional del Mercado Bursátil con Machine Learning

**Autor:** Adrià Fernández López
**Universidad:** Universitat de Lleida (UdL) - Grado en Ingeniería Informática  
**Título completo:** *Análisis del impacto de la cantidad y variedad de datos en la predicción direccional del mercado bursátil mediante Machine Learning*

---

## Descripción

Sistema de predicción direccional del mercado bursátil que determina si una acción subirá o bajará en los próximos 30 días. Combina redes LSTM y XGBoost en 4 experimentos progresivos, un modelo generalista pre-entrenado con 20 acciones y un módulo de análisis de sentimiento basado en FinBERT.

La predicción se basa en dos condiciones:

1. **Filtro de fiabilidad** - el modelo debe superar unos umbrales mínimos sobre el conjunto de test (AUC ≥ 0.55, Precisión UP > 55%, Especificidad > 50%).
2. **Predicción actual** - si supera el filtro, se predice sobre los últimos 20 días reales disponibles para generar la señal BUY / DO NOT BUY / INSUFFICIENT.

---

## Estructura del proyecto

```
TFG_26-Prediction_Stocks/
├── src/
│   ├── lstm_v1_baseline.py       # Exp.1: OHLCV (10 features)
│   ├── lstm_v2_technical.py      # Exp.2: + Indicadores técnicos (20 features)
│   ├── lstm_v3_macro.py          # Exp.3: + Macro FRED (29 features)
│   └── lstm_v4_market.py         # Exp.4: + Contexto de mercado (35 features)
│   └── xgb_universal.py          # XGBoost para todos los experimentos
├── models/
│   ├── model_lstm_generalista.keras
│   ├── model_xgb_generalista.pkl
│   ├── scaler_generalista.pkl
│   └── threshold_generalista.json
├── app.py                        # Aplicación web Streamlit
├── train_generalista.py          # Entrena el modelo generalista
├── predict_generalista.py        # Predicción con modelo generalista
├── neuron_search.py              # Búsqueda de arquitectura LSTM
├── sentiment_module.py           # Módulo FinBERT (análisis de noticias)
└── requirements.txt
```

---

## Experimentos

| Exp. | Features | N | Modelo |
|------|----------|---|--------|
| Exp.1 | OHLCV derivadas | 10 | LSTM + XGBoost |
| Exp.2 | + Indicadores técnicos | 20 | LSTM + XGBoost |
| Exp.3 | + Variables macro FRED | 29 | LSTM + XGBoost |
| Exp.4 | + Contexto de mercado | 35 | LSTM + XGBoost |

**Acciones de referencia:** AAPL, MSFT, KO  
**Período:** 2010-07-01 → 2026-03-01  
**Split temporal:** 70% train / 15% validación / 15% test

---

## Instalación

```bash
git clone https://github.com/usuario/TFG_26-Prediction_Stocks.git
cd TFG_26-Prediction_Stocks
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Uso

### Experimentos individuales (LSTM)

```bash
python src/lstm_v1_baseline.py --ticker AAPL --horizon 30
python src/lstm_v2_technical.py --ticker MSFT --horizon 30
python src/lstm_v3_macro.py --ticker KO --horizon 30
python src/lstm_v4_market.py --ticker AAPL --horizon 30

# Versión Stacked LSTM
python src/lstm_v4_market.py --ticker AAPL --horizon 30 --stacked
```

### Experimentos XGBoost

```bash
python src/xgb_universal.py --ticker AAPL --horizon 30 --exp 1
python src/xgb_universal.py --ticker AAPL --horizon 30 --exp 2
python src/xgb_universal.py --ticker AAPL --horizon 30 --exp 3
python src/xgb_universal.py --ticker AAPL --horizon 30 --exp 4
```

### Modelo generalista

```bash
# Entrenar (solo necesario una vez)
python train_generalista.py

# Predecir con modelo pre-entrenado
python predict_generalista.py --ticker AAPL
python predict_generalista.py --ticker V       # acción no entrenada
```

### Búsqueda de arquitectura

```bash
python neuron_search.py --ticker AAPL --exp 1
python neuron_search.py --ticker AAPL --exp 2
python neuron_search.py --ticker AAPL --exp 3
python neuron_search.py --ticker AAPL --exp 4
```

### Aplicación web

```bash
streamlit run app.py
```

---

## Arquitectura LSTM

| Parámetro | Exp.1 / Exp.2         | Exp.3 / Exp.4 |
|-----------|-----------------------|--------------|
| Units | 48                    | 64 |
| Dropout | 0.4 / 0.5             | 0.5 |
| L2 | 0.002                 | 0.003 |
| Window size | 20 días               | 20 días |
| Epochs máx. | 150                   | 150 |
| Early stopping | val_auc (patience=20) | val_auc (patience=20) |

*Exp.1 usa dropout=0.4, Exp.2 usa dropout=0.5

**Stacked LSTM:** 2 capas (64 + 32 unidades), `return_sequences=True` en la primera capa.

---

## Variable objetivo

```
target = 1 si mean(Close[t+1 : t+31]) > mean(Close[t-29 : t+1])
target = 0 en caso contrario
```

Horizonte de 30 días. El scaler se ajusta únicamente sobre el conjunto de train.

---

## Señal de inversión

```
Si AUC < 0.55                          -> INSUFFICIENT
Si pred_today == UP
   y Precisión UP > 55%
   y Especificidad > 50%               -> BUY
Si pred_today == DOWN
   y Especificidad > 55%               -> DO NOT BUY
En cualquier otro caso                 -> INSUFFICIENT
```

---

## Módulo de sentimiento (FinBERT)

Analiza las últimas noticias relacionadas con la acción y devuelve:
- Puntuación media de sentimiento (−1 a +1)
- Distribución positivo / negativo / neutral
- Ajuste cualitativo sobre la señal del modelo

El sentimiento **no** es un input del modelo, actúa como información complementaria post-predicción.

---

## Notas metodológicas

- **Survivorship bias:** Las acciones del modelo generalista han sobrevivido hasta 2026.
- **Bull market 2023–2026:** Las métricas pueden estar infladas si el porcentaje de subidas en el test supera el 60%.
- **XGBoost con features macro:** El XGBoost no tiene memoria temporal, por lo que su rendimiento degrada al añadir variables macroeconómicas (Exp.3, Exp.4). Este es un resultado esperado y válido.
- **Stacked LSTM:** Mejora el AUC pero puede colapsar el threshold en acciones con fuerte sesgo alcista.
- **Fechas fijas:** START=2010-07-01, END=2026-03-01 para garantizar reproducibilidad.

---

## Requisitos

- Python 3.10+
- TensorFlow 2.x
- XGBoost
- Optuna
- yfinance
- fredapi
- Streamlit
- transformers (FinBERT)

Ver `requirements.txt` para versiones exactas.

---

## Aviso legal

Las predicciones generadas son orientativas y se basan en el rendimiento histórico del modelo. El rendimiento pasado no garantiza resultados futuros. Este sistema es un proyecto académico y no constituye ningún consejo de inversión. El usuario es el único responsable de las decisiones tomadas.