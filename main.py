
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import yfinance as yf
from datetime import datetime

# Загрузка данных QQQ
ticker = yf.Ticker("QQQ")
hist = ticker.history(interval="1m", period="10d", prepost=True)

# Обработка временной зоны
if getattr(hist.index, 'tz', None) is None:
    hist.index = hist.index.tz_localize("UTC").tz_convert("Europe/Berlin")
else:
    hist.index = hist.index.tz_convert("Europe/Berlin")

hist['Date'] = hist.index.date

def extract_features_and_labels(df):
    result = []
    for date in df['Date'].unique():
        day = df[df['Date'] == date]
        morning = day.between_time("08:00", "09:00")
        full_premarket = day.between_time("09:00", "15:30")
        if len(morning) < 10 or len(full_premarket) < 10:
            continue
        open_morning = morning['Open'].iloc[0]
        close_morning = morning['Close'].iloc[-1]
        change_morning = (close_morning - open_morning) / open_morning
        volume_morning = morning['Volume'].mean()
        open_full = full_premarket['Open'].iloc[0]
        close_full = full_premarket['Close'].iloc[-1]
        change_full = (close_full - open_full) / open_full
        label = "up" if change_full > 0.003 else "down" if change_full < -0.003 else "flat"
        result.append({
            "change_pre": change_morning,
            "volume_pre": volume_morning,
            "label": label
        })
    return pd.DataFrame(result)

data = extract_features_and_labels(hist)
X = data[["change_pre", "volume_pre"]]
y = data["label"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Прогноз на сегодня
now = datetime.now().astimezone(tz=pd.Timestamp.now(tz='Europe/Berlin').tz)
today_data = hist[hist.index.date == now.date()]
morning_now = today_data.between_time("08:00", "09:00")
full_premarket_now = today_data.between_time("09:00", "15:30")

if len(morning_now) >= 5:
    open_pre = morning_now['Open'].iloc[0]
    close_pre = morning_now['Close'].iloc[-1]
    change_pre = (close_pre - open_pre) / open_pre
    volume_pre = morning_now['Volume'].mean()
    pred = model.predict([[change_pre, volume_pre]])[0]
    proba = model.predict_proba([[change_pre, volume_pre]])[0]
    confidence = round(max(proba) * 100, 1)

    print(f"📊 Утреннее изменение: {change_pre*100:.2f}%")
    print(f"📦 Средний объём: {volume_pre:.0f}")
    print(f"🔮 Прогноз от ИИ: {pred.upper()} (уверенность {confidence}%)")

    if len(full_premarket_now) > 5:
        open_real = full_premarket_now['Open'].iloc[0]
        close_real = full_premarket_now['Close'].iloc[-1]
        change_real = (close_real - open_real) / open_real
        actual = "up" if change_real > 0.003 else "down" if change_real < -0.003 else "flat"
        print(f"✅ Факт: {actual.upper()} ({change_real*100:.2f}%)")
else:
    print("⏳ Недостаточно данных для прогноза (08:00–09:00).")
