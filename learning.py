import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta

# データの読み込み
df = pd.read_csv('S&P500_40years.csv')

# 日付を datetime 型に変換
df['日付け'] = pd.to_datetime(df['日付け'], format='%Y%m%d')

# Prophet用にデータフレームを準備
prophet_df = df[['日付け', '終値']].rename(columns={'日付け': 'ds', '終値': 'y'})
prophet_df = prophet_df.sort_values('ds')

# Prophetモデルの作成と学習
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(prophet_df)

# 未来の日付を生成
future_dates = pd.date_range(start=prophet_df['ds'].max() + timedelta(days=1), periods=3650)  # 10年分(365日*10)
future = pd.DataFrame({'ds': future_dates})

# 予測の実行
forecast = model.predict(future)

# 実際のデータと予測データを結合
all_data = pd.concat([prophet_df, forecast[['ds', 'yhat']]])

# グラフの作成
plt.figure(figsize=(15, 8))
plt.plot(prophet_df['ds'], prophet_df['y'], label='Actual', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)

plt.title('S&P500 Index: Historical Data and 10-Year Forecast')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()

# x軸の日付フォーマットを設定
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))

plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 最新の実際の値と10年後の予測値を表示
last_actual = prophet_df.iloc[-1]
ten_years_later = forecast.iloc[-1]

print(f"最新の実際の値 ({last_actual['ds'].strftime('%Y-%m-%d')}): {last_actual['y']:.2f}")
print(f"10年後の予測値 ({ten_years_later['ds'].strftime('%Y-%m-%d')}): {ten_years_later['yhat']:.2f}")
print(f"予測される10年間の成長率: {((ten_years_later['yhat'] / last_actual['y']) - 1) * 100:.2f}%")