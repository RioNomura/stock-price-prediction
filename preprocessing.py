import pandas as pd

# データの読み込み
df1 = pd.read_csv("S&P500_20years.csv")
df2 = pd.read_csv("S&P500_20-40years.csv")

# データフレームの結合
df = pd.concat([df1, df2], ignore_index=True)

# 以下、元のコードと同じ前処理を行う
# 日付を8桁の数字に変換
df['日付け'] = pd.to_datetime(df['日付け']).dt.strftime('%Y%m%d')

# "終値","始値","高値","安値"の千の位と百の位の間の","を削除
for col in ['終値', '始値', '高値', '安値']:
    df[col] = df[col].astype(str).str.replace(',', '')

# "出来高"列を削除
df = df.drop('出来高', axis=1)

# 変化率の%を削除
df['変化率 %'] = df['変化率 %'].str.rstrip('%').astype('float') / 100.0

# 列名を変更
df = df.rename(columns={'変化率 %': '変化率'})

# 前処理後のデータを保存
df.to_csv("S&P500_40years.csv", index=False)