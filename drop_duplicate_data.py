import pandas as pd
import glob
import os

pwd = os.getcwd()
data_dir = "data/BINANCE_AMPUSDT_60"

csv_files = glob.glob(os.path.join(pwd, data_dir, "BINANCE_AMPUSDT_60*.csv"))

df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index= True)

# 需要注意tradingview导出数据的 unix time 单位
df['time'] = pd.to_datetime(df['time'], unit='s') 

df = df.sort_values(by='time').reset_index(drop=True)

df = df.drop_duplicates(subset=['time'], keep='first')

df.to_csv(os.path.join(pwd, data_dir, "merged_base_data.csv"), index=False)