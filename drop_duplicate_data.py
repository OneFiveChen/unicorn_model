import pandas as pd
import glob
import os


# pwd = os.getcwd()
# data_dir = "data/BINANCE_ETHUSDT_60_20240101"

# csv_files = glob.glob(os.path.join(pwd, data_dir, "kline_*.csv"))

# df_list = [pd.read_csv(file) for file in csv_files]
# df = pd.concat(df_list, ignore_index= True)

# # 需要注意tradingview导出数据的 unix time 单位
# df['time'] = pd.to_datetime(df['time'], unit='s') 

# df = df.sort_values(by='time').reset_index(drop=True)

# df = df.drop_duplicates(subset=['time'], keep='first')

# df.to_csv(os.path.join(pwd, data_dir, "merged_kline_data.csv"), index=False)

def deduplicate_and_merge(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "BINANCE_*.csv"))
    if not csv_files:
        print(f"No files found in {data_dir}")
        return
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index= True)
    # 需要注意tradingview导出数据的 unix time 单位
    df['time'] = pd.to_datetime(df['time'], unit='s') 
    df = df.sort_values(by='time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['time'], keep='first')
    output_file = os.path.join(data_dir, "merged_kline_data.csv")
    df.to_csv(output_file, index=False)
    print(f"Merged file saved to: {output_file}")


def main():
    root_dir = os.path.join(os.getcwd(), "data")
    for sub_dir in os.listdir(root_dir):
        fullpath = os.path.join(root_dir, sub_dir)

        if os.path.isdir(fullpath):
            print(f"Processing directory: {fullpath}")
            deduplicate_and_merge(fullpath)

if __name__ == "__main__":
    main()