import pandas as pd 
import numpy as np
import os

def create_dataset(data_dir):
    #精度保留
    # pd.set_option('display.float_format', lambda x: '%.16f' % x)

    kline_df = pd.read_csv(os.path.join(data_dir, "merged_kline_data.csv"))
    trade_df = pd.read_csv(os.path.join(data_dir, "trade_data.csv"))

    # 需要配置的数据项目
    titles = ['open', 'high', 'low', 'close', 'Volume', 'Plot', 'Plot.1']
    k_num = 5

    kline_df['time'] = pd.to_datetime(kline_df['time'])
    trade_df['日期/时间'] = pd.to_datetime(trade_df['日期/时间'])
    kline_df = kline_df.sort_values(by='time')
    trade_df = trade_df.sort_values(by='日期/时间')

    # 计算 K线的时间间隔
    kline_df['time_diff'] = kline_df['time'].diff()
    kline_period = kline_df['time_diff'].dropna().iloc[0]  # 获取第一个非 NaT 的时间间隔
    print(f"每根 K线的时间间隔是：{kline_period}")

    # 根据trade匹配对应的k线基础数据
    # def get_previous_k_lines(trade_timestamp, n=5):
    #     end_time = trade_timestamp
    #     start_time = end_time - n * kline_period
    #     subset = kline_df[(kline_df['time'] > start_time) & (kline_df['time'] <= end_time)]
    #     if len(subset) == n:
    #         return subset[titles].values.flatten()
    #     else:
    #         return np.nan * np.ones(n * len(titles))

    # 暂时构造为前5根k线，以及对应的band值
    features = []
    for index, trade_row in trade_df.iterrows():
        # 只读取买入的数据行
        # 交易 #	类型	信号	日期/时间	价格 USDT	合约	获利 USDT	获利 %	累计获利 USDT	累计获利 %	最大交易获利 USDT	最大交易获利 %	交易亏损 USDT	交易亏损 %
        # 393	多头进场	l	2024-12-13 04:00	0.009808	10195.758	2.06	2.06	363.81	0.45	2.06	2.06	1.57	1.57
        if trade_row['信号'] != 'l': 
            continue
        trade_timestamp = trade_row['日期/时间']
        # kline_features = get_previous_k_lines(trade_timestamp, k_num)

        # 根据trade匹配对应的k线基础数据, 取窗口n根k线数据
        start_time = trade_timestamp - k_num * kline_period
        kline_window = kline_df[(kline_df['time'] > start_time) & (kline_df['time'] <= trade_timestamp)]

        # 防止不足n根k线，污染数据集
        if kline_window.shape[0] != k_num:
            continue

        # 以第一根k线为基准, Plot和Plot.1为tradingview里面的参考band上下沿
        base_kline = kline_window.iloc[0]
        base_price = (base_kline['open'] + base_kline['close']) / 2
        base_volume = base_kline['Volume']
        base_band = (base_kline['Plot'] + base_kline['Plot.1']) / 2

        # 防止基线为0
        if base_price == 0 or base_volume == 0:
            continue

        # 特征处理
        normalized_features = []
        for _, row in kline_window.iterrows():

            # 归一化(以第一根k线为基准)
            normalized_features.extend([
                row['open'] / base_price,
                row['high'] / base_price,
                row['low'] / base_price,
                row['close'] / base_price,
                row['Volume'] / base_volume,
                row['Plot'] / base_band,
                row['Plot.1'] / base_band
            ])

            # 衍生特征
            distance_to_bandlow = (row['close'] - row['Plot']) / row['Plot']
            distance_to_bandup = (row['close'] - row['Plot.1']) / row['Plot.1']
            price_range = (row['high']- row['low']) / row['open']
            band_width = 2 * (row['Plot.1'] - row['Plot']) / (row['Plot.1'] + row['Plot']) 
            price_trend = (row['close'] - base_kline['close']) / base_kline['close']
            volume_trend = (row['Volume'] - base_kline['Volume']) / base_kline['Volume']
            extend_features = [
                distance_to_bandlow,
                distance_to_bandup,
                price_range,
                band_width,
                price_trend,
                volume_trend
            ]
            normalized_features.extend(extend_features)

        # 数据非NAN才有效，去除不完整数据
        if not np.isnan(normalized_features).any():
            profit = trade_row[['获利 USDT']].values
            profit_label = np.where(profit > 0, 1, 0)
            features.append(np.concatenate([normalized_features, profit_label]))
        # else:
        #     features.append([np.nan] * (len(titles) * k_num + 1)) #如果缺少 K线数据，填充 NaN

    # 构建DataFrame
    cols = [f"feature_{i}" for i in range(len(features[0]) - 1)]
    print(f"总共特征数量：{len(features[0]) - 1}")
    # for i in range(1, 6):
    #     cols.extend([f'open_{i}', f'high_{i}', f'low_{i}', f'close_{i}', f'volume_{i}', f'plot_{i}', f'plot.1_{i}'])
    cols.append('profit_label')

    features_df = pd.DataFrame(features, columns=cols)
    output_file = os.path.join(data_dir, 'normalized_dataset.csv')
    features_df.to_csv(output_file, index=False)
    print(f"crate dataset down: {output_file}")


def main():
    root_dir = os.path.join(os.getcwd(), "data")
    for sub_dir in os.listdir(root_dir):
        fullpath = os.path.join(root_dir, sub_dir)

        if os.path.isdir(fullpath):
            print(f"Processing directory: {fullpath}")
            create_dataset(fullpath)

if __name__ == "__main__":
    main()