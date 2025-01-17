数据来源(tradingview的数据导出)
1. 只能导出当前页面窗口内包含的k线
2. 可以提前用pine写需要的数据，会在导出的同时一并导出
3. 不同时间段的数据导出之后需要去重，这里需要单独写一个去重的脚本

数据处理：
1. 训练数据k线周期的选择。因为找超买买点是短期趋势的分析，考虑使用20根k线以下的数量级。（暂时考虑先用20/10/5）
2. 可能存在一些信息重复的问题，直接使用大量k线一定会增加噪声和计算复杂度（相邻k线数据高度相关，open\close一定程度上被high\low包含）
3. 用一些技术指标作为原始数据的补充
     a.暂时可以选择作为特征的基础指标：MA/EMA/MACD/RSI/参考band/ATR/成交量均线
     b.衍生指标：MACD 的逐步扩散或收敛、MA 的斜率变化
4. 由于不同标的以及同一标的的不同时期的价格成交量差距会很大，需要单独对k线数据进行归一化标准化，暂时先考虑直接使用滚动归一化，
     a.滚动归一化的问题是会当滚动窗口的振幅实际较小时，归一化会放大其实际波动，因此放弃
     b.进一步考虑，可以采用基准归一化，以第一根k线为基准，后面的k线除以第一根k线数据，以此归一化在目前的寻找超跌买入机会的情况，更加适用
5. 特征数量，控制在20-50维度吧

模型的选择：
Knn效率较低，可以直接先用随机森林或者xgboost建立基线模型，看后续数据量可以逐步引入lstm试试

训练过程：
先做过去N根k线训练，然后增加基础指标特征，最后增加衍生指标特征，逐步增加模型的复杂性，观察模型性能的提升效果

模型的应用和测试：
tradingview单独输出数据确认买卖点，然后再通过写strategy的方式，在tradingview回测


第一版：

数据：使用包括交易k线在内前5根k线的open/close/low/high, volume, 以及band上下沿为数据，且未经过归一化等处理
模型：随机森林
结果：
```
训练集大小: (318, 35), 测试集大小: (80, 35)
训练集准确率: 1.0000
测试集准确率: 0.7125
分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.25      0.11      0.15        19
         1.0       0.76      0.90      0.83        61

    accuracy                           0.71        80
   macro avg       0.51      0.50      0.49        80
weighted avg       0.64      0.71      0.67        80
```
结论分析：
数据量不足，尤其是标签为0的样本数据量不足，考虑归一化之后，加入其他的标的数据
没有经过归一化的处理，没有加入各种指标，后续可以持续添加


第二版：

数据：
以第一根k线的open+close /2进行基线处理，volume以第一根为基准，参考系以第一根的均值为基准，进行归一化处理，方便对多标的数据进行整合，增大数据量
模型：随机森林
结果：
```
训练集大小: (318, 35), 测试集大小: (80, 35)
训练集准确率: 1.0000
测试集准确率: 0.7875
分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.62      0.26      0.37        19
         1.0       0.81      0.95      0.87        61

    accuracy                           0.79        80
   macro avg       0.72      0.61      0.62        80
weighted avg       0.76      0.79      0.75        80
```
结论：
归一化之后性能明显提升，尤其是对0标签的数据预测能力有较大改善


第三版：
数据：
1. 增加多个标的数据,AMP,AVAX,BNB,DOGE,ETH,FIL,KAS,PEPE,SHIB,SOL,SUI,XRP
2. 增加多个衍生特征：distance_to_bandlow,distance_to_bandup,price_range,band_width,price_trend,volume_trend
3. 对0样本进行过采样
4. 采样方法更改为平衡采样
5. 增加5折交叉验证
```
平衡后数据集大小: (4938, 65)
交叉验证准确率: [0.74493927 0.73076923 0.74898785 0.79635258 0.78926039]
交叉验证平均准确率: 0.7621
测试集准确率: 0.8924

分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.79      0.90      0.84       235
         1.0       0.95      0.89      0.92       490

    accuracy                           0.89       725
   macro avg       0.87      0.89      0.88       725
weighted avg       0.90      0.89      0.89       725

```
结论：
1. 模型的测试集准确率提高，且 F1-score 和精确度/召回率表现优秀。
2. 针对0样本数据的分类准确度有所提高。过采样和平衡采样处理样本不平衡，效果显著


第四版：
数据：
1. 扩展数据量，从2024年数据扩展到2022-2024数据，数据量大约增到3倍，目前数据条数：9246
2. 增加过采样开关, 修改随机森林参数
3. 增加梯度提升、支持向量机、逻辑回归、xgboost来做训练

随机森林
```
平衡后数据集大小: (12142, 65)
交叉验证准确率: [0.74163664 0.73288729 0.75656202 0.77239959 0.7507724 ]
交叉验证平均准确率: 0.7509
测试集准确率: 0.7526

分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.75      0.78      0.76      1241
         1.0       0.76      0.73      0.74      1188

    accuracy                           0.75      2429
   macro avg       0.75      0.75      0.75      2429
weighted avg       0.75      0.75      0.75      2429
```

梯度提升
```
平衡后数据集大小: (12142, 65)
交叉验证准确率: [0.76634071 0.74575399 0.75965003 0.77085479 0.75437693]
交叉验证平均准确率: 0.7594
测试集准确率: 0.7629

分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.76      0.78      0.77      1241
         1.0       0.77      0.74      0.75      1188

    accuracy                           0.76      2429
   macro avg       0.76      0.76      0.76      2429
weighted avg       0.76      0.76      0.76      2429
```

SVM支持向量机
```
平衡后数据集大小: (12142, 65)
交叉验证准确率: [0.51055069 0.50334534 0.51981472 0.50720906 0.50978373]
交叉验证平均准确率: 0.5101
测试集准确率: 0.5117

分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.51      0.86      0.64      1241
         1.0       0.50      0.14      0.22      1188

    accuracy                           0.51      2429
   macro avg       0.51      0.50      0.43      2429
weighted avg       0.51      0.51      0.44      2429
```

逻辑回归
```
平衡后数据集大小: (12142, 65)
交叉验证准确率: [0.6860525  0.67678847 0.69686052 0.71524202 0.68743563]
交叉验证平均准确率: 0.6925
测试集准确率: 0.6970

分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.70      0.71      0.70      1241
         1.0       0.69      0.69      0.69      1188

    accuracy                           0.70      2429
   macro avg       0.70      0.70      0.70      2429
weighted avg       0.70      0.70      0.70      2429
```

XGBoost
```
平衡后数据集大小: (12142, 65)
交叉验证准确率: [0.76737005 0.76582604 0.76994339 0.78836251 0.7584964 ]
交叉验证平均准确率: 0.7700
测试集准确率: 0.7727

分类报告 (测试集):
              precision    recall  f1-score   support

         0.0       0.78      0.78      0.78      1241
         1.0       0.77      0.77      0.77      1188

    accuracy                           0.77      2429
   macro avg       0.77      0.77      0.77      2429
weighted avg       0.77      0.77      0.77      2429
```
结论：
1. 新增大量数据之后，随机森林测试集和验证集准确率几乎等同，大大减少过拟合，模型开始稳定，但是各项指标下降，数据集可能存在无效、反向数据
2. 新增的支持向量机和逻辑回归效果弱，算法本身对高维数据（尤其是特征较多、分布复杂的数据）不够有效。
3. xgboost和梯度增强算法相比于随机森林有些微优势，指标效果略优，特别是xgboost

有问题的地方：
1. unicorn针对不同的标的有不同的参数，目前对所有标的仅采用同一套unicorn参数来训练，直接做整合模型训练存在问题
2. 同样是由于不同标的同一套参数的原因，有些标的存在大量这根k线买下一根k线卖的情况，产生大量垃圾数据