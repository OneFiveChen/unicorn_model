import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset_path = os.path.join(os.getcwd(), 'data/BINANCE_AMPUSDT_60', 'normalized_dataset.csv')

features_df = pd.read_csv(dataset_path)
y = features_df['profit_label']
x = features_df.drop(columns=['profit_label'])

# 暂时用random_state固定随机种子，观察结果
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"训练集大小: {x_train.shape}, 测试集大小: {x_test.shape}")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(x_train, y_train)

y_train_pred = rf_model.predict(x_train)
y_test_pred = rf_model.predict(x_test)

# 训练集
train_acc = accuracy_score(y_train, y_train_pred)
print(f"训练集准确率: {train_acc:.4f}")

# 测试集
test_acc = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_acc:.4f}")

print("\n分类报告 (测试集):")
print(classification_report(y_test, y_test_pred))