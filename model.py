from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import os

# 数据加载
dataset_path = os.path.join(os.getcwd(), 'dataset.csv')
features_df = pd.read_csv(dataset_path)
y = features_df['profit_label']
x = features_df.drop(columns=['profit_label'])

# 使用 SMOTE 进行过采样
smote = SMOTE(random_state=42)
x_balanced, y_balanced = smote.fit_resample(x, y)

print(f"平衡后数据集大小: {x_balanced.shape}")

# 设定随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# 进行5折交叉验证
cv_scores = cross_val_score(rf_model, x_balanced, y_balanced, cv=5, scoring='accuracy')

# 输出交叉验证的结果
print(f"交叉验证准确率: {cv_scores}")
print(f"交叉验证平均准确率: {cv_scores.mean():.4f}")

# 在整个平衡后的数据上进行训练，最终模型
rf_model.fit(x_balanced, y_balanced)

# 测试集评估
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_test_pred = rf_model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_acc:.4f}")

print("\n分类报告 (测试集):")
print(classification_report(y_test, y_test_pred))
