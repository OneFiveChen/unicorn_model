from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import os
import shap  

# 数据加载
dataset_path = os.path.join(os.getcwd(), 'dataset.csv')
features_df = pd.read_csv(dataset_path)
y = features_df['profit_label']
x = features_df.drop(columns=['profit_label'])

# 开关控制是否使用 SMOTE
use_smote = True
if use_smote:
    smote = SMOTE(random_state=42)
    x, y = smote.fit_resample(x, y)
    print(f"平衡后数据集大小: {x.shape}")

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 配置训练方法开关
train_method = "RandomForest"  # 可选值: "RandomForest", "GradientBoosting", "SVM", "LogisticRegression", "XGBoost"

# 配置各方法的参数
if train_method == "RandomForest":
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=15,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
elif train_method == "GradientBoosting":
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
elif train_method == "SVM":
    model = SVC(
        kernel='rbf',
        C=0.1,                
        gamma='scale',        
        class_weight='balanced', 
        probability=True,
        random_state=42
    )
elif train_method == "LogisticRegression":
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
elif train_method == "XGBoost":
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,  
        max_depth=8,        
        min_child_weight=4,  
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
else:
    raise ValueError("Unsupported training method selected!")

# 进行5折交叉验证
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {cv_scores}")
print(f"交叉验证平均准确率: {cv_scores.mean():.4f}")

# 模型训练
model.fit(x_train, y_train)

# 测试集评估
y_test_pred = model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_acc:.4f}")

print("\n分类报告 (测试集):")
print(classification_report(y_test, y_test_pred))

# shap分析特征影响权重
# if train_method == "RandomForest":
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(x_test)
#     shap.summary_plot(shap_values, x_test)
