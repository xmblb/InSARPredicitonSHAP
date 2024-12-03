import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv(r"E:\essays\deep_learning\linzhi\data_kun.csv")

# 确定训练集和测试集的索引
train_idx = np.argwhere(data['Time'].values == 1).ravel()

test_idx = np.argwhere(
    ((data['Time'].values == 2) & (data['PS_count'].values > 9) & (data['VLOS.std.'].values < 2)) |
    ((data['Time'].values == 3) & (data['PS_count'].values > 9) & (data['VLOS.std.'].values < 2)) |
    ((data['Time'].values == 4) & (data['PS_count'].values > 9) & (data['VLOS.std.'].values < 2))).ravel()

# 提取训练集和测试集的特征和标签
X_train = data.values[train_idx, 1:-5]
y_train = data.values[train_idx, -1]

X_test = data.values[test_idx, 1:-5]
y_test = data.values[test_idx, -1]

# 定义基本模型
models = [
    RandomForestRegressor(random_state=1, n_estimators=500, min_samples_split=2,
                          min_samples_leaf=1, max_features='sqrt', max_depth=30),
    XGBRegressor(random_state=1, n_estimators=300, subsample=0.6,
                 max_depth=5, learning_rate=0.05, gamma=0, colsample_bytree=0.7),

]

# 用5折交叉验证训练基本模型，并生成元特征
meta_features_train = np.zeros((len(X_train), len(models)))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    for i, model in enumerate(models):
        model.fit(X_train_fold, y_train_fold)
        meta_features_train[val_index, i] = model.predict(X_val_fold)


# 训练元模型（随机森林）
# meta_model = RandomForestRegressor(random_state=1, n_estimators=500)
meta_model = XGBRegressor(random_state=1, n_estimators=300, subsample=0.6,
                 max_depth=5, learning_rate=0.05, gamma=0, colsample_bytree=0.7)
meta_model.fit(meta_features_train, y_train)

# 用基本模型生成测试集的元特征并预测
meta_features_test = np.zeros((len(X_test), len(models)))
for i, model in enumerate(models):
    meta_features_test[:, i] = model.predict(X_test)

# 使用元模型进行预测
y_pred_test = meta_model.predict(meta_features_test)

# 评估模型
test_corr = pearsonr(y_test, y_pred_test)[0]
print("Test Pearson Correlation Coefficient:", test_corr)

# 画测试集的散点图
plt.figure(figsize=(5, 5))
plt.plot([0, 20], [0, 20], color="b", linestyle="-", linewidth=1)
plt.scatter(y_test, y_pred_test, c='k', marker='o', label="Predictions", s=15)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xticks(np.linspace(0, 20, 5))
plt.yticks(np.linspace(0, 20, 5))
plt.rcParams.update({'font.size': 10})
plt.legend(loc='upper left')
plt.ylabel("Predicted mean LOS velocity (mm/year)")
plt.xlabel("Calculated mean LOS velocity (mm/year)")
plt.title("Stacked Model with 5-Fold Cross-Validation: True vs Predicted")
plt.show()
