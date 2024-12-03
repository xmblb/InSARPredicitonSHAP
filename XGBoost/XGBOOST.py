import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取数据
data = pd.read_csv(r"E:\essays\deep_learning\linzhi\data_kun.csv")

# 定义超参数范围
param_grid_random = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
}

# 使用随机搜索进行初步探索
random_search = RandomizedSearchCV(XGBRegressor(random_state=1),
                                   param_distributions=param_grid_random,
                                   n_iter=1000,
                                   cv=5,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,
                                   verbose=2)

# 提取训练数据和标签
idx = np.argwhere((data['Time'].values == 1)).ravel()
test_idx= np.argwhere(
    ((data['Time'].values == 2) ) |
    ((data['Time'].values == 3) ) |
    ((data['Time'].values == 4) )).ravel()
data_x = data.values[:, 1:-5][idx, :]
data_y = data.values[:, -1][idx]

# 进行随机搜索
random_search.fit(data_x, data_y)

# 获取随机搜索的最佳参数
best_params_random = random_search.best_params_

# 定义网格搜索的超参数范围
param_grid_grid = {
    'n_estimators': [best_params_random['n_estimators'] - 50, best_params_random['n_estimators'], best_params_random['n_estimators'] + 50],
    'max_depth': [best_params_random['max_depth'], best_params_random['max_depth'] + 1, best_params_random['max_depth'] + 2],
    'learning_rate': [best_params_random['learning_rate']],
    'subsample': [best_params_random['subsample']],
    'colsample_bytree': [best_params_random['colsample_bytree']],
    'gamma': [best_params_random['gamma']]
}

# 使用网格搜索进行精细调整
grid_search = GridSearchCV(XGBRegressor(random_state=1),
                           param_grid=param_grid_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1,
                           verbose=2)

# 进行网格搜索
grid_search.fit(data_x, data_y)

# 获取最佳参数和最佳模型
best_params_grid = grid_search.best_params_
best_model = grid_search.best_estimator_

# 训练和测试数据的预测
test_x = data.values[:, 1:-5][test_idx, :]
test_y = data.values[:, -1][test_idx]

y_train_pred = best_model.predict(data_x)
y_test_pred = best_model.predict(test_x)


# 计算各种评估指标
# 计算均方误差（MSE）
mse = mean_squared_error(data_y, y_train_pred)

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(data_y, y_train_pred)

corr_test = pearsonr(test_y, y_test_pred)

print("Pearson Correlation Coefficient (PCC) on Test Set:", corr_test)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Best Parameters Random Search:", best_params_random)
print("Best Parameters Grid Search:", best_params_grid)

