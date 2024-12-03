import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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

# 训练基本模型
rf_model = RandomForestRegressor(random_state=1, n_estimators=300, min_samples_split=2,
                                 min_samples_leaf=1, max_features='sqrt', max_depth=30)
gb_model = XGBRegressor(random_state=1,n_estimators=300, subsample=0.6,
                        max_depth=5,learning_rate=0.05,gamma=0,colsample_bytree=0.7)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# 得到基本模型在训练集上的预测结果
pred_train_rf = rf_model.predict(X_train)
pred_train_gb = gb_model.predict(X_train)

# 将基本模型的预测结果堆叠起来作为元特征
meta_features_train = np.column_stack((pred_train_rf, pred_train_gb))
meta_features_train = torch.tensor(meta_features_train, dtype=torch.float32)

# 定义神经网络元模型
# 定义卷积神经网络元模型
class CNNMetaModel(nn.Module):
    def __init__(self):
        super(CNNMetaModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 6, 128)  # 通过池化层后的输出大小为 64*6
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 在输入数据上增加一个维度以适应卷积操作
        print("Input size:", x.size())
        x = nn.functional.relu(self.conv1(x))
        print("Conv1 output size:", x.size())
        x = nn.functional.max_pool1d(x, 2)
        print("Max pool output size:", x.size())
        x = nn.functional.relu(self.conv2(x))
        print("Conv2 output size:", x.size())
        x = nn.functional.max_pool1d(x, 2)
        x = x.view(-1, 64 * 6)  # 将特征展平以供全连接层处理
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建并训练卷积神经网络元模型
cnn_meta_model = CNNMetaModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_meta_model.parameters(), lr=0.001)
num_epochs = 800

for epoch in range(num_epochs):
    cnn_meta_model.train()
    optimizer.zero_grad()
    outputs = cnn_meta_model(meta_features_train)
    loss = criterion(outputs.flatten(), torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 得到基本模型在测试集上的预测结果
pred_test_rf = rf_model.predict(X_test)
pred_test_gb = gb_model.predict(X_test)

# 将基本模型的预测结果堆叠起来作为元特征
meta_features_test = np.column_stack((pred_test_rf, pred_test_gb))
meta_features_test = torch.tensor(meta_features_test, dtype=torch.float32)

# 使用训练好的神经网络元模型进行预测
cnn_meta_model.eval()
with torch.no_grad():
    meta_features_test_tensor = torch.tensor(meta_features_test, dtype=torch.float32).unsqueeze(1)
    outputs = cnn_meta_model(meta_features_test_tensor)
y_pred_test = outputs.flatten().numpy()

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
plt.title("Blending with Neural Network as Meta Model: True vs Predicted")
plt.show()
