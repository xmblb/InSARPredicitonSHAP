import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
# 加载数据
data = pd.read_csv("E:/essays/deep_learning/linzhi/data_kun.csv")  # 请根据实际路径修改

# 确定用于训练和测试的索引
idx = np.argwhere((data['Time'].values == 1)).ravel()
test_idx = np.argwhere(
    ((data['Time'].values == 2) & (data['PS_count'].values > 9) & (data['VLOS.std.'].values < 2)) |
    ((data['Time'].values == 3) & (data['PS_count'].values > 9) & (data['VLOS.std.'].values < 2)) |
    ((data['Time'].values == 4) & (data['PS_count'].values > 9) & (data['VLOS.std.'].values < 2))
).ravel()

# 获取训练和测试数据
data_x = data.values[:, 1:-5][idx, :]
data_y = data.values[:, -1][idx]

test_x = data.values[:, 1:-5][test_idx, :]
test_y = data.values[:, -1][test_idx]

# 训练随机森林模型
model = RandomForestRegressor(random_state=1,n_estimators=500,min_samples_split=2,
                                 min_samples_leaf=1,max_features='sqrt',max_depth=30)
model.fit(data_x, data_y)

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_x)

# 计算全局重要性
global_shap_values = np.abs(shap_values).mean(axis=0)

# 特征排名
feature_names = data.columns[1:-5]
sorted_indices = np.argsort(global_shap_values)[::-1]

# 打印特征排名
print("Feature Ranking:")
for i, idx in enumerate(sorted_indices):
    print(f"{i+1}. {feature_names[idx]}: {global_shap_values[idx]}")

# replace sample_idx with the index of the sample you want to explain
sample_idx = 0

# 计算单个预测的SHAP值
sample_shap_values = shap_values[sample_idx]

# 获取特定样本的特征值
sample_features = test_x[sample_idx]

# 打印每个特征的局部贡献
print("Local Contributions to Prediction:")
for i, (feature_name, shap_value) in enumerate(zip(feature_names, sample_shap_values)):
    print(f"{feature_name}: {shap_value}")

# 计算总体偏差，即模型的基准预测值
base_value = explainer.expected_value

# 计算单个样本的预测值
predicted_value = base_value + np.sum(sample_shap_values)

# 判断预测值的正负
prediction_direction = "Positive" if predicted_value > base_value else "Negative"

# 打印预测值及其方向
print(f"Predicted Value: {predicted_value}")
print(f"Prediction Direction: {prediction_direction}")

#############################################################################################
# 临时设置字体
plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial

# SHAP摘要图
shap.summary_plot(shap_values, test_x, feature_names=feature_names)

# 变量重要性图
shap.summary_plot(shap_values, test_x, feature_names=data.columns[1:-5])

# 定义要分析的特征的索引
features = [0, 1, 2]  # 例如，选择前三个特征进行分析

# 绘制局部敏感性分析图
plot_partial_dependence(model, data_x, features, feature_names=data.columns[1:-5],
                        grid_resolution=50)  # grid_resolution为分辨率，即要绘制的曲线的点数
plt.suptitle('Partial dependence of class probability on features')
plt.subplots_adjust(top=0.9)  # 调整子图之间的间距
plt.show()

# SHAP值的平均图 绝对值的均值
shap.summary_plot(shap_values, test_x, plot_type="bar",feature_names=data.columns[1:-5])

shap.summary_plot(shap_values, test_x, plot_type="violin", feature_names=data.columns[1:-5])


#waterfall图
shap_exp1 = shap.Explanation(
    values=shap_values,  # 将整个 SHAP 值列表传递给 Explanation
    base_values=explainer.expected_value,
    data=test_x
)

# 绘制SHAP Waterfall plot
shap.plots.waterfall(shap_exp1[0])  # 绘制第一个实例的 SHAP Waterfall 图
plt.show()

shap_values_mean = np.mean(shap_values, axis=0)
# 将平均 SHAP 值转换为 Explanation 对象
shap_exp2 = shap.Explanation(
    values=[shap_values_mean],  # 注意这里传递的是一个二维数组
    base_values=explainer.expected_value,
    data=test_x
)
shap.plots.waterfall(shap_exp2[0], max_display=14)



# 定义特征名称列表
feature_names = ['Slope.mean.', 'Slope.std.', 'VRM.mean.', 'VRM.std.', 'planCurv.mean.', 'planCurv.std.', 'profCurv.mean.', 'profCurv.std.', 'dist2fault', 'PGA', 'Lithology', 'Total.precipitation', 'SnowCov', 'Tempdiff']
# 绘制SHAP依赖图
shap.dependence_plot('Slope.mean.', shap_values, test_x, interaction_index='Slope.std.', feature_names=feature_names)


# 绘制力图用于局部分析的
# 获取 Time=2 的样本索引
sample_indices = range(130,133)
# 从 SHAP 值中提取出选中样本的数据
selected_shap_values = shap_values[sample_indices]
# 计算选中样本的平均 SHAP 值
mean_selected_shap_values = np.mean(selected_shap_values, axis=0)
# 绘制 SHAP 响应图
force_plot = shap.force_plot(explainer.expected_value, mean_selected_shap_values, feature_names=data.columns[1:-5])
plt.show()
shap.save_html('force_plot_time_4.html', force_plot)



# 选择一个特征作为横轴
feature_index = 0
feature_name = 'Slope.mean.'
# 对测试数据进行预测
test_predictions = model.predict(test_x)
# 提取选择的特征
feature_values = test_x[:, feature_index]
# 绘制响应图
plt.scatter(feature_values, test_predictions)
plt.xlabel('Your_Selected_Feature_Name')
plt.ylabel('Model_Predictions')
plt.title('Response Plot')
plt.show()
plt.hexbin(feature_values, test_predictions, gridsize=30, cmap='Blues')
plt.colorbar(label='Density')
plt.xlabel('Your_Selected_Feature_Name')
plt.ylabel('Model_Predictions')
plt.title('Response Plot (Hexbin)')
plt.show()


# 将特征值范围划分成等间隔的区间
num_bins = 25  # 可以根据实际情况修改区间数量
bin_edges = np.linspace(np.min(feature_values), np.max(feature_values), num_bins + 1)
# 计算每个区间的平均预测值
bin_means = []
for i in range(num_bins):
    mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
    bin_mean = np.mean(test_predictions[mask])
    bin_means.append(bin_mean)
# 绘制响应曲线
plt.plot(bin_edges[:-1], bin_means, marker='o')
plt.xlabel(feature_name)
plt.ylabel('Average Model Prediction')
plt.title('Response Curve')
plt.show()




# 获取每个特征的SHAP值的均值
mean_shap_values = np.mean(shap_values, axis=0)
import matplotlib.colors as mcolors
# 定义天蓝色和玫瑰红色的 RGB 或 RGBA 值
skyblue = (0/255, 139/255, 251/255)  # 天蓝色的 RGB 值
rosepink = (255/255, 0/255, 80/255)   # 玫瑰红色的 RGB 值
# 创建颜色列表，其中包含天蓝色和玫瑰红色
colors = [skyblue, rosepink]
# 创建颜色映射
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
# 绘制条形图，并设置颜色
plt.figure(figsize=(10, 9))
bars = plt.barh(data.columns[1:-5], mean_shap_values, color=cmap((mean_shap_values > 0).astype(float)))
plt.tight_layout()
plt.yticks(rotation=45)
plt.xlabel('Mean SHAP Value')
plt.ylabel('Feature Name')
plt.title('Mean SHAP Values by Feature')
plt.show()


# 获取 Time=2 的样本索引
sample_indices = range(130,133)

# 从 SHAP 值中提取出选中样本的数据
selected_shap_values = shap_values[sample_indices]

# 计算选中样本的平均 SHAP 值
mean_selected_shap_values = np.mean(selected_shap_values, axis=0)
# 绘制 SHAP 力图
force_plot = shap.force_plot(explainer.expected_value, mean_selected_shap_values, feature_names=data.columns[1:-5])
plt.show()
shap.save_html('force_plot_time_4.html', force_plot)
