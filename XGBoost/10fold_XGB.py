import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
import seaborn as sns
from xgboost import XGBRegressor

def scatter(dataset):
    mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    y_real = dataset['y_test']  # 修改这里的列名为正确的列名
    y_rf = dataset['y_pred']  # 修改这里的列名为正确的列名

    plt.figure(dpi=150)
    plt.plot([0, 20], [0, 20], color="b", linestyle="-", linewidth=1)  # 图线的设置
    #plt.scatter(y_rf, y_real, c='m', marker='^', label="BRP")
    plt.scatter(y_rf, y_real, c='k', marker='o', label="BRP")
    # 手动设置横纵坐标轴的范围和刻度，让原点是同一个
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xticks(np.linspace(0, 20, 5))
    plt.yticks(np.linspace(0, 20, 5))

    # 图例
    plt.rcParams.update({'font.size': 10})  # 图例的大小
    plt.legend(loc='upper left')  # 图例的位置
    plt.ylabel("Predicted mean LOS velocity (mm/year)")  # y轴标签
    plt.xlabel("Calculated mean LOS velocity (mm/year)")  # x轴标签
    plt.show()



if __name__ == '__main__':
    data = pd.read_csv(r"E:\essays\deep_learning\linzhi\data_kun.csv")

    idx1 = np.argwhere((data['Time'].values == 1) & (data['PS_count'].values > 15) & (data['VLOS.std.'].values < 2)).ravel()
    #idx1 = np.argwhere((data['Time'].values == 4)).ravel()
    #idx1 = np.argwhere((data['Time'].values != 0)).ravel()
    data_x = data.values[:, 1:-5][idx1, :]
    data_y = data.values[:, -1][idx1]

    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    mse_scores = []
    pearson_scores = []
    all_true_labels = []
    all_predicted_labels = []

    for fold_num, (train_index, test_index) in enumerate(kf.split(data_x), 1):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

        brt_model = XGBRegressor(random_state=1,n_estimators=300, subsample=0.6,
                        max_depth=5,learning_rate=0.05,gamma=0,colsample_bytree=0.7)
        brt_model.fit(X_train, y_train)
        y_pred = brt_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

        pearson_corr, _ = pearsonr(y_test, y_pred)
        pearson_scores.append(pearson_corr)

        all_true_labels.extend(y_test)
        all_predicted_labels.extend(y_pred)

        fold_results = pd.DataFrame({'True Labels': y_test, 'Predicted Labels': y_pred})
        fold_results.to_csv(f'fold_results_fold_{fold_num}.csv', index=False)

    average_mse = np.mean(mse_scores)
    average_pearson = np.mean(pearson_scores)
    std_pearson = np.std(pearson_scores)

    print(f'Average Mean Squared Error across {n_folds}-fold cross-validation: {average_mse}')
    print("平均十折交叉验证得分（Pearson相关系数）:", average_pearson)
    print("十折交叉验证得分的标准差:", std_pearson)

    results_df = pd.DataFrame({'y_test': all_true_labels, 'y_pred': all_predicted_labels})
    results_df.to_csv('combined_results.csv', index=False)

    scatter(results_df)

# Manually set the range for both x and y axes
plt.xlim(0, 20)
plt.ylim(0, 20)

# Plot the density plot with 13 levels
sns.kdeplot(x=all_true_labels, y=all_predicted_labels, cmap="Blues", fill=True, thresh=0, levels=13)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Density Plot of True vs Predicted Labels')
plt.show()
