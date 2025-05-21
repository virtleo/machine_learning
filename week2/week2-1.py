from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
data = load_diabetes()
X, y = data.data, data.target

# 先划分数据集，再标准化（修正数据泄漏）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 添加偏置项
X_b_train = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]

# 解析解
theta_best = np.linalg.pinv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)

# 预测函数
def predict(X):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta_best)

# 梯度下降（带损失记录）
def gradient_descent(X, y, learning_rate=0.001, n_iters=1000):
    m = X.shape[0]
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(X_b.shape[1], 1)
    loss_history = []
    
    for _ in range(n_iters):
        y_pred = X_b.dot(theta).flatten()
        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y.reshape(-1,1))
        theta -= learning_rate * gradients
    return theta, loss_history

# 调用梯度下降（学习率对比）
learning_rates = [0.001, 0.01, 0.1]
results = {}

for lr in learning_rates:
    theta_gd, losses = gradient_descent(X_train_scaled, y_train, learning_rate=lr, n_iters=1000)
    y_pred_gd = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled].dot(theta_gd).flatten()
    mse = mean_squared_error(y_test, y_pred_gd)
    results[f"GD (lr={lr})"] = {
        "theta": theta_gd,
        "losses": losses,
        "MSE": mse,
        "MAE": mean_absolute_error(y_test, y_pred_gd),
        "R²": r2_score(y_test, y_pred_gd)
    }

# 解析解评估
y_pred = predict(X_test_scaled)
results["Analytical"] = {
    "MSE": mean_squared_error(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R²": r2_score(y_test, y_pred)
}

# 输出结果
print("="*40)
print("{:<15} {:<10} {:<10} {:<10}".format("Method", "MSE", "MAE", "R²"))
print("-"*40)
for method, metrics in results.items():
    if "Analytical" in method:
        print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(
            method, metrics["MSE"], metrics["MAE"], metrics["R²"]))
    else:
        print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(
            method, metrics["MSE"], metrics["MAE"], metrics["R²"]))
print("="*40)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
for method, metrics in results.items():
    if "GD" in method:
        plt.plot(metrics["losses"], label=method)
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.title("Loss Curve for Different Learning Rates")
plt.legend()
plt.show()

# 可视化预测结果
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Analytical Solution: True vs Predicted')
plt.show()

# 尝试单特征模型
feature_names = data.feature_names
single_feature_results = {}

for i in range(X.shape[1]):
    # 选择单个特征
    X_single = X_train_scaled[:, i].reshape(-1, 1)
    
    # 训练模型
    theta_gd, _ = gradient_descent(X_single, y_train, learning_rate=0.1, n_iters=1000)
    X_test_single = X_test_scaled[:, i].reshape(-1, 1)
    y_pred_gd = np.c_[np.ones((X_test_single.shape[0], 1)), X_test_single].dot(theta_gd).flatten()
    
    # 记录结果
    single_feature_results[feature_names[i]] = {
        "MSE": mean_squared_error(y_test, y_pred_gd),
        "R²": r2_score(y_test, y_pred_gd)
    }

# 输出结果
print("{:<15} {:<10} {:<10}".format("Feature", "MSE", "R²"))
print("-"*35)
for feature, metrics in single_feature_results.items():
    print("{:<15} {:<10.2f} {:<10.2f}".format(feature, metrics["MSE"], metrics["R²"]))