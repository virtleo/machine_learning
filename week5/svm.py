from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer() 
x, y = data.data, data.target
print("样本数量:", len(x))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

print("标准化后样本数量:", len(x_scaled))

from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix
c_values = [0.1, 1, 10]

print("\n测试集结果:")
for c in c_values:
    print(f"\n正则化参数 c={c}:")
    model = SVC(kernel='linear', C=c, class_weight='balanced') 
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_test)
    print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}") 
    print("测试集混淆矩阵:\n", confusion_matrix(y_test, y_pred))

print("\n训练集结果:")
for c in c_values:
    print(f"\n正则化参数 c={c}:")
    model = SVC(kernel='linear', C=c, class_weight='balanced') 
    model.fit(x_train, y_train) 
    y_pred = model.predict(x_train)
    print(f"训练集准确率: {accuracy_score(y_train, y_pred):.4f}") 
    print("训练集混淆矩阵:\n", confusion_matrix(y_train, y_pred))

# 不同核函数比较
print("\n不同核函数性能比较:")
model_rbf = SVC(kernel='rbf', gamma=0.1, C=1.0, class_weight='balanced') 
model_rbf.fit(x_train, y_train) 
y_pred_rbf = model_rbf.predict(x_test)
print("\nrbf核函数结果:")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_rbf):.4f}") 
print("测试集混淆矩阵:\n", confusion_matrix(y_test, y_pred_rbf))

model_poly = SVC(kernel='poly', degree=3, C=1.0) 
model_poly.fit(x_train, y_train) 
y_pred_poly = model_poly.predict(x_test)
print("\n多项式核函数结果:")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_poly):.4f}") 
print("测试集混淆矩阵:\n", confusion_matrix(y_test, y_pred_poly))

# 可视化支持向量
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
print("\n可视化支持向量...")
pca = PCA(n_components=2) 
x_pca = pca.fit_transform(x_train) 
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_train, cmap='coolwarm') 
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
            s=100, facecolors='none', edgecolors='k', label='support vectors') 
plt.legend() 
plt.title("support vectors visualization")
plt.show()

# 学习曲线
print("\n生成学习曲线...")
from sklearn.model_selection import learning_curve
import numpy as np
train_sizes, train_scores, test_scores = learning_curve(
    SVC(kernel='linear', C=1.0), x_scaled, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title('learning curve')
plt.xlabel("training examples")
plt.ylabel("score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="cross-validation score")
plt.legend(loc="best")
plt.show()

# 网格搜索
print("\n开始网格搜索...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("\n网格搜索结果:")
print("最优参数组合:", grid_search.best_params_)
print("最优模型得分:", grid_search.best_score_)

# 自定义核函数
print("\n自定义核函数测试...")
def linear_kernel(x, y):
    return np.dot(x, y.T)

def poly_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y.T) + coef0) ** degree

def custom_kernel(x, y):
    alpha = 0.3
    beta = 0.4
    return alpha * linear_kernel(x, y) + beta * poly_kernel(x, y) 

model_custom = SVC(kernel=custom_kernel)
model_custom.fit(x_train, y_train) 
y_pred_custom = model_custom.predict(x_test)
print("\n自定义核函数结果:")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_custom):.4f}") 
print("测试集混淆矩阵:\n", confusion_matrix(y_test, y_pred_custom))