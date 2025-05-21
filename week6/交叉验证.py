import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, KFold, 
                                     StratifiedKFold, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from matplotlib import rcParams

# 设置字体为 SimHei（黑体），以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 设置随机种子保证结果可复现
np.random.seed(42)

# 1. 数据准备
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 查看数据基本信息
print(f"数据集形状: {X.shape}")
print(f"特征数量: {len(feature_names)}")
print(f"类别分布:\n{pd.Series(y).value_counts()}")
print(f"类别名称: {target_names}")

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 3. 交叉验证实现
# 创建不同交叉验证策略
cv_methods = {
    '5-Fold': KFold(n_splits=5, shuffle=True, random_state=42),
    '10-Fold': KFold(n_splits=10, shuffle=True, random_state=42),
    'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified 10-Fold': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
}

# 初始化SVM模型
svm_model = SVC(kernel='linear', random_state=42)

# 比较不同交叉验证方法
cv_results = {}
for name, cv in cv_methods.items():
    scores = cross_val_score(svm_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores
    }
    print(f"{name}交叉验证结果: 平均准确率={cv_results[name]['mean']:.4f} (±{cv_results[name]['std']:.4f})")

# 修复代码：对齐数组长度
max_len = max(len(v['scores']) for v in cv_results.values())  # 找到最长的数组长度

# 填充较短的数组，使其与最长数组长度一致
aligned_scores = {
    k: np.pad(v['scores'], (0, max_len - len(v['scores'])), constant_values=np.nan)
    for k, v in cv_results.items()
}

# 创建 DataFrame
cv_results_df = pd.DataFrame(aligned_scores)

# 可视化
plt.figure(figsize=(12, 6))
sns.boxplot(data=cv_results_df)
plt.title('不同交叉验证方法的准确率分布比较')
plt.ylabel('准确率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 可视化分析：绘制不同 k 值下交叉验证准确率的分布箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(data=cv_results_df)
plt.title('不同 k 值下交叉验证准确率的分布箱线图')
plt.ylabel('准确率')
plt.xlabel('交叉验证方法')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. 模型选择与超参数优化
# SVM网格搜索
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

svm_grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid=svm_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)

svm_grid_search.fit(X_train, y_train)

print("\nSVM网格搜索结果:")
print(f"最优参数组合: {svm_grid_search.best_params_}")
print(f"最优模型交叉验证准确率: {svm_grid_search.best_score_:.4f}")

# SVM随机搜索
svm_random_search = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions=svm_param_grid,
    n_iter=20,  # 随机尝试20组参数
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

svm_random_search.fit(X_train, y_train)

print("\nSVM随机搜索结果:")
print(f"最优参数组合: {svm_random_search.best_params_}")
print(f"最优模型交叉验证准确率: {svm_random_search.best_score_:.4f}")

# 随机森林网格搜索
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],  # 森林中树的数量
    'max_depth': [5, 10, 15, 20, None]   # 树的最大深度
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)

# 执行网格搜索
rf_grid_search.fit(X_train, y_train)

# 输出最优参数和最优模型的交叉验证准确率
print("\n随机森林网格搜索结果:")
print(f"最优参数组合: {rf_grid_search.best_params_}")
print(f"最优模型交叉验证准确率: {rf_grid_search.best_score_:.4f}")

# 5. 多模型比较
models = {
    'SVM': svm_grid_search.best_estimator_,
    'Random Forest': rf_grid_search.best_estimator_,
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42)
}

model_results = {}
for name, model in models.items():
    # 使用分层5折交叉验证评估
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    model_results[name] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores
    }
    print(f"\n{name}模型评估:")
    print(f"交叉验证准确率: {model_results[name]['mean']:.4f} (±{model_results[name]['std']:.4f})")
    
    # 在测试集上评估
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {test_acc:.4f}")
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.title(f'{name}混淆矩阵')
    plt.show()

# 可视化多模型比较
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame({k: v['scores'] for k, v in model_results.items()}))
plt.title('不同模型的交叉验证准确率分布比较')
plt.ylabel('准确率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. 扩展任务：时间序列交叉验证（假设数据是按时间顺序收集的）
# 自定义时间序列交叉验证
class TimeSeriesKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            yield train_indices, test_indices

# 使用时间序列交叉验证
time_cv = TimeSeriesKFold(n_splits=5)
time_scores = cross_val_score(
    svm_grid_search.best_estimator_, 
    X_train, y_train, 
    cv=time_cv, 
    scoring='accuracy'
)

print("\n时间序列交叉验证结果:")
print(f"平均准确率: {np.mean(time_scores):.4f} (±{np.std(time_scores):.4f})")