"""
感知机模型综合实验系统
包含原始形式（支持SGD/BGD/Mini-batch GD）和对偶形式实现
实现学习率策略、训练过程可视化及对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import warnings

# 配置可视化样式
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
warnings.filterwarnings('ignore')  # 忽略警告信息

#%% 数据准备模块
class DataGenerator:
    """
    数据生成与预处理类
    功能：
    1. 生成线性可分/不可分数据
    2. 数据标准化
    3. 训练测试集划分
    """
    @staticmethod
    def generate_data(noise=False, n_samples=100, n_features=2):
        """
        生成二分类数据集
        :param noise: 是否添加噪声
        :param n_samples: 样本量
        :param n_features: 特征维度
        :return: 标准化后的训练集、测试集
        """
        # 设置不同的随机种子保证实验可重复性
        if noise:
            X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                     n_redundant=0, flip_y=0.3, random_state=42)
        else:
            X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                     n_redundant=0, n_clusters_per_class=1, random_state=42)
        
        y = np.where(y == 0, -1, 1)  # 标签转换为{-1, 1}
        
        # 数据标准化（重要：需先拟合训练集再转换测试集）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练测试集
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#%% 感知机模型实现
class PrimalPerceptron:
    """
    原始形式感知机（支持三种梯度下降策略）
    
    参数说明：
    lr: 初始学习率（支持常数或衰减策略）
    max_iter: 最大迭代次数
    grad_type: 优化策略 ['sgd'|'bgd'|'mini-batch']
    batch_size: 小批量样本数（仅mini-batch模式有效）
    decay: 是否启用学习率衰减（线性衰减策略）
    tolerance: 早停阈值（连续n次损失不下降则停止）
    
    属性说明：
    w: 权重向量
    b: 偏置项
    losses: 各epoch损失记录（误分类数）
    """
    def __init__(self, lr=0.01, max_iter=1000, grad_type='sgd', 
                 batch_size=32, decay=False, tolerance=5):
        self.lr = lr
        self.max_iter = max_iter
        self.grad_type = grad_type.lower()
        self.batch_size = batch_size
        self.decay = decay
        self.tolerance = tolerance
        
        # 模型参数初始化
        self.w = None
        self.b = 0
        self.losses = []
        self.acc_history = []  # 记录准确率变化
    
    def fit(self, X, y):
        """
        训练入口方法
        实现三种梯度下降策略的切换
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        current_lr = self.lr  # 当前学习率（考虑衰减）
        best_loss = float('inf')
        no_improve = 0  # 早停计数器
        
        for epoch in range(self.max_iter):
            # 根据当前学习率更新权重
            loss = self._update_weights(X, y, current_lr)
            self.acc_history.append(self.score(X, y))
            
            # 学习率衰减（线性衰减）
            if self.decay:
                current_lr = self.lr / (1 + epoch * 0.1)
            
            # 记录损失并检查早停
            self.losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= self.tolerance:
                print(f"早停触发，第{epoch}次迭代")
                break
            if loss == 0:
                break
        
        return self
    
    def _update_weights(self, X, y, lr):
        """根据梯度下降类型更新权重"""
        if self.grad_type == 'sgd':
            return self._sgd_update(X, y, lr)
        elif self.grad_type == 'bgd':
            return self._bgd_update(X, y, lr)
        else:
            return self._mini_batch_update(X, y, lr)
    
    def _sgd_update(self, X, y, lr):
        """随机梯度下降更新策略"""
        loss = 0
        indices = np.random.permutation(len(X))
        for i in indices:
            xi, yi = X[i], y[i]
            if yi * (np.dot(self.w, xi) + self.b) <= 0:
                self.w += lr * yi * xi
                self.b += lr * yi
                loss += 1
        return loss
    
    def _bgd_update(self, X, y, lr):
        """批量梯度下降更新策略"""
        grad_w = np.zeros_like(self.w)
        grad_b = 0
        loss = 0
        
        for xi, yi in zip(X, y):
            margin = yi * (np.dot(self.w, xi) + self.b)
            if margin <= 0:
                grad_w += yi * xi
                grad_b += yi
                loss += 1
        
        self.w += lr * grad_w
        self.b += lr * grad_b
        return loss
    
    def _mini_batch_update(self, X, y, lr):
        """小批量梯度下降更新策略"""
        loss = 0
        indices = np.random.permutation(len(X))
        
        for i in range(0, len(X), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            grad_w = np.zeros_like(self.w)
            grad_b = 0
            
            for xi, yi in zip(X_batch, y_batch):
                margin = yi * (np.dot(self.w, xi) + self.b)
                if margin <= 0:
                    grad_w += yi * xi
                    grad_b += yi
                    loss += 1
            
            self.w += lr * grad_w
            self.b += lr * grad_b
        
        return loss
    
    def predict(self, X):
        """预测方法"""
        return np.sign(X @ self.w + self.b)
    
    def score(self, X, y):
        """模型准确率评估"""
        return np.mean(self.predict(X) == y)
class DualPerceptron:
    """
    对偶形式感知机
    特点：通过Gram矩阵加速内积计算
    
    参数说明：
    lr: 学习率（实际为参数更新步长）
    max_iter: 最大迭代次数
    
    属性说明：
    alpha: 拉格朗日乘子向量
    b: 偏置项
    gram: Gram矩阵（空间换时间）
    """
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.gram = None
        self.X_train = None
        self.y_train = None
        self.losses = []  # 新增损失记录列表
        self.acc_history = []  # 记录准   
    def fit(self, X, y):
        """训练方法"""
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.gram = X @ X.T  # 预计算Gram矩阵
        self.X_train = X      # 保存训练数据用于预测
        self.y_train = y
        
        for epoch in range(self.max_iter):
            errors = 0
            for i in range(n_samples):
                margin = y[i] * (np.sum(self.alpha * y * self.gram[i]) + self.b)
                if margin <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
                    errors += 1
            self.losses.append(errors)  # 新增损失记录
            self.acc_history.append(self.score(X, y))
            if errors == 0:
                break
        
        return self
    
    def predict(self, X):
        """预测方法（需与训练样本计算内积）"""
        scores = np.array([
            np.sum(self.alpha * self.y_train * (self.X_train @ x)) + self.b
            for x in X
        ])
        return np.sign(scores)
    
    def score(self, X, y):
        """模型准确率评估"""
        return np.mean(self.predict(X) == y)

#%% 可视化模块
class Visualizer:
    """
    可视化工具类
    功能：
    1. 绘制损失曲线对比图
    2. 绘制决策边界动态变化
    3. 模型性能对比柱状图
    """
    @staticmethod
    def plot_loss_curves(models, model_names, title=''):
        """
        绘制多模型损失曲线对比
        :param models: 已训练模型列表
        :param model_names: 模型名称列表
        :param title: 图表标题
        """
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple']
        
        for model, name, color in zip(models, model_names, colors):
            losses = model.losses
            plt.plot(range(len(losses)), losses, 
                    label=name, color=color, alpha=0.8)
        
        plt.title(title)
        plt.xlabel('迭代次数')
        plt.ylabel('误分类数量')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    @staticmethod
    def plot_decision_surface(models, model_names, X, y):
        """
        绘制多模型决策边界对比
        :param models: 已训练模型列表
        :param model_names: 模型名称列表
        :param X: 特征数据
        :param y: 真实标签
        """
        plt.figure(figsize=(15, 10))
        
        # 生成网格数据
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        for idx, (model, name) in enumerate(zip(models, model_names), 1):
            plt.subplot(2, 2, idx)
            
            # 预测网格点
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            # 绘制决策区域
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            # 绘制数据点
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', edgecolors='k')
            # 高亮支持向量（对偶形式）
            if isinstance(model, DualPerceptron):
                sv_indices = model.alpha > 0
                plt.scatter(model.X_train[sv_indices, 0], 
                           model.X_train[sv_indices, 1],
                           s=100, facecolors='none', edgecolors='k', 
                           label='支持向量')
            
            plt.title(f"{name}\n准确率: {model.score(X, y):.2f}")
            plt.xlabel('特征1')
            plt.ylabel('特征2')
        
        plt.tight_layout()
        plt.show()

#%% 实验分析模块
class Experiment:
    """
    综合实验执行类
    包含预设的实验流程：
    1. 不同学习率对比实验
    2. 不同优化策略对比实验
    3. 噪声数据实验
    4. 计算效率对比实验
    """
    @staticmethod
    def learning_rate_experiment():
        """不同学习率对比实验"""
        X_train, X_test, y_train, y_test = DataGenerator.generate_data()
        
        # 初始化模型配置
        models = [
            PrimalPerceptron(lr=0.001, grad_type='sgd', max_iter=200),
            PrimalPerceptron(lr=0.01, grad_type='sgd', max_iter=200),
            PrimalPerceptron(lr=0.1, grad_type='sgd', max_iter=200),
            DualPerceptron(lr=0.1, max_iter=200)
        ]
        model_names = ['SGD (lr=0.001)', 'SGD (lr=0.01)', 
                      'SGD (lr=0.1)', 'Dual (lr=0.1)']
        
        # 训练模型
        trained_models = []
        for model in models:
            model.fit(X_train, y_train)
            trained_models.append(model)
        
        # 可视化结果
        Visualizer.plot_loss_curves(trained_models, model_names, 
                                  '不同学习率下的训练损失对比')
        Visualizer.plot_decision_surface(trained_models, model_names, X_test, y_test)
    
    @staticmethod
    def optimization_comparison():
        """不同优化策略对比实验"""
        X_train, X_test, y_train, y_test = DataGenerator.generate_data()
        
        # 初始化四种优化策略模型
        models = [
            PrimalPerceptron(lr=0.1, grad_type='sgd', max_iter=200),
            PrimalPerceptron(lr=0.1, grad_type='bgd', max_iter=200),
            PrimalPerceptron(lr=0.1, grad_type='mini-batch', max_iter=200),
            DualPerceptron(lr=0.1, max_iter=200)
        ]
        model_names = ['随机梯度下降', '批量梯度下降', 
                      '小批量梯度下降', '对偶形式']
        
        # 训练并记录时间
        training_times = []
        trained_models = []
        for model in models:
            start = time.time()
            model.fit(X_train, y_train)
            training_times.append(time.time() - start)
            trained_models.append(model)
        
        # 输出性能报告
        print("\n=== 训练时间对比 ===")
        for name, time_cost in zip(model_names, training_times):
            print(f"{name}: {time_cost:.4f}秒")
        
        # 可视化对比
        Visualizer.plot_loss_curves(trained_models, model_names, 
                                  '不同优化策略损失曲线对比')
        Visualizer.plot_decision_surface(trained_models, model_names, X_test, y_test)
    
    @staticmethod
    def noise_data_experiment():
        """噪声数据实验"""
        X_train, X_test, y_train, y_test = DataGenerator.generate_data(noise=True)
        
        # 初始化模型
        models = [
            PrimalPerceptron(lr=0.1, grad_type='sgd', max_iter=500),
            DualPerceptron(lr=0.1, max_iter=500)
        ]
        model_names = ['原始形式(SGD)', '对偶形式']
        
        # 训练模型
        trained_models = []
        for model in models:
            model.fit(X_train, y_train)
            trained_models.append(model)
        
        # 可视化噪声数据下的表现
        Visualizer.plot_loss_curves(trained_models, model_names, 
                                  '噪声数据训练损失对比')
        Visualizer.plot_decision_surface(trained_models, model_names, X_test, y_test)
    
    @staticmethod
    def efficiency_analysis():
        """计算效率对比实验"""
        data_configs = [
            (1000, 100),   # 高维数据
            (5000, 10),    # 大样本
            (10000, 50)    # 中等规模
        ]
        
        for n_samples, n_features in data_configs:
            X, y = make_classification(n_samples=n_samples, n_features=n_features)
            y = np.where(y == 0, -1, 1)
            
            # 原始形式计时
            start = time.time()
            PrimalPerceptron(max_iter=100).fit(X, y)
            primal_time = time.time() - start
            
            # 对偶形式计时
            start = time.time()
            DualPerceptron(max_iter=100).fit(X, y)
            dual_time = time.time() - start
            
            print(f"n={n_samples}, d={n_features} | "
                 f"原始形式: {primal_time:.2f}s, 对偶形式: {dual_time:.2f}s")

#%% 主程序入口
if __name__ == "__main__":
    # 执行完整实验流程
    print("=== 学习率对比实验 ===")
    Experiment.learning_rate_experiment()
    
    print("\n=== 优化策略对比实验 ===")
    Experiment.optimization_comparison()
    
    print("\n=== 噪声数据实验 ===")
    Experiment.noise_data_experiment()
    
    print("\n=== 计算效率分析 ===")
    Experiment.efficiency_analysis()
def learning_rate_experiment():
    """不同学习率对收敛速度的影响实验"""
    X_train, X_test, y_train, y_test = DataGenerator.generate_data()
    
    # 实验参数配置
    learning_rates = [0.001, 0.01, 0.1, 0.5]  # 测试不同学习率
    model_types = {
        'SGD': lambda lr: PrimalPerceptron(lr=lr, grad_type='sgd', max_iter=200),
        'BGD': lambda lr: PrimalPerceptron(lr=lr, grad_type='bgd', max_iter=200),
        'MiniBatch': lambda lr: PrimalPerceptron(lr=lr, grad_type='mini-batch', max_iter=200),
        'Dual': lambda lr: DualPerceptron(lr=lr, max_iter=200)
    }
    
    # 结果存储
    results = {name: {'losses': [], 'final_acc': []} for name in model_types}
    
    # 执行实验
    for lr in learning_rates:
        plt.figure(figsize=(15, 10))
        
        for idx, (name, builder) in enumerate(model_types.items(), 1):
            model = builder(lr)
            model.fit(X_train, y_train)
            
            # 记录结果
            results[name]['losses'].append(model.losses)
            acc = model.score(X_test, y_test)
            results[name]['final_acc'].append(acc)
            
            # 绘制学习曲线
            plt.subplot(2, 2, idx)
            plt.plot(model.losses, label=f'LR={lr}', alpha=0.7)
            plt.title(f'{name} (最终准确率: {acc:.2%})')
            plt.xlabel('迭代次数')
            plt.ylabel('误分类数')
            plt.legend()
            plt.grid(True)
        
        plt.suptitle(f'学习率对比实验 (α={lr})', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # 生成分析报告
    print("\n=== 学习率影响分析报告 ===")
    for name in model_types:
        best_lr_idx = np.argmax(results[name]['final_acc'])
        best_lr = learning_rates[best_lr_idx]
        print(f"{name}模型:")
        print(f"最佳学习率: {best_lr} (准确率: {results[name]['final_acc'][best_lr_idx]:.2%})")
        print("各学习率收敛速度:")
        for lr, losses in zip(learning_rates, results[name]['losses']):
            conv_step = len(losses) if losses[-1]==0 else '未完全收敛'
            print(f"α={lr}: 收敛步数={conv_step}")
        print("-"*40)
def plot_convergence_analysis(results, learning_rates):
    """绘制多模型收敛分析图"""
    plt.figure(figsize=(15, 8))
    
    # 绘制收敛速度对比
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        conv_steps = [len(loss) if loss[-1]==0 else 200 for loss in data['losses']]
        plt.plot(learning_rates, conv_steps, 'o-', label=name)
    
    plt.xlabel('学习率')
    plt.ylabel('收敛所需迭代次数')
    plt.title('不同学习率下的收敛速度')
    plt.legend()
    plt.grid(True)
    
    # 绘制最终准确率对比
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(learning_rates, data['final_acc'], 'o-', label=name)
    
    plt.xlabel('学习率')
    plt.ylabel('测试集准确率')
    plt.title('不同学习率下的模型性能')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
# 执行实验
results = learning_rate_experiment()

# 高级分析
plot_convergence_analysis(results, learning_rates=[0.001, 0.01, 0.1, 0.5])
