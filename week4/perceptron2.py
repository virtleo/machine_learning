import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 设置中文字体（需要系统安装对应字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
# 数据准备和预处理
def prepare_data(noise=False):
    """生成并预处理数据"""
    if not noise:
        # 生成线性可分数据
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                 n_clusters_per_class=1, random_state=42)
    else:
        # 生成含噪声数据
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                 flip_y=0.3, random_state=42)
    
    y = np.where(y == 0, -1, 1)  # 标签转换为{-1, 1}
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 原始形式感知机（支持三种梯度下降）
class PrimalPerceptron:
    """原始形式感知机实现
    
    Attributes:
        lr: 学习率
        max_iter: 最大迭代次数
        grad_type: 梯度下降类型（sgd/bgd/mini-batch）
        batch_size: 小批量大小
        decay: 是否启用学习率衰减
    """
    def __init__(self, lr=0.01, max_iter=1000, grad_type='sgd', batch_size=32, decay=False, tolerance=5):
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
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        current_lr = self.lr  # 当前学习率（考虑衰减）
        
        for epoch in range(self.max_iter):
            loss = 0
            
            if self.grad_type == 'sgd':
                # 随机梯度下降
                indices = np.random.permutation(n_samples)
                for i in indices:
                    self._update(X[i], y[i], current_lr)
                    loss += self._compute_loss(X, y)  # 计算当前误分类数
            
            elif self.grad_type == 'bgd':
                # 批量梯度下降
                grad_w, grad_b = self._compute_gradient(X, y)
                self.w += current_lr * grad_w
                self.b += current_lr * grad_b
                loss = self._compute_loss(X, y)
            
            elif self.grad_type == 'mini-batch':
                # 小批量梯度下降
                indices = np.random.permutation(n_samples)
                for i in range(0, n_samples, self.batch_size):
                    batch_idx = indices[i:i+self.batch_size]
                    grad_w, grad_b = self._compute_gradient(X[batch_idx], y[batch_idx])
                    self.w += current_lr * grad_w
                    self.b += current_lr * grad_b
                loss = self._compute_loss(X, y)
            
            # 学习率衰减
            if self.decay:
                current_lr = self.lr / (1 + epoch*0.1)
            
            self.losses.append(loss)
            if loss == 0:
                break  # 提前停止
        
        return self
    
    def _update(self, xi, yi, lr):
        """单样本参数更新"""
        if yi * (np.dot(self.w, xi) + self.b) <= 0:
            self.w += lr * yi * xi
            self.b += lr * yi
    
    def _compute_gradient(self, X, y):
        """计算梯度"""
        grad_w = np.zeros_like(self.w)
        grad_b = 0
        for xi, yi in zip(X, y):
            if yi * (np.dot(self.w, xi) + self.b) <= 0:
                grad_w += yi * xi
                grad_b += yi
        return grad_w, grad_b
    
    def _compute_loss(self, X, y):
        """计算误分类数量"""
        return np.sum(y * (X @ self.w + self.b) <= 0)
    
    def predict(self, X):
        """预测类别"""
        return np.sign(X @ self.w + self.b)

# 对偶形式感知机
class DualPerceptron:
    """对偶形式感知机实现
    
    特点：通过Gram矩阵加速计算
    """
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.alpha = None  # 拉格朗日乘子
        self.b = 0         # 偏置项
        self.gram = None   # Gram矩阵
        self.losses = []
        self.acc_history = []  # 记录准确率变化       
    def score(self, X, y):
        return np.mean(self.predict(X) == y)    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.gram = X @ X.T  # 预计算Gram矩阵
        
        for _ in range(self.max_iter):
            errors = 0
            for i in range(n_samples):
                margin = y[i] * (np.sum(self.alpha * y * self.gram[i]) + self.b)
                if margin <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
                    errors += 1
            if errors == 0:
                break
        
        # 保存训练数据用于预测
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """预测时需要与训练样本计算内积"""
        scores = np.array([
            np.sum(self.alpha * self.y_train * (self.X_train @ x)) + self.b
            for x in X
        ])
        return np.sign(scores)

# 可视化工具函数
def plot_decision_boundary(model, X, y, title=""):
    """绘制决策边界"""
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()
import matplotlib.gridspec as gridspec
# 实验1：四种模型的学习率比较（需要已添加中文字体支持）
def experiment1():
    X_train, X_test, y_train, y_test = prepare_data()
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])    

    plt.figure(figsize=(15, 5))
    model_configs = [
        {
            'name': '随机梯度下降 (SGD)',
            'constructor': lambda lr: PrimalPerceptron(lr=lr, grad_type='sgd'),
            'color': 'orange'
        },
        {
            'name': '批量梯度下降 (BGD)',
            'constructor': lambda lr: PrimalPerceptron(lr=lr, grad_type='bgd'),
            'color': 'green'
        },
        {
            'name': '小批量梯度下降 (Mini-batch)',
            'constructor': lambda lr: PrimalPerceptron(lr=lr, grad_type='mini-batch', batch_size=32),
            'color': 'purple'
        },
        {
            'name': '对偶形式感知机',
            'constructor': lambda lr: DualPerceptron(lr=lr),
            'color': 'brown'
        }
    ]

    # 为每个学习率创建子图
    for lr_idx, lr in enumerate([0.001, 0.01, 0.1], 1):
        plt.subplot(1, 3, lr_idx)
        
        # 遍历所有模型类型
        for config in model_configs:
            # 训练模型
            model = config['constructor'](lr)
            model.fit(X_train, y_train)
            
            # 截取前50次迭代显示更清晰
            losses = model.losses[:50] if len(model.losses) > 50 else model.losses
            plt.plot(losses, 
                    label=config['name'],
                    color=config['color'],
                    linestyle='-',
                    alpha=0.8)

        plt.title(f'学习率 α={lr} 的收敛情况对比')
        plt.xlabel('迭代次数')
        plt.ylabel('误分类数量')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.tight_layout()
    plt.show()

def experiment2():
    """四种模型决策边界对比"""
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 初始化四种模型
    models = [
        ('SGD感知机', PrimalPerceptron(lr=0.01, grad_type='sgd')),
        ('BGD感知机', PrimalPerceptron(lr=0.01, grad_type='bgd')),
        ('Mini-batch感知机', PrimalPerceptron(lr=0.01, grad_type='mini-batch', batch_size=32)),
        ('对偶形式感知机', DualPerceptron(lr=0.01))
    ]
    
    # 训练并绘图
    plt.figure(figsize=(15, 10))
    for idx, (name, model) in enumerate(models, 1):
        model.fit(X_train, y_train)
        plt.subplot(2, 2, idx)
        plot_decision_boundary(model, X_train, y_train)
        plt.title(f"{name}\n准确率: {model.score(X_test, y_test):.2f}")
    
    plt.tight_layout()
    plt.show()




# 改进后的决策边界绘制函数
def plot_decision_boundary(model, X, y, title=""):
    """增强版决策边界可视化"""
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 添加样本点到背景色
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # 绘制支持向量（对偶形式特有）
    if isinstance(model, DualPerceptron):
        sv_indices = model.alpha > 0
        plt.scatter(X[sv_indices, 0], X[sv_indices, 1], 
                   s=100, facecolors='none', edgecolors='k', label='支持向量')
    
    # 绘制数据点
    plt.scatter(X[:,0], X[:,1], c=y, cmap='Paired', edgecolors='k')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
# 实验分析示例
if __name__ == "__main__":
    # 实验1：不同学习率比较
    # X_train, X_test, y_train, y_test = prepare_data()
    
    # plt.figure(figsize=(10,6))
    # for lr in [0.001, 0.01, 0.1]:
    #     model = PrimalPerceptron(lr=lr, grad_type='sgd')
    #     model.fit(X_train, y_train)
    #     plt.plot(model.losses, label=f'lr={lr}')
    # plt.title('不同学习率的收敛速度')
    # plt.xlabel('Epoch')
    # plt.ylabel('Misclassifications')
    # plt.legend()
    # plt.show()
    # 执行实验
    experiment1()
    # 实验2：对偶形式可视化
    # dual_model = DualPerceptron()
    # dual_model.fit(X_train, y_train)
    # plot_decision_boundary(dual_model, X_train, y_train, "Dual Form Decision Boundary")
    experiment2()
    # 实验3：噪声数据表现
    X_train_n, X_test_n, y_train_n, y_test_n = prepare_data(noise=True)
    model = PrimalPerceptron(max_iter=500)
    model.fit(X_train_n, y_train_n)
    plt.plot(model.losses)
    plt.title('Noisy Data Training')
    plt.show()