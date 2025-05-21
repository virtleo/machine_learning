from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# 加载数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 训练模型
model = GaussianNB()
model.fit(X_train, y_train)
# 评估
print("Accuracy:", model.score(X_test, y_test))
from sklearn.naive_bayes import MultinomialNB

# 文本分类：使用新闻组数据集和TF-IDF+朴素贝叶斯
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载文本数据
categories = ['sci.space', 'rec.autos']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
X_text, y_text = newsgroups.data, newsgroups.target
# 构建流水线（TF-IDF + 分类器）
pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB(alpha=0.1))
pipeline.fit(X_text, y_text)
