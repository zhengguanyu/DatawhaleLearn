&emsp;&emsp;在本章我们将开始使用机器学习模型来解决文本分类。机器学习发展比较广，且包括多个分支，本章侧重使用传统机器学习，从下一章开始是基于深度学习的文本分类。
## 学习目标
- 使用sklearn的机器学习模型完成文本分类
- 学会TF-IDF的原理和使用
## 机器学习模型
&emsp;&emsp;机器学习是对能通过经验自动改进的计算机算法的研究。机器学习通过历史数据训练出模型对应于人类对经验进行归纳的过程，机器学习利用模型对新数据进行预测对应于人类利用总结的规律对新问题进行预测的过程。
&emsp;&emsp;机器学习有很多种分支，对于学习者来说应该优先掌握机器学习算法的分类，然后再其中一种机器学习算法进行学习。由于机器学习算法的分支和细节实在是太多，所以如果你一开始就被细节迷住了眼，你就很难知道全局是什么情况的。
&emsp;&emsp;如果你是机器学习初学者，你应该知道如下的事情：
1. 机器学习能解决一定的问题，但不能奢求机器学习是万能的；
2. 机器学习算法有很多种，看具体问题需要什么，再来进行选择；
3. 每种机器学习算法有一定的偏好，需要具体问题具体分析；


## 文本表示方法 Part1
&emsp;&emsp;文本表示成计算机能够运算的数字或向量的方法一般称为词嵌入（Word Embedding）方法。词嵌入将不定长的文本转换到定长的空间内，是文本分类的第一步。
### One-hot
&emsp;&emsp;这里的One-hot与数据挖掘任务中的操作是一致的，即将每一个单词使用一个离散的向量表示。具体将每个字/词编码一个索引，然后根据索引进行赋值。

&emsp;&emsp;One-hot表示方法的例子如下：

>句子1：我 爱 北 京 天 安 门
句子2：我 喜 欢 上 海

&emsp;&emsp;首先对所有句子的字进行索引，即将每个字确定一个编号：

>{
    '我': 1, '爱': 2, '北': 3, '京': 4, '天': 5,
  '安': 6, '门': 7, '喜': 8, '欢': 9, '上': 10, '海': 11
}

&emsp;&emsp;在这里共包括11个字，因此每个字可以转换为一个11维度稀疏向量：

>我：[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
爱：[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
海：[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

### Bag of Words
&emsp;&emsp;Bag of Words（词袋表示），也称为Count Vectors，每个文档的字/词可以使用其出现次数来进行表示。

>句子1：我 爱 北 京 天 安 门
句子2：我 喜 欢 上 海

&emsp;&emsp;直接统计每个字出现的次数，并进行赋值：

>句子1：我 爱 北 京 天 安 门
转换为 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
>句子2：我 喜 欢 上 海
转换为 [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

&emsp;&emsp;在sklearn中可以直接``CountVectorizer``来实现这一步骤：

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus).toarray()
```

### N-gram
&emsp;&emsp;N-gram与Count Vectors类似，不过加入了相邻单词组合成为新的单词，并进行计数。

&emsp;&emsp;如果N取值为2，则句子1和句子2就变为：

>句子1：我爱 爱北 北京 京天 天安 安门
句子2：我喜 喜欢 欢上 上海

### TF-IDF
&emsp;&emsp;TF-IDF 分数由两部分组成：第一部分是词语频率（Term Frequency），第二部分是逆文档频率（Inverse Document Frequency）。其中计算语料库中文档总数除以含有该词语的文档数量，然后再取对数就是逆文档频率。

>TF(t)= 该词语在当前文档出现的次数 / 当前文档中词语的总数
IDF(t)= log_e（文档总数 / 出现该词语的文档总数）

## 基于机器学习的文本分类
&emsp;&emsp;接下来我们将对比不同文本表示算法的精度，通过本地构建验证集计算F1得分

```python
# Count Vectors + RidgeClassifier

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=15000)

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.74
```

```python
# TF-IDF +  RidgeClassifier

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('../data/train_set.csv', sep='\t', nrows=15000)

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# 0.87
```

## SVM
