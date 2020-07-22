## 学习框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722222144211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)

## 使用Pandas读取数据
&emsp;&emsp;根据[Pandas中文文档](https://www.pypandas.cn)的描述，Pandas内置了CSV的数据读取接口，对于赛题数据可直接读取。

```python
import pandas as pd
train_df = pd.read_csv('../data/train_set.csv', sep='\t')
print(train_df.head())
```

&emsp;&emsp;``nrows=100``是读取前一百行数据，这里我直接读取了20w全部的数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722222714147.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)
&emsp;&emsp;这里的``read_csv``由三部分构成：

 - 读取的文件路径，这里需要根据改成你本地的路径，可以使用相对路径或绝对路径；
 - 分隔符sep，为每列分割的字符，设置为``\t``即可；
 - 读取行数nrows，为此次读取文件的函数，是数值类型（由于数据集比较大，建议先设置为100）；

&emsp;&emsp;此外有一个小坑，也就是展示数据。如果用的是pycharm，需要在``print()``内输出结果。

## 数据分析
### 句子长度分析

```python
#%%pylab inline
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722223320168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)
&emsp;&emsp;每个句子平均由907个字符构成，最短的句子长度为2，最长的句子长度为57921。

&emsp;&emsp;下图将句子长度绘制了直方图

```python
import matplotlib.pyplot as plt

_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")

plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722225003149.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)
&emsp;&emsp;这里针对于pycharm又有一个小坑，那就是即使安装了matplotlib，图像也无法显示。在google之后发现，可能是由于mac本身自带py2.7，而我现在用的是py3.8，相当于版本重复，导致后端绘图版本对不上～此时只需要加一行``plt.show()``即可解决～

### 新闻类别分布
&emsp;&emsp;接下来可以对数据集的类别进行分布统计，具体统计每类新闻的样本个数。

```python
train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722225344939.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)
&emsp;&emsp;在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
&emsp;&emsp;从统计结果可以看出，赛题的数据集类别分布存在较为不均匀的情况。在训练集中科技类新闻最多，其次是股票类新闻，最少的新闻是星座新闻。

### 字符分布统计
&emsp;&emsp;接下来可以统计每个字符出现的次数，首先可以将训练集中所有的句子进行拼接进而划分为字符，并统计每个字符的个数。从统计结果中可以看出，在训练集中总共包括6869个字，其中编号3750的字出现的次数最多，编号3133的字出现的次数最少。

```python
from collections import Counter
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))

print(word_count[0])

print(word_count[-1]
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722225531220.png)

&emsp;&emsp;这里还可以根据字在每个句子的出现情况，反推出标点符号。下面代码统计了不同字符在句子中出现的次数，其中字符3750，字符900和字符648在20w新闻的覆盖率接近99%，很有可能是标点符号。

```python
from collections import Counter
train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])

print(word_count[1])

print(word_count[2])
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200722225614677.png)

### 结论
&emsp;&emsp;通过上述分析我们可以得出以下结论：

1. 赛题中每个新闻包含的字符个数平均为1000个，还有一些新闻字符较长；
2. 赛题中新闻类别分布不均匀，科技类新闻样本量接近4w，星座类新闻样本量不到1k；
3. 赛题总共包括7000-8000个字符；

&emsp;&emsp;通过数据分析，我们还可以得出以下结论：

5. 每个新闻平均字符个数较多，可能需要截断；

6. 由于类别不均衡，会严重影响模型的精度；

## 作业
&emsp;&emsp;不是很懂，正在学～～～




>如果有错误或者不严谨的地方，请务必给予指正，十分感谢。



