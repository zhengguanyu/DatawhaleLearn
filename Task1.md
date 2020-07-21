## [题目概述](https://tianchi.aliyun.com/competition/entrance/531810/introduction)
&emsp;&emsp;赛题数据为新闻文本，并按照字符级别进行**匿名处理**。赛题数据来源为互联网上的新闻，通过收集并匿名处理得到。因此选手可以自行进行数据分析，可以充分发挥自己的特长来完成各种特征工程，不限制使用任何外部数据和模型。

<br/>

## 学习框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200721200002817.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)

<br/>

## 基于[*Task1*](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.6406111aIKCSLV&postId=118252)的简要分析
 &emsp;&emsp;简单理解，就是要求我们读取数据，然后提取数据特征，最后做数据分类。当然，越精准越好。
 &emsp;&emsp;这里，题目反复强调赛题数据为**匿名处理**后的新闻数据，而[*Task1*解析](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.6406111aIKCSLV&postId=118252)提到：对于文本分类问题，需要根据每句的字符进行分类，通常可使用中文分词的方式，但由于赛题数据经过匿名化处理，所以不能直接采用。那什么是[中文分词](https://baike.baidu.com/item/中文分词/371496?fr=aladdin)呢？

> 中文分词，可以简单理解为让计算机读懂中文词句（当然不仅限于中文）的第一步。这就像我们最开始上c语言课一样，老师会告诉我们要用计算机思维编程。为什么要这么做？因为很简单，计算机笨啊，你要告诉它怎么计算出结果。那么回到中文分词，就是让计算机像我们一样知道哪些是词哪些不是词。
> 那么说到中文分词与西文分词的区别，主要还是在于相对于西文，中文的复杂程度要更胜一筹。在我们的中文语句中，一般会淡化词与词之间的界限，从而提高计算机识别语句的难度，这就是为什么要有中文分词去处理语句。


 &emsp;&emsp;那么为什么不能用中文分词呢？我通过检索一些概述来阐述我的理解，欢迎大佬指点迷津。
 1. 首先，当然由于这是字符，不是中文（狗头）；
 2. 其次，中文分词长期存在两个技术难题，一个是分词标准，另一个是歧义，显然遇到匿名字符就更难处理了；
 3. 最后，我简单查阅了中文分词的其中一种方法——[基于*Trie*树的字符串匹配](http://zhanghonglun.cn/blog/中文分词一些思路的总结/)。***Trie*树的本质就是利用字符串之间的公共前缀，将重复的前缀合并在一起**。显然，针对于本题已经匿名化处理的数据，是很难获取准确的公共前缀。


&emsp;&emsp;本人本科毕设做过车牌分割，也就是通过图像处理把车牌分割成单个字符图片，便于后续车牌字符识别。在实际应用场景中，车牌图像会遇到诸多干扰，比如车牌上的铆钉、阳光照射阴影以及泥巴痕迹等等。这时候，普遍的做法是先对车牌图像进行一个预处理（由于编码能力所限，我用的是最大类间方差法，当然深度学习相关的方法更牛了～），然后再二值化处理图像。我认为万法归一，既然无法一步到位，所幸对赛题数据进行预处理。也就是*Task1*所言，我们要对匿名字符进行**建模**。

<br/>

## 基于思路1的初步认识
&emsp;&emsp;**特征提取**和**分类模型**显然就是我们在建模之后要着重解决的两个部分了，简单理解就是获取数据的某种统一规律，然后按一定规则进行分类。这里谈一下对于思路1的初步认识。

##### TF-IDF
&emsp;&emsp;*TF-IDF*（*Term Frequency-inverse Document Frequency*）是一种针对关键词的统计分析方法，用于评估一个词对一个文件集或者一个语料库的重要程度。一个词的重要程度跟它在文章中出现的次数成正比，跟它在语料库出现的次数成反比。
&emsp;&emsp;*TF*（单词频率）是指我们计算一个查询关键字中某一个单词在目标文档中出现的次数。举例说来，如果我们要查询 “*Car Insurance*”，那么对于每一个文档，我们都计算“*Car*” 这个单词在其中出现了多少次，“*Insurance*”这个单词在其中出现了多少次。这个就是 TF 的计算方法。*TF*背后的隐含的假设是，查询关键字中的单词应该相对于其他单词更加重要，而文档的重要程度，也就是相关度，与单词在文档中出现的次数成正比。比如，“*Car*” 这个单词在文档 A 里出现了 5 次，而在文档 B 里出现了 20 次，那么*TF*计算就认为文档 B 可能更相关。
&emsp;&emsp;*IDF*（逆文档频率）需要去 “惩罚”那些出现在太多文档中的单词。也就是说，真正携带 “相关” 信息的单词仅仅出现在相对比较少，有时候可能是极少数的文档里。这个信息，很容易用 “文档频率” 来计算，也就是，有多少文档涵盖了这个单词。很明显，如果有太多文档都涵盖了某个单词，这个单词也就越不重要，或者说是这个单词就越没有信息量。因此，我们需要对*TF*的值进行修正，而* IDF*的想法是用*DF*的倒数来进行修正。倒数的应用正好表达了这样的思想，*DF*值越大越不重要。引用[检索资料](http://easyai.tech/ai-definition/tf-idf/)的一张图就是
![&emsp;&emsp;](https://img-blog.csdnimg.cn/20200721215209421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNzcxNA==,size_16,color_FFFFFF,t_70)

##### SVM
&emsp;&emsp;支持向量机（*support vector machines*, *SVM*）是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；*SVM*还包括核技巧，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。

&emsp;&emsp;看不懂没关系，毕竟一切刚开始。这里推荐一篇[**知乎高赞回答**](https://www.zhihu.com/question/21094489)，先理解一下这个东西是干嘛的～

<br/>


## 参考链接
>- [https://baike.baidu.com/item/中文分词/371496?fr=aladdin](https://baike.baidu.com/item/中文分词/371496?fr=aladdin)
>- [https://www.zhihu.com/search?type=content&q=中文分词](https://www.zhihu.com/search?type=content&q=中文分词)
>- [https://zhuanlan.zhihu.com/p/60267896](https://zhuanlan.zhihu.com/p/60267896)
>- [http://zhanghonglun.cn/blog/中文分词一些思路的总结/](http://zhanghonglun.cn/blog/中文分词一些思路的总结/)
>- [https://www.zhihu.com/question/19578687](https://www.zhihu.com/question/19578687)
>- [http://easyai.tech/ai-definition/tf-idf/](http://easyai.tech/ai-definition/tf-idf/)
>- [https://www.zhihu.com/question/21094489](https://www.zhihu.com/question/21094489)

<br/>

>如果有错误或者不严谨的地方，请务必给予指正，十分感谢。
