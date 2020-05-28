## 神经网络文本分类

### TextCNN

#### 网络结构

#### 实验结果

实验在两方面做了变化：

+ 学习率更新，尝试了不用 scheduler、exponential scheduler、cosine scheduler。
+ 尝试了基于字、基于 jieba 切分词的模型。

默认参数训练完在验证集上的结果：

| class | precision | recall | F1 score |
| ----- | --------- | ------ | -------- |
| 奥运  | 0.9656    | 0.8682 | 0.9143   |
| 房产  | 0.8955    | 0.9142 | 0.9047   |
| 商业  | 0.8446    | 0.9058 | 0.8741   |
| 娱乐  | 0.9454    | 0.9445 | 0.9450   |

可以看到，商业的 precision 是最差的，它涵盖的文本内容与其他交叉最多，容易将其他的也分成商业。

奥运的 recall 最差。


### BiLSTM

### BiLSTM + CNN

### BiLSTM + Attention