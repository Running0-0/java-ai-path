# 6.3 Transformer架构：注意力机制的革命

> "Attention Is All You Need——这篇论文改变了NLP的一切。"

## 从RNN到Transformer

### RNN的问题

```
RNN处理序列的方式：
x1 → h1 → x2 → h2 → x3 → h3 → ...

问题：
1. 顺序计算：无法并行
2. 长距离依赖：信息衰减
3. 信息瓶颈：隐状态容量有限
```

### Transformer的突破

```
Transformer处理序列的方式：
x1, x2, x3 → Self-Attention → 并行处理

优势：
1. 并行计算：所有位置同时处理
2. 直接连接：任意两位置直接交互
3. 灵活建模：注意力权重动态计算
```

## 注意力机制的核心

### 直观理解

想象你在读一句话：

```
"小明在公园里遛狗，他很高兴"

读到"他"时，你的注意力会指向"小明"
这就是注意力机制在做的事
```

### 数学表达

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Q (Query): 查询向量
K (Key): 键向量  
V (Value): 值向量

类比：
- Q: 我想找什么
- K: 每个东西的标签
- V: 每个东西的内容
```

### 用Java实现注意力

```java
package com.example.ai.chapter06;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 注意力机制实现
 */
public class Attention {
    
    private INDArray Wq;  // Query权重
    private INDArray Wk;  // Key权重
    private INDArray Wv;  // Value权重
    private int dK;       // Key维度
    
    public Attention(int inputSize, int attentionSize) {
        this.dK = attentionSize;
        
        // 初始化权重
        Wq = Nd4j.randn(inputSize, attentionSize).mul(0.1);
        Wk = Nd4j.randn(inputSize, attentionSize).mul(0.1);
        Wv = Nd4j.randn(inputSize, attentionSize).mul(0.1);
    }
    
    /**
     * 计算注意力
     */
    public INDArray forward(INDArray input) {
        // input: [batchSize, seqLen, inputSize]
        
        // 1. 计算Q, K, V
        INDArray Q = input.mmul(Wq);  // [batchSize, seqLen, attentionSize]
        INDArray K = input.mmul(Wk);
        INDArray V = input.mmul(Wv);
        
        // 2. 计算注意力分数
        // scores = QK^T / √d_k
        INDArray scores = Q.mmul(K.transpose())
                          .div(Math.sqrt(dK));
        
        // 3. Softmax归一化
        INDArray attentionWeights = softmax(scores);
        
        // 4. 加权求和
        INDArray output = attentionWeights.mmul(V);
        
        return output;
    }
    
    private INDArray softmax(INDArray x) {
        // 数值稳定的softmax
        INDArray max = x.max(true, -1);
        INDArray exp = Transforms.exp(x.subColumnVector(max));
        INDArray sum = exp.sum(true, -1);
        return exp.divColumnVector(sum);
    }
}
```

## 自注意力（Self-Attention）

### 核心思想

```
自注意力：序列中的每个元素都与其他所有元素交互

输入序列: [x1, x2, x3, x4]

x1 关注: x1, x2, x3, x4 → 加权求和 → h1
x2 关注: x1, x2, x3, x4 → 加权求和 → h2
...

每个位置的表示都融合了全局信息
```

### 可视化示例

```
句子: "我 爱 编程"

        我    爱    编程
我     [0.5  0.3   0.2]
爱     [0.2  0.5   0.3]
编程   [0.1  0.2   0.7]

"编程"对"编程"的注意力最高(0.7)
"我"对"我"的注意力最高(0.5)
```

## 多头注意力

### 为什么需要多头

```
单头注意力：学习一种关系模式
多头注意力：学习多种关系模式

例如：
- 头1：关注语法关系
- 头2：关注语义关系
- 头3：关注位置关系
```

### 多头注意力实现

```java
/**
 * 多头注意力实现
 */
public class MultiHeadAttention {
    
    private int numHeads;
    private int headSize;
    private Attention[] heads;
    private INDArray Wo;  // 输出投影
    
    public MultiHeadAttention(int inputSize, int numHeads, int totalSize) {
        this.numHeads = numHeads;
        this.headSize = totalSize / numHeads;
        
        heads = new Attention[numHeads];
        for (int i = 0; i < numHeads; i++) {
            heads[i] = new Attention(inputSize, headSize);
        }
        
        Wo = Nd4j.randn(totalSize, totalSize).mul(0.1);
    }
    
    public INDArray forward(INDArray input) {
        INDArray[] headOutputs = new INDArray[numHeads];
        
        // 每个头独立计算
        for (int i = 0; i < numHeads; i++) {
            headOutputs[i] = heads[i].forward(input);
        }
        
        // 拼接所有头的输出
        INDArray concat = Nd4j.concat(1, headOutputs);
        
        // 输出投影
        return concat.mmul(Wo);
    }
}
```

## Transformer编码器

### 完整结构

```
输入嵌入
    ↓
位置编码
    ↓
┌─────────────────────┐
│   多头自注意力       │
│         ↓           │
│   残差连接 + 层归一化 │
│         ↓           │
│   前馈网络           │
│         ↓           │
│   残差连接 + 层归一化 │
└─────────────────────┘
    ↓
(重复N次)
```

### 位置编码

```java
/**
 * 位置编码
 */
public class PositionalEncoding {
    
    public static INDArray encode(int maxLen, int dModel) {
        INDArray pe = Nd4j.zeros(maxLen, dModel);
        
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dModel; i += 2) {
                double divTerm = Math.exp(-i * Math.log(10000.0) / dModel);
                
                pe.putScalar(pos, i, Math.sin(pos * divTerm));
                if (i + 1 < dModel) {
                    pe.putScalar(pos, i + 1, Math.cos(pos * divTerm));
                }
            }
        }
        
        return pe;
    }
}
```

### 前馈网络

```java
/**
 * 前馈网络
 */
public class FeedForward {
    
    private INDArray W1, b1;
    private INDArray W2, b2;
    
    public FeedForward(int dModel, int dFF) {
        W1 = Nd4j.randn(dModel, dFF).mul(0.1);
        b1 = Nd4j.zeros(dFF);
        W2 = Nd4j.randn(dFF, dModel).mul(0.1);
        b2 = Nd4j.zeros(dModel);
    }
    
    public INDArray forward(INDArray x) {
        // 两层全连接 + ReLU
        return x.mmul(W1).add(b1)
                .transform(Transforms.relu(true))
                .mmul(W2).add(b2);
    }
}
```

## Transformer vs RNN

### 对比分析

| 特性 | RNN | Transformer |
|------|-----|-------------|
| 并行性 | 顺序处理 | 完全并行 |
| 长距离依赖 | 需要传递 | 直接连接 |
| 计算复杂度 | O(n) | O(n²) |
| 内存占用 | O(1) | O(n²) |
| 位置感知 | 天然 | 需要编码 |

### 适用场景

```
RNN适合：
- 实时流式处理
- 内存受限场景
- 短序列

Transformer适合：
- 离线批处理
- 长序列建模
- 需要全局依赖
```

## 设计思考：注意力的哲学

### 选择性关注

```
人类视觉：选择性注意
- 不处理所有信息
- 关注重要部分
- 忽略无关信息

注意力机制：让机器学会"关注"
```

### 全局视野

```
传统方法：局部窗口
Transformer：全局视野

每个位置都能"看到"整个序列
这是建模复杂依赖的关键
```

## 小结

本章我们学习了：

1. **注意力机制**：Q、K、V的计算
2. **自注意力**：序列内部交互
3. **多头注意力**：多种关系模式
4. **Transformer结构**：编码器组成

**核心概念：**

| 概念 | 说明 |
|------|------|
| 注意力 | 加权聚合信息 |
| 自注意力 | 序列内部注意力 |
| 多头 | 多种注意力模式 |
| 位置编码 | 注入位置信息 |

**下一步：** 我们将总结Transformer对NLP的影响。

---

**思考题：**

1. 为什么Transformer比RNN更适合处理长序列？
2. 多头注意力的"多头"有什么意义？
3. 为什么需要位置编码？
