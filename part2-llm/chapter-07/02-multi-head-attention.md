<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 7.1 自注意力机制：让词与词对话](01-self-attention.md)</span>

<span>[7.3 位置编码：给序列注入顺序信息 &rarr;](03-positional-encoding.md)</span>

</div>
---

# 7.2 多头注意力：并行捕捉多种关系

> "多头注意力让模型同时关注不同的信息维度——就像多人从不同角度观察同一事物。"

## 为什么需要多头

### 单一注意力的局限

```
单一注意力的问题：
- 只能捕捉一种关系
- 可能错过重要信息
- 表达能力有限

类比：
一个人观察一幅画，只能关注一个方面
多人观察，可以看到色彩、构图、笔触等不同维度
```

### 多头的优势

```
多头注意力的核心思想：

将查询、键、值投影到多个子空间
每个头学习不同的注意力模式
最后拼接所有头的输出

优势：
1. 并行学习多种关系
2. 增强表达能力
3. 提高模型鲁棒性
```

## 多头注意力机制

### 数学原理

```
多头注意力计算：

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

其中每个头：
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

参数：
- h: 头数（通常8）
- d_k = d_model / h: 每个头的维度
```

### Java实现

```java
package com.example.ai.chapter07;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 多头注意力实现
 */
public class MultiHeadAttention {
    
    private final int numHeads;
    private final int dModel;
    private final int dHead;
    
    private final INDArray wQ;
    private final INDArray wK;
    private final INDArray wV;
    private final INDArray wO;
    
    public MultiHeadAttention(int numHeads, int dModel) {
        this.numHeads = numHeads;
        this.dModel = dModel;
        this.dHead = dModel / numHeads;
        
        // 初始化投影矩阵
        this.wQ = Nd4j.rand(dModel, dModel);
        this.wK = Nd4j.rand(dModel, dModel);
        this.wV = Nd4j.rand(dModel, dModel);
        this.wO = Nd4j.rand(dModel, dModel);
    }
    
    /**
     * 前向传播
     */
    public INDArray forward(INDArray query, INDArray key, INDArray value) {
        int batchSize = query.size(0);
        int seqLen = query.size(1);
        
        // 线性投影
        INDArray q = query.mmul(wQ);  // (batch, seq, d_model)
        INDArray k = key.mmul(wK);
        INDArray v = value.mmul(wV);
        
        // 分割成多个头
        // (batch, seq, d_model) -> (batch, num_heads, seq, d_head)
        q = splitHeads(q, batchSize, seqLen);
        k = splitHeads(k, batchSize, seqLen);
        v = splitHeads(v, batchSize, seqLen);
        
        // 计算注意力
        INDArray attentionOutput = scaledDotProductAttention(q, k, v);
        
        // 合并头
        // (batch, num_heads, seq, d_head) -> (batch, seq, d_model)
        attentionOutput = mergeHeads(attentionOutput, batchSize, seqLen);
        
        // 最终线性投影
        return attentionOutput.mmul(wO);
    }
    
    /**
     * 分割成多个头
     */
    private INDArray splitHeads(INDArray x, int batchSize, int seqLen) {
        // 重塑为 (batch, seq, num_heads, d_head)
        x = x.reshape(batchSize, seqLen, numHeads, dHead);
        // 转置为 (batch, num_heads, seq, d_head)
        return x.permute(0, 2, 1, 3);
    }
    
    /**
     * 合并头
     */
    private INDArray mergeHeads(INDArray x, int batchSize, int seqLen) {
        // 转置为 (batch, seq, num_heads, d_head)
        x = x.permute(0, 2, 1, 3);
        // 重塑为 (batch, seq, d_model)
        return x.reshape(batchSize, seqLen, dModel);
    }
    
    /**
     * 缩放点积注意力
     */
    private INDArray scaledDotProductAttention(INDArray q, INDArray k, INDArray v) {
        // q, k, v: (batch, num_heads, seq, d_head)
        
        // 计算注意力分数: Q * K^T / sqrt(d_k)
        INDArray scores = q.matmul(k.permute(0, 1, 3, 2))
            .div(Math.sqrt(dHead));
        
        // Softmax
        INDArray attentionWeights = softmax(scores);
        
        // 加权求和
        return attentionWeights.matmul(v);
    }
    
    private INDArray softmax(INDArray x) {
        // 简化实现
        return Nd4j.nn.softmax(x, -1);
    }
}
```

## 多头的作用分析

### 不同头的 specialization

```
研究表明，不同头学习不同的模式：

头1：关注相邻词（局部语法）
头2：关注远距离依赖（长程关系）
头3：关注特定词性（名词、动词）
头4：关注语义相似性
...

这种分工让模型更全面理解文本
```

### 可视化示例

```java
/**
 * 注意力权重可视化
 */
public class AttentionVisualizer {
    
    public static void visualize(INDArray attentionWeights, String[] tokens) {
        // attentionWeights: (num_heads, seq_len, seq_len)
        int numHeads = (int) attentionWeights.size(0);
        int seqLen = (int) attentionWeights.size(1);
        
        for (int h = 0; h < numHeads; h++) {
            System.out.println("\n=== Head " + h + " ===");
            
            INDArray headWeights = attentionWeights.getRow(h);
            
            // 打印注意力热力图
            for (int i = 0; i < seqLen; i++) {
                System.out.printf("%-10s ", tokens[i]);
                for (int j = 0; j < seqLen; j++) {
                    double weight = headWeights.getDouble(i, j);
                    // 用颜色深浅表示权重
                    System.out.printf("%.2f ", weight);
                }
                System.out.println();
            }
        }
    }
}
```

## 实践建议

### 头数选择

```
头数选择指南：

- 小模型（d_model=512）：8头，d_head=64
- 中模型（d_model=768）：12头，d_head=64
- 大模型（d_model=1024）：16头，d_head=64

经验法则：
- d_head 通常保持64或128
- 头数 = d_model / d_head
- 头数最好是2的幂次，便于GPU优化
```

### 性能优化

```java
/**
 * 多头注意力优化技巧
 */
public class MultiHeadAttentionOptimization {
    
    // 1. 批量矩阵乘法
    // 使用ND4J的批量matmul优化
    
    // 2. 内存布局优化
    // 确保数据在内存中连续存储
    
    // 3. 并行计算
    // 不同头可以并行计算
    
    // 4. 缓存友好
    // 重复使用的矩阵缓存起来
}
```

## 小结

本章我们学习了：

1. **多头动机**：捕捉多种关系，增强表达能力
2. **实现原理**：投影→分头→注意力→合并
3. **头的作用**：不同头学习不同模式
4. **实践建议**：头数选择、性能优化

**关键认识：**
多头注意力是Transformer表达能力的核心，让模型能够全方位理解文本。

**下一步：** 我们将学习位置编码。

---

**练习题：**

1. 为什么d_head通常设置为64？
2. 多头注意力相比单头有哪些优势？
3. 如何实现多头注意力的并行计算？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 7.1 自注意力机制：让词与词对话](01-self-attention.md)</span>

<span>[7.3 位置编码：给序列注入顺序信息 &rarr;](03-positional-encoding.md)</span>

</div>