<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-06/04-why-transformer-changed-nlp.md">&larr; 6.4 设计思考：为什么Transformer改变了NLP</a></td>
      <td align="right"><a href="02-multi-head-attention.md">7.2 多头注意力：并行捕捉多种关系 &rarr;</a></td>
   </tr>
</table>
---

# 7.1 自注意力机制：让词与词对话

> "自注意力是Transformer的灵魂——它让序列中的每个元素都能与所有其他元素直接交流。"

## 自注意力的直觉

### 一个阅读理解的例子

当你读到这句话：

```
"小明把苹果给了小红，因为她饿了。"

问题："她"指的是谁？
答案：小红

你的大脑是如何理解的？
- "她"需要找到一个女性
- "小红"是女性
- 语境支持这个关联

自注意力就在做类似的事情：让"她"关注到"小红"
```

### 注意力权重可视化

```
句子: "小明 把 苹果 给了 小红"

        小明   把   苹果   给了   小红
小明    0.3   0.1   0.1    0.2    0.3
把      0.1   0.2   0.3    0.3    0.1
苹果    0.1   0.2   0.4    0.2    0.1
给了    0.2   0.1   0.2    0.2    0.3
小红    0.3   0.1   0.1    0.2    0.3

每个词都在"看"其他所有词
```

## 自注意力的数学

### 核心公式

```
SelfAttention(X) = softmax(XWq × (XWk)^T / √d) × XWv

分解：
1. X: 输入序列 [seq_len, d_model]
2. Wq, Wk, Wv: 三个投影矩阵
3. Q = XWq: Query矩阵
4. K = XWk: Key矩阵
5. V = XWv: Value矩阵
```

### 逐步计算

```java
package com.example.ai.chapter07;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 自注意力机制实现
 */
public class SelfAttention {
    
    private INDArray Wq;  // Query投影
    private INDArray Wk;  // Key投影
    private INDArray Wv;  // Value投影
    private int dK;       // Key的维度
    
    public SelfAttention(int dModel, int dK) {
        this.dK = dK;
        
        // Xavier初始化
        double scale = Math.sqrt(2.0 / (dModel + dK));
        Wq = Nd4j.randn(dModel, dK).mul(scale);
        Wk = Nd4j.randn(dModel, dK).mul(scale);
        Wv = Nd4j.randn(dModel, dK).mul(scale);
    }
    
    /**
     * 前向传播
     * @param X 输入 [seqLen, dModel]
     * @return 输出 [seqLen, dK]
     */
    public AttentionOutput forward(INDArray X) {
        int seqLen = (int) X.size(0);
        
        // Step 1: 计算Q, K, V
        INDArray Q = X.mmul(Wq);  // [seqLen, dK]
        INDArray K = X.mmul(Wk);  // [seqLen, dK]
        INDArray V = X.mmul(Wv);  // [seqLen, dK]
        
        // Step 2: 计算注意力分数
        // scores = QK^T / √dK
        INDArray scores = Q.mmul(K.transpose())
                          .div(Math.sqrt(dK));
        
        // Step 3: Softmax归一化
        INDArray attentionWeights = softmax(scores);
        
        // Step 4: 加权求和
        INDArray output = attentionWeights.mmul(V);
        
        return new AttentionOutput(output, attentionWeights, Q, K, V);
    }
    
    /**
     * 数值稳定的softmax
     */
    private INDArray softmax(INDArray x) {
        // 减去最大值防止溢出
        INDArray max = x.max(true, 1);
        INDArray exp = Transforms.exp(x.subColumnVector(max));
        INDArray sum = exp.sum(true, 1);
        return exp.divColumnVector(sum);
    }
    
    /**
     * 注意力输出
     */
    public record AttentionOutput(
        INDArray output,           // 输出向量
        INDArray attentionWeights, // 注意力权重矩阵
        INDArray Q, INDArray K, INDArray V
    ) {}
}
```

## Query、Key、Value的含义

### 类比理解

```
图书馆检索系统：
- Query: 你想找什么书
- Key: 每本书的标签/索引
- Value: 书的实际内容

计算过程：
1. 用Query匹配所有Key
2. 得到匹配程度（注意力分数）
3. 根据匹配程度取Value的加权和
```

### 具体例子

```
句子: "猫 坐在 垫子 上"

对于"猫"这个位置：
- Query: "我想找与猫相关的内容"
- Key(猫): "我是猫"
- Key(坐在): "我是坐在"
- Key(垫子): "我是垫子"
- Key(上): "我是上"

匹配结果：
- 猫-猫: 高分（自身）
- 猫-坐在: 中分（动作关联）
- 猫-垫子: 中分（位置关联）
- 猫-上: 低分

最终"猫"的表示融合了这些关联信息
```

## 缩放因子√d_k的作用

### 为什么需要缩放

```java
/**
 * 缩放因子的重要性演示
 */
public class ScalingDemo {
    
    public static void main(String[] args) {
        int dK = 64;  // 典型的维度
        
        // 不缩放时，点积的方差
        double variance = dK;  // 点积方差约为dK
        double std = Math.sqrt(variance);
        
        System.out.println("维度 dK = " + dK);
        System.out.println("点积标准差 = " + std);
        
        // 问题：点积值可能很大
        // 导致softmax的输入差异很大
        // softmax输出会趋向于one-hot
        // 梯度会变得很小
        
        // 缩放后
        double scaledStd = 1.0;
        System.out.println("缩放后标准差 = " + scaledStd);
    }
}
```

### 数学解释

```
假设Q和K的元素独立同分布，均值为0，方差为1

点积 QK^T = Σ q_i * k_i

期望: E[Σ q_i * k_i] = 0
方差: Var[Σ q_i * k_i] = d_k

当d_k很大时，点积的值会很大
→ softmax输入差异大
→ softmax输出接近one-hot
→ 梯度消失

除以√d_k后，方差变为1
→ softmax输入范围合理
→ 梯度正常传播
```

## 自注意力 vs 其他注意力

### 对比

| 类型 | Query来源 | Key来源 | Value来源 |
|------|-----------|---------|-----------|
| 自注意力 | 输入X | 输入X | 输入X |
| 交叉注意力 | 解码器 | 编码器 | 编码器 |
| 掩码自注意力 | 输入X | 输入X（掩码） | 输入X（掩码） |

### 掩码自注意力

```java
/**
 * 掩码自注意力（用于解码器）
 */
public class MaskedSelfAttention extends SelfAttention {
    
    public MaskedSelfAttention(int dModel, int dK) {
        super(dModel, dK);
    }
    
    @Override
    public AttentionOutput forward(INDArray X) {
        int seqLen = (int) X.size(0);
        
        INDArray Q = X.mmul(Wq);
        INDArray K = X.mmul(Wk);
        INDArray V = X.mmul(Wv);
        
        INDArray scores = Q.mmul(K.transpose()).div(Math.sqrt(dK));
        
        // 应用掩码：上三角设为负无穷
        INDArray mask = createCausalMask(seqLen);
        scores = scores.add(mask);
        
        INDArray attentionWeights = softmax(scores);
        INDArray output = attentionWeights.mmul(V);
        
        return new AttentionOutput(output, attentionWeights, Q, K, V);
    }
    
    /**
     * 创建因果掩码
     */
    private INDArray createCausalMask(int seqLen) {
        INDArray mask = Nd4j.zeros(seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                mask.putScalar(i, j, Double.NEGATIVE_INFINITY);
            }
        }
        return mask;
    }
}
```

## 自注意力的计算复杂度

### 时间复杂度分析

```
输入: [seqLen, dModel]

计算Q, K, V: O(seqLen × dModel × dK)
计算注意力分数: O(seqLen² × dK)
加权求和: O(seqLen² × dK)

总复杂度: O(seqLen² × dModel)

问题：序列长度增加时，计算量平方增长
```

### 空间复杂度

```
注意力权重矩阵: [seqLen, seqLen]

对于seqLen=4096:
权重矩阵大小 = 4096 × 4096 = 16M 个浮点数
约64MB内存（float32）
```

## 设计思考：自注意力的哲学

### 平等交流

```
传统模型：信息单向流动
自注意力：每个位置平等地与其他位置交流

这打破了序列的"距离"概念
任何两个位置都可以直接交互
```

### 动态权重

```
CNN: 固定的卷积核权重
RNN: 固定的转移权重
自注意力: 动态计算的权重

权重取决于内容，而非位置
这使得模型更加灵活
```

## 小结

本章我们学习了：

1. **自注意力原理**：Q、K、V的计算
2. **缩放因子**：防止梯度消失
3. **掩码注意力**：因果约束
4. **复杂度分析**：O(n²)的代价

**核心公式：**

```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

**下一步：** 我们将学习多头注意力机制。

---

**思考题：**

1. 为什么Query、Key、Value要用不同的投影矩阵？
2. 掩码自注意力为什么能保证因果性？
3. 如何优化自注意力的计算复杂度？

---

<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-06/04-why-transformer-changed-nlp.md">&larr; 6.4 设计思考：为什么Transformer改变了NLP</a></td>
      <td align="right"><a href="02-multi-head-attention.md">7.2 多头注意力：并行捕捉多种关系 &rarr;</a></td>
   </tr>
</table>