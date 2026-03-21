<table width="100%">
   <tr>
      <td align="left"><a href="02-multi-head-attention.md">&larr; 7.2 多头注意力：并行捕捉多种关系</a></td>
      <td align="right"><a href="04-encoder-decoder-architecture.md">7.4 编码器-解码器架构 &rarr;</a></td>
   </tr>
</table>
---

# 7.3 位置编码：给序列注入顺序信息

> "Transformer没有递归，没有卷积，位置编码是它理解顺序的关键。"

## 为什么需要位置编码

### 问题背景

```
Transformer的问题：

自注意力机制是位置无关的
"我爱你"和"你爱我"对模型是一样的
因为每个词都与其他所有词直接连接

解决方案：
显式注入位置信息
让每个位置有唯一的表示
```

### 位置编码的作用

```
位置编码的目标：

1. 唯一性：每个位置有独特编码
2. 相对性：能表达相对位置关系
3. 外推性：能处理训练时未见过的长度
4. 连续性：相邻位置编码相似
```

## 正弦位置编码

### 原理

```
原始Transformer使用正弦/余弦函数：

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: 位置
- i: 维度索引
- d_model: 模型维度

特点：
- 波长随维度变化
- 可以表示相对位置
- 能外推到任意长度
```

### Java实现

```java
package com.example.ai.chapter07;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 正弦位置编码
 */
public class PositionalEncoding {
    
    private final int dModel;
    private final int maxLen;
    private final INDArray encoding;
    
    public PositionalEncoding(int dModel, int maxLen) {
        this.dModel = dModel;
        this.maxLen = maxLen;
        this.encoding = createEncoding();
    }
    
    /**
     * 创建位置编码矩阵
     */
    private INDArray createEncoding() {
        INDArray pe = Nd4j.zeros(maxLen, dModel);
        
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dModel; i += 2) {
                double angle = pos / Math.pow(10000, (2.0 * i) / dModel);
                
                // 偶数维度用sin
                pe.putScalar(pos, i, Math.sin(angle));
                
                // 奇数维度用cos
                if (i + 1 < dModel) {
                    pe.putScalar(pos, i + 1, Math.cos(angle));
                }
            }
        }
        
        return pe;
    }
    
    /**
     * 获取位置编码
     */
    public INDArray getEncoding(int seqLen) {
        return encoding.getRows(0, seqLen);
    }
    
    /**
     * 将位置编码加到输入上
     */
    public INDArray addPositionalEncoding(INDArray input) {
        // input: (batch_size, seq_len, d_model)
        int seqLen = (int) input.size(1);
        INDArray pe = getEncoding(seqLen);
        
        // 广播加法
        return input.add(pe);
    }
    
    /**
     * 可视化位置编码
     */
    public void visualize() {
        System.out.println("位置编码矩阵形状: " + encoding.shapeInfoToString());
        
        // 打印前5个位置的编码
        for (int pos = 0; pos < Math.min(5, maxLen); pos++) {
            System.out.printf("位置 %d: ", pos);
            INDArray row = encoding.getRow(pos);
            for (int i = 0; i < Math.min(8, dModel); i++) {
                System.out.printf("%.3f ", row.getDouble(i));
            }
            System.out.println("...");
        }
    }
}
```

### 相对位置特性

```java
/**
 * 位置编码的相对位置特性
 */
public class PositionalEncodingProperties {
    
    /**
     * 展示相对位置关系
     */
    public static void demonstrateRelativePosition() {
        int dModel = 512;
        int maxLen = 100;
        
        PositionalEncoding pe = new PositionalEncoding(dModel, maxLen);
        
        // 位置编码的点积可以表示相对位置
        // PE(pos + k) · PE(pos) ≈ f(k)
        
        System.out.println("相对位置关系示例：");
        System.out.println("位置5和位置10的相似度: " + 
            calculateSimilarity(pe, 5, 10));
        System.out.println("位置5和位置6的相似度: " + 
            calculateSimilarity(pe, 5, 6));
        System.out.println("位置5和位置50的相似度: " + 
            calculateSimilarity(pe, 5, 50));
    }
    
    private static double calculateSimilarity(PositionalEncoding pe, 
                                             int pos1, int pos2) {
        INDArray encoding1 = pe.getEncoding(pos1 + 1).getRow(pos1);
        INDArray encoding2 = pe.getEncoding(pos2 + 1).getRow(pos2);
        
        // 点积
        return encoding1.mul(encoding2).sumNumber().doubleValue();
    }
}
```

## 可学习位置编码

### 对比

```
正弦位置编码 vs 可学习位置编码：

正弦编码：
- 固定函数，无需学习
- 可以外推到任意长度
- 有明确的数学解释

可学习编码：
- 通过训练学习
- 可能更适合特定任务
- 受限于训练时的最大长度
```

### 实现

```java
/**
 * 可学习位置编码
 */
public class LearnablePositionalEncoding {
    
    private final INDArray positionEmbeddings;
    private final int maxLen;
    private final int dModel;
    
    public LearnablePositionalEncoding(int maxLen, int dModel) {
        this.maxLen = maxLen;
        this.dModel = dModel;
        // 随机初始化
        this.positionEmbeddings = Nd4j.rand(maxLen, dModel);
    }
    
    public INDArray forward(int seqLen) {
        if (seqLen > maxLen) {
            throw new IllegalArgumentException(
                "序列长度超过最大长度: " + maxLen);
        }
        return positionEmbeddings.getRows(0, seqLen);
    }
}
```

## 旋转位置编码（RoPE）

### 简介

```
RoPE（Rotary Position Embedding）：

现代大模型（如LLaMA、ChatGLM）使用的位置编码
通过旋转矩阵注入位置信息

优势：
- 更好的外推性
- 与注意力机制更兼容
- 计算效率更高
```

### 核心思想

```
RoPE将位置信息编码为旋转：

对于二维向量 (x, y)，位置m的旋转：
[x_m]   [cos(mθ)  -sin(mθ)] [x]
[y_m] = [sin(mθ)   cos(mθ)] [y]

其中θ与维度相关，形成等比数列
```

## 实践建议

### 选择指南

```
位置编码选择：

1. 正弦编码
   - 经典选择
   - 需要外推能力
   - 解释性要求高

2. 可学习编码
   - 固定长度场景
   - 追求最佳性能
   - 有足够训练数据

3. RoPE
   - 现代大模型首选
   - 长文本场景
   - 需要良好外推性
```

## 小结

本章我们学习了：

1. **位置编码必要性**：为Transformer注入顺序信息
2. **正弦编码**：经典方法，具有良好数学性质
3. **可学习编码**：通过训练优化
4. **RoPE**：现代方法，更好的外推性

**关键认识：**
位置编码是Transformer理解序列顺序的关键，选择合适的位置编码对模型性能有重要影响。

**下一步：** 我们将学习编码器-解码器架构。

---

**练习题：**

1. 为什么正弦位置编码可以表示相对位置？
2. 可学习位置编码和正弦编码各有什么优缺点？
3. 如何实现RoPE位置编码？

---

<table width="100%">
   <tr>
      <td align="left"><a href="02-multi-head-attention.md">&larr; 7.2 多头注意力：并行捕捉多种关系</a></td>
      <td align="right"><a href="04-encoder-decoder-architecture.md">7.4 编码器-解码器架构 &rarr;</a></td>
   </tr>
</table>