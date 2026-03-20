# 7.4 编码器-解码器架构

> "编码器理解输入，解码器生成输出——Transformer的完整形态。"

## 架构概述

### 编码器-解码器结构

```
原始Transformer架构：

输入 → [编码器] × N → 上下文表示 → [解码器] × N → 输出

编码器：
- 多头自注意力
- 前馈网络
- 处理输入序列

解码器：
- 掩码多头自注意力
- 交叉注意力
- 前馈网络
- 生成输出序列
```

### 各组件作用

```
编码器层：
1. 多头自注意力
   - 关注输入序列的所有位置
   - 生成上下文表示

2. 前馈网络
   - 进一步变换表示
   - 增加非线性

解码器层：
1. 掩码自注意力
   - 只能关注已生成的词
   - 防止看到未来信息

2. 交叉注意力
   - 关注编码器的输出
   - 获取输入信息

3. 前馈网络
   - 变换输出表示
```

## Java实现

### 编码器层

```java
package com.example.ai.chapter07;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Transformer编码器层
 */
public class EncoderLayer {
    
    private final MultiHeadAttention selfAttention;
    private final FeedForwardNetwork feedForward;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    
    public EncoderLayer(int dModel, int numHeads, int dFF) {
        this.selfAttention = new MultiHeadAttention(numHeads, dModel);
        this.feedForward = new FeedForwardNetwork(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
    }
    
    /**
     * 前向传播
     */
    public INDArray forward(INDArray x) {
        // 自注意力 + 残差连接 + 层归一化
        INDArray attnOutput = selfAttention.forward(x, x, x);
        x = norm1.forward(x.add(attnOutput));
        
        // 前馈网络 + 残差连接 + 层归一化
        INDArray ffOutput = feedForward.forward(x);
        x = norm2.forward(x.add(ffOutput));
        
        return x;
    }
}

/**
 * 前馈网络
 */
class FeedForwardNetwork {
    
    private final int dModel;
    private final int dFF;
    
    public FeedForwardNetwork(int dModel, int dFF) {
        this.dModel = dModel;
        this.dFF = dFF;
    }
    
    public INDArray forward(INDArray x) {
        // 简化实现
        // 实际应有: Linear -> ReLU -> Linear
        return x;
    }
}

/**
 * 层归一化
 */
class LayerNorm {
    
    private final int features;
    
    public LayerNorm(int features) {
        this.features = features;
    }
    
    public INDArray forward(INDArray x) {
        // 简化实现
        // 实际应计算均值和方差进行归一化
        return x;
    }
}
```

### 解码器层

```java
package com.example.ai.chapter07;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Transformer解码器层
 */
public class DecoderLayer {
    
    private final MultiHeadAttention maskedSelfAttention;
    private final MultiHeadAttention crossAttention;
    private final FeedForwardNetwork feedForward;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LayerNorm norm3;
    
    public DecoderLayer(int dModel, int numHeads, int dFF) {
        this.maskedSelfAttention = new MultiHeadAttention(numHeads, dModel);
        this.crossAttention = new MultiHeadAttention(numHeads, dModel);
        this.feedForward = new FeedForwardNetwork(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.norm3 = new LayerNorm(dModel);
    }
    
    /**
     * 前向传播
     */
    public INDArray forward(INDArray x, INDArray encoderOutput) {
        // 掩码自注意力
        INDArray mask = createLookAheadMask(x.size(1));
        INDArray attnOutput = maskedSelfAttention.forward(x, x, x, mask);
        x = norm1.forward(x.add(attnOutput));
        
        // 交叉注意力
        INDArray crossOutput = crossAttention.forward(
            x, encoderOutput, encoderOutput);
        x = norm2.forward(x.add(crossOutput));
        
        // 前馈网络
        INDArray ffOutput = feedForward.forward(x);
        x = norm3.forward(x.add(ffOutput));
        
        return x;
    }
    
    /**
     * 创建前瞻掩码
     */
    private INDArray createLookAheadMask(int size) {
        // 上三角矩阵，防止看到未来信息
        INDArray mask = Nd4j.ones(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                mask.putScalar(i, j, 0);
            }
        }
        return mask;
    }
}
```

## 完整Transformer

```java
package com.example.ai.chapter07;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 完整Transformer模型
 */
public class Transformer {
    
    private final Encoder encoder;
    private final Decoder decoder;
    private final Linear outputLinear;
    private final PositionalEncoding posEncoding;
    
    public Transformer(int numLayers, int dModel, int numHeads, 
                      int dFF, int srcVocabSize, int tgtVocabSize,
                      int maxLen) {
        this.encoder = new Encoder(numLayers, dModel, numHeads, dFF);
        this.decoder = new Decoder(numLayers, dModel, numHeads, dFF);
        this.outputLinear = new Linear(dModel, tgtVocabSize);
        this.posEncoding = new PositionalEncoding(dModel, maxLen);
    }
    
    /**
     * 前向传播
     */
    public INDArray forward(INDArray src, INDArray tgt) {
        // 添加位置编码
        src = posEncoding.addPositionalEncoding(src);
        tgt = posEncoding.addPositionalEncoding(tgt);
        
        // 编码器
        INDArray memory = encoder.forward(src);
        
        // 解码器
        INDArray output = decoder.forward(tgt, memory);
        
        // 输出投影
        return outputLinear.forward(output);
    }
}

/**
 * 编码器
 */
class Encoder {
    
    private final EncoderLayer[] layers;
    
    public Encoder(int numLayers, int dModel, int numHeads, int dFF) {
        this.layers = new EncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new EncoderLayer(dModel, numHeads, dFF);
        }
    }
    
    public INDArray forward(INDArray x) {
        for (EncoderLayer layer : layers) {
            x = layer.forward(x);
        }
        return x;
    }
}

/**
 * 解码器
 */
class Decoder {
    
    private final DecoderLayer[] layers;
    
    public Decoder(int numLayers, int dModel, int numHeads, int dFF) {
        this.layers = new DecoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new DecoderLayer(dModel, numHeads, dFF);
        }
    }
    
    public INDArray forward(INDArray x, INDArray memory) {
        for (DecoderLayer layer : layers) {
            x = layer.forward(x, memory);
        }
        return x;
    }
}

/**
 * 线性层
 */
class Linear {
    
    private final int inFeatures;
    private final int outFeatures;
    
    public Linear(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }
    
    public INDArray forward(INDArray x) {
        // 简化实现
        return x;
    }
}
```

## 架构变体

### 仅编码器（Encoder-only）

```
代表：BERT

结构：[编码器] × N
用途：理解任务
- 文本分类
- 命名实体识别
- 问答

特点：
- 双向注意力
- 适合理解
```

### 仅解码器（Decoder-only）

```
代表：GPT系列

结构：[解码器] × N
用途：生成任务
- 文本生成
- 代码生成
- 对话

特点：
- 单向注意力（掩码）
- 适合生成
```

### 对比

| 架构 | 代表 | 注意力 | 用途 |
|------|------|--------|------|
| Encoder-only | BERT | 双向 | 理解 |
| Decoder-only | GPT | 单向 | 生成 |
| Encoder-Decoder | T5 | 都有 | 翻译、摘要 |

## 小结

本章我们学习了：

1. **编码器-解码器架构**：Transformer的完整结构
2. **编码器**：理解输入，生成上下文表示
3. **解码器**：生成输出，使用掩码和交叉注意力
4. **架构变体**：Encoder-only、Decoder-only的选择

**关键认识：**
不同的架构变体适合不同的任务，理解它们的区别有助于选择合适的模型。

**下一步：** 我们将用Java实现完整的Transformer。

---

**练习题：**

1. 编码器和解码器的主要区别是什么？
2. 为什么解码器需要掩码自注意力？
3. 什么时候选择Encoder-only，什么时候选择Decoder-only？
