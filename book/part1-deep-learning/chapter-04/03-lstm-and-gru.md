# 4.3 LSTM与GRU：解决长期依赖的钥匙

> "LSTM是深度学习领域最优雅的设计之一——它用门控机制实现了选择性的记忆与遗忘。"

## 长期依赖问题的本质

### 回顾：简单RNN的问题

```
h_t = tanh(W·x_t + U·h_{t-1} + b)

问题：每一步都要经过tanh压缩
      梯度在反向传播时不断衰减
      长距离信息逐渐消失
```

### 一个形象的比喻

想象你在传话游戏：

```
原始信息：今天下午3点开会
第1个人 → 第2个人 → ... → 第20个人

经过多人传递后，信息可能变成：
"今天...好像有事..."

信息在传递过程中逐渐丢失
```

### 解决思路

```
如果有一条"直通通道"，信息可以直接传递呢？

原始路径：信息被不断处理和压缩
直通路径：信息可以直接"跳过"中间步骤
```

## LSTM：长短期记忆网络

### 核心思想

LSTM引入了**门控机制**，让网络自己学习：
- 哪些信息需要**记住**
- 哪些信息需要**遗忘**
- 哪些信息需要**输出**

### LSTM的结构

```
LSTM单元结构：

     ┌─────────────────────────────────────┐
     │                                      │
     │   ┌───┐   ┌───┐   ┌───┐            │
c_t-1├──→│ × │←──│ f │   │ i │←──┐        │
     │   └─┬─┘   └───┘   └─┬─┘   │        │
     │     │      遗忘门    │     │        │
     │     │              ↓     │        │
     │     │   ┌───┐   ┌───┐   │        │
     │     └──→│ + │←──│ × │←──┘        │
     │         └─┬─┘   └───┘             │
     │           │      输入门            │
     │           ↓                       │
     │         ┌───┐   ┌───┐   ┌───┐    │
     │         │tanh│  │ o │→──│ × │──→h_t
     │         └───┘   └───┘   └───┘     │
     │                  输出门            │
     └─────────────────────────────────────┘
```

### 三个门的作用

| 门 | 作用 | 公式 |
|------|------|------|
| 遗忘门 | 决定丢弃哪些旧信息 | f_t = σ(W_f·[h_{t-1}, x_t]) |
| 输入门 | 决定存储哪些新信息 | i_t = σ(W_i·[h_{t-1}, x_t]) |
| 输出门 | 决定输出哪些信息 | o_t = σ(W_o·[h_{t-1}, x_t]) |

### LSTM的完整计算

```java
/**
 * LSTM单元实现
 */
public class LSTMCell {
    
    private int hiddenSize;
    // 四组权重：遗忘门、输入门、候选值、输出门
    private INDArray Wf, Wi, Wc, Wo;
    private INDArray Uf, Ui, Uc, Uo;
    private INDArray bf, bi, bc, bo;
    
    public LSTMCell(int inputSize, int hiddenSize) {
        this.hiddenSize = hiddenSize;
        
        // 初始化权重（省略具体初始化代码）
        // W: 输入权重，U: 隐状态权重
        // f: 遗忘门，i: 输入门，c: 候选值，o: 输出门
    }
    
    /**
     * LSTM前向计算
     */
    public LSTMOutput step(INDArray x, INDArray hPrev, INDArray cPrev) {
        // 拼接输入和上一时刻隐状态
        INDArray concat = Nd4j.hstack(hPrev, x);
        
        // 1. 遗忘门：决定丢弃哪些信息
        INDArray f = Transforms.sigmoid(concat.mmul(Wf).add(bf));
        
        // 2. 输入门：决定存储哪些新信息
        INDArray i = Transforms.sigmoid(concat.mmul(Wi).add(bi));
        
        // 3. 候选值：新信息的候选
        INDArray cTilde = Transforms.tanh(concat.mmul(Wc).add(bc));
        
        // 4. 更新细胞状态
        INDArray c = f.mul(cPrev).add(i.mul(cTilde));
        
        // 5. 输出门：决定输出哪些信息
        INDArray o = Transforms.sigmoid(concat.mmul(Wo).add(bo));
        
        // 6. 计算隐状态输出
        INDArray h = o.mul(Transforms.tanh(c));
        
        return new LSTMOutput(h, c);
    }
    
    public record LSTMOutput(INDArray hidden, INDArray cell) {}
}
```

### 细胞状态：信息的直通通道

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

关键：细胞状态c可以无损传递！
     c_t = c_{t-1} 当 f_t=1, i_t=0 时

这解决了长期依赖问题
```

## GRU：简化版的LSTM

### 设计思想

LSTM有三个门，计算复杂。GRU简化为两个门：

```
GRU结构：

     ┌─────────────────────────────┐
     │   ┌───┐                     │
     │   │ r │ 重置门              │
     │   └─┬─┘                     │
     │     ↓                       │
     │   ┌───┐   ┌───┐            │
h_t-1├──→│ × │←──│ z │更新门       │
     │   └─┬─┘   └───┘            │
     │     │       │               │
     │     ↓       ↓               │
     │   ┌───┐   ┌───┐            │
     │   │1-z│   │ z │            │
     │   └─┬─┘   └─┬─┘            │
     │     │       │               │
     │   ┌─┴───────┴─┐            │
     │   │    +      │──→ h_t      │
     │   └───────────┘             │
     └─────────────────────────────┘
```

### GRU的计算

```java
/**
 * GRU单元实现
 */
public class GRUCell {
    
    private int hiddenSize;
    private INDArray Wr, Wz, Wh;  // 重置门、更新门、候选值
    private INDArray Ur, Uz, Uh;
    private INDArray br, bz, bh;
    
    /**
     * GRU前向计算
     */
    public INDArray step(INDArray x, INDArray hPrev) {
        // 1. 重置门：决定忽略多少过去信息
        INDArray r = Transforms.sigmoid(
            x.mmul(Wr).add(hPrev.mmul(Ur)).add(br)
        );
        
        // 2. 更新门：决定新旧信息的混合比例
        INDArray z = Transforms.sigmoid(
            x.mmul(Wz).add(hPrev.mmul(Uz)).add(bz)
        );
        
        // 3. 候选隐状态
        INDArray hTilde = Transforms.tanh(
            x.mmul(Wh).add(r.mul(hPrev).mmul(Uh)).add(bh)
        );
        
        // 4. 最终隐状态
        INDArray h = z.mul(hPrev).add(z.rsub(1).mul(hTilde));
        // z.rsub(1) = 1 - z
        
        return h;
    }
}
```

### GRU vs LSTM

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 状态数量 | 2个(h, c) | 1个(h) |
| 参数量 | 较多 | 较少 |
| 计算速度 | 较慢 | 较快 |
| 表达能力 | 较强 | 相当 |

## 用Deeplearning4j实现LSTM

```java
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.LSTM;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * 使用DL4J构建LSTM网络
 */
public class LSTMNetwork {
    
    public static MultiLayerNetwork buildTextClassifier(
            int vocabSize, int embeddingSize, int hiddenSize, int numClasses) {
        
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.001))
            
            .list()
            // 词嵌入层
            .layer(new EmbeddingLayer.Builder()
                .nIn(vocabSize)
                .nOut(embeddingSize)
                .build())
            
            // LSTM层
            .layer(new LSTM.Builder()
                .nIn(embeddingSize)
                .nOut(hiddenSize)
                .activation(Activation.TANH)
                .build())
            
            // 取最后一个时间步的输出
            .layer(new LastTimeStep(new LSTM.Builder()
                .nIn(hiddenSize)
                .nOut(hiddenSize)
                .activation(Activation.TANH)
                .build()))
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .nIn(hiddenSize)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            
            .build();
        
        return new MultiLayerNetwork(config);
    }
}
```

## 为什么LSTM能解决长期依赖

### 梯度流动分析

```
LSTM的梯度流动：

∂c_t/∂c_{t-1} = f_t

当f_t ≈ 1时，梯度可以直接传递
当f_t ≈ 0时，梯度被阻断

网络学会了在需要时保持f_t接近1
```

### 信息选择性

```
LSTM学会了选择性处理信息：

遗忘门：什么信息已经过时了？
输入门：什么新信息很重要？
输出门：当前应该输出什么？

这种选择性是长期记忆的关键
```

## 设计思考：门控的哲学

### 门控的本质

```
门控 = 选择性

不是所有信息都同等重要
门控让网络学会"关注重要信息，忽略无关信息"
```

### 与人类认知的类比

| LSTM机制 | 人类认知 |
|----------|----------|
| 遗忘门 | 选择性遗忘 |
| 输入门 | 注意力聚焦 |
| 输出门 | 信息表达 |
| 细胞状态 | 长期记忆 |

### 设计原则

```
1. 信息要有"直通通道"：避免过度处理
2. 选择要有"门控机制"：学会取舍
3. 记忆要有"容量限制"：防止过载
```

## 小结

本章我们学习了：

1. **长期依赖问题**：梯度消失导致信息丢失
2. **LSTM结构**：三个门控机制
3. **GRU结构**：简化的两个门控
4. **为什么有效**：直通通道和选择性记忆

**核心概念：**

| 概念 | 说明 |
|------|------|
| 遗忘门 | 选择性遗忘旧信息 |
| 输入门 | 选择性接收新信息 |
| 输出门 | 选择性输出信息 |
| 细胞状态 | 信息的直通通道 |

**下一步：** 我们将用Java实现一个文本生成模型。

---

**思考题：**

1. LSTM的细胞状态为什么能解决长期依赖？
2. GRU相比LSTM做了哪些简化？为什么仍然有效？
3. 在什么情况下你会选择GRU而不是LSTM？
