# 4.2 RNN的核心思想：记忆与遗忘

> "记忆是智能的基础，遗忘是智慧的艺术。RNN在两者之间寻找平衡。"

## 从一个故事说起

想象你在读一本小说：

```
第1章：主角小明出场，性格开朗
第2章：小明遇到了困难
...
第50章：小明回忆起第1章的经历，做出决定
```

要理解第50章，你需要记住第1章的信息。但如果你记住所有细节，大脑会过载。所以你会：
- **记住**重要信息（主角性格）
- **遗忘**不重要信息（天气细节）

RNN也是如此。

## RNN的核心机制

### 隐状态：记忆的载体

隐状态h承载了序列的历史信息：

```
h_t = f(h_{t-1}, x_t)

当前记忆 = 更新函数(旧记忆, 新输入)
```

### 信息流动

```
时间步t的信息流动：

         ┌──────────────────┐
         │                  ↓
    [h_{t-1}] ──→ [更新] ──→ [h_t]
         ↑            ↑
    [历史记忆]    [新输入x_t]
```

### 完整的RNN计算

```java
/**
 * RNN单元的完整计算
 */
public class RNNCell {
    
    private int hiddenSize;
    private INDArray Wxh, Whh, bh;
    private INDArray Why, by;
    
    /**
     * 单步计算
     */
    public RNNOutput step(INDArray x, INDArray hPrev) {
        /*
         * 核心公式：
         * h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)
         * y_t = Why·h_t + by
         */
        
        // 1. 计算新的隐藏状态
        INDArray h = x.mmul(Wxh)           // 输入变换
                    .add(hPrev.mmul(Whh))  // 历史信息
                    .add(bh)               // 偏置
                    .transform(Transforms.tanh(true));  // 激活
        
        // 2. 计算输出
        INDArray y = h.mmul(Why).add(by);
        
        return new RNNOutput(y, h);
    }
}
```

## 记忆的更新过程

### 逐步追踪

让我们追踪一个简单例子：

```
序列: [1, 2, 3]
隐藏层大小: 2
```

```java
/**
 * 记忆更新演示
 */
public class MemoryUpdateDemo {
    
    public static void main(String[] args) {
        // 假设权重
        double wxh = 0.5;  // 输入权重
        double whh = 0.8;  // 隐状态权重
        
        double h = 0;  // 初始记忆为0
        
        // 处理序列 [1, 2, 3]
        double[] sequence = {1, 2, 3};
        
        for (int t = 0; t < sequence.length; t++) {
            double x = sequence[t];
            
            // 记忆更新
            double newH = Math.tanh(wxh * x + whh * h);
            
            System.out.printf("t=%d: 输入=%.1f, 旧记忆=%.4f, 新记忆=%.4f%n",
                t + 1, x, h, newH);
            
            h = newH;
        }
    }
}
```

**输出：**
```
t=1: 输入=1.0, 旧记忆=0.0000, 新记忆=0.4621
t=2: 输入=2.0, 旧记忆=0.4621, 新记忆=0.9051
t=3: 输入=3.0, 旧记忆=0.9051, 新记忆=0.9890
```

### 信息衰减

注意到旧记忆的影响逐渐衰减：

```
h_t = tanh(0.5·x_t + 0.8·h_{t-1})

h_1 = tanh(0.5·1 + 0) = 0.46
h_2 = tanh(0.5·2 + 0.8·0.46) = 0.91
h_3 = tanh(0.5·3 + 0.8·0.91) = 0.99

x_1对h_3的贡献：0.5 × 0.8 × 0.8 = 0.32（衰减了）
```

## 长期依赖问题

### 梯度消失

当序列很长时，早期信息会逐渐消失：

```java
/**
 * 梯度消失演示
 */
public class VanishingGradientDemo {
    
    public static void main(String[] args) {
        double tanhDerivative = 0.25;  // tanh导数最大值
        double whh = 0.8;
        
        double gradient = 1.0;
        
        for (int t = 1; t <= 20; t++) {
            gradient *= whh * tanhDerivative;
            System.out.printf("第%d步梯度: %.10f%n", t, gradient);
        }
    }
}
```

**输出：**
```
第1步梯度: 0.2000000000
第2步梯度: 0.0400000000
第3步梯度: 0.0080000000
...
第10步梯度: 0.0000001024
第20步梯度: 0.0000000000  // 几乎为0！
```

### 长期依赖的例子

```
句子："我出生在法国...（很长的描述）...所以我说一口流利的法语"

要预测"法语"，需要记住开头的"法国"
但经过很多步后，这个信息可能已经丢失
```

## 双向RNN

### 动机

有时候，理解当前词需要看后面的内容：

```
"他拿起电话，拨打了110报警"

读到"拨打"时，不知道拨打什么
读到"110"后，才理解是在报警
```

### 结构

```
前向RNN：从左到右处理
后向RNN：从右到左处理
合并：拼接或求和两个方向的隐状态

    →→→→→→→
x1  x2  x3  x4
    ←←←←←←←
```

### 代码实现

```java
/**
 * 双向RNN
 */
public class BidirectionalRNN {
    
    private RNNCell forwardRNN;
    private RNNCell backwardRNN;
    
    public INDArray[] forward(INDArray[] sequence) {
        int T = sequence.length;
        
        // 前向传播
        INDArray[] forwardH = new INDArray[T];
        INDArray hForward = Nd4j.zeros(forwardRNN.getHiddenSize());
        for (int t = 0; t < T; t++) {
            RNNOutput out = forwardRNN.step(sequence[t], hForward);
            forwardH[t] = out.hidden();
            hForward = out.hidden();
        }
        
        // 后向传播
        INDArray[] backwardH = new INDArray[T];
        INDArray hBackward = Nd4j.zeros(backwardRNN.getHiddenSize());
        for (int t = T - 1; t >= 0; t--) {
            RNNOutput out = backwardRNN.step(sequence[t], hBackward);
            backwardH[t] = out.hidden();
            hBackward = out.hidden();
        }
        
        // 合并
        INDArray[] hidden = new INDArray[T];
        for (int t = 0; t < T; t++) {
            hidden[t] = Nd4j.hstack(forwardH[t], backwardH[t]);
        }
        
        return hidden;
    }
}
```

## 深层RNN

### 结构

```
输入层
    ↓
RNN层1
    ↓
RNN层2
    ↓
RNN层3
    ↓
输出层
```

### 代码实现

```java
/**
 * 多层RNN
 */
public class StackedRNN {
    
    private List<RNNCell> layers;
    
    public StackedRNN(int inputSize, int hiddenSize, int numLayers) {
        layers = new ArrayList<>();
        
        for (int i = 0; i < numLayers; i++) {
            int inSize = (i == 0) ? inputSize : hiddenSize;
            layers.add(new RNNCell(inSize, hiddenSize));
        }
    }
    
    public List<INDArray> forward(List<INDArray> sequence) {
        List<INDArray> current = sequence;
        
        for (RNNCell layer : layers) {
            current = processLayer(layer, current);
        }
        
        return current;
    }
    
    private List<INDArray> processLayer(RNNCell layer, List<INDArray> inputs) {
        List<INDArray> outputs = new ArrayList<>();
        INDArray h = Nd4j.zeros(layer.getHiddenSize());
        
        for (INDArray x : inputs) {
            RNNOutput out = layer.step(x, h);
            outputs.add(out.hidden());
            h = out.hidden();
        }
        
        return outputs;
    }
}
```

## 设计思考：记忆的哲学

### 记忆与遗忘的平衡

```
完全记住：信息过载，无法处理
完全遗忘：没有上下文，无法理解

RNN通过tanh激活函数实现"软遗忘"：
- 信息被压缩到[-1, 1]
- 不重要的信息被压缩到接近0
```

### 与人类记忆的类比

| RNN机制 | 人类记忆 |
|---------|----------|
| 隐状态 | 工作记忆 |
| 参数共享 | 学习能力 |
| 梯度消失 | 遗忘曲线 |
| 深层RNN | 多层认知 |

### 设计原则

```
1. 状态大小要适中：太小记不住，太大难训练
2. 激活函数要合适：tanh适合记忆，ReLU适合特征
3. 层数不要太多：RNN本身就有时序深度
```

## 小结

本章我们学习了：

1. **隐状态机制**：记忆的载体
2. **信息流动**：历史与当前的融合
3. **长期依赖问题**：梯度消失
4. **RNN变体**：双向RNN、深层RNN

**核心概念：**

| 概念 | 说明 |
|------|------|
| 隐状态 | 压缩的历史信息 |
| 记忆更新 | h_t = f(h_{t-1}, x_t) |
| 梯度消失 | 长期信息丢失 |
| 双向RNN | 利用未来信息 |

**下一步：** 我们将学习LSTM和GRU——解决长期依赖的关键技术。

---

**思考题：**

1. 为什么tanh激活函数比ReLU更适合RNN？
2. 双向RNN在什么场景下有用？
3. 如何判断一个序列任务是否存在长期依赖问题？
