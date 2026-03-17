# 4.1 序列数据的挑战：时间维度的引入

> "序列数据无处不在——文本、语音、股票价格、天气变化。理解序列，就是理解时间的智慧。"

## 什么是序列数据

### 序列的定义

序列是按时间或顺序排列的数据点：

```
时间序列：股票价格 [100, 102, 98, 105, 107, ...]
文本序列：["我", "爱", "编程"]
语音序列：音频波形 [0.1, 0.3, 0.5, 0.2, ...]
DNA序列：[A, T, G, C, A, G, ...]
```

### 序列数据的特点

| 特点 | 说明 | 示例 |
|------|------|------|
| 顺序性 | 顺序改变意义不同 | "狗咬人" vs "人咬狗" |
| 变长性 | 长度不固定 | 不同长度的句子 |
| 时序依赖 | 当前值依赖历史 | 股价预测 |
| 长期依赖 | 需要记住远距离信息 | 文章开头的伏笔 |

## 为什么传统网络处理不好序列

### 全连接网络的困境

假设用全连接网络处理长度为T的序列：

```java
// 传统全连接处理序列
public class FcForSequence {
    
    public double[] process(double[] sequence) {
        // 问题1：输入长度必须固定
        if (sequence.length != fixedLength) {
            throw new IllegalArgumentException("输入长度不匹配");
        }
        
        // 问题2：参数数量爆炸
        // 如果序列长度1000，隐藏层256，参数 = 1000 × 256 = 256,000
        
        // 问题3：无法处理时序依赖
        // 每个时间步独立处理，忽略顺序信息
        
        return null;
    }
}
```

### CNN处理序列的局限

```java
// CNN处理序列
public class CnnForSequence {
    
    public double[] process(double[] sequence) {
        // CNN可以捕捉局部模式
        // 但感受野有限，难以建模长期依赖
        
        // 例如：在文本中
        // "小明出生在1990年，...（很长一段）...他今年34岁了"
        // CNN难以关联"1990年"和"34岁"
        
        return null;
    }
}
```

### 核心问题

| 问题 | 全连接 | CNN | RNN |
|------|--------|-----|-----|
| 变长输入 | × | ✓ | ✓ |
| 参数共享 | × | ✓ | ✓ |
| 时序依赖 | × | 局部 | ✓ |
| 长期依赖 | × | × | △ |

## 序列建模的核心思想

### 条件概率视角

序列可以看作条件概率的链式展开：

```
P(x1, x2, x3, ..., xT) = P(x1) × P(x2|x1) × P(x3|x1,x2) × ... × P(xT|x1,...,xT-1)

即：每个时刻的输出依赖于之前所有时刻的输入
```

### 隐状态的思想

引入隐状态来"记忆"历史信息：

```
h_t = f(h_{t-1}, x_t)

h_t: 当前时刻的隐状态（记忆）
h_{t-1}: 上一时刻的隐状态
x_t: 当前时刻的输入
f: 状态转移函数
```

### 类比理解

想象你在读一本书：

```
读到第10页时，你脑中有之前9页的记忆（隐状态）
这个记忆帮你理解第10页的内容
每读一页，记忆就更新一次
```

## RNN的基本结构

### 展开图

```
时间步展开：
     t=1      t=2      t=3
    ┌───┐    ┌───┐    ┌───┐
x1→│RNN│──→│RNN│──→│RNN│→y3
    └───┘    └───┘    └───┘
      ↓        ↓        ↓
      y1       y2       y3

每个RNN单元共享相同的参数
```

### 数学表达

```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

### 用Java实现基础RNN

```java
/**
 * 简单RNN单元
 */
public class SimpleRNNCell {
    
    private double[][] Wxh;  // 输入到隐藏层的权重
    private double[][] Whh;  // 隐藏层到隐藏层的权重
    private double[] bh;     // 隐藏层偏置
    private double[][] Why;  // 隐藏层到输出层的权重
    private double[] by;     // 输出层偏置
    
    private int hiddenSize;
    
    public SimpleRNNCell(int inputSize, int hiddenSize, int outputSize) {
        this.hiddenSize = hiddenSize;
        
        // 初始化权重
        Wxh = initWeights(inputSize, hiddenSize);
        Whh = initWeights(hiddenSize, hiddenSize);
        Why = initWeights(hiddenSize, outputSize);
        bh = new double[hiddenSize];
        by = new double[outputSize];
    }
    
    /**
     * 单步前向传播
     */
    public RNNOutput step(double[] x, double[] hPrev) {
        // 计算新的隐藏状态
        double[] h = new double[hiddenSize];
        
        for (int i = 0; i < hiddenSize; i++) {
            h[i] = bh[i];
            
            // 输入贡献
            for (int j = 0; j < x.length; j++) {
                h[i] += Wxh[j][i] * x[j];
            }
            
            // 隐藏状态贡献
            for (int j = 0; j < hiddenSize; j++) {
                h[i] += Whh[j][i] * hPrev[j];
            }
            
            // 激活函数
            h[i] = Math.tanh(h[i]);
        }
        
        // 计算输出
        double[] y = new double[by.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = by[i];
            for (int j = 0; j < hiddenSize; j++) {
                y[i] += Why[j][i] * h[j];
            }
        }
        
        return new RNNOutput(y, h);
    }
    
    /**
     * 处理整个序列
     */
    public List<RNNOutput> forward(List<double[]> sequence) {
        List<RNNOutput> outputs = new ArrayList<>();
        double[] h = new double[hiddenSize];  // 初始隐藏状态为0
        
        for (double[] x : sequence) {
            RNNOutput out = step(x, h);
            outputs.add(out);
            h = out.hidden;
        }
        
        return outputs;
    }
    
    private double[][] initWeights(int rows, int cols) {
        double[][] w = new double[rows][cols];
        double scale = Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                w[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return w;
    }
    
    public record RNNOutput(double[] output, double[] hidden) {}
}
```

## RNN的应用场景

### 1. 文本生成

```
输入: "今天天气"
输出: "很好，适合出门"
```

### 2. 机器翻译

```
输入: "I love programming"
输出: "我爱编程"
```

### 3. 语音识别

```
输入: 音频波形序列
输出: 文字序列
```

### 4. 时间序列预测

```
输入: 过去7天的股价
输出: 明天的股价预测
```

## RNN的变体

### 多对一（Many-to-One）

```
序列输入 → 单个输出
应用：情感分析、文本分类

[我, 爱, 编程] → "正面情感"
```

### 一对多（One-to-Many）

```
单个输入 → 序列输出
应用：图像描述生成

[图像] → "一只猫坐在沙发上"
```

### 多对多（Many-to-Many）

```
序列输入 → 序列输出
应用：机器翻译、视频分类

[I, love, you] → [我, 爱, 你]
```

## 设计思考：时间维度的引入

### 从空间到时间

| 维度 | 数据类型 | 网络类型 |
|------|----------|----------|
| 空间 | 图像 | CNN |
| 时间 | 序列 | RNN |
| 时空 | 视频 | CNN + RNN |

### 记忆的本质

RNN引入了"记忆"的概念：

```
记忆 = 压缩的历史信息

h_t = 压缩(x_1, x_2, ..., x_t)

每一步都将新信息压缩到固定大小的状态中
```

### 设计哲学

```
传统网络：静态映射 f(x) = y
RNN：动态过程 f(x, h) = (y, h')

引入状态，让网络有了"历史"
```

## 小结

本章我们学习了：

1. **序列数据的特点**：顺序性、变长性、时序依赖
2. **传统网络的局限**：无法处理变长和时序依赖
3. **RNN的核心思想**：隐状态记忆历史
4. **RNN的应用场景**：文本、语音、时间序列

**核心概念：**

| 概念 | 说明 |
|------|------|
| 隐状态 | 压缩的历史信息 |
| 时序依赖 | 当前依赖历史 |
| 参数共享 | 所有时间步共享参数 |

**下一步：** 我们将深入RNN的核心机制——记忆与遗忘。

---

**思考题：**

1. 为什么RNN能处理变长序列？
2. RNN的参数共享与CNN有什么异同？
3. 举出生活中需要"记忆"才能理解的序列例子。
