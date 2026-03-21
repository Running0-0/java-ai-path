<table width="100%">
   <tr>
      <td align="left"><a href="01-perceptron-to-neural-network.md">← 2.1 从感知机到神经网络</a></td>
      <td align="right"><a href="03-backpropagation.md">2.3 反向传播 →</a></td>
   </tr>
</table>

---

# 2.2 前向传播：数据流动的艺术

> "前向传播就像一条流水线，数据从输入端进入，经过层层加工，最终变成我们想要的输出。"

## 从工厂流水线理解前向传播

想象一个汽车制造厂：

```
原材料 → 冲压车间 → 焊接车间 → 涂装车间 → 总装车间 → 成品汽车
```

每个车间都对材料进行加工，最终产出汽车。

神经网络的前向传播也是类似的：

```
输入数据 → 第1层处理 → 第2层处理 → ... → 输出结果
```

**关键区别：**
- 工厂加工的是物质材料
- 神经网络加工的是数据（数字）

## 前向传播的数学基础

### 单个神经元的计算

一个神经元的工作流程：

```
输入 → 加权求和 → 加偏置 → 激活函数 → 输出
```

用Java代码表示：

```java
public class Neuron {
    private double[] weights;
    private double bias;
    
    public double forward(double[] inputs) {
        // Step 1: 加权求和
        double z = 0;
        for (int i = 0; i < inputs.length; i++) {
            z += inputs[i] * weights[i];
        }
        
        // Step 2: 加偏置
        z += bias;
        
        // Step 3: 激活函数
        double output = activation(z);
        
        return output;
    }
    
    private double activation(double x) {
        return 1.0 / (1.0 + Math.exp(-x));  // Sigmoid
    }
}
```

### 向量化计算

当输入和神经元数量增加时，用矩阵运算更高效：

```java
// 使用ND4J进行矩阵运算
public class VectorizedNeuron {
    private INDArray weights;  // 权重矩阵
    private INDArray bias;     // 偏置向量
    
    public INDArray forward(INDArray inputs) {
        // z = W·x + b
        INDArray z = inputs.mmul(weights.transpose()).add(bias);
        
        // a = σ(z)
        return sigmoid(z);
    }
    
    private INDArray sigmoid(INDArray x) {
        return Transforms.sigmoid(x);
    }
}
```

**为什么向量化重要？**

| 方式 | 处理1000个输入 |
|------|----------------|
| 循环 | 逐个计算，慢 |
| 向量化 | 矩阵运算，快100倍+ |

## 完整的前向传播过程

### 三层网络示例

假设一个简单的网络结构：
- 输入层：3个神经元
- 隐藏层：4个神经元
- 输出层：2个神经元

```
输入层        隐藏层        输出层
  x1 ────────┐
             ├──→ h1 ────┐
  x2 ────────┤           ├──→ y1
             ├──→ h2 ────┤
  x3 ────────┤           ├──→ y2
             ├──→ h3 ────┘
             │
             └──→ h4
```

### 逐步计算

```java
public class ForwardPropagation {
    
    public static void main(String[] args) {
        // 初始化网络参数
        double[][] W1 = {  // 输入层到隐藏层的权重 (4×3)
            {0.1, 0.2, -0.1},
            {-0.1, 0.1, 0.9},
            {0.1, 0.4, 0.1},
            {0.3, 0.1, 0.2}
        };
        double[] b1 = {0.1, 0.2, 0.3, 0.4};  // 隐藏层偏置
        
        double[][] W2 = {  // 隐藏层到输出层的权重 (2×4)
            {0.3, 0.1, -0.2, 0.1},
            {0.2, -0.1, 0.1, 0.3}
        };
        double[] b2 = {0.1, 0.2};  // 输出层偏置
        
        // 输入数据
        double[] x = {0.5, 0.3, 0.2};
        
        // === 前向传播 ===
        
        // Step 1: 输入层 → 隐藏层
        double[] z1 = new double[4];  // 隐藏层加权输入
        double[] a1 = new double[4];  // 隐藏层激活输出
        
        for (int j = 0; j < 4; j++) {
            z1[j] = b1[j];
            for (int i = 0; i < 3; i++) {
                z1[j] += x[i] * W1[j][i];
            }
            a1[j] = sigmoid(z1[j]);  // 激活函数
        }
        
        System.out.println("隐藏层输出: " + Arrays.toString(a1));
        
        // Step 2: 隐藏层 → 输出层
        double[] z2 = new double[2];  // 输出层加权输入
        double[] a2 = new double[2];  // 输出层激活输出
        
        for (int j = 0; j < 2; j++) {
            z2[j] = b2[j];
            for (int i = 0; i < 4; i++) {
                z2[j] += a1[i] * W2[j][i];
            }
            a2[j] = sigmoid(z2[j]);  // 激活函数
        }
        
        System.out.println("输出层输出: " + Arrays.toString(a2));
    }
    
    static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
```

### 使用ND4J的简洁实现

```java
public class VectorizedForwardPropagation {
    
    public static void main(String[] args) {
        // 定义网络结构
        int inputSize = 3;
        int hiddenSize = 4;
        int outputSize = 2;
        
        // 初始化权重和偏置
        INDArray W1 = Nd4j.randn(hiddenSize, inputSize).mul(0.1);
        INDArray b1 = Nd4j.zeros(hiddenSize);
        
        INDArray W2 = Nd4j.randn(outputSize, hiddenSize).mul(0.1);
        INDArray b2 = Nd4j.zeros(outputSize);
        
        // 输入数据（批量处理）
        INDArray X = Nd4j.create(new double[][]{
            {0.5, 0.3, 0.2},
            {0.1, 0.9, 0.4},
            {0.8, 0.2, 0.6}
        });
        
        // 前向传播
        INDArray z1 = X.mmul(W1.transpose()).addRowVector(b1);
        INDArray a1 = Transforms.sigmoid(z1);
        
        INDArray z2 = a1.mmul(W2.transpose()).addRowVector(b2);
        INDArray a2 = Transforms.sigmoid(z2);
        
        System.out.println("输出结果:\n" + a2);
    }
}
```

## 激活函数：引入非线性

### 为什么需要激活函数？

如果没有激活函数，多层网络会退化成单层：

```
y = W2(W1·x + b1) + b2
  = W2·W1·x + W2·b1 + b2
  = W'·x + b'  （等价于单层！）
```

激活函数打破这种线性关系，让网络能够学习复杂模式。

### 常用激活函数

#### 1. Sigmoid

```java
double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
}
```

**特点：**
- 输出范围：(0, 1)
- 优点：输出有界，适合概率输出
- 缺点：梯度消失问题

**适用场景：** 二分类输出层

#### 2. Tanh

```java
double tanh(double x) {
    return Math.tanh(x);  // 或 2*sigmoid(2x) - 1
}
```

**特点：**
- 输出范围：(-1, 1)
- 优点：零中心化
- 缺点：仍有梯度消失

**适用场景：** 隐藏层（RNN常用）

#### 3. ReLU (Rectified Linear Unit)

```java
double relu(double x) {
    return Math.max(0, x);
}
```

**特点：**
- 输出范围：[0, +∞)
- 优点：计算快，缓解梯度消失
- 缺点：Dead ReLU问题

**适用场景：** 隐藏层（最常用）

#### 4. Leaky ReLU

```java
double leakyRelu(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}
```

**特点：**
- 解决Dead ReLU问题
- alpha通常取0.01

#### 5. Softmax

```java
double[] softmax(double[] x) {
    double[] result = new double[x.length];
    double sum = 0;
    
    // 数值稳定性：减去最大值
    double max = Arrays.stream(x).max().getAsDouble();
    
    for (int i = 0; i < x.length; i++) {
        result[i] = Math.exp(x[i] - max);
        sum += result[i];
    }
    
    for (int i = 0; i < x.length; i++) {
        result[i] /= sum;
    }
    
    return result;
}
```

**特点：**
- 输出和为1
- 适合多分类

**适用场景：** 多分类输出层

### 激活函数对比

| 激活函数 | 公式 | 输出范围 | 主要用途 |
|----------|------|----------|----------|
| Sigmoid | 1/(1+e^(-x)) | (0,1) | 二分类输出 |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1,1) | RNN隐藏层 |
| ReLU | max(0,x) | [0,+∞) | 隐藏层（首选） |
| Leaky ReLU | max(αx,x) | (-∞,+∞) | 隐藏层 |
| Softmax | e^x/Σe^x | (0,1),和为1 | 多分类输出 |

## 批处理：高效的数据处理

### 为什么使用批处理？

| 方式 | 处理1000个样本 |
|------|----------------|
| 单样本 | 1000次前向传播 |
| 批处理 | 1次矩阵运算 |

批处理利用矩阵运算的并行性，大幅提升效率。

### 批处理实现

```java
public class BatchForwardPropagation {
    
    public static void main(String[] args) {
        int batchSize = 32;
        int inputSize = 784;  // MNIST图像大小
        int hiddenSize = 256;
        int outputSize = 10;
        
        // 初始化权重
        INDArray W1 = Nd4j.randn(hiddenSize, inputSize).mul(Math.sqrt(2.0 / inputSize));
        INDArray b1 = Nd4j.zeros(hiddenSize);
        INDArray W2 = Nd4j.randn(outputSize, hiddenSize).mul(Math.sqrt(2.0 / hiddenSize));
        INDArray b2 = Nd4j.zeros(outputSize);
        
        // 模拟一个批次的数据
        INDArray batch = Nd4j.randn(batchSize, inputSize);
        
        // 批量前向传播
        long startTime = System.currentTimeMillis();
        
        INDArray z1 = batch.mmul(W1.transpose()).addRowVector(b1);
        INDArray a1 = Transforms.relu(z1);  // ReLU激活
        
        INDArray z2 = a1.mmul(W2.transpose()).addRowVector(b2);
        INDArray a2 = softmax(z2);  // Softmax输出
        
        long endTime = System.currentTimeMillis();
        
        System.out.printf("批处理 %d 个样本，耗时: %d ms%n", batchSize, endTime - startTime);
        System.out.printf("输出形状: %s%n", Arrays.toString(a2.shape()));
    }
    
    static INDArray softmax(INDArray x) {
        INDArray max = x.max(true, 1);
        INDArray exp = Transforms.exp(x.subColumnVector(max));
        INDArray sum = exp.sum(true, 1);
        return exp.divColumnVector(sum);
    }
}
```

## 设计思考：数据流动的设计哲学

### 信息瓶颈原理

每一层都在对信息进行压缩和提取：

```
原始数据 → 特征提取 → 抽象表示 → 最终决策
  高维      中维        低维        标量
```

**设计原则：**
- 逐层减少维度（大部分情况）
- 每层学习更高级的抽象
- 避免信息过度压缩

### 残差连接：让信息流动更顺畅

深层网络可能出现信息丢失，残差连接提供"直通通道"：

```java
// 普通层
a = activation(W·x + b)

// 残差层
a = activation(W·x + b) + x  // 加上原始输入
```

这解决了深层网络的梯度消失问题，是ResNet的核心创新。

## 实践：完整的前向传播Demo

```java
package com.example.ai.chapter02;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 完整的前向传播示例
 */
public class ForwardPropagationDemo {
    
    public static void main(String[] args) {
        // 创建一个3层网络
        NeuralNetwork nn = new NeuralNetwork(
            new int[]{784, 256, 128, 10},  // 层大小
            ActivationFunction.RELU,        // 隐藏层激活
            ActivationFunction.SOFTMAX      // 输出层激活
        );
        
        // 模拟输入（MNIST图像展平）
        INDArray input = Nd4j.rand(1, 784);
        
        // 前向传播
        INDArray output = nn.forward(input);
        
        System.out.println("网络输出（各类别概率）:");
        System.out.println(output);
        
        // 获取预测类别
        int predictedClass = output.argMax(1).getInt(0);
        System.out.printf("预测类别: %d%n", predictedClass);
    }
    
    enum ActivationFunction {
        RELU, SIGMOID, TANH, SOFTMAX
    }
    
    static class NeuralNetwork {
        private INDArray[] weights;
        private INDArray[] biases;
        private ActivationFunction hiddenActivation;
        private ActivationFunction outputActivation;
        
        public NeuralNetwork(int[] layerSizes, 
                           ActivationFunction hiddenActivation,
                           ActivationFunction outputActivation) {
            this.hiddenActivation = hiddenActivation;
            this.outputActivation = outputActivation;
            
            int numLayers = layerSizes.length - 1;
            weights = new INDArray[numLayers];
            biases = new INDArray[numLayers];
            
            // Xavier初始化
            for (int i = 0; i < numLayers; i++) {
                weights[i] = Nd4j.randn(layerSizes[i + 1], layerSizes[i])
                    .mul(Math.sqrt(2.0 / (layerSizes[i] + layerSizes[i + 1])));
                biases[i] = Nd4j.zeros(layerSizes[i + 1]);
            }
        }
        
        public INDArray forward(INDArray input) {
            INDArray current = input;
            
            for (int i = 0; i < weights.length; i++) {
                // 线性变换: z = W·x + b
                current = current.mmul(weights[i].transpose()).addRowVector(biases[i]);
                
                // 激活函数
                if (i < weights.length - 1) {
                    current = activate(current, hiddenActivation);
                } else {
                    current = activate(current, outputActivation);
                }
            }
            
            return current;
        }
        
        private INDArray activate(INDArray x, ActivationFunction func) {
            return switch (func) {
                case RELU -> Transforms.relu(x);
                case SIGMOID -> Transforms.sigmoid(x);
                case TANH -> Transforms.tanh(x);
                case SOFTMAX -> softmax(x);
            };
        }
        
        private INDArray softmax(INDArray x) {
            INDArray max = x.max(true, 1);
            INDArray exp = Transforms.exp(x.subColumnVector(max));
            INDArray sum = exp.sum(true, 1);
            return exp.divColumnVector(sum);
        }
    }
}
```

## 小结

本章我们学习了：

1. **前向传播的本质**：数据在网络中的流动过程
2. **计算步骤**：加权求和 → 加偏置 → 激活函数
3. **向量化**：用矩阵运算提升效率
4. **激活函数**：引入非线性的关键
5. **批处理**：并行处理多个样本

**核心公式：**

```
z = W·x + b        （线性变换）
a = f(z)           （非线性激活）
```

**下一步：** 我们将学习反向传播——神经网络如何"学习"调整参数。

---

**思考题：**

1. 为什么激活函数对神经网络如此重要？
2. ReLU相比Sigmoid有什么优势？
3. 批处理为什么能提升计算效率？尝试解释矩阵运算的并行性。

---

<table width="100%">
   <tr>
      <td align="left"><a href="01-perceptron-to-neural-network.md">← 2.1 从感知机到神经网络</a></td>
      <td align="right"><a href="03-backpropagation.md">2.3 反向传播 →</a></td>
   </tr>
</table>