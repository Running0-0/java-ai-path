<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 3.2 卷积操作的本质](02-convolution-operation.md)</span>

<span>[3.4 经典CNN架构解析 →](04-classic-cnn-architectures.md)</span>

</div>

---

# 3.3 池化与全连接：信息压缩的智慧

> "池化是智慧的遗忘，全连接是全局的整合。两者结合，让网络既高效又强大。"

## 为什么需要池化

### 问题：特征图太大

经过卷积后，特征图可能仍然很大：

```
输入图像: 224×224×3
卷积后: 222×222×64  （使用3×3卷积核）

参数量: 222×222×64 ≈ 316万个值！
```

这会导致：
- 计算量大
- 参数多，容易过拟合
- 丢失空间层次信息

**解决方案：池化（Pooling）**

## 池化操作

### 最大池化

取窗口内的最大值：

```
输入:           2×2池化:
┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 4 │    ┌───┬───┐
├───┼───┼───┼───┤    │ 6 │ 8 │
│ 5 │ 6 │ 7 │ 8 │ →  ├───┼───┤
├───┼───┼───┼───┤    │12 │16 │
│ 9 │10 │11 │12 │    └───┴───┘
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘

每个窗口取最大值：
max(1,3,5,6)=6, max(2,4,7,8)=8
max(9,10,13,14)=12, max(11,12,15,16)=16
```

### 平均池化

取窗口内的平均值：

```java
/**
 * 池化操作实现
 */
public class Pooling {
    
    /**
     * 最大池化
     */
    public static double[][] maxPool(double[][] input, int poolSize, int stride) {
        int h = input.length;
        int w = input[0].length;
        
        int outH = (h - poolSize) / stride + 1;
        int outW = (w - poolSize) / stride + 1;
        
        double[][] output = new double[outH][outW];
        
        for (int i = 0; i < outH; i++) {
            for (int j = 0; j < outW; j++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int pi = 0; pi < poolSize; pi++) {
                    for (int pj = 0; pj < poolSize; pj++) {
                        max = Math.max(max, input[i * stride + pi][j * stride + pj]);
                    }
                }
                output[i][j] = max;
            }
        }
        
        return output;
    }
    
    /**
     * 平均池化
     */
    public static double[][] avgPool(double[][] input, int poolSize, int stride) {
        int h = input.length;
        int w = input[0].length;
        
        int outH = (h - poolSize) / stride + 1;
        int outW = (w - poolSize) / stride + 1;
        
        double[][] output = new double[outH][outW];
        
        for (int i = 0; i < outH; i++) {
            for (int j = 0; j < outW; j++) {
                double sum = 0;
                for (int pi = 0; pi < poolSize; pi++) {
                    for (int pj = 0; pj < poolSize; pj++) {
                        sum += input[i * stride + pi][j * stride + pj];
                    }
                }
                output[i][j] = sum / (poolSize * poolSize);
            }
        }
        
        return output;
    }
}
```

### 池化的作用

| 作用 | 说明 |
|------|------|
| 降维 | 减少计算量 |
| 平移不变性 | 小的位置变化不影响结果 |
| 扩大感受野 | 后续层能看到更大区域 |
| 防止过拟合 | 减少参数数量 |

## 全连接层

### 从特征到分类

卷积和池化提取了特征，全连接层负责分类：

```
特征图 [H, W, C] → 展平 → [H×W×C] → 全连接 → [类别数]
```

### 全连接的结构

```java
/**
 * 全连接层
 */
public class FullyConnectedLayer {
    
    private double[][] weights;  // [输入维度, 输出维度]
    private double[] bias;       // [输出维度]
    
    public FullyConnectedLayer(int inputDim, int outputDim) {
        weights = new double[inputDim][outputDim];
        bias = new double[outputDim];
        
        // Xavier初始化
        double scale = Math.sqrt(2.0 / (inputDim + outputDim));
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                weights[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
    }
    
    public double[] forward(double[] input) {
        double[] output = new double[bias.length];
        
        for (int j = 0; j < output.length; j++) {
            output[j] = bias[j];
            for (int i = 0; i < input.length; i++) {
                output[j] += input[i] * weights[i][j];
            }
        }
        
        return output;
    }
}
```

### 全连接的作用

```
卷积层：提取局部特征
池化层：聚合空间信息
全连接层：整合全局信息，做出最终决策
```

## 完整的CNN结构

### 典型架构

```
输入图像
    ↓
卷积层1 → 激活 → 池化
    ↓
卷积层2 → 激活 → 池化
    ↓
卷积层3 → 激活 → 池化
    ↓
展平
    ↓
全连接层1 → 激活
    ↓
全连接层2（输出层）→ Softmax
    ↓
类别概率
```

### 用DL4J构建CNN

```java
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * 使用DL4J构建CNN
 */
public class CnnBuilder {
    
    public static MultiLayerNetwork buildCnn(int inputHeight, int inputWidth, int numClasses) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            
            .list()
            // 卷积块1
            .layer(new ConvolutionLayer.Builder(3, 3)  // 3×3卷积核
                .nIn(3)          // 输入通道数（RGB）
                .nOut(32)        // 输出通道数
                .stride(1, 1)    // 步长
                .padding(1, 1)   // 填充
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)  // 2×2池化
                .stride(2, 2)
                .build())
            
            // 卷积块2
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 卷积块3
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 全连接层
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new DropoutLayer(0.5))  // Dropout防止过拟合
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutional(inputHeight, inputWidth, 3))
            .build();
        
        return new MultiLayerNetwork(config);
    }
}
```

## 全局平均池化

### 传统方法的问题

全连接层参数量大：

```
特征图: 7×7×512 = 25088
全连接层: 25088 × 4096 ≈ 1亿参数！
```

### 全局平均池化

对每个特征图取全局平均值：

```java
/**
 * 全局平均池化
 */
public class GlobalAveragePooling {
    
    /**
     * 输入: [H, W, C]
     * 输出: [C]
     */
    public static double[] forward(double[][][] input) {
        int h = input.length;
        int w = input[0].length;
        int c = input[0][0].length;
        
        double[] output = new double[c];
        
        for (int ch = 0; ch < c; ch++) {
            double sum = 0;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    sum += input[i][j][ch];
                }
            }
            output[ch] = sum / (h * w);
        }
        
        return output;
    }
}
```

**优势：**
- 无参数
- 减少过拟合
- 直接输出类别分数

## 设计思考：信息流动的设计

### 尺度金字塔

CNN形成特征尺度金字塔：

```
浅层：高分辨率，低语义
    ↓
中层：中分辨率，中语义
    ↓
深层：低分辨率，高语义
```

### 感受野的概念

感受野是指输出像素"看到"的输入区域：

```
卷积核3×3，步长1：
第1层：感受野3×3
第2层：感受野5×5
第3层：感受野7×7

越深的层，感受野越大，能看到更全局的信息
```

### 信息压缩的平衡

| 操作 | 信息变化 |
|------|----------|
| 卷积 | 提取特征，保持空间 |
| 池化 | 压缩空间，保留主要特征 |
| 全连接 | 压缩空间，整合全局 |

**设计原则：在信息压缩和信息保留之间找到平衡。**

## 实践：完整的CNN示例

```java
package com.example.ai.chapter03;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * CNN识别MNIST
 */
public class MnistCnnDemo {
    
    public static void main(String[] args) throws Exception {
        // 数据
        DataSetIterator trainData = new MnistDataSetIterator(64, true, 42);
        DataSetIterator testData = new MnistDataSetIterator(64, false, 42);
        
        // CNN配置
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.001))
            
            .list()
            // 卷积块1: 28x28x1 -> 14x14x32
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(1)
                .nOut(32)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 卷积块2: 14x14x32 -> 7x7x64
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 全连接层
            .layer(new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(new DropoutLayer(0.5))
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        
        System.out.println(model.summary());
        
        // 训练
        for (int epoch = 0; epoch < 5; epoch++) {
            model.fit(trainData);
            
            var eval = model.evaluate(testData);
            System.out.printf("Epoch %d - 准确率: %.2f%%%n", 
                epoch + 1, eval.accuracy() * 100);
            
            trainData.reset();
            testData.reset();
        }
    }
}
```

## 小结

本章我们学习了：

1. **池化操作**：降维、扩大感受野
2. **全连接层**：整合全局信息
3. **CNN架构**：卷积-池化-全连接的组合
4. **全局平均池化**：无参数的分类方法

**核心概念：**

| 概念 | 作用 |
|------|------|
| 最大池化 | 保留最显著特征 |
| 平均池化 | 保留整体信息 |
| 全连接 | 全局整合分类 |
| 感受野 | 特征"看到"的范围 |

**下一步：** 我们将学习经典的CNN架构。

---

**思考题：**

1. 最大池化和平均池化各有什么优缺点？
2. 为什么要在全连接层前展平特征图？
3. 全局平均池化相比全连接有什么优势？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 3.2 卷积操作的本质](02-convolution-operation.md)</span>

<span>[3.4 经典CNN架构解析 →](04-classic-cnn-architectures.md)</span>

</div>
