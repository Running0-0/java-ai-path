<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 3.3 池化与全连接](03-pooling-and-fully-connected.md)</span>

<span>[3.5 实战：用Java构建图像分类器 →](05-build-image-classifier.md)</span>

</div>

---

# 3.4 经典CNN架构解析：LeNet、AlexNet、ResNet

> "站在巨人的肩膀上，理解经典架构的设计智慧。"

## CNN架构演进史

```
1998: LeNet-5      → 开创性工作，手写数字识别
2012: AlexNet      → 深度学习爆发的起点
2014: VGG          → 更深的网络
2014: GoogLeNet    → Inception模块
2015: ResNet       → 残差连接，突破深度限制
2017: MobileNet    → 轻量化，移动端部署
2020: Vision Transformer → 将图像切成Patch后交给Transformer建模
```

## LeNet-5：CNN的鼻祖

### 架构设计

Yann LeCun在1998年设计，用于手写数字识别：

```
输入: 32×32×1（灰度图像）
    ↓
卷积层C1: 6个5×5卷积核 → 28×28×6
    ↓
池化层S2: 2×2平均池化 → 14×14×6
    ↓
卷积层C3: 16个5×5卷积核 → 10×10×16
    ↓
池化层S4: 2×2平均池化 → 5×5×16
    ↓
全连接层C5: 120个神经元
    ↓
全连接层F6: 84个神经元
    ↓
输出层: 10个类别
```

### 用DL4J实现LeNet

```java
/**
 * LeNet-5实现
 */
public class LeNet {
    
    public static MultiLayerNetwork build() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            
            .list()
            // C1: 卷积层
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(1)
                .nOut(6)
                .activation(Activation.TANH)  // 原始用Tanh
                .build())
            
            // S2: 池化层
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // C3: 卷积层
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nOut(16)
                .activation(Activation.TANH)
                .build())
            
            // S4: 池化层
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // C5: 全连接层（实际是卷积）
            .layer(new DenseLayer.Builder()
                .nOut(120)
                .activation(Activation.TANH)
                .build())
            
            // F6: 全连接层
            .layer(new DenseLayer.Builder()
                .nOut(84)
                .activation(Activation.TANH)
                .build())
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .build();
        
        return new MultiLayerNetwork(config);
    }
}
```

### LeNet的设计智慧

| 设计 | 意义 |
|------|------|
| 卷积+池化交替 | 逐步提取高级特征 |
| 逐层增加通道 | 学习更多特征 |
| Tanh激活 | 当时主流选择 |
| 参数约6万 | 在当时算力下可行 |

## AlexNet：深度学习的爆发

### 历史背景

2012年ImageNet竞赛，AlexNet将错误率从26%降到15%，震惊世界。

### 架构创新

```
输入: 227×227×3
    ↓
卷积层(11×11, stride=4) → 55×55×96
    ↓ 最大池化
卷积层(5×5) → 27×27×256
    ↓ 最大池化
卷积层(3×3) → 13×13×384
    ↓
卷积层(3×3) → 13×13×384
    ↓
卷积层(3×3) → 13×13×256
    ↓ 最大池化
全连接层(4096)
    ↓ Dropout
全连接层(4096)
    ↓ Dropout
输出层(1000)
```

### 关键创新

```java
/**
 * AlexNet的关键特性
 */
public class AlexNetFeatures {
    
    // 1. ReLU激活函数
    // 解决了Sigmoid的梯度消失问题
    .activation(Activation.RELU)
    
    // 2. Dropout正则化
    .layer(new DropoutLayer(0.5))
    
    // 3. 数据增强
    // 随机裁剪、水平翻转
    
    // 4. GPU并行训练
    // 将网络分成两部分，在两个GPU上训练
    
    // 5. 局部响应归一化（LRN）
    .layer(new LocalResponseNormalization.Builder().build())
}
```

### 用DL4J实现简化版AlexNet

```java
public class AlexNet {
    
    public static MultiLayerNetwork build(int numClasses) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            
            .list()
            // 第1卷积块
            .layer(new ConvolutionLayer.Builder(11, 11)
                .nIn(3)
                .nOut(96)
                .stride(4, 4)
                .activation(Activation.RELU)
                .build())
            .layer(new LocalResponseNormalization.Builder()
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(3, 3)
                .stride(2, 2)
                .build())
            
            // 第2卷积块
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nOut(256)
                .padding(2, 2)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(3, 3)
                .stride(2, 2)
                .build())
            
            // 第3-5卷积块
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(384)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(384)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(256)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(3, 3)
                .stride(2, 2)
                .build())
            
            // 全连接层
            .layer(new DenseLayer.Builder()
                .nOut(4096)
                .activation(Activation.RELU)
                .build())
            .layer(new DropoutLayer(0.5))
            .layer(new DenseLayer.Builder()
                .nOut(4096)
                .activation(Activation.RELU)
                .build())
            .layer(new DropoutLayer(0.5))
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutional(224, 224, 3))
            .build();
        
        return new MultiLayerNetwork(config);
    }
}
```

## ResNet：突破深度限制

### 深度网络的困境

网络越深，效果反而变差：

```
20层网络：训练准确率 90%
56层网络：训练准确率 85%（更差！）
```

这不是过拟合，是训练困难。

### 残差学习的思想

```
普通网络学习：H(x)
残差网络学习：F(x) = H(x) - x

即：H(x) = F(x) + x
```

**直觉：** 学习残差比学习完整映射更容易。

### 残差块

下面用简化示意代码说明残差块的核心思想。这里重点是“两层卷积 + 跳跃连接”，不是某个框架可以直接运行的完整实现。

```java
/**
 * 残差块实现
 */
public class ResidualBlock {
    
    private Conv2d conv1;
    private Conv2d conv2;
    
    /**
     * 前向传播
     */
    public Tensor forward(Tensor x) {
        Tensor identity = x;
        
        Tensor out = conv1.forward(x);
        out = batchNorm(out);
        out = relu(out);
        
        out = conv2.forward(out);
        out = batchNorm(out);
        
        // 残差连接：加上原始输入
        out = out.add(identity);
        
        return relu(out);
    }
}
```

### ResNet架构

```
输入 224×224×3
    ↓
卷积 7×7, 64, stride=2 → 112×112×64
    ↓ 池化
残差块 ×3 (64通道)  → 56×56×64
    ↓
残差块 ×4 (128通道) → 28×28×128
    ↓
残差块 ×6 (256通道) → 14×14×256
    ↓
残差块 ×3 (512通道) → 7×7×512
    ↓
全局平均池化 → 512
    ↓
全连接 → 类别数
```

### ResNet的变体

| 模型 | 层数 | 参数量 | Top-5准确率 |
|------|------|--------|-------------|
| ResNet-18 | 18 | 11.7M | 89.1% |
| ResNet-34 | 34 | 21.8M | 90.4% |
| ResNet-50 | 50 | 25.6M | 92.2% |
| ResNet-101 | 101 | 44.5M | 92.5% |
| ResNet-152 | 152 | 60.2M | 92.9% |

## 设计思考：架构演进背后的智慧

### 设计原则的演变

| 时代 | 设计理念 | 代表 |
|------|----------|------|
| LeNet | 卷积+池化交替 | 层次特征提取 |
| AlexNet | 更深+ReLU+Dropout | 训练技巧 |
| VGG | 小卷积核堆叠 | 简单即美 |
| GoogLeNet | 多尺度特征 | Inception模块 |
| ResNet | 跳跃连接 | 解决训练问题 |
| MobileNet | 深度可分离卷积 | 效率优先 |

### 核心设计模式

**1. 特征金字塔**

```
高分辨率低语义 → 低分辨率高语义
```

**2. 跳跃连接**

```
允许梯度直接流向浅层
```

**3. 批归一化**

```
稳定训练，加速收敛
```

**4. 瓶颈结构**

```
1×1卷积降维 → 3×3卷积 → 1×1卷积升维
减少参数，增加深度
```

## 实践：使用预训练模型

```java
package com.example.ai.chapter03;

import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 使用预训练模型
 */
public class PretrainedModelDemo {
    
    public static void main(String[] args) throws Exception {
        // 加载预训练的ResNet50
        ResNet50 resNet = ResNet50.builder()
            .numClasses(1000)
            .build();
        
        // 下载并加载预训练权重
        var model = resNet.initPretrained(PretrainedType.IMAGENET);
        
        System.out.println("模型加载完成");
        System.out.println("参数数量: " + model.numParams());
        
        // 使用模型进行预测
        INDArray image = loadImage("test.jpg");
        INDArray[] predictions = model.output(image);
        
        // 获取预测类别
        int predictedClass = predictions[0].argMax().getInt(0);
        System.out.println("预测类别: " + predictedClass);
    }
    
    private static INDArray loadImage(String path) {
        // 图像加载和预处理代码
        // ...
        return null;
    }
}
```

## 小结

本章我们学习了：

1. **LeNet**：CNN的开创性工作
2. **AlexNet**：深度学习爆发的起点
3. **ResNet**：残差连接突破深度限制

**架构演进的核心洞察：**

| 问题 | 解决方案 |
|------|----------|
| 梯度消失 | ReLU、残差连接 |
| 训练困难 | 批归一化 |
| 过拟合 | Dropout、数据增强 |
| 计算量大 | 瓶颈结构、深度可分离卷积 |

**下一步：** 我们将用Java构建实际的图像分类器。

---

**思考题：**

1. 为什么ResNet能训练更深的网络？
2. AlexNet的哪些创新至今仍在使用？
3. 如果你要设计一个图像分类网络，会参考哪个架构？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 3.3 池化与全连接](03-pooling-and-fully-connected.md)</span>

<span>[3.5 实战：用Java构建图像分类器 →](05-build-image-classifier.md)</span>

</div>
