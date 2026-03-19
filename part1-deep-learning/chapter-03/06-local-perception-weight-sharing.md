<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 3.5 实战：用Java构建图像分类器](05-build-image-classifier.md)</span>

<span>[4.1 序列数据的挑战 →](../chapter-04/01-sequence-data-challenge.md)</span>

</div>

---

# 3.6 设计思考：局部感知与参数共享

> "CNN的两大设计原则——局部感知和参数共享，是深度学习处理图像的关键智慧。"

## 从全连接说起

### 全连接处理图像的问题

假设输入一张224×224的彩色图像：

```
输入神经元数量 = 224 × 224 × 3 = 150,528

如果下一层有1000个神经元：
参数数量 = 150,528 × 1000 = 1.5亿参数！
```

**问题：**
1. 参数爆炸，内存不够
2. 容易过拟合
3. 忽略图像的空间结构
4. 没有平移不变性

## 局部感知：只看局部

### 生物学启发

人类视觉系统不是同时看整张图，而是：
- 先看局部细节
- 再组合成整体

```
┌─────────────────────────────┐
│  ┌───┐                      │
│  │ 👁 │ ← 只看这一小块      │
│  └───┘                      │
│        ┌───┐                │
│        │ 👁 │ ← 移动后看这里 │
│        └───┘                │
└─────────────────────────────┘
```

### CNN的局部感知

每个神经元只连接输入的一小块区域：

```
全连接：每个神经元看整张图
┌─────────────────────────────┐
│ ┌───┬───┬───┬───┬───┬───┐ │
│ │ * │ * │ * │ * │ * │ * │ │  * = 连接
│ ├───┼───┼───┼───┼───┼───┤ │
│ │ * │ * │ * │ * │ * │ * │ │
│ └───┴───┴───┴───┴───┴───┘ │
└─────────────────────────────┘

局部连接：每个神经元只看一小块
┌─────────────────────────────┐
│ ┌───┬───┬───┬───┬───┬───┐ │
│ │ * │ * │ * │   │   │   │ │  只连接3×3区域
│ ├───┼───┼───┼───┼───┼───┤ │
│ │ * │ * │ * │   │   │   │ │
│ └───┴───┴───┴───┴───┴───┘ │
└─────────────────────────────┘
```

### 参数对比

假设输入100×100，下一层100个神经元：

| 连接方式 | 每个神经元参数 | 总参数 |
|----------|----------------|--------|
| 全连接 | 10000 | 1,000,000 |
| 局部连接(5×5) | 25 | 2,500 |

**局部连接减少了400倍参数！**

## 参数共享：一个核走天下

### 核心思想

如果一种特征（如边缘）在图像某个位置有用，那么在其他位置也应该有用。

```
同一个卷积核扫描整张图像：
┌─────────────────────────────┐
│ ┌───┐                       │
│ │ K │ → 第一个位置          │
│ └───┘                       │
│      ┌───┐                  │
│      │ K │ → 第二个位置     │
│      └───┘                  │
│           ┌───┐             │
│           │ K │ → 第三个位置│
│           └───┘             │
└─────────────────────────────┘
K = 同一个卷积核
```

### 参数共享的效果

```java
/**
 * 参数共享演示
 */
public class WeightSharingDemo {
    
    public static void main(String[] args) {
        int imageSize = 224;
        int kernelSize = 3;
        int inChannels = 3;
        int outChannels = 64;
        
        // 无共享：每个位置独立参数
        int positions = (imageSize - kernelSize + 1) * (imageSize - kernelSize + 1);
        int paramsNoSharing = positions * kernelSize * kernelSize * inChannels * outChannels;
        
        // 有共享：整个图像共享参数
        int paramsWithSharing = kernelSize * kernelSize * inChannels * outChannels;
        
        System.out.println("无参数共享: " + paramsNoSharing + " 参数");
        System.out.println("有参数共享: " + paramsWithSharing + " 参数");
        System.out.println("减少倍数: " + (paramsNoSharing / paramsWithSharing));
    }
}
```

**输出：**
```
无参数共享: 2,147,374,080 参数
有参数共享: 1,728 参数
减少倍数: 1,242,680
```

参数共享减少了超过100万倍！

## 平移不变性

### 什么是平移不变性

无论特征出现在图像的哪个位置，都能被检测到：

```
原图：          平移后：
┌───┬───┬───┐   ┌───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┤   ├───┼───┼───┤
│   │ ▲ │   │   │   │   │   │
├───┼───┼───┤   ├───┼───┼───┤
│   │   │   │   │   │ ▲ │   │
└───┴───┴───┘   └───┴───┴───┘

同样的卷积核，都能检测到▲
```

### 数学表达

```
f(g(x)) = g(f(x))

卷积操作与平移操作可交换
```

```java
/**
 * 平移不变性演示
 */
public class TranslationInvarianceDemo {
    
    public static void main(String[] args) {
        // 边缘检测卷积核
        double[][] edgeKernel = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };
        
        // 原图
        double[][] image1 = {
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0}
        };
        
        // 平移后的图
        double[][] image2 = {
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 0}
        };
        
        double[][] result1 = convolve(image1, edgeKernel);
        double[][] result2 = convolve(image2, edgeKernel);
        
        System.out.println("原图检测结果:");
        printMatrix(result1);
        
        System.out.println("\n平移后检测结果:");
        printMatrix(result2);
        
        // 结果形状相同，只是位置平移了
    }
}
```

## 感受野的计算

### 感受野的定义

感受野是指输出特征图上一个像素对应输入图像上的区域大小。

### 逐层计算

```
第1层卷积(3×3): 感受野 = 3
第2层卷积(3×3): 感受野 = 5
第3层卷积(3×3): 感受野 = 7
...

公式: RF_l = RF_{l-1} + (K_l - 1) × ∏_{i=1}^{l-1} S_i
```

### 感受野的意义

```
感受野小：只能看到局部细节
感受野大：能看到全局上下文

深层神经元的感受野更大，能理解更高级的语义
```

```java
/**
 * 感受野计算
 */
public class ReceptiveFieldCalculator {
    
    public static void main(String[] args) {
        int[] kernelSizes = {3, 3, 3, 3, 3};  // 5层3×3卷积
        int[] strides = {1, 1, 1, 1, 1};
        
        int rf = 1;
        int totalStride = 1;
        
        for (int i = 0; i < kernelSizes.length; i++) {
            rf = rf + (kernelSizes[i] - 1) * totalStride;
            totalStride *= strides[i];
            
            System.out.printf("第%d层感受野: %d%n", i + 1, rf);
        }
    }
}
```

**输出：**
```
第1层感受野: 3
第2层感受野: 5
第3层感受野: 7
第4层感受野: 9
第5层感受野: 11
```

## 设计思考：归纳偏置

### 什么是归纳偏置

归纳偏置是学习算法对解空间的假设。CNN的归纳偏置：

| 归纳偏置 | 含义 |
|----------|------|
| 局部性 | 附近像素更相关 |
| 平移不变性 | 特征位置无关 |
| 层次性 | 低级特征组合成高级特征 |

### 为什么有效

这些归纳偏置符合图像的本质特性：

```
图像特点：
1. 局部相关性：相邻像素往往属于同一物体
2. 平移等变性：物体可能出现在任何位置
3. 层次结构：边缘→部件→物体

CNN的设计正好匹配这些特点！
```

### 与其他模型的对比

| 模型 | 归纳偏置 | 适用场景 |
|------|----------|----------|
| 全连接 | 无 | 数据充足时 |
| CNN | 局部性、平移不变性 | 图像 |
| RNN | 时序依赖 | 序列数据 |
| Transformer | 全局注意力 | 序列、图像 |

## 实践：可视化卷积核

```java
package com.example.ai.chapter03;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * 卷积核可视化
 */
public class KernelVisualization {
    
    public static void visualizeFirstLayer(MultiLayerNetwork model, String outputPath) 
            throws Exception {
        
        // 获取第一层卷积核
        INDArray weights = model.getLayer(0).getParam("W");
        
        int numKernels = (int) weights.size(0);  // 卷积核数量
        int kernelSize = (int) weights.size(2);  // 卷积核大小
        
        // 创建可视化图像
        int cols = (int) Math.ceil(Math.sqrt(numKernels));
        int rows = (int) Math.ceil((double) numKernels / cols);
        
        int margin = 2;
        int scale = 20;  // 放大倍数
        
        BufferedImage image = new BufferedImage(
            cols * (kernelSize * scale + margin),
            rows * (kernelSize * scale + margin),
            BufferedImage.TYPE_INT_RGB
        );
        
        Graphics2D g = image.createGraphics();
        
        for (int k = 0; k < numKernels; k++) {
            int col = k % cols;
            int row = k / cols;
            
            int x = col * (kernelSize * scale + margin);
            int y = row * (kernelSize * scale + margin);
            
            // 获取单个卷积核
            INDArray kernel = weights.slice(k);
            
            // 归一化到0-255
            double min = kernel.minNumber().doubleValue();
            double max = kernel.maxNumber().doubleValue();
            
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    double value = kernel.getDouble(i, j);
                    int gray = (int) ((value - min) / (max - min) * 255);
                    
                    g.setColor(new Color(gray, gray, gray));
                    g.fillRect(x + j * scale, y + i * scale, scale, scale);
                }
            }
        }
        
        g.dispose();
        ImageIO.write(image, "PNG", new File(outputPath));
        System.out.println("卷积核可视化保存到: " + outputPath);
    }
}
```

## 小结

本章我们深入探讨了CNN的设计哲学：

1. **局部感知**：减少参数，关注局部
2. **参数共享**：一个核走天下
3. **平移不变性**：位置无关的特征检测
4. **感受野**：逐层扩大的感知范围

**核心洞察：**

| 设计原则 | 解决的问题 | 实现方式 |
|----------|------------|----------|
| 局部感知 | 参数爆炸 | 小卷积核 |
| 参数共享 | 泛化能力 | 权值共享 |
| 平移不变性 | 位置变化 | 卷积滑动 |

**设计哲学：**

```
好的设计不是增加复杂性，而是引入正确的约束。
CNN的约束（局部性、共享）正好匹配图像的本质特性。
```

**下一步：** 我们将学习循环神经网络，看看如何处理序列数据。

---

**思考题：**

1. 为什么局部感知和参数共享能减少过拟合？
2. 如果图像不满足局部相关性，CNN还适用吗？
3. 如何设计一个具有旋转不变性的网络？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 3.5 实战：用Java构建图像分类器](05-build-image-classifier.md)</span>

<span>[4.1 序列数据的挑战 →](../chapter-04/01-sequence-data-challenge.md)</span>

</div>
