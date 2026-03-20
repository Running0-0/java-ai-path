# 9.2 模型量化：让大模型变小

> "量化是模型工程的魔法——用少量精度损失换取巨大的效率提升。"

## 量化原理

### 什么是量化

```
量化 = 将高精度浮点数转换为低精度整数

示例：
FP32: 3.1415926535
INT8: 201  (映射到0-255范围)

为什么有效？
- 神经网络权重分布集中
- 低精度足以表示大部分信息
- 现代CPU/GPU对整数运算优化更好
```

### 量化类型

```
1. 训练后量化（PTQ）
   - 模型训练完成后量化
   - 简单快速，无需重新训练
   - 精度损失相对较大

2. 量化感知训练（QAT）
   - 训练时模拟量化
   - 精度损失小
   - 需要重新训练

3. 动态量化
   - 运行时动态确定量化参数
   - 灵活但速度较慢
```

## 量化方法详解

### 线性量化

```
最简单的量化方法：

r = S * (q - Z)

其中：
- r: 原始浮点数
- q: 量化后的整数
- S: 缩放因子（scale）
- Z: 零点（zero point）

计算过程：
1. 确定最小值min和最大值max
2. S = (max - min) / (2^n - 1)
3. Z = round(-min / S)
4. q = round(r / S) + Z
```

### Java实现

```java
package com.example.quantization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 线性量化实现
 */
public class LinearQuantization {
    
    private final int bits;
    private final int qmin;
    private final int qmax;
    
    public LinearQuantization(int bits) {
        this.bits = bits;
        this.qmin = -(1 << (bits - 1));
        this.qmax = (1 << (bits - 1)) - 1;
    }
    
    /**
     * 计算量化参数
     */
    public QuantizationParams computeParams(INDArray tensor) {
        float min = tensor.minNumber().floatValue();
        float max = tensor.maxNumber().floatValue();
        
        float scale = (max - min) / (qmax - qmin);
        int zeroPoint = Math.round(qmin - min / scale);
        
        return new QuantizationParams(scale, zeroPoint, min, max);
    }
    
    /**
     * 量化
     */
    public INDArray quantize(INDArray tensor, QuantizationParams params) {
        INDArray quantized = tensor.div(params.scale)
            .add(params.zeroPoint)
            .castTo(org.nd4j.linalg.api.buffer.DataType.INT8);
        
        // 裁剪到有效范围
        return Nd4j.clip(quantized, qmin, qmax);
    }
    
    /**
     * 反量化
     */
    public INDArray dequantize(INDArray quantized, QuantizationParams params) {
        return quantized.sub(params.zeroPoint)
            .mul(params.scale)
            .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
    }
}

class QuantizationParams {
    final float scale;
    final int zeroPoint;
    final float min;
    final float max;
    
    QuantizationParams(float scale, int zeroPoint, float min, float max) {
        this.scale = scale;
        this.zeroPoint = zeroPoint;
        this.min = min;
        this.max = max;
    }
}
```

## GGUF/GGML格式

### 简介

```
GGML/GGUF：llama.cpp使用的量化格式

特点：
- 支持多种量化策略
- 针对CPU推理优化
- 文件格式紧凑

量化类型：
- Q4_0: 4位，每块32个权重
- Q4_1: 4位，带最小值/最大值
- Q5_0, Q5_1: 5位
- Q8_0: 8位
- F16: 半精度
```

### 使用llama.cpp

```bash
# 克隆仓库
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 编译
make

# 转换模型为GGUF格式
python convert.py models/7B/

# 量化
./quantize models/7B/ggml-model-f16.bin \
    models/7B/ggml-model-q4_0.bin q4_0

# 运行
./main -m models/7B/ggml-model-q4_0.bin \
    -p "Hello, my name is"
```

## 量化实践

### 选择量化策略

```java
/**
 * 量化策略选择
 */
public class QuantizationStrategy {
    
    /**
     * 根据需求选择量化方案
     */
    public String selectStrategy(Requirements req) {
        if (req.getPriority() == Priority.QUALITY) {
            // 优先质量
            if (req.getGpuMemory() >= 16) {
                return "int8";  // 8位，质量接近原模型
            } else {
                return "q5_1";  // 5位，平衡方案
            }
        } else {
            // 优先速度/内存
            if (req.getGpuMemory() >= 8) {
                return "q4_1";  // 4位，推荐方案
            } else {
                return "q4_0";  // 最小内存
            }
        }
    }
}
```

### 量化效果评估

```java
/**
 * 量化质量评估
 */
public class QuantizationEvaluator {
    
    /**
     * 计算量化误差
     */
    public double computeError(INDArray original, INDArray quantized) {
        INDArray dequantized = dequantize(quantized);
        INDArray diff = original.sub(dequantized);
        
        // 均方误差
        double mse = diff.mul(diff).meanNumber().doubleValue();
        
        // 信噪比
        double signalPower = original.mul(original).meanNumber().doubleValue();
        double snr = 10 * Math.log10(signalPower / mse);
        
        return snr;
    }
    
    /**
     * 困惑度评估
     */
    public double evaluatePerplexity(Model model, Dataset testSet) {
        double totalLoss = 0;
        int totalTokens = 0;
        
        for (Sample sample : testSet) {
            INDArray logits = model.forward(sample.input);
            double loss = computeCrossEntropy(logits, sample.target);
            totalLoss += loss;
            totalTokens += sample.length;
        }
        
        // 困惑度 = exp(平均损失)
        return Math.exp(totalLoss / totalTokens);
    }
}
```

## 小结

本章我们学习了：

1. **量化原理**：将浮点数转换为整数
2. **量化方法**：PTQ、QAT、动态量化
3. **GGUF格式**：llama.cpp的量化方案
4. **实践策略**：如何选择合适的量化级别

**关键认识：**
量化是大模型部署的关键技术，让大模型能在资源受限的环境中运行。

**下一步：** 我们将学习LangChain4j框架。

---

**练习题：**

1. 量化为什么能减少模型大小？
2. PTQ和QAT各有什么优缺点？
3. 在你的机器上尝试量化一个模型。
