# 7.5 用Java实现简化版Transformer

> "纸上得来终觉浅，绝知此事要躬行——让我们动手实现一个简化版Transformer。"

## 实现目标

### 功能范围

```
我们将实现：
- 简化的Transformer编码器
- 支持前向传播
- 可配置的层数和维度
- 完整的Java实现

不实现：
- 完整的训练流程
- 反向传播
- 大规模模型优化
```

## 完整实现

### 项目结构

```
src/main/java/com/example/transformer/
├── Transformer.java          # 主类
├── Encoder.java              # 编码器
├── EncoderLayer.java         # 编码器层
├── MultiHeadAttention.java   # 多头注意力
├── FeedForward.java          # 前馈网络
├── LayerNorm.java            # 层归一化
└── PositionalEncoding.java   # 位置编码
```

### Maven依赖

```xml
<dependencies>
    <!-- ND4J for tensor operations -->
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
</dependencies>
```

### 核心实现

```java
package com.example.transformer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 简化版Transformer实现
 */
public class SimpleTransformer {
    
    private final int dModel;
    private final int numHeads;
    private final int numLayers;
    
    private final Encoder encoder;
    private final PositionalEncoding posEncoding;
    
    public SimpleTransformer(int numLayers, int dModel, int numHeads, 
                            int dFF, int maxLen) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        
        this.encoder = new Encoder(numLayers, dModel, numHeads, dFF);
        this.posEncoding = new PositionalEncoding(dModel, maxLen);
    }
    
    /**
     * 前向传播
     */
    public INDArray forward(INDArray input) {
        // input: (batch_size, seq_len, d_model)
        
        // 添加位置编码
        INDArray x = posEncoding.addPositionalEncoding(input);
        
        // 通过编码器
        x = encoder.forward(x);
        
        return x;
    }
    
    public static void main(String[] args) {
        // 测试
        int batchSize = 2;
        int seqLen = 10;
        int dModel = 512;
        
        // 创建随机输入
        INDArray input = Nd4j.rand(batchSize, seqLen, dModel);
        
        // 创建Transformer
        SimpleTransformer transformer = new SimpleTransformer(
            6,      // 6层
            512,    // 模型维度
            8,      // 8个头
            2048,   // 前馈网络维度
            100     // 最大长度
        );
        
        // 前向传播
        INDArray output = transformer.forward(input);
        
        System.out.println("输入形状: " + input.shapeInfoToString());
        System.out.println("输出形状: " + output.shapeInfoToString());
    }
}

/**
 * 编码器
 */
class Encoder {
    private final EncoderLayer[] layers;
    
    public Encoder(int numLayers, int dModel, int numHeads, int dFF) {
        layers = new EncoderLayer[numLayers];
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
 * 编码器层
 */
class EncoderLayer {
    private final MultiHeadAttention selfAttention;
    private final FeedForward feedForward;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    
    public EncoderLayer(int dModel, int numHeads, int dFF) {
        this.selfAttention = new MultiHeadAttention(dModel, numHeads);
        this.feedForward = new FeedForward(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
    }
    
    public INDArray forward(INDArray x) {
        // 自注意力 + 残差 + 归一化
        INDArray attn = selfAttention.forward(x, x, x);
        x = norm1.forward(x.add(attn));
        
        // 前馈 + 残差 + 归一化
        INDArray ff = feedForward.forward(x);
        x = norm2.forward(x.add(ff));
        
        return x;
    }
}

/**
 * 多头注意力
 */
class MultiHeadAttention {
    private final int dModel;
    private final int numHeads;
    private final int dHead;
    
    private final INDArray wQ;
    private final INDArray wK;
    private final INDArray wV;
    private final INDArray wO;
    
    public MultiHeadAttention(int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dHead = dModel / numHeads;
        
        // 初始化权重
        this.wQ = Nd4j.rand(dModel, dModel).mul(0.01);
        this.wK = Nd4j.rand(dModel, dModel).mul(0.01);
        this.wV = Nd4j.rand(dModel, dModel).mul(0.01);
        this.wO = Nd4j.rand(dModel, dModel).mul(0.01);
    }
    
    public INDArray forward(INDArray query, INDArray key, INDArray value) {
        int batchSize = (int) query.size(0);
        int seqLen = (int) query.size(1);
        
        // 线性投影
        INDArray q = query.reshape(batchSize * seqLen, dModel).mmul(wQ)
            .reshape(batchSize, seqLen, numHeads, dHead)
            .permute(0, 2, 1, 3);
        INDArray k = key.reshape(batchSize * seqLen, dModel).mmul(wK)
            .reshape(batchSize, seqLen, numHeads, dHead)
            .permute(0, 2, 1, 3);
        INDArray v = value.reshape(batchSize * seqLen, dModel).mmul(wV)
            .reshape(batchSize, seqLen, numHeads, dHead)
            .permute(0, 2, 1, 3);
        
        // 缩放点积注意力
        INDArray scores = q.matmul(k.permute(0, 1, 3, 2))
            .div(Math.sqrt(dHead));
        INDArray attnWeights = softmax(scores);
        INDArray context = attnWeights.matmul(v);
        
        // 合并头
        context = context.permute(0, 2, 1, 3)
            .reshape(batchSize, seqLen, dModel);
        
        // 最终投影
        return context.reshape(batchSize * seqLen, dModel)
            .mmul(wO)
            .reshape(batchSize, seqLen, dModel);
    }
    
    private INDArray softmax(INDArray x) {
        INDArray exp = Transforms.exp(x);
        INDArray sum = exp.sum(-1, true);
        return exp.div(sum);
    }
}

/**
 * 前馈网络
 */
class FeedForward {
    private final int dModel;
    private final int dFF;
    
    public FeedForward(int dModel, int dFF) {
        this.dModel = dModel;
        this.dFF = dFF;
    }
    
    public INDArray forward(INDArray x) {
        // 简化实现：两层线性变换 + ReLU
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);
        
        // 展平处理
        INDArray flat = x.reshape(batchSize * seqLen, dModel);
        
        // 第一层 (d_model -> d_ff)
        INDArray w1 = Nd4j.rand(dModel, dFF).mul(0.01);
        INDArray b1 = Nd4j.zeros(dFF);
        INDArray h = flat.mmul(w1).add(b1);
        h = Transforms.relu(h);
        
        // 第二层 (d_ff -> d_model)
        INDArray w2 = Nd4j.rand(dFF, dModel).mul(0.01);
        INDArray b2 = Nd4j.zeros(dModel);
        INDArray output = h.mmul(w2).add(b2);
        
        return output.reshape(batchSize, seqLen, dModel);
    }
}

/**
 * 层归一化
 */
class LayerNorm {
    private final int features;
    private final INDArray gamma;
    private final INDArray beta;
    
    public LayerNorm(int features) {
        this.features = features;
        this.gamma = Nd4j.ones(features);
        this.beta = Nd4j.zeros(features);
    }
    
    public INDArray forward(INDArray x) {
        // 计算均值和方差
        INDArray mean = x.mean(-1, true);
        INDArray var = x.var(-1, true);
        
        // 归一化
        INDArray normalized = x.sub(mean).div(Transforms.sqrt(var.add(1e-6)));
        
        // 缩放和平移
        return normalized.mul(gamma).add(beta);
    }
}

/**
 * 位置编码
 */
class PositionalEncoding {
    private final int dModel;
    private final int maxLen;
    private final INDArray encoding;
    
    public PositionalEncoding(int dModel, int maxLen) {
        this.dModel = dModel;
        this.maxLen = maxLen;
        this.encoding = createEncoding();
    }
    
    private INDArray createEncoding() {
        INDArray pe = Nd4j.zeros(maxLen, dModel);
        
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dModel; i += 2) {
                double angle = pos / Math.pow(10000, (2.0 * i) / dModel);
                pe.putScalar(pos, i, Math.sin(angle));
                if (i + 1 < dModel) {
                    pe.putScalar(pos, i + 1, Math.cos(angle));
                }
            }
        }
        
        return pe;
    }
    
    public INDArray addPositionalEncoding(INDArray input) {
        int seqLen = (int) input.size(1);
        INDArray pe = encoding.getRows(0, seqLen);
        return input.add(pe);
    }
}
```

## 测试与验证

### 单元测试

```java
package com.example.transformer;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Transformer测试
 */
public class TransformerTest {
    
    @Test
    public void testTransformerForward() {
        // 创建输入
        int batchSize = 2;
        int seqLen = 10;
        int dModel = 512;
        
        INDArray input = Nd4j.rand(batchSize, seqLen, dModel);
        
        // 创建Transformer
        SimpleTransformer transformer = new SimpleTransformer(
            2, 512, 8, 2048, 100
        );
        
        // 前向传播
        INDArray output = transformer.forward(input);
        
        // 验证输出形状
        assertEquals(batchSize, output.size(0));
        assertEquals(seqLen, output.size(1));
        assertEquals(dModel, output.size(2));
        
        System.out.println("测试通过！输出形状: " + output.shapeInfoToString());
    }
    
    @Test
    public void testMultiHeadAttention() {
        int batchSize = 2;
        int seqLen = 10;
        int dModel = 512;
        int numHeads = 8;
        
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads);
        
        INDArray x = Nd4j.rand(batchSize, seqLen, dModel);
        INDArray output = mha.forward(x, x, x);
        
        assertEquals(batchSize, output.size(0));
        assertEquals(seqLen, output.size(1));
        assertEquals(dModel, output.size(2));
    }
}
```

## 性能优化建议

```java
/**
 * 优化建议
 */
public class OptimizationTips {
    
    // 1. 使用GPU加速
    // ND4J支持CUDA，可以显著加速矩阵运算
    
    // 2. 批量处理
    // 尽量增大batch size，提高GPU利用率
    
    // 3. 内存管理
    // 及时释放不用的INDArray，避免内存泄漏
    
    // 4. 混合精度
    // 使用float16代替float32，减少内存占用
    
    // 5. 模型压缩
    // 使用量化技术，减少模型大小
}
```

## 小结

本章我们完成了：

1. **完整实现**：用Java实现了简化版Transformer
2. **组件实现**：多头注意力、前馈网络、层归一化、位置编码
3. **测试验证**：单元测试确保正确性
4. **优化建议**：性能优化方向

**关键认识：**
通过动手实现，我们更深入理解了Transformer的工作原理。

**下一步：** 我们将学习GPT和BERT两大模型。

---

**练习题：**

1. 运行代码，观察输出形状是否正确
2. 修改层数和头数，观察性能变化
3. 添加dropout正则化
