# 8.1 GPT系列：生成式预训练模型

> "GPT证明了：给Transformer足够的数据和算力，它就能学会生成流畅的人类语言。"

## GPT概述

### 什么是GPT

```
GPT = Generative Pre-trained Transformer

核心思想：
- 仅使用Transformer的解码器部分
- 通过语言建模任务进行预训练
- 通过自回归方式生成文本

特点：
- 单向注意力（只能看前面的词）
- 适合生成任务
- 参数量巨大
```

### GPT发展历程

| 版本 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-1 | 2018 | 1.17亿 | 首次证明预训练有效性 |
| GPT-2 | 2019 | 15亿 | 零样本学习能力 |
| GPT-3 | 2020 | 1750亿 | 涌现能力，Few-shot学习 |
| GPT-4 | 2023 | 未公开 | 多模态，推理能力更强 |

## GPT架构详解

### 解码器结构

```
GPT架构：

输入嵌入 + 位置编码
    ↓
[解码器层] × N
    - 掩码多头自注意力
    - 前馈网络
    - 层归一化
    ↓
输出层（Softmax）

关键：使用掩码防止看到未来信息
```

### 掩码自注意力

```java
/**
 * GPT的掩码自注意力
 */
public class GPTMaskedAttention {
    
    /**
     * 创建上三角掩码
     */
    public INDArray createCausalMask(int seqLen) {
        // 上三角矩阵（包括对角线）
        INDArray mask = Nd4j.zeros(seqLen, seqLen);
        
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j <= i; j++) {
                mask.putScalar(i, j, 1.0);
            }
        }
        
        return mask;
    }
    
    /**
     * 应用掩码
     */
    public INDArray applyMask(INDArray scores, INDArray mask) {
        // 掩码为0的位置设为负无穷
        INDArray masked = scores.mul(mask)
            .add(mask.mul(-1).add(1).mul(-1e9));
        return masked;
    }
}
```

## 预训练任务

### 语言建模

```
目标：预测下一个词

输入："今天 天气 很"
目标："好"

损失函数：交叉熵
L = -Σ log P(x_t | x_<t)

训练数据：
- 海量互联网文本
- 书籍、文章、网页
- 代码（GPT-3以后）
```

### 训练过程

```java
/**
 * GPT训练伪代码
 */
public class GPTTraining {
    
    public void train(GPTModel model, Dataset dataset) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (Batch batch : dataset) {
                // 前向传播
                INDArray logits = model.forward(batch.input);
                
                // 计算损失
                INDArray loss = crossEntropy(logits, batch.target);
                
                // 反向传播
                INDArray gradients = computeGradients(loss);
                
                // 更新参数
                model.update(gradients);
            }
        }
    }
    
    /**
     * 交叉熵损失
     */
    private INDArray crossEntropy(INDArray logits, INDArray targets) {
        // 对每个位置计算交叉熵
        // logits: (batch, seq, vocab)
        // targets: (batch, seq)
        
        INDArray logProbs = softmax(logits).log();
        // 取目标词的负对数概率
        return negativeLogLikelihood(logProbs, targets);
    }
}
```

## 文本生成

### 自回归生成

```java
/**
 * GPT文本生成
 */
public class GPTTextGeneration {
    
    private final GPTModel model;
    private final Tokenizer tokenizer;
    
    /**
     * 生成文本
     */
    public String generate(String prompt, int maxLength) {
        // 编码输入
        int[] inputIds = tokenizer.encode(prompt);
        
        for (int i = 0; i < maxLength; i++) {
            // 前向传播
            INDArray logits = model.forward(inputIds);
            
            // 取最后一个位置的预测
            INDArray nextTokenLogits = logits.getRow(logits.rows() - 1);
            
            // 采样下一个词
            int nextToken = sample(nextTokenLogits);
            
            // 添加到序列
            inputIds = append(inputIds, nextToken);
            
            // 检查是否生成结束符
            if (nextToken == tokenizer.eosTokenId()) {
                break;
            }
        }
        
        return tokenizer.decode(inputIds);
    }
    
    /**
     * 采样策略：贪婪解码
     */
    private int greedySample(INDArray logits) {
        return Nd4j.argMax(logits, 0).getInt(0);
    }
    
    /**
     * 采样策略：温度采样
     */
    private int temperatureSample(INDArray logits, double temperature) {
        // 温度缩放
        INDArray scaled = logits.div(temperature);
        INDArray probs = softmax(scaled);
        
        // 按概率采样
        return multinomialSample(probs);
    }
}
```

### 采样策略对比

```
1. 贪婪解码
   - 总是选择概率最高的词
   - 确定性输出
   - 可能陷入重复

2. 温度采样
   - temperature > 1: 更随机，有创意
   - temperature < 1: 更确定，更保守
   - temperature = 0: 等价于贪婪

3. Top-k采样
   - 只从概率最高的k个词中采样
   - 避免选择概率极低的词

4. Top-p (Nucleus)采样
   - 从累积概率达到p的最小集合中采样
   - 动态调整候选词数量
```

## GPT-3的Few-shot学习

### 上下文学习

```
GPT-3的关键创新：

不需要微调，通过上下文示例学习

示例：

翻译任务：
英语: Hello
中文: 你好

英语: Thank you
中文: 谢谢

英语: Good morning
中文: 

模型通过前面的示例，学会翻译模式
```

### Java调用示例

```java
/**
 * 使用GPT-3 API
 */
public class GPT3Example {
    
    private final OpenAIClient client;
    
    public GPT3Example(String apiKey) {
        this.client = new OpenAIClient(apiKey);
    }
    
    /**
     * Few-shot分类
     */
    public String classify(String text) {
        String prompt = """
            判断以下评论的情感：
            
            评论: 这个产品太棒了！
            情感: 正面
            
            评论: 质量很差，浪费钱
            情感: 负面
            
            评论: 一般般，没什么特别的
            情感: 中性
            
            评论: %s
            情感:""".formatted(text);
        
        CompletionRequest request = CompletionRequest.builder()
            .model("text-davinci-003")
            .prompt(prompt)
            .maxTokens(10)
            .temperature(0.0)
            .build();
        
        return client.complete(request);
    }
}
```

## 小结

本章我们学习了：

1. **GPT架构**：仅解码器，掩码自注意力
2. **预训练**：语言建模，预测下一个词
3. **文本生成**：自回归，多种采样策略
4. **Few-shot学习**：通过上下文学习新任务

**关键认识：**
GPT展示了大规模预训练+自回归生成的强大能力，开启了生成式AI的新时代。

**下一步：** 我们将学习BERT模型。

---

**练习题：**

1. 为什么GPT使用掩码自注意力？
2. 温度参数如何影响生成结果？
3. Few-shot学习与微调有什么区别？
