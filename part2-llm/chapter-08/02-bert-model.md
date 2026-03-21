<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 8.1 GPT系列：生成式预训练模型](01-gpt-series.md)</span>

<span>[8.3 预训练与微调范式 &rarr;](03-pretraining-finetuning.md)</span>

</div>
---

# 8.2 BERT：双向编码器表示

> "BERT告诉我们：理解语言需要双向上下文——既要看前面，也要看后面。"

## BERT概述

### 什么是BERT

```
BERT = Bidirectional Encoder Representations from Transformers

核心思想：
- 仅使用Transformer的编码器部分
- 双向注意力（可以看到所有位置的词）
- 通过掩码语言模型预训练

特点：
- 深度双向表示
- 适合理解任务
- 需要微调
```

### BERT vs GPT

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | 仅编码器 | 仅解码器 |
| 注意力 | 双向 | 单向 |
| 预训练 | 掩码语言模型 | 自回归语言模型 |
| 优势 | 理解任务 | 生成任务 |
| 使用方式 | 微调 | 提示/微调 |

## BERT架构

### 编码器结构

```
BERT架构：

输入嵌入 + 位置编码 + 段落编码
    ↓
[编码器层] × N（Base:12, Large:24）
    - 多头自注意力（双向）
    - 前馈网络
    - 层归一化
    ↓
上下文表示

关键：真正的双向注意力
```

### 输入表示

```java
/**
 * BERT输入表示
 */
public class BERTInput {
    
    /**
     * BERT输入包含三部分：
     * 1. Token嵌入
     * 2. 位置嵌入
     * 3. 段落嵌入（用于句子对）
     */
    public INDArray createInput(String text) {
        // Tokenize
        String[] tokens = tokenize(text);
        
        // [CLS] token + tokens + [SEP] token
        String[] bertTokens = new String[tokens.length + 2];
        bertTokens[0] = "[CLS]";
        System.arraycopy(tokens, 0, bertTokens, 1, tokens.length);
        bertTokens[bertTokens.length - 1] = "[SEP]";
        
        // 转换为ID
        int[] inputIds = tokensToIds(bertTokens);
        
        // 创建三种嵌入
        INDArray tokenEmbeddings = embeddingLayer.forward(inputIds);
        INDArray positionEmbeddings = positionEmbedding(inputIds.length);
        INDArray segmentEmbeddings = segmentEmbedding(0, inputIds.length);
        
        // 相加
        return tokenEmbeddings.add(positionEmbeddings).add(segmentEmbeddings);
    }
}
```

## 预训练任务

### 掩码语言模型（MLM）

```
MLM任务：

输入：今天 [MASK] 气很 [MASK]
目标：天，好

掩码策略：
- 80% 概率替换为 [MASK]
- 10% 概率替换为随机词
- 10% 概率保持不变

为什么这样做？
- 防止模型只学习[MASK]的表示
- 让模型更好地理解上下文
```

### 下一句预测（NSP）

```
NSP任务：

输入两个句子，判断第二个是否是第一个的下一句

示例：
句子A：今天天气很好
句子B：我们出去散步吧
标签：IsNext

句子A：今天天气很好
句子B：机器学习很有趣
标签：NotNext

注意：后续研究发现NSP作用有限，RoBERTa等模型已移除
```

### Java实现

```java
/**
 * BERT预训练
 */
public class BERTPretraining {
    
    /**
     * 创建MLM训练样本
     */
    public MLMSample createMLMSample(String[] tokens, double maskProb) {
        int[] labels = new int[tokens.length];
        String[] maskedTokens = tokens.clone();
        
        for (int i = 0; i < tokens.length; i++) {
            // 跳过特殊token
            if (isSpecialToken(tokens[i])) continue;
            
            if (Math.random() < maskProb) {
                labels[i] = tokenToId(tokens[i]);
                
                double rand = Math.random();
                if (rand < 0.8) {
                    maskedTokens[i] = "[MASK]";
                } else if (rand < 0.9) {
                    maskedTokens[i] = getRandomToken();
                }
                // 10%概率保持不变
            } else {
                labels[i] = -1; // 不计算损失
            }
        }
        
        return new MLMSample(maskedTokens, labels);
    }
    
    /**
     * MLM损失计算
     */
    public double mlmLoss(INDArray logits, int[] labels) {
        double loss = 0;
        int count = 0;
        
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] >= 0) {
                // 只计算被掩码位置的损失
                INDArray tokenLogits = logits.getRow(i);
                loss += crossEntropy(tokenLogits, labels[i]);
                count++;
            }
        }
        
        return loss / count;
    }
}
```

## 微调（Fine-tuning）

### 分类任务

```java
/**
 * BERT文本分类
 */
public class BERTClassifier {
    
    private final BERTModel bert;
    private final Linear classifier;
    
    public BERTClassifier(BERTModel bert, int numClasses) {
        this.bert = bert;
        this.classifier = new Linear(bert.getHiddenSize(), numClasses);
    }
    
    /**
     * 前向传播
     */
    public INDArray forward(String text) {
        // BERT编码
        INDArray embeddings = bert.encode(text);
        
        // 取[CLS] token的表示
        INDArray clsEmbedding = embeddings.getRow(0);
        
        // 分类
        return classifier.forward(clsEmbedding);
    }
    
    /**
     * 微调训练
     */
    public void fineTune(Dataset dataset, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Sample sample : dataset) {
                // 前向
                INDArray logits = forward(sample.text);
                
                // 损失
                INDArray loss = crossEntropy(logits, sample.label);
                
                // 反向传播并更新
                backprop(loss);
            }
        }
    }
}
```

### 命名实体识别（NER）

```java
/**
 * BERT NER
 */
public class BERTNER {
    
    private final BERTModel bert;
    private final Linear tokenClassifier;
    
    public BERTNER(BERTModel bert, int numEntityTypes) {
        this.bert = bert;
        // 每个token预测一个实体标签
        this.tokenClassifier = new Linear(
            bert.getHiddenSize(), numEntityTypes);
    }
    
    /**
     * 预测每个token的实体标签
     */
    public String[] predict(String text) {
        INDArray embeddings = bert.encode(text);
        
        // 对每个token分类
        String[] labels = new String[(int)embeddings.size(0)];
        for (int i = 0; i < embeddings.size(0); i++) {
            INDArray tokenEmbedding = embeddings.getRow(i);
            INDArray logits = tokenClassifier.forward(tokenEmbedding);
            int labelId = argMax(logits);
            labels[i] = idToLabel(labelId);
        }
        
        return labels;
    }
}
```

## BERT变体

### 主要变体对比

| 模型 | 改进点 | 特点 |
|------|--------|------|
| RoBERTa | 优化训练 | 移除NSP，更大batch |
| ALBERT | 参数共享 | 减少参数量，更快训练 |
| DistilBERT | 知识蒸馏 | 轻量级，保持97%能力 |
| ELECTRA | 替换检测 | 更高效的预训练 |

### 选择合适的模型

```
选择建议：

追求效果：RoBERTa > BERT-Large
追求速度：DistilBERT > ALBERT
追求效率：ELECTRA训练更快
资源受限：DistilBERT或ALBERT
```

## 小结

本章我们学习了：

1. **BERT架构**：双向编码器，深度上下文理解
2. **预训练任务**：MLM和NSP
3. **微调方法**：分类、NER等任务
4. **BERT变体**：RoBERTa、DistilBERT等

**关键认识：**
BERT的双向表示使其在理解任务上表现出色，是NLP理解任务的基准模型。

**下一步：** 我们将学习预训练与微调。

---

**练习题：**

1. 为什么BERT使用双向注意力而GPT使用单向？
2. MLM的掩码策略为什么那样设计？
3. 什么时候选择BERT，什么时候选择GPT？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 8.1 GPT系列：生成式预训练模型](01-gpt-series.md)</span>

<span>[8.3 预训练与微调范式 &rarr;](03-pretraining-finetuning.md)</span>

</div>