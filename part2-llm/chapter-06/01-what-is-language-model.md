<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 5.5 部署与集成](../../part1-deep-learning/chapter-05/05-deployment-integration.md)</span>

<span>[6.2 从N-gram到Word2Vec →](02-ngram-to-word2vec.md)</span>

</div>

---

# 6.1 语言模型是什么：让机器理解语言

> "语言是人类智慧的结晶，语言模型是机器理解人类语言的钥匙。"

## 从一个简单问题开始

当你读到"今天天气很好，适合出门___"时，你会自然地想到"散步"、"运动"、"郊游"等词。

这种"预测下一个词"的能力，正是语言模型的核心。

## 语言模型的定义

### 数学定义

语言模型计算一个句子出现的概率：

```
P(w1, w2, w3, ..., wn)

或者，预测下一个词的概率：
P(wt | w1, w2, ..., wt-1)

即：给定前面的词，预测下一个词的概率
```

### 直观理解

语言模型就像一个"语言预言家"：

```java
/**
 * 语言模型的抽象接口
 */
public interface LanguageModel {
    
    /**
     * 预测下一个词的概率分布
     * @param context 前文 ["今天", "天气", "很好"]
     * @return 下一个词的概率分布 {"散步": 0.3, "运动": 0.2, ...}
     */
    Map<String, Double> predictNext(String[] context);
    
    /**
     * 计算句子的概率
     * @param sentence 句子
     * @return 句子出现的概率
     */
    double sentenceProbability(String sentence);
}
```

## 语言模型的应用

### 核心应用场景

| 应用 | 描述 | 示例 |
|------|------|------|
| 文本生成 | 生成连贯的文本 | GPT写文章 |
| 机器翻译 | 翻译自然语言 | Google翻译 |
| 语音识别 | 语音转文字 | Siri |
| 拼写纠错 | 纠正拼写错误 | "teh" → "the" |
| 智能输入法 | 预测下一个词 | 手机输入法 |

### 实际例子

```java
// 智能输入法的预测
输入: "我想去"
预测: {"北京": 0.3, "上海": 0.2, "旅游": 0.15, "吃饭": 0.1}

// 机器翻译的解码
源语言: "I love programming"
候选翻译: 
  "我爱编程" → P = 0.85
  "我爱程序" → P = 0.10
  "我喜欢编程" → P = 0.05
```

## 语言模型的发展历程

### 时间线

```
1940s: 信息论基础（香农）
    ↓
1980s: N-gram模型
    ↓
2000s: 神经网络语言模型
    ↓
2013: Word2Vec（词向量）
    ↓
2017: Transformer
    ↓
2018: BERT、GPT
    ↓
2020+: 大语言模型（GPT-3/4、Claude等）
```

### 性能演进

| 时代 | 模型 | 特点 | 困惑度(PPL) |
|------|------|------|-------------|
| 1980s | N-gram | 简单、稀疏 | ~200 |
| 2000s | NNLM | 分布式表示 | ~150 |
| 2013 | Word2Vec | 词向量 | ~100 |
| 2017 | Transformer | 注意力机制 | ~25 |
| 2020 | GPT-3 | 大规模预训练 | ~20 |

## 语言模型的评估

### 困惑度（Perplexity）

困惑度是评估语言模型的核心指标：

```
PPL = exp(-1/N × Σlog P(wi|w1,...,wi-1))

困惑度越低，模型越好
```

```java
/**
 * 困惑度计算
 */
public class PerplexityCalculator {
    
    public static double calculate(LanguageModel model, List<String[]> sentences) {
        double totalLogProb = 0;
        long totalWords = 0;
        
        for (String[] sentence : sentences) {
            for (int i = 1; i < sentence.length; i++) {
                String[] context = Arrays.copyOfRange(sentence, 0, i);
                String nextWord = sentence[i];
                
                Map<String, Double> probs = model.predictNext(context);
                Double prob = probs.getOrDefault(nextWord, 1e-10);
                
                totalLogProb += Math.log(prob);
                totalWords++;
            }
        }
        
        return Math.exp(-totalLogProb / totalWords);
    }
}
```

### 困惑度的直观理解

```
困惑度 = 模型在预测时的"困惑程度"

PPL = 1: 完美预测
PPL = 10: 每次在10个候选词中选
PPL = 100: 每次在100个候选词中选
PPL = ∞: 完全随机
```

## 语言模型的核心挑战

### 1. 稀疏性问题

```
问题：语言组合是无限的
"I love ___" 后面可以接任何词

大多数词组合在训练数据中从未出现
→ 如何预测未见过的组合？
```

### 2. 长距离依赖

```
问题：句子开头的词可能影响结尾

"小明出生在1990年，......（很长一段话）......他今年___岁了"

要预测"34"，需要记住"1990年"
```

### 3. 上下文理解

```
问题：同一个词在不同上下文中含义不同

"苹果"：
- "我吃了一个苹果" → 水果
- "苹果发布了新手机" → 公司
- "苹果股价下跌" → 股票
```

## 语言模型的本质

### 从概率到理解

```
早期观点：语言模型 = 统计概率
现代观点：语言模型 = 语言理解

一个好的语言模型需要：
1. 语法知识
2. 语义理解
3. 世界知识
4. 推理能力
```

### 涌现能力

大语言模型展现出了"涌现"能力：

```
小模型：只能做简单的词预测
大模型：涌现出推理、创作、编程等能力

这种现象被称为"涌现"（Emergence）
```

## 设计思考：语言模型的意义

### 语言即思维

```
维特根斯坦：语言的边界就是世界的边界

语言模型学习语言，实际上是在学习：
- 人类的知识
- 思维的方式
- 世界的规律
```

### 从工具到伙伴

```
传统软件：工具（执行指令）
语言模型：伙伴（理解意图）

这是AI范式的根本转变
```

## 小结

本章我们学习了：

1. **语言模型的定义**：预测下一个词的概率
2. **应用场景**：生成、翻译、识别等
3. **发展历程**：从N-gram到大语言模型
4. **评估指标**：困惑度
5. **核心挑战**：稀疏性、长距离依赖、上下文理解

**核心概念：**

| 概念 | 说明 |
|------|------|
| 语言模型 | 预测词序列概率 |
| 困惑度 | 模型评估指标 |
| 长距离依赖 | 远距离词之间的关系 |
| 涌现能力 | 大模型展现的新能力 |

**下一步：** 我们将学习N-gram模型和Word2Vec，了解语言模型的早期发展。

---

**思考题：**

1. 为什么预测下一个词的能力能体现语言理解？
2. 困惑度低是否一定意味着模型更好？
3. 大语言模型的涌现能力从何而来？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 5.5 部署与集成](../../part1-deep-learning/chapter-05/05-deployment-integration.md)</span>

<span>[6.2 从N-gram到Word2Vec →](02-ngram-to-word2vec.md)</span>

</div>
