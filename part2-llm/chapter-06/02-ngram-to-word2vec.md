# 6.2 从N-gram到Word2Vec：词向量的诞生

> "词向量是自然语言处理的基石——它让计算机第一次'理解'了词的含义。"

## N-gram模型：统计的智慧

### 基本思想

用前N-1个词预测第N个词：

```
Unigram (N=1): P(w1, w2, ...) = P(w1) × P(w2) × ...
Bigram (N=2): P(w2|w1)
Trigram (N=3): P(w3|w1, w2)
```

### Java实现

```java
package com.example.ai.chapter06;

import java.util.*;

/**
 * N-gram语言模型
 */
public class NGramModel {
    
    private int n;
    private Map<String, Integer> ngramCounts;
    private Map<String, Integer> contextCounts;
    private int totalTokens;
    private double smoothing = 0.1;  // 平滑参数
    
    public NGramModel(int n) {
        this.n = n;
        this.ngramCounts = new HashMap<>();
        this.contextCounts = new HashMap<>();
    }
    
    /**
     * 训练模型
     */
    public void train(List<String[]> corpus) {
        for (String[] sentence : corpus) {
            // 添加开始和结束标记
            String[] tokens = new String[sentence.length + 2];
            tokens[0] = "<s>";
            tokens[tokens.length - 1] = "</s>";
            System.arraycopy(sentence, 0, tokens, 1, sentence.length);
            
            // 统计N-gram
            for (int i = 0; i <= tokens.length - n; i++) {
                String ngram = String.join(" ", Arrays.copyOfRange(tokens, i, i + n));
                String context = String.join(" ", Arrays.copyOfRange(tokens, i, i + n - 1));
                
                ngramCounts.merge(ngram, 1, Integer::sum);
                contextCounts.merge(context, 1, Integer::sum);
                totalTokens++;
            }
        }
    }
    
    /**
     * 预测下一个词的概率
     */
    public double probability(String[] context, String nextWord) {
        String contextStr = String.join(" ", context);
        String ngram = contextStr + " " + nextWord;
        
        int ngramCount = ngramCounts.getOrDefault(ngram, 0);
        int contextCount = contextCounts.getOrDefault(contextStr, 0);
        
        // 加一平滑（Laplace Smoothing）
        double vocabSize = ngramCounts.size();
        return (ngramCount + smoothing) / (contextCount + smoothing * vocabSize);
    }
    
    /**
     * 预测最可能的下一个词
     */
    public List<Map.Entry<String, Double>> predictTop(String[] context, int topK) {
        Map<String, Double> probs = new HashMap<>();
        
        for (String ngram : ngramCounts.keySet()) {
            String[] parts = ngram.split(" ");
            if (parts.length == n) {
                String word = parts[n - 1];
                probs.put(word, probability(context, word));
            }
        }
        
        return probs.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .collect(Collectors.toList());
    }
    
    /**
     * 计算句子概率
     */
    public double sentenceProbability(String[] sentence) {
        double prob = 1.0;
        
        for (int i = n - 1; i < sentence.length; i++) {
            String[] context = Arrays.copyOfRange(sentence, i - n + 1, i);
            prob *= probability(context, sentence[i]);
        }
        
        return prob;
    }
    
    public static void main(String[] args) {
        // 训练语料
        List<String[]> corpus = Arrays.asList(
            "我 爱 编程".split(" "),
            "我 爱 学习".split(" "),
            "我 喜欢 编程".split(" "),
            "他 爱 编程".split(" ")
        );
        
        // 训练Bigram模型
        NGramModel model = new NGramModel(2);
        model.train(corpus);
        
        // 预测
        String[] context = {"我"};
        System.out.println("给定 '" + context[0] + "'，预测下一个词:");
        
        for (var entry : model.predictTop(context, 5)) {
            System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
        }
    }
}
```

### N-gram的问题

| 问题 | 说明 |
|------|------|
| 稀疏性 | 大部分N-gram组合从未出现 |
| 存储爆炸 | N增大时，组合数量爆炸 |
| 无法泛化 | "猫吃鱼"和"狗吃肉"无法共享知识 |
| 维数灾难 | 词表大小V，N-gram数量为V^N |

## 词向量：分布式表示的革命

### 从离散到连续

```
传统表示（One-hot）：
"猫" → [0, 0, 1, 0, 0, ...]  (维度=词表大小)
"狗" → [0, 1, 0, 0, 0, ...]

问题：
1. 维度极高
2. 所有词正交，无法表达相似性
3. 无法泛化
```

```
词向量表示：
"猫" → [0.2, -0.1, 0.5, ...]  (维度=100-300)
"狗" → [0.3, -0.2, 0.4, ...]

优势：
1. 维度低
2. 相似词向量相近
3. 可以泛化
```

### 分布式假设

```
"你可以通过一个词的上下文来理解它的含义"
—— John Rupert Firth (1957)

核心思想：出现在相似上下文中的词，具有相似的含义
```

## Word2Vec：词向量的里程碑

### 两种架构

```
CBOW (Continuous Bag of Words):
    上下文 → 目标词
    "我 爱 ___" → "编程"

Skip-gram:
    目标词 → 上下文
    "编程" → "我", "爱"
```

### Skip-gram详解

```
输入: 中心词 "编程"
输出: 上下文词 ["我", "爱", "学习", "Java"]

训练目标: 最大化 P(上下文|中心词)
```

### Java实现简化版Word2Vec

```java
package com.example.ai.chapter06;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * 简化版Word2Vec实现
 */
public class SimpleWord2Vec {
    
    private int embeddingSize;
    private Map<String, Integer> wordToIndex;
    private INDArray embeddings;  // 词嵌入矩阵
    
    public SimpleWord2Vec(int vocabSize, int embeddingSize) {
        this.embeddingSize = embeddingSize;
        this.wordToIndex = new HashMap<>();
        this.embeddings = Nd4j.randn(vocabSize, embeddingSize).mul(0.1);
    }
    
    /**
     * 获取词向量
     */
    public INDArray getVector(String word) {
        Integer index = wordToIndex.get(word);
        if (index == null) {
            return null;
        }
        return embeddings.getRow(index);
    }
    
    /**
     * 计算词相似度（余弦相似度）
     */
    public double similarity(String word1, String word2) {
        INDArray v1 = getVector(word1);
        INDArray v2 = getVector(word2);
        
        if (v1 == null || v2 == null) {
            return 0;
        }
        
        return Transforms.cosineSim(v1, v2);
    }
    
    /**
     * 找最相似的词
     */
    public List<Map.Entry<String, Double>> mostSimilar(String word, int topK) {
        INDArray target = getVector(word);
        if (target == null) {
            return Collections.emptyList();
        }
        
        Map<String, Double> similarities = new HashMap<>();
        
        for (Map.Entry<String, Integer> entry : wordToIndex.entrySet()) {
            if (!entry.getKey().equals(word)) {
                INDArray vec = embeddings.getRow(entry.getValue());
                double sim = Transforms.cosineSim(target, vec);
                similarities.put(entry.getKey(), sim);
            }
        }
        
        return similarities.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .collect(Collectors.toList());
    }
    
    /**
     * 词类比：king - man + woman = queen
     */
    public String analogy(String word1, String word2, String word3, int topK) {
        INDArray v1 = getVector(word1);
        INDArray v2 = getVector(word2);
        INDArray v3 = getVector(word3);
        
        if (v1 == null || v2 == null || v3 == null) {
            return null;
        }
        
        // target = v1 - v2 + v3
        INDArray target = v1.sub(v2).add(v3);
        
        // 找最相似的词（排除输入词）
        Set<String> exclude = Set.of(word1, word2, word3);
        
        Map<String, Double> similarities = new HashMap<>();
        for (Map.Entry<String, Integer> entry : wordToIndex.entrySet()) {
            if (!exclude.contains(entry.getKey())) {
                INDArray vec = embeddings.getRow(entry.getValue());
                double sim = Transforms.cosineSim(target, vec);
                similarities.put(entry.getKey(), sim);
            }
        }
        
        return similarities.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .findFirst()
            .map(Map.Entry::getKey)
            .orElse(null);
    }
    
    public static void main(String[] args) {
        SimpleWord2Vec w2v = new SimpleWord2Vec(10000, 100);
        
        // 添加词汇（实际使用时通过训练得到）
        w2v.wordToIndex.put("编程", 0);
        w2v.wordToIndex.put("代码", 1);
        w2v.wordToIndex.put("猫", 2);
        w2v.wordToIndex.put("狗", 3);
        
        // 相似度测试
        System.out.println("相似度测试:");
        System.out.println("编程 vs 代码: " + w2v.similarity("编程", "代码"));
        System.out.println("猫 vs 狗: " + w2v.similarity("猫", "狗"));
    }
}
```

### 词向量的神奇特性

```java
// 词类比
"国王" - "男人" + "女人" ≈ "女王"
"北京" - "中国" + "日本" ≈ "东京"
"编程" - "程序员" + "画家" ≈ "绘画"

// 语义聚类
相似的词在向量空间中聚集：
[猫, 狗, 兔子, ...] → 动物区域
[编程, 代码, 程序, ...] → 计算机区域
```

## 使用预训练词向量

### 在Java中加载GloVe/Word2Vec

```java
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;

/**
 * 使用预训练词向量
 */
public class PretrainedVectors {
    
    public static void main(String[] args) throws Exception {
        // 加载Google News词向量
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(
            new File("GoogleNews-vectors-negative300.bin")
        );
        
        // 获取词向量
        double[] vector = word2Vec.getWordVector("programming");
        System.out.println("'programming' 向量维度: " + vector.length);
        
        // 找相似词
        Collection<String> similar = word2Vec.wordsNearest("programming", 10);
        System.out.println("与 'programming' 最相似的词:");
        similar.forEach(System.out::println);
        
        // 词类比
        Collection<String> analogy = word2Vec.wordsNearest(
            Arrays.asList("king", "woman"),
            Arrays.asList("man"),
            5
        );
        System.out.println("king - man + woman = " + analogy);
    }
}
```

## 词向量的局限

### 1. 静态表示

```
问题：同一个词在不同上下文中向量相同

"苹果"：
- "我吃了一个苹果" → 水果含义
- "苹果发布了新手机" → 公司含义

Word2Vec无法区分这两种含义
```

### 2. OOV问题

```
问题：词表中没有的词无法获得向量

新词、专有名词、拼写错误等
```

### 3. 无法处理短语

```
问题："人工智能"不是"人工"+"智能"的简单相加

短语的整体含义无法从词向量推导
```

## 设计思考：从离散到连续的意义

### 表示学习的本质

```
传统方法：人工设计特征
词向量：自动学习表示

这是深度学习"端到端"思想的体现
```

### 几何视角

```
词向量将词映射到高维空间：
- 相似的词距离近
- 关系可以用向量运算表达
- 语言结构在空间中有几何对应

这是"意义的几何化"
```

## 小结

本章我们学习了：

1. **N-gram模型**：统计语言模型
2. **词向量**：分布式表示
3. **Word2Vec**：Skip-gram和CBOW
4. **词向量特性**：相似度、类比

**核心概念：**

| 概念 | 说明 |
|------|------|
| N-gram | 连续N个词的组合 |
| 词向量 | 词的分布式表示 |
| 分布式假设 | 上下文相似的词含义相似 |
| 词类比 | 向量运算表达语义关系 |

**下一步：** 我们将学习Transformer架构——注意力机制的革命。

---

**思考题：**

1. 为什么词向量能捕捉语义关系？
2. Word2Vec的两种架构各有什么优缺点？
3. 如何解决词向量的静态表示问题？
