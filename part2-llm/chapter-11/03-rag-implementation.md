# 11.3 RAG检索实现

> "检索是RAG的核心——找到最相关的信息，才能生成最准确的答案。"

## 检索流程

### 整体流程

```
检索流程：

用户查询
    ↓
[查询预处理] → 分词、扩展、纠错
    ↓
[查询向量化] → 生成Embedding
    ↓
[向量检索] → 相似度搜索
    ↓
[结果重排序] → 精排
    ↓
[上下文构建] → 格式化
    ↓
返回结果
```

## 查询预处理

### 查询理解

```java
package com.example.rag.retrieval;

/**
 * 查询预处理器
 */
public class QueryPreprocessor {
    
    private final QueryExpander expander;
    private final SpellChecker spellChecker;
    
    /**
     * 预处理查询
     */
    public ProcessedQuery preprocess(String rawQuery) {
        String query = rawQuery.trim();
        
        // 1. 拼写检查
        query = spellChecker.correct(query);
        
        // 2. 查询扩展
        List<String> expanded = expander.expand(query);
        
        // 3. 提取关键词
        List<String> keywords = extractKeywords(query);
        
        // 4. 意图识别
        QueryIntent intent = detectIntent(query);
        
        return new ProcessedQuery(query, expanded, keywords, intent);
    }
    
    /**
     * 查询扩展
     */
    public List<String> expandQuery(String query) {
        List<String> expansions = new ArrayList<>();
        expansions.add(query);
        
        // 同义词扩展
        expansions.addAll(getSynonyms(query));
        
        // 相关词扩展
        expansions.addAll(getRelatedTerms(query));
        
        return expansions;
    }
}
```

## 向量检索

### 相似度计算

```java
package com.example.rag.retrieval;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 相似度计算器
 */
public class SimilarityCalculator {
    
    /**
     * 余弦相似度
     */
    public double cosineSimilarity(INDArray a, INDArray b) {
        return Transforms.cosineSim(a, b);
    }
    
    /**
     * 欧氏距离
     */
    public double euclideanDistance(INDArray a, INDArray b) {
        return a.distance2(b);
    }
    
    /**
     * 点积相似度
     */
    public double dotProduct(INDArray a, INDArray b) {
        return a.mul(b).sumNumber().doubleValue();
    }
}
```

### 检索实现

```java
package com.example.rag.retrieval;

import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 向量检索服务
 */
public class VectorRetrievalService {
    
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingGenerator embeddingGenerator;
    private final SimilarityCalculator similarityCalculator;
    
    /**
     * 检索相关文档
     */
    public RetrievalResult retrieve(String query, RetrievalConfig config) {
        // 1. 生成查询向量
        Embedding queryEmbedding = embeddingGenerator.generate(query);
        
        // 2. 向量搜索
        List<EmbeddingMatch<TextSegment>> matches = embeddingStore
            .findRelevant(queryEmbedding, config.getTopK() * 2);
        
        // 3. 过滤低相似度结果
        List<ScoredDocument> filtered = matches.stream()
            .filter(m -> m.score() >= config.getMinScore())
            .map(m -> new ScoredDocument(
                m.embedded().text(),
                m.score(),
                m.embedded().metadata()
            ))
            .collect(Collectors.toList());
        
        // 4. 重排序
        List<ScoredDocument> reranked = rerank(query, filtered);
        
        // 5. 取TopK
        return new RetrievalResult(
            reranked.subList(0, Math.min(config.getTopK(), reranked.size()))
        );
    }
    
    /**
     * 混合检索：向量 + 关键词
     */
    public RetrievalResult hybridRetrieve(String query, 
                                          RetrievalConfig config) {
        // 向量检索
        RetrievalResult vectorResults = retrieve(query, config);
        
        // 关键词检索（BM25）
        List<ScoredDocument> keywordResults = 
            keywordSearch(query, config.getTopK());
        
        // 融合结果
        return fuseResults(vectorResults, keywordResults);
    }
}
```

## 重排序

### 重排序策略

```java
package com.example.rag.retrieval;

/**
 * 重排序器
 */
public class Reranker {
    
    private final CrossEncoder crossEncoder;
    
    /**
     * 交叉编码器重排序
     */
    public List<ScoredDocument> rerank(String query, 
                                        List<ScoredDocument> candidates) {
        // 使用交叉编码器计算查询-文档相关性
        List<ScoredDocument> reranked = candidates.stream()
            .map(doc -> {
                double score = crossEncoder.score(query, doc.getText());
                return new ScoredDocument(doc.getText(), score, doc.getMetadata());
            })
            .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
            .collect(Collectors.toList());
        
        return reranked;
    }
    
    /**
     * 多样性重排序（MMR）
     */
    public List<ScoredDocument> diversify(List<ScoredDocument> candidates,
                                          double lambda) {
        List<ScoredDocument> selected = new ArrayList<>();
        List<ScoredDocument> remaining = new ArrayList<>(candidates);
        
        while (!remaining.isEmpty() && selected.size() < 5) {
            ScoredDocument best = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            for (ScoredDocument doc : remaining) {
                double score = lambda * doc.getScore() - 
                    (1 - lambda) * maxSimilarity(doc, selected);
                
                if (score > bestScore) {
                    bestScore = score;
                    best = doc;
                }
            }
            
            if (best != null) {
                selected.add(best);
                remaining.remove(best);
            }
        }
        
        return selected;
    }
}
```

## 上下文构建

### 上下文组装

```java
package com.example.rag.retrieval;

/**
 * 上下文构建器
 */
public class ContextBuilder {
    
    /**
     * 构建上下文
     */
    public String buildContext(List<ScoredDocument> documents,
                               ContextConfig config) {
        StringBuilder context = new StringBuilder();
        
        for (int i = 0; i < documents.size(); i++) {
            ScoredDocument doc = documents.get(i);
            
            // 添加文档标记
            context.append(config.getDocumentPrefix())
                   .append(i + 1)
                   .append(config.getDocumentSuffix())
                   .append("\\n");
            
            // 添加文档内容
            context.append(doc.getText()).append("\\n\\n");
            
            // 添加来源信息（可选）
            if (config.isIncludeSource()) {
                context.append("来源: ")
                       .append(doc.getMetadata().getString("source"))
                       .append("\\n\\n");
            }
        }
        
        return context.toString().trim();
    }
    
    /**
     * 智能截断
     */
    public String truncateToFit(String context, int maxTokens) {
        // 估算token数（简化：1token ≈ 4字符）
        int estimatedTokens = context.length() / 4;
        
        if (estimatedTokens <= maxTokens) {
            return context;
        }
        
        // 按段落截断
        String[] paragraphs = context.split("\\n\\n");
        StringBuilder result = new StringBuilder();
        
        for (String para : paragraphs) {
            if ((result.length() + para.length()) / 4 > maxTokens) {
                break;
            }
            result.append(para).append("\\n\\n");
        }
        
        return result.toString().trim();
    }
}
```

## 检索优化

### 缓存策略

```java
package com.example.rag.retrieval;

/**
 * 检索缓存
 */
public class RetrievalCache {
    
    private final Cache<String, RetrievalResult> cache;
    
    public RetrievalCache() {
        this.cache = Caffeine.newBuilder()
            .maximumSize(1000)
            .expireAfterWrite(Duration.ofMinutes(10))
            .build();
    }
    
    /**
     * 获取缓存或检索
     */
    public RetrievalResult getOrRetrieve(String query,
                                         Function<String, RetrievalResult> retriever) {
        return cache.get(query, retriever);
    }
}
```

## 小结

本章我们学习了：

1. **检索流程**：从查询到结果的完整流程
2. **查询预处理**：扩展、纠错、意图识别
3. **向量检索**：相似度计算，TopK搜索
4. **重排序**：精排和多样性排序
5. **上下文构建**：格式化检索结果

**关键认识：**
检索质量决定RAG效果，需要多环节优化。

**下一步：** 我们将实现答案生成。

---

**练习题：**

1. 如何评估检索质量？
2. 混合检索如何平衡向量检索和关键词检索？
3. 设计一个支持多语言的检索系统。
