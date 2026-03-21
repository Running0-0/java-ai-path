<table width="100%">
   <tr>
      <td align="left"><a href="04-conversation-system.md">&larr; 11.4 对话系统实现</a></td>
      <td align="right"><a href="../../part3-agents/chapter-12/01-what-is-agent.md">12.1 智能体概念与架构 &rarr;</a></td>
   </tr>
</table>
---

# 11.5 部署与优化

> "从开发到生产——性能、稳定性、成本的全面优化。"

## 性能优化

### 检索优化

```java
/**
 * 检索性能优化
 */
public class RetrievalOptimization {
    
    // 1. 索引优化
    public void optimizeIndex() {
        // 使用HNSW索引（近似最近邻）
        // 比暴力搜索快100倍，精度损失<1%
    }
    
    // 2. 缓存策略
    private final Cache<String, RetrievalResult> queryCache = 
        Caffeine.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(Duration.ofMinutes(30))
            .build();
    
    // 3. 批量处理
    public List<RetrievalResult> batchRetrieve(List<String> queries) {
        // 批量生成嵌入
        List<Embedding> embeddings = embeddingGenerator
            .generateBatch(queries);
        
        // 批量检索
        return embeddingStore.searchBatch(embeddings);
    }
    
    // 4. 异步检索
    public CompletableFuture<RetrievalResult> retrieveAsync(String query) {
        return CompletableFuture.supplyAsync(
            () -> retrieve(query),
            executorService
        );
    }
}
```

### 生成优化

```java
/**
 * 生成性能优化
 */
public class GenerationOptimization {
    
    // 1. 模型量化
    public void useQuantizedModel() {
        // INT8量化：速度提升2-4倍
        // INT4量化：速度提升4-8倍
    }
    
    // 2. 批处理生成
    public List<String> batchGenerate(List<String> prompts) {
        // 合并多个请求，提高GPU利用率
        return model.generateBatch(prompts);
    }
    
    // 3. 流式生成
    public Stream<String> streamGenerate(String prompt) {
        // 首token延迟降低，用户体验更好
        return model.generateStream(prompt);
    }
    
    // 4. 投机解码
    public String speculativeDecode(String prompt) {
        // 小模型预测，大模型验证
        // 速度提升2-3倍
    }
}
```

## 系统架构优化

### 分层缓存

```
缓存架构：

L1: 本地缓存（Caffeine）
    - 查询结果缓存
    - 嵌入向量缓存
    - 延迟：<1ms
    
L2: 分布式缓存（Redis）
    - 跨实例共享
    - 会话状态
    - 延迟：<5ms
    
L3: 向量数据库（Chroma）
    - 文档向量
    - 持久化存储
    - 延迟：<50ms
```

### 异步处理

```java
/**
 * 异步文档处理
 */
@Service
public class AsyncDocumentProcessor {
    
    @Async("taskExecutor")
    public CompletableFuture<Void> processDocumentAsync(
            DocumentSource source) {
        return CompletableFuture.runAsync(() -> {
            // 1. 解析
            ParsedDocument doc = parser.parse(source);
            
            // 2. 分割
            List<Chunk> chunks = splitter.split(doc);
            
            // 3. 嵌入（批量）
            List<Embedding> embeddings = embeddingGenerator
                .generateBatch(chunks.stream()
                    .map(Chunk::getText)
                    .collect(Collectors.toList()));
            
            // 4. 存储
            vectorStorage.storeBatch(chunks, embeddings);
            
            // 5. 通知
            eventPublisher.publishEvent(
                new DocumentProcessedEvent(source.getId()));
        });
    }
}
```

## 成本控制

### 混合部署策略

```java
/**
 * 智能路由（成本优化）
 */
public class CostOptimizedRouter {
    
    private final ChatLanguageModel localModel;
    private final ChatLanguageModel cloudModel;
    
    /**
     * 智能选择模型
     */
    public String generate(String query, TaskComplexity complexity) {
        switch (complexity) {
            case SIMPLE:
                // 简单任务用本地模型（免费）
                return localModel.generate(query);
                
            case MEDIUM:
                // 中等任务，先尝试本地
                String localResult = localModel.generate(query);
                if (qualityChecker.isGoodEnough(localResult)) {
                    return localResult;
                }
                // 本地效果不好，降级到云端
                return cloudModel.generate(query);
                
            case COMPLEX:
                // 复杂任务直接用云端
                return cloudModel.generate(query);
                
            default:
                return cloudModel.generate(query);
        }
    }
}
```

### Token优化

```java
/**
 * Token使用优化
 */
public class TokenOptimizer {
    
    // 1. 提示压缩
    public String compressPrompt(String prompt) {
        // 移除冗余空格
        // 简化表达
        // 保留关键信息
        return promptCompressor.compress(prompt);
    }
    
    // 2. 上下文截断
    public String truncateContext(String context, int maxTokens) {
        // 智能截断，保留最重要信息
        return contextTruncator.truncate(context, maxTokens);
    }
    
    // 3. 缓存相似查询
    public String getCachedOrGenerate(String query) {
        // 计算查询相似度
        String similarQuery = findSimilarQuery(query);
        if (similarQuery != null) {
            return adaptAnswer(cachedAnswers.get(similarQuery), query);
        }
        return generateNew(query);
    }
}
```

## 监控与可观测性

### 指标收集

```java
/**
 * 性能监控
 */
@Component
public class RAGMetrics {
    
    private final MeterRegistry meterRegistry;
    
    // 检索指标
    public void recordRetrieval(String query, int resultCount, long latency) {
        meterRegistry.counter("rag.retrieval.total").increment();
        meterRegistry.timer("rag.retrieval.latency").record(latency, 
            TimeUnit.MILLISECONDS);
        meterRegistry.gauge("rag.retrieval.result_count", resultCount);
    }
    
    // 生成指标
    public void recordGeneration(String model, int inputTokens, 
                                  int outputTokens, long latency) {
        meterRegistry.counter("rag.generation.total", 
            "model", model).increment();
        meterRegistry.counter("rag.tokens.input", 
            "model", model).increment(inputTokens);
        meterRegistry.counter("rag.tokens.output", 
            "model", model).increment(outputTokens);
        meterRegistry.timer("rag.generation.latency", 
            "model", model).record(latency, TimeUnit.MILLISECONDS);
    }
    
    // 用户满意度
    public void recordFeedback(String query, boolean helpful) {
        meterRegistry.counter("rag.feedback", 
            "helpful", String.valueOf(helpful)).increment();
    }
}
```

### 日志追踪

```java
/**
 * 请求追踪
 */
@Component
public class RequestTracing {
    
    public void traceRAGRequest(String requestId, String query) {
        MDC.put("requestId", requestId);
        
        log.info("RAG请求开始: query={}", query);
        
        // 记录各环节耗时
        long startTime = System.currentTimeMillis();
        
        // 检索
        long retrievalStart = System.currentTimeMillis();
        RetrievalResult result = retrievalService.retrieve(query);
        log.info("检索完成: latency={}, results={}", 
            System.currentTimeMillis() - retrievalStart,
            result.getDocuments().size());
        
        // 生成
        long generationStart = System.currentTimeMillis();
        String answer = generationService.generate(query, result);
        log.info("生成完成: latency={}", 
            System.currentTimeMillis() - generationStart);
        
        log.info("RAG请求完成: totalLatency={}", 
            System.currentTimeMillis() - startTime);
        
        MDC.clear();
    }
}
```

## 部署架构

### Kubernetes部署

```yaml
# rag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-service
        image: rag-service:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: VECTOR_DB_URL
          value: "chroma-service:8000"
        - name: LLM_API_URL
          value: "ollama-service:11434"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 小结

本章我们学习了：

1. **性能优化**：检索和生成优化策略
2. **系统架构**：分层缓存、异步处理
3. **成本控制**：混合部署、Token优化
4. **监控观测**：指标收集、日志追踪
5. **生产部署**：K8s配置、自动扩缩容

**关键认识：**
生产环境的RAG系统需要在性能、成本、稳定性之间找到平衡。

---

**练习题：**

1. 设计一个RAG系统的性能测试方案
2. 如何实现RAG系统的降级策略？
3. 计算你的RAG系统的成本模型。

---

<table width="100%">
   <tr>
      <td align="left"><a href="04-conversation-system.md">&larr; 11.4 对话系统实现</a></td>
      <td align="right"><a href="../../part3-agents/chapter-12/01-what-is-agent.md">12.1 智能体概念与架构 &rarr;</a></td>
   </tr>
</table>