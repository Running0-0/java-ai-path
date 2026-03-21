<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-10/05-prompt-as-programming.md">&larr; 10.5 设计思考：提示即编程</a></td>
      <td align="right"><a href="02-document-parsing-vectorization.md">11.2 文档解析与向量化 &rarr;</a></td>
   </tr>
</table>
---

# 11.1 项目架构设计

> "好的架构是项目成功的一半——让我们设计一个可扩展的RAG系统。"

## RAG系统架构

### 整体架构

```
RAG系统架构图：

┌─────────────────────────────────────────────────┐
│                   用户界面层                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Web界面  │  │ API接口  │  │ 管理后台 │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼─────────────┼─────────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
┌─────────────────────┼───────────────────────────┐
│                   业务逻辑层                      │
│  ┌──────────────────┴──────────────────┐       │
│  │           问答服务 (QAService)         │       │
│  │  - 问题理解  - 答案生成  - 会话管理     │       │
│  └──────────────────┬──────────────────┘       │
│                     │                           │
│  ┌──────────────────┼──────────────────┐       │
│  │         检索服务 (RetrievalService)   │       │
│  │  - 向量化  - 相似度搜索  - 结果排序    │       │
│  └──────────────────┬──────────────────┘       │
└─────────────────────┼───────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────┐
│                   数据层                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ 向量数据库│  │ 文档存储 │  │ 元数据   │       │
│  │ (Chroma) │  │ (MinIO)  │  │ (MySQL)  │       │
│  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────┘
```

### 核心组件

```java
/**
 * 系统组件定义
 */
public class RAGSystemComponents {
    
    // 1. 文档处理管道
    interface DocumentPipeline {
        Document load(String source);
        List<Chunk> split(Document doc);
        Embedding embed(Chunk chunk);
        void store(Embedding embedding, Chunk chunk);
    }
    
    // 2. 检索服务
    interface RetrievalService {
        List<SearchResult> search(String query, int topK);
        List<SearchResult> hybridSearch(String query, 
                                        Map<String, Object> filters);
    }
    
    // 3. 生成服务
    interface GenerationService {
        String generate(String query, List<Context> contexts);
        Stream<String> generateStream(String query, 
                                       List<Context> contexts);
    }
    
    // 4. 会话管理
    interface SessionManager {
        String createSession();
        void addMessage(String sessionId, Message message);
        List<Message> getHistory(String sessionId);
    }
}
```

## 技术选型

### 技术栈对比

| 组件 | 选项1 | 选项2 | 选项3 | 推荐 |
|------|-------|-------|-------|------|
| 向量数据库 | Chroma | Milvus | Pinecone | Chroma |
| 嵌入模型 | OpenAI | 本地模型 | HuggingFace | 本地模型 |
| LLM | GPT-4 | Claude | 本地LLM | 混合 |
| 框架 | LangChain4j | Spring AI | 自研 | LangChain4j |
| 文档存储 | 本地文件 | MinIO | S3 | MinIO |

### 选型理由

```
Chroma选择理由：
- 开源免费
- 轻量级，易于部署
- Java客户端支持
- 支持内存和持久化模式

本地嵌入模型：
- 数据隐私
- 无网络依赖
- 成本可控

混合LLM策略：
- 简单任务用本地模型
- 复杂任务用云端API
- 成本与效果平衡
```

## 模块设计

### 文档处理模块

```java
package com.example.rag.document;

/**
 * 文档处理流程
 */
public class DocumentProcessingModule {
    
    private final DocumentLoader loader;
    private final DocumentSplitter splitter;
    private final EmbeddingGenerator embedder;
    private final VectorStore vectorStore;
    
    /**
     * 处理并索引文档
     */
    public void processDocument(DocumentSource source) {
        // 1. 加载
        Document doc = loader.load(source);
        
        // 2. 分割
        List<Chunk> chunks = splitter.split(doc, 
            SplitConfig.builder()
                .chunkSize(500)
                .overlap(50)
                .build()
        );
        
        // 3. 嵌入并存储
        for (Chunk chunk : chunks) {
            Embedding embedding = embedder.generate(chunk.getText());
            vectorStore.store(embedding, chunk);
        }
        
        // 4. 记录元数据
        saveMetadata(source, chunks.size());
    }
}
```

### 检索模块

```java
package com.example.rag.retrieval;

/**
 * 检索服务实现
 */
public class RetrievalModule {
    
    private final VectorStore vectorStore;
    private final EmbeddingGenerator embedder;
    private final Reranker reranker;
    
    /**
     * 检索相关文档
     */
    public RetrievalResult retrieve(String query, RetrievalConfig config) {
        // 1. 查询向量化
        Embedding queryEmbedding = embedder.generate(query);
        
        // 2. 向量搜索
        List<SearchResult> candidates = vectorStore.search(
            queryEmbedding, 
            config.getTopK() * 2  // 多召回一些用于重排序
        );
        
        // 3. 重排序
        List<SearchResult> reranked = reranker.rerank(
            query, candidates);
        
        // 4. 返回TopK
        return new RetrievalResult(
            reranked.subList(0, config.getTopK())
        );
    }
}
```

### 生成模块

```java
package com.example.rag.generation;

/**
 * 答案生成服务
 */
public class GenerationModule {
    
    private final ChatLanguageModel chatModel;
    private final PromptTemplate promptTemplate;
    
    /**
     * 生成答案
     */
    public String generate(String query, RetrievalResult context) {
        // 构建提示
        Map<String, Object> params = new HashMap<>();
        params.put("query", query);
        params.put("contexts", formatContexts(context));
        
        String prompt = promptTemplate.render(params);
        
        // 生成答案
        return chatModel.generate(prompt);
    }
    
    /**
     * 流式生成
     */
    public Stream<String> generateStream(String query, 
                                          RetrievalResult context) {
        String prompt = buildPrompt(query, context);
        return chatModel.generateStream(prompt);
    }
}
```

## 数据流设计

### 索引流程

```
文档索引数据流：

原始文档
    ↓
[文档加载器] → 解析不同格式
    ↓
[文档清洗] → 去除噪声、格式化
    ↓
[文档分割] → 语义分块
    ↓
[元数据提取] → 标题、标签、时间
    ↓
[向量化] → 生成Embedding
    ↓
[向量存储] → 存入Chroma
    ↓
[元数据存储] → 存入MySQL
```

### 查询流程

```
用户查询数据流：

用户问题
    ↓
[查询理解] → 意图识别、扩展
    ↓
[查询向量化] → 生成Embedding
    ↓
[向量检索] → Chroma搜索
    ↓
[结果重排序] → 精排TopK
    ↓
[上下文构建] → 格式化检索结果
    ↓
[答案生成] → LLM生成
    ↓
[后处理] → 格式化、引用标注
    ↓
返回用户
```

## 部署架构

### 容器化部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - VECTOR_DB_URL=chroma:8000
      - LLM_API_URL=ollama:11434
    depends_on:
      - chroma
      - ollama

  chroma:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: rag_system
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  chroma_data:
  ollama_models:
  mysql_data:
```

## 小结

本章我们设计了：

1. **整体架构**：分层设计，职责清晰
2. **技术选型**：开源优先，混合策略
3. **模块设计**：文档处理、检索、生成
4. **数据流**：索引和查询流程
5. **部署架构**：容器化方案

**关键认识：**
好的架构设计是RAG项目成功的基础，要考虑可扩展性、可维护性和成本。

**下一步：** 我们将实现文档解析和向量化。

---

**练习题：**

1. 为什么RAG系统需要分层架构？
2. 向量数据库选型要考虑哪些因素？
3. 设计一个支持多租户的RAG架构。

---

<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-10/05-prompt-as-programming.md">&larr; 10.5 设计思考：提示即编程</a></td>
      <td align="right"><a href="02-document-parsing-vectorization.md">11.2 文档解析与向量化 &rarr;</a></td>
   </tr>
</table>