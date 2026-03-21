<table width="100%">
   <tr>
      <td align="left"><a href="02-dialog-memory-management.md">&larr; 15.2 对话记忆管理</a></td>
      <td align="right"><a href="04-build-memory-agent.md">15.4 构建记忆智能体 &rarr;</a></td>
   </tr>
</table>
---

# 15.3 向量数据库

> "向量数据库是智能体的长期记忆——语义存储，相似检索。"

## 向量数据库原理

### 核心概念

```
向量数据库：

1. 向量化
   文本 → 嵌入模型 → 向量（768维或更高）
   
2. 存储
   向量 + 原始文本 + 元数据
   
3. 检索
   查询向量化 → 相似度计算 → 返回最相似的

相似度度量：
- 余弦相似度（最常用）
- 欧氏距离
- 点积
```

### 近似最近邻（ANN）

```
ANN算法：

1. HNSW（Hierarchical Navigable Small World）
   - 图结构索引
   - 查询快，精度高
   - 内存占用较大

2. IVF（Inverted File Index）
   - 聚类索引
   - 平衡速度和精度
   - 适合大规模数据

3. PQ（Product Quantization）
   - 量化压缩
   - 节省存储
   - 适合高维向量
```

## Java集成

### Chroma使用

```java
package com.example.agent.vectorstore;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;

/**
 * Chroma向量存储
 */
public class ChromaVectorStore {
    
    private final EmbeddingStore<TextSegment> store;
    
    public ChromaVectorStore(String baseUrl, String collectionName) {
        this.store = ChromaEmbeddingStore.builder()
            .baseUrl(baseUrl)
            .collectionName(collectionName)
            .build();
    }
    
    /**
     * 添加文档
     */
    public void addDocument(String text, Map<String, String> metadata) {
        TextSegment segment = TextSegment.from(text);
        metadata.forEach((k, v) -> segment.metadata().put(k, v));
        
        // 需要配合EmbeddingModel使用
        // store.add(embedding, segment);
    }
    
    /**
     * 相似度搜索
     */
    public List<EmbeddingMatch<TextSegment>> search(
            Embedding queryEmbedding, int maxResults) {
        return store.findRelevant(queryEmbedding, maxResults);
    }
}
```

### Milvus使用

```java
/**
 * Milvus向量存储
 */
public class MilvusVectorStore {
    
    private final MilvusClient client;
    private final String collectionName;
    
    public MilvusVectorStore(String host, int port, String collection) {
        ConnectParam connectParam = ConnectParam.newBuilder()
            .withHost(host)
            .withPort(port)
            .build();
        
        this.client = new MilvusServiceClient(connectParam);
        this.collectionName = collection;
    }
    
    /**
     * 创建集合
     */
    public void createCollection(int dimension) {
        FieldType idField = FieldType.newBuilder()
            .withName("id")
            .withDataType(DataType.Int64)
            .withPrimaryKey(true)
            .withAutoID(true)
            .build();
        
        FieldType vectorField = FieldType.newBuilder()
            .withName("embedding")
            .withDataType(DataType.FloatVector)
            .withDimension(dimension)
            .build();
        
        CreateCollectionParam createParam = CreateCollectionParam.newBuilder()
            .withCollectionName(collectionName)
            .withFieldTypes(Arrays.asList(idField, vectorField))
            .build();
        
        client.createCollection(createParam);
    }
}
```

## 性能优化

### 索引选择

```java
/**
 * 索引配置建议
 */
public class IndexConfiguration {
    
    /**
     * 根据数据量选择索引
     */
    public static IndexType recommendIndex(long dataSize, int dimension) {
        if (dataSize < 10000) {
            // 小数据集：暴力搜索
            return IndexType.FLAT;
        } else if (dataSize < 1000000) {
            // 中等数据集：HNSW
            return IndexType.HNSW;
        } else {
            // 大数据集：IVF + PQ
            return IndexType.IVF_PQ;
        }
    }
}
```

## 小结

本章我们学习了：

1. **向量数据库原理**：向量化、存储、检索
2. **ANN算法**：HNSW、IVF、PQ
3. **Java集成**：Chroma、Milvus使用
4. **性能优化**：索引选择

**关键认识：**
向量数据库是智能体长期记忆的基础设施，选择合适的方案很重要。

**下一步：** 我们将学习构建记忆智能体。

---

**练习题：**

1. 比较不同ANN算法的优缺点
2. 如何评估向量检索的质量？
3. 设计一个混合索引策略。

---

<table width="100%">
   <tr>
      <td align="left"><a href="02-dialog-memory-management.md">&larr; 15.2 对话记忆管理</a></td>
      <td align="right"><a href="04-build-memory-agent.md">15.4 构建记忆智能体 &rarr;</a></td>
   </tr>
</table>