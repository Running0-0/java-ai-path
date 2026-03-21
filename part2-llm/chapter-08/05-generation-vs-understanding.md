<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 8.4 实战：调用OpenAI API](04-openai-api-practice.md)</span>

<span>[9.1 开源大模型生态 &rarr;](../chapter-09/01-open-source-llm-ecosystem.md)</span>

</div>
---

# 8.5 设计思考：生成式与理解式模型

> "生成与理解是AI的两大能力——理解是思考，生成是表达。"

## 两类模型的本质区别

### 核心差异

```
理解式模型（如BERT）：
- 双向注意力，看完整上下文
- 输出是表示/分类结果
- 适合：分类、NER、问答、相似度

生成式模型（如GPT）：
- 单向注意力，自左向右生成
- 输出是序列生成
- 适合：写作、对话、代码生成
```

### 对比分析

| 维度 | 理解式（BERT） | 生成式（GPT） |
|------|---------------|--------------|
| 注意力 | 双向 | 单向（掩码） |
| 架构 | 仅编码器 | 仅解码器 |
| 预训练 | MLM | 自回归 |
| 输出 | 表示向量 | 生成序列 |
| 使用方式 | 微调为主 | 提示/微调 |
| 数据需求 | 需要标注数据 | 零样本/Few-shot |

## 能力边界分析

### 理解式模型的边界

```
擅长：
✓ 文本分类（情感、主题）
✓ 命名实体识别
✓ 问答（给定上下文）
✓ 语义相似度
✓ 文本蕴含

局限：
✗ 不能生成新文本
✗ 需要针对任务微调
✗ 没有创造性输出

适用场景：
- 需要精确理解的任务
- 有标注数据的场景
- 对确定性要求高的应用
```

### 生成式模型的边界

```
擅长：
✓ 文本生成（文章、代码）
✓ 开放式对话
✓ 创意写作
✓ 代码补全
✓ 多轮推理

局限：
✗ 可能产生幻觉（错误信息）
✗ 输出不确定
✗ 难以精确控制
✗ 计算成本高

适用场景：
- 需要创造性的任务
- 开放式问题
- 人机交互应用
```

## 架构选择的决策框架

### 决策树

```
任务类型？
├── 理解/分析任务
│   ├── 有标注数据？
│   │   ├── 是 → BERT + 微调
│   │   └── 否 → GPT + Few-shot
│   └── 需要精确输出？
│       ├── 是 → BERT
│       └── 否 → GPT
│
└── 生成任务
    ├── 需要事实准确？
    │   ├── 是 → GPT + RAG（检索增强）
    │   └── 否 → GPT
    └── 需要特定风格？
        ├── 是 → GPT + 微调
        └── 否 → GPT + 提示工程
```

### 实际案例

```java
/**
 * 任务与模型选择
 */
public class ModelSelectionGuide {
    
    /**
     * 情感分析 → 理解式
     */
    public String sentimentAnalysis(String text) {
        // BERT更适合：需要精确分类
        return bertClassifier.predict(text);
    }
    
    /**
     * 客服对话 → 生成式
     */
    public String customerService(String query) {
        // GPT更适合：需要自然对话
        return gpt.chat("客服：" + query);
    }
    
    /**
     * 文档检索 → 理解式
     */
    public List<Document> documentSearch(String query, 
                                          List<Document> docs) {
        // BERT嵌入计算相似度
        INDArray queryEmbedding = bert.encode(query);
        return docs.stream()
            .sorted(Comparator.comparingDouble(d -> 
                -cosineSimilarity(queryEmbedding, d.embedding)))
            .limit(5)
            .collect(Collectors.toList());
    }
    
    /**
     * 代码生成 → 生成式
     */
    public String codeGeneration(String description) {
        // GPT更适合：需要创造性生成
        return gpt.generate("生成Java代码：" + description);
    }
}
```

## 混合使用策略

### 理解+生成的流水线

```
RAG（检索增强生成）模式：

用户问题
    ↓
[理解模型] → 编码查询
    ↓
向量检索 → 找到相关文档
    ↓
[生成模型] → 基于检索结果生成回答
    ↓
最终答案

优势：
- 生成模型有事实依据
- 减少幻觉
- 可溯源
```

### Java实现示例

```java
/**
 * RAG流水线
 */
public class RAGPipeline {
    
    private final BERTModel encoder;      // 理解式
    private final GPTModel generator;     // 生成式
    private final VectorStore vectorStore;
    
    /**
     * 回答用户问题
     */
    public String answer(String question) {
        // 1. 编码问题
        INDArray questionEmbedding = encoder.encode(question);
        
        // 2. 检索相关文档
        List<Document> relevantDocs = vectorStore.search(
            questionEmbedding, 3);
        
        // 3. 构建上下文
        String context = relevantDocs.stream()
            .map(d -> d.content)
            .collect(Collectors.joining("\n\n"));
        
        // 4. 生成回答
        String prompt = String.format("""
            基于以下信息回答问题：
            
            信息：
            %s
            
            问题：%s
            
            回答：""", context, question);
        
        return generator.generate(prompt);
    }
}
```

## 未来趋势

### 统一模型

```
趋势：理解与生成的融合

GPT-4已经展现出：
- 强大的理解能力
- 优秀的生成能力
- 推理和规划能力

未来方向：
- 单一模型处理多种任务
- 动态切换理解和生成模式
- 更高效的架构设计
```

### 对Java程序员的启示

```
1. 不要局限于单一模型
   - 根据任务选择合适工具
   - 学会组合使用

2. 关注能力边界
   - 了解模型的优势和局限
   - 设计合适的应用架构

3. 持续学习
   - 模型快速迭代
   - 保持对新技术的关注

4. 工程实践
   - 封装模型调用
   - 设计可切换的架构
   - 关注成本和性能
```

## 小结

本章我们思考了：

1. **两类模型的本质**：理解式vs生成式，双向vs单向
2. **能力边界**：各自的优势和局限
3. **选择框架**：根据任务选择合适模型
4. **混合策略**：RAG等组合使用方法
5. **未来趋势**：统一模型的发展

**关键认识：**
理解式和生成式模型各有优势，聪明的工程师会根据任务特点选择和组合使用。

**下一步：** 我们将进入第三部分，学习大语言模型的应用。

---

**思考题：**

1. 你的项目中哪些任务适合理解式模型，哪些适合生成式？
2. 如何设计一个同时使用两类模型的系统？
3. 未来统一模型会取代专门的模型吗？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 8.4 实战：调用OpenAI API](04-openai-api-practice.md)</span>

<span>[9.1 开源大模型生态 &rarr;](../chapter-09/01-open-source-llm-ecosystem.md)</span>

</div>