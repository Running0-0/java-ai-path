<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 9.3 LangChain4j：Java的LLM应用框架](03-langchain4j-framework.md)</span>

<span>[9.5 设计思考：云端vs本地部署 &rarr;](05-cloud-vs-local-deployment.md)</span>

</div>
---

# 9.4 实战：构建本地问答系统

> "一个完全本地运行的问答系统——保护隐私，无需联网，随时可用。"

## 系统设计

### 架构概览

```
本地问答系统架构：

用户问题
    ↓
[问题理解] → 意图识别
    ↓
[知识检索] → 向量数据库
    ↓
[答案生成] → 本地LLM
    ↓
格式化回答

技术栈：
- Ollama：本地模型服务
- LangChain4j：应用框架
- Chroma/Milvus：向量数据库
- Java：业务逻辑
```

### 组件清单

```java
/**
 * 系统组件
 */
public class QASystemComponents {
    
    // 1. 嵌入模型（用于向量化）
    // 使用轻量级模型如 all-MiniLM-L6-v2
    
    // 2. 聊天模型（用于生成答案）
    // 使用本地模型如 llama2 或 qwen
    
    // 3. 向量存储
    // 内存存储或Chroma
    
    // 4. 文档加载器
    // 支持txt、pdf、md等格式
}
```

## 完整实现

### 项目结构

```
src/main/java/com/example/qa/
├── QASystem.java           # 主类
├── DocumentLoader.java     # 文档加载
├── VectorStore.java        # 向量存储
├── EmbeddingService.java   # 嵌入服务
├── LLMService.java         # LLM服务
└── QAService.java          # 问答服务
```

### 核心代码

```java
package com.example.qa;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.util.List;

/**
 * 本地问答系统
 */
public class LocalQASystem {
    
    private final EmbeddingModel embeddingModel;
    private final OllamaChatModel chatModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    
    public LocalQASystem() {
        // 初始化嵌入模型（用于文档向量化）
        this.embeddingModel = OllamaEmbeddingModel.builder()
            .baseUrl("http://localhost:11434")
            .modelName("nomic-embed-text")  // 轻量级嵌入模型
            .build();
        
        // 初始化聊天模型（用于生成答案）
        this.chatModel = OllamaChatModel.builder()
            .baseUrl("http://localhost:11434")
            .modelName("llama2")
            .temperature(0.3)  // 低温度，更确定
            .build();
        
        // 内存向量存储
        this.embeddingStore = new InMemoryEmbeddingStore<>();
    }
    
    /**
     * 加载知识库
     */
    public void loadKnowledgeBase(List<Path> documentPaths) {
        for (Path path : documentPaths) {
            loadDocument(path);
        }
        System.out.println("知识库加载完成，共 " + 
            embeddingStore.size() + " 个段落");
    }
    
    /**
     * 加载单个文档
     */
    private void loadDocument(Path path) {
        // 解析文档
        Document document = new TextDocumentParser().parse(path);
        
        // 分割成段落
        List<TextSegment> segments = DocumentSplitters
            .recursive(500, 50)
            .split(document);
        
        // 向量化并存储
        for (TextSegment segment : segments) {
            Embedding embedding = embeddingModel.embed(segment);
            embeddingStore.add(embedding, segment);
        }
        
        System.out.println("已加载: " + path.getFileName());
    }
    
    /**
     * 回答问题
     */
    public String answer(String question) {
        // 1. 检索相关段落
        List<EmbeddingMatch<TextSegment>> matches = retrieve(question);
        
        // 2. 构建上下文
        String context = buildContext(matches);
        
        // 3. 生成答案
        String prompt = buildPrompt(question, context);
        
        return chatModel.generate(prompt);
    }
    
    /**
     * 检索相关段落
     */
    private List<EmbeddingMatch<TextSegment>> retrieve(String question) {
        Embedding questionEmbedding = embeddingModel.embed(question);
        
        return embeddingStore.findRelevant(
            questionEmbedding, 
            3,  // 返回前3个最相关的
            0.7 // 相似度阈值
        );
    }
    
    /**
     * 构建上下文
     */
    private String buildContext(List<EmbeddingMatch<TextSegment>> matches) {
        StringBuilder context = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : matches) {
            context.append(match.embedded().text()).append("\n\n");
        }
        return context.toString();
    }
    
    /**
     * 构建提示词
     */
    private String buildPrompt(String question, String context) {
        return String.format("""
            基于以下信息回答问题。如果信息不足，请说明。
            
            信息：
            %s
            
            问题：%s
            
            请用中文回答：""", context, question);
    }
}
```

### 主程序

```java
package com.example.qa;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * 问答系统入口
 */
public class QASystemApp {
    
    public static void main(String[] args) {
        // 创建系统
        LocalQASystem qaSystem = new LocalQASystem();
        
        // 加载知识库
        List<Path> documents = Arrays.asList(
            Paths.get("docs/company-policy.txt"),
            Paths.get("docs/product-manual.txt"),
            Paths.get("docs/faq.md")
        );
        
        System.out.println("正在加载知识库...");
        qaSystem.loadKnowledgeBase(documents);
        
        // 交互式问答
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n问答系统已就绪，输入问题（输入'退出'结束）：");
        
        while (true) {
            System.out.print("\n问题: ");
            String question = scanner.nextLine();
            
            if (question.equalsIgnoreCase("退出")) {
                break;
            }
            
            System.out.println("思考中...");
            String answer = qaSystem.answer(question);
            System.out.println("回答: " + answer);
        }
        
        System.out.println("再见！");
        scanner.close();
    }
}
```

## 功能扩展

### 添加对话历史

```java
/**
 * 带记忆的问答系统
 */
public class ConversationalQASystem extends LocalQASystem {
    
    private final List<String> history = new ArrayList<>();
    private static final int MAX_HISTORY = 5;
    
    @Override
    public String answer(String question) {
        // 构建带历史的提示
        StringBuilder prompt = new StringBuilder();
        
        // 添加历史
        for (String h : history) {
            prompt.append(h).append("\n");
        }
        
        // 添加当前问题
        String currentAnswer = super.answer(question);
        
        // 更新历史
        history.add("Q: " + question);
        history.add("A: " + currentAnswer);
        
        // 限制历史长度
        while (history.size() > MAX_HISTORY * 2) {
            history.remove(0);
            history.remove(0);
        }
        
        return currentAnswer;
    }
}
```

### 多格式文档支持

```java
/**
 * 多格式文档加载器
 */
public class MultiFormatLoader {
    
    public Document load(Path path) {
        String extension = getExtension(path);
        
        return switch (extension) {
            case "txt" -> new TextDocumentParser().parse(path);
            case "md" -> new MarkdownDocumentParser().parse(path);
            case "pdf" -> new PdfDocumentParser().parse(path);
            case "docx" -> new DocxDocumentParser().parse(path);
            default -> throw new UnsupportedOperationException(
                "不支持的格式: " + extension);
        };
    }
}
```

## 部署与优化

### 性能优化

```java
/**
 * 性能优化建议
 */
public class OptimizationTips {
    
    // 1. 预加载模型
    // 系统启动时加载模型，避免首次请求延迟
    
    // 2. 批量嵌入
    // 多个段落一起嵌入，提高效率
    
    // 3. 缓存热点查询
    // 缓存常见问题的答案
    
    // 4. 异步处理
    // 文档加载使用后台线程
    
    // 5. 增量更新
    // 只更新变化的文档，而非全量重建
}
```

## 小结

本章我们完成了：

1. **系统设计**：本地问答系统架构
2. **完整实现**：从文档加载到答案生成
3. **功能扩展**：对话历史、多格式支持
4. **部署优化**：性能提升建议

**关键认识：**
完全本地运行的问答系统既保护了隐私，又提供了可靠的AI能力。

**下一步：** 我们将对比云端和本地部署。

---

**练习题：**

1. 如何优化向量检索的速度？
2. 如何支持PDF等格式的文档？
3. 设计一个增量更新知识库的方案。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 9.3 LangChain4j：Java的LLM应用框架](03-langchain4j-framework.md)</span>

<span>[9.5 设计思考：云端vs本地部署 &rarr;](05-cloud-vs-local-deployment.md)</span>

</div>