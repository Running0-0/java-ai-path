# 9.3 LangChain4j：Java的LLM应用框架

> "LangChain4j让Java程序员也能轻松构建LLM应用——链式调用、组件化设计，开发效率倍增。"

## LangChain4j概述

### 什么是LangChain4j

```
LangChain4j = Java版的LangChain

核心价值：
- 统一接口：支持多种LLM（OpenAI、本地模型等）
- 链式调用：将多个操作组合成流水线
- 内存管理：对话历史自动维护
- 工具集成：轻松扩展LLM能力

与Python LangChain的关系：
- 功能对等，API相似
- 专为Java生态设计
- 更好的类型安全
```

### 核心组件

```
LangChain4j核心组件：

1. LanguageModel
   - 与LLM交互的接口
   - 支持聊天和文本生成

2. Memory
   - 对话历史管理
   - 短期和长期记忆

3. Chain
   - 操作链
   - 组合多个步骤

4. Tool
   - 外部工具
   - 扩展LLM能力

5. Document
   - 文档处理
   - 支持RAG
```

## 快速开始

### 添加依赖

```xml
<!-- pom.xml -->
<dependencies>
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j</artifactId>
        <version>0.24.0</version>
    </dependency>
    
    <!-- OpenAI支持 -->
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-open-ai</artifactId>
        <version>0.24.0</version>
    </dependency>
    
    <!-- 本地模型支持 -->
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-ollama</artifactId>
        <version>0.24.0</version>
    </dependency>
</dependencies>
```

### 基础使用

```java
package com.example.langchain4j;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;

/**
 * LangChain4j基础示例
 */
public class BasicExample {
    
    /**
     * 使用OpenAI
     */
    public void useOpenAI() {
        ChatLanguageModel model = OpenAiChatModel.builder()
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .modelName("gpt-3.5-turbo")
            .temperature(0.7)
            .build();
        
        String response = model.generate("Hello, how are you?");
        System.out.println(response);
    }
    
    /**
     * 使用本地模型（Ollama）
     */
    public void useLocalModel() {
        ChatLanguageModel model = OllamaChatModel.builder()
            .baseUrl("http://localhost:11434")
            .modelName("llama2")
            .temperature(0.7)
            .build();
        
        String response = model.generate("Hello, how are you?");
        System.out.println(response);
    }
}
```

## 对话与记忆

### 对话管理

```java
package com.example.langchain4j;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

/**
 * 对话助手
 */
interface Assistant {
    
    @SystemMessage("你是一个友好的Java编程助手。")
    String chat(@UserMessage String message);
}

public class ConversationExample {
    
    public void createAssistant() {
        // 创建记忆（保留最近10条消息）
        ChatMemory chatMemory = MessageWindowChatMemory.builder()
            .maxMessages(10)
            .build();
        
        // 创建助手
        Assistant assistant = AiServices.builder(Assistant.class)
            .chatLanguageModel(model)
            .chatMemory(chatMemory)
            .build();
        
        // 多轮对话
        System.out.println(assistant.chat("什么是Java的Stream API？"));
        System.out.println(assistant.chat("给我一个例子"));
        System.out.println(assistant.chat("还有更复杂的用法吗？"));
        // 助手能记住之前的对话内容
    }
}
```

### 持久化记忆

```java
/**
 * 长期记忆
 */
public class PersistentMemory {
    
    /**
     * 使用数据库存储对话历史
     */
    public ChatMemory createPersistentMemory(String userId) {
        return PersistentChatMemory.builder()
            .userId(userId)
            .storage(new JdbcChatMemoryStore(dataSource))
            .maxMessages(100)
            .build();
    }
}
```

## 工具使用（Tools）

### 定义工具

```java
package com.example.langchain4j;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.service.AiServices;

import java.time.LocalDateTime;

/**
 * 工具定义
 */
class Tools {
    
    @Tool("获取当前日期时间")
    public String getCurrentDateTime() {
        return LocalDateTime.now().toString();
    }
    
    @Tool("计算两个数的和")
    public double add(double a, double b) {
        return a + b;
    }
    
    @Tool("查询天气")
    public String getWeather(String city) {
        // 调用天气API
        return weatherApi.query(city);
    }
}

/**
 * 使用工具的助手
 */
interface ToolUsingAssistant {
    String chat(String message);
}

public class ToolExample {
    
    public void createToolAssistant() {
        ToolUsingAssistant assistant = AiServices.builder(ToolUsingAssistant.class)
            .chatLanguageModel(model)
            .tools(new Tools())
            .build();
        
        // LLM会自动决定何时使用工具
        String response = assistant.chat("现在几点了？北京天气怎么样？");
        System.out.println(response);
    }
}
```

## 文档处理与RAG

### 文档加载

```java
package com.example.langchain4j;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * 文档处理
 */
public class DocumentProcessing {
    
    /**
     * 加载和分割文档
     */
    public List<TextSegment> processDocument(String filePath) {
        // 加载文档
        Path path = Paths.get(filePath);
        Document document = FileSystemDocumentLoader.loadDocument(
            path, new TextDocumentParser());
        
        // 分割成段落
        return DocumentSplitters.recursive(500, 50)
            .split(document);
    }
    
    /**
     * 构建向量存储
     */
    public EmbeddingStore<TextSegment> buildVectorStore(
            List<TextSegment> segments,
            EmbeddingModel embeddingModel) {
        
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        
        for (TextSegment segment : segments) {
            Embedding embedding = embeddingModel.embed(segment);
            store.add(embedding, segment);
        }
        
        return store;
    }
}
```

### RAG实现

```java
/**
 * RAG助手
 */
interface RAGAssistant {
    @SystemMessage("""
        你是一个知识库助手。
        基于提供的文档内容回答问题。
        如果文档中没有相关信息，请说明。
        """)
    String answer(String question);
}

public class RAGExample {
    
    public void createRAGAssistant() {
        // 1. 加载文档
        List<TextSegment> segments = documentProcessing
            .processDocument("knowledge-base.txt");
        
        // 2. 构建向量存储
        EmbeddingStore<TextSegment> store = documentProcessing
            .buildVectorStore(segments, embeddingModel);
        
        // 3. 创建内容检索器
        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
            .embeddingStore(store)
            .embeddingModel(embeddingModel)
            .maxResults(3)
            .build();
        
        // 4. 创建RAG助手
        RAGAssistant assistant = AiServices.builder(RAGAssistant.class)
            .chatLanguageModel(chatModel)
            .contentRetriever(retriever)
            .build();
        
        // 5. 使用
        String answer = assistant.answer("公司的年假政策是什么？");
    }
}
```

## 小结

本章我们学习了：

1. **LangChain4j概述**：Java的LLM应用框架
2. **基础使用**：连接OpenAI和本地模型
3. **对话管理**：记忆维护，持久化存储
4. **工具使用**：扩展LLM能力
5. **RAG实现**：文档处理，检索增强生成

**关键认识：**
LangChain4j让Java开发者能够快速构建复杂的LLM应用，是Java AI开发的重要工具。

**下一步：** 我们将构建一个本地问答系统。

---

**练习题：**

1. LangChain4j与直接使用API有什么区别？
2. 如何实现对话记忆的持久化？
3. 设计一个使用工具的LangChain4j应用。
