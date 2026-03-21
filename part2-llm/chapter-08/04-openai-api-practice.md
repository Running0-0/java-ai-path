<table width="100%">
   <tr>
      <td align="left"><a href="03-pretraining-finetuning.md">&larr; 8.3 预训练与微调范式</a></td>
      <td align="right"><a href="05-generation-vs-understanding.md">8.5 设计思考：生成式与理解式模型 &rarr;</a></td>
   </tr>
</table>
---

# 8.4 实战：调用OpenAI API

> "学会使用API是现代AI开发者的必备技能——让强大的模型为你所用。"

## OpenAI API概述

### 提供的模型

| 类型 | 用途 | 说明 |
|------|------|------|
| Chat Model | 对话、摘要、代码生成 | 适合大多数聊天与助手场景 |
| Reasoning Model | 复杂推理、规划 | 成本更高，延迟通常也更高 |
| Embedding Model | 文本向量化 | 用于检索、RAG、相似度计算 |
| Whisper | 语音转文字 | 适合音频转录 |
| Image Model | 图像生成/编辑 | 适合多模态创作场景 |

> 模型名称和价格更新很快。实际项目中不要把模型名和单价硬编码在代码里，而是通过配置注入，并定期对照官方文档更新。

### 获取API密钥

```
1. 访问 https://platform.openai.com
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的API Key
5. 保存好密钥（只显示一次）

注意事项：
- API调用需要付费
- 保护好API Key，不要泄露
- 设置使用限额防止意外消费
```

## Java调用OpenAI API

### 添加依赖

```xml
<!-- pom.xml -->
<dependencies>
    <!-- HTTP客户端 -->
    <dependency>
        <groupId>com.squareup.okhttp3</groupId>
        <artifactId>okhttp</artifactId>
        <version>4.12.0</version>
    </dependency>
    
    <!-- JSON处理 -->
    <dependency>
        <groupId>com.google.code.gson</groupId>
        <artifactId>gson</artifactId>
        <version>2.10.1</version>
    </dependency>
</dependencies>
```

### 基础客户端

```java
package com.example.openai;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * OpenAI API客户端
 */
public class OpenAIClient {
    
    private static final String BASE_URL = "https://api.openai.com/v1";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    
    private final OkHttpClient httpClient;
    private final Gson gson;
    private final String apiKey;
    
    public OpenAIClient(String apiKey) {
        this.apiKey = apiKey;
        this.gson = new Gson();
        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();
    }
    
    /**
     * 发送JSON请求并返回JSON响应
     */
    public JsonObject post(String endpoint, JsonObject payload) throws IOException {
        RequestBody body = RequestBody.create(payload.toString(), JSON);
        
        Request request = new Request.Builder()
            .url(BASE_URL + endpoint)
            .header("Authorization", "Bearer " + apiKey)
            .header("Content-Type", "application/json")
            .post(body)
            .build();
        
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Unexpected code: " + response);
            }
            String responseBody = response.body().string();
            return gson.fromJson(responseBody, JsonObject.class);
        }
    }
}
```

## 文本生成（Chat Completions）

### 基础调用

```java
package com.example.openai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.util.List;

/**
 * 聊天完成API
 */
public class ChatCompletion {
    
    private final OpenAIClient client;
    private final String chatModel;
    
    public ChatCompletion(OpenAIClient client, String chatModel) {
        this.client = client;
        this.chatModel = chatModel;
    }
    
    /**
     * 简单对话
     */
    public String chat(String message) throws IOException {
        return chat(List.of(new Message("user", message)));
    }
    
    /**
     * 多轮对话
     */
    public String chat(List<Message> messages) throws IOException {
        return chat(messages, 0.7, 500);
    }

    public String chat(List<Message> messages,
                       double temperature,
                       int maxTokens) throws IOException {
        JsonObject request = new JsonObject();
        request.addProperty("model", chatModel);

        JsonArray jsonMessages = new JsonArray();
        for (Message message : messages) {
            JsonObject item = new JsonObject();
            item.addProperty("role", message.role());
            item.addProperty("content", message.content());
            jsonMessages.add(item);
        }

        request.add("messages", jsonMessages);
        request.addProperty("temperature", temperature);
        request.addProperty("max_tokens", maxTokens);
        
        JsonObject response = client.post("/chat/completions", request);
        return response.getAsJsonArray("choices")
            .get(0).getAsJsonObject()
            .getAsJsonObject("message")
            .get("content").getAsString();
    }
}

/**
 * 消息类
 */
public record Message(String role, String content) {}
```

### 高级参数

```java
package com.example.openai;

import java.io.IOException;
import java.util.List;

/**
 * 高级调用示例
 */
public class AdvancedChat {

    private final ChatCompletion chat;

    public AdvancedChat(ChatCompletion chat) {
        this.chat = chat;
    }
    
    public String generateWithParameters(String prompt) 
            throws IOException {
        List<Message> messages = List.of(
            new Message("system", "你是一个专业的Java程序员，擅长解释技术概念。"),
            new Message("user", prompt)
        );

        return chat.chat(messages, 0.7, 1000);
    }
}
```

## 实际应用场景

### 代码助手

```java
package com.example.openai;

import java.io.IOException;
import java.util.List;

/**
 * AI代码助手
 */
public class AICodeAssistant {
    
    private final ChatCompletion chat;

    public AICodeAssistant(ChatCompletion chat) {
        this.chat = chat;
    }
    
    /**
     * 解释代码
     */
    public String explainCode(String code) throws IOException {
        List<Message> messages = List.of(
            new Message("system", 
                "你是一个Java专家，用简洁的语言解释代码。"),
            new Message("user", 
                "请解释这段代码：\n```java\n" + code + "\n```")
        );
        
        return chat.chat(messages);
    }
    
    /**
     * 生成代码
     */
    public String generateCode(String description) throws IOException {
        List<Message> messages = List.of(
            new Message("system", 
                "你是一个Java程序员，生成高质量、带注释的代码。"),
            new Message("user", description)
        );
        
        return chat.chat(messages);
    }
    
    /**
     * 重构建议
     */
    public String refactorSuggestion(String code) throws IOException {
        List<Message> messages = List.of(
            new Message("system", 
                "你是一个代码审查专家，提供重构建议。"),
            new Message("user", 
                "请审查这段代码并提供重构建议：\n" + code)
        );
        
        return chat.chat(messages);
    }
}
```

### 文本处理工具

```java
package com.example.openai;

import java.io.IOException;
import java.util.List;

/**
 * 文本处理工具
 */
public class TextProcessor {
    
    private final ChatCompletion chat;

    public TextProcessor(ChatCompletion chat) {
        this.chat = chat;
    }
    
    /**
     * 文本摘要
     */
    public String summarize(String text) throws IOException {
        return chat.chat(List.of(
            new Message("user", 
                "请用一段话总结以下内容：\n" + text)
        ));
    }
    
    /**
     * 情感分析
     */
    public String sentiment(String text) throws IOException {
        return chat.chat(List.of(
            new Message("user", 
                "判断以下文本的情感（正面/负面/中性）：\n" + text)
        ));
    }
    
    /**
     * 翻译
     */
    public String translate(String text, String targetLang) 
            throws IOException {
        return chat.chat(List.of(
            new Message("user", 
                "将以下内容翻译成" + targetLang + "：\n" + text)
        ));
    }
}
```

## 错误处理与优化

### 异常处理

```java
package com.example.openai;

import java.io.IOException;
import java.util.List;

/**
 * 健壮的错误处理
 */
public class RobustOpenAIClient {
    
    private static final int MAX_RETRIES = 3;
    private static final long RETRY_DELAY = 1000;
    private final ChatCompletion chat;

    public RobustOpenAIClient(ChatCompletion chat) {
        this.chat = chat;
    }
    
    public String chatWithRetry(List<Message> messages) {
        int retries = 0;
        
        while (retries < MAX_RETRIES) {
            try {
                return chat.chat(messages);
            } catch (IOException e) {
                retries++;
                
                if (retries >= MAX_RETRIES) {
                    throw new RuntimeException(
                        "API调用失败，已重试" + MAX_RETRIES + "次", e);
                }
                
                // 指数退避
                try {
                    Thread.sleep(RETRY_DELAY * retries);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        
        return null;
    }
}
```

### 成本控制

价格经常调整，下面演示的是“如何做预算控制”，不是一份长期有效的价目表。生产环境里建议把单价放到配置中心，并定期和官方定价页面同步。

```java
package com.example.openai;

import java.util.Map;

/**
 * API调用成本控制
 */
public class CostController {
    
    private final Map<String, Double> pricesPer1KTokens;
    private double totalCost = 0;
    private final double costLimit;

    public CostController(Map<String, Double> pricesPer1KTokens,
                          double costLimit) {
        this.pricesPer1KTokens = pricesPer1KTokens;
        this.costLimit = costLimit;
    }
    
    /**
     * 检查是否超出预算
     */
    public boolean canMakeRequest(String model, int estimatedTokens) {
        double price = pricesPer1KTokens.getOrDefault(model, 0.0);
        double estimatedCost = (estimatedTokens / 1000.0) * price;
        
        return (totalCost + estimatedCost) < costLimit;
    }
    
    /**
     * 记录实际消耗
     */
    public void recordUsage(String model, int tokens) {
        double price = pricesPer1KTokens.getOrDefault(model, 0.0);
        totalCost += (tokens / 1000.0) * price;
    }
}
```

## 小结

本章我们学习了：

1. **OpenAI API**：模型选择，密钥获取
2. **Java调用**：HTTP客户端，JSON处理
3. **文本生成**：基础调用，高级参数
4. **实际应用**：代码助手，文本处理
5. **错误处理**：重试机制，成本控制

**关键认识：**
API调用让强大的AI模型触手可及，掌握API使用是现代AI开发的基础。

**下一步：** 我们将对比生成式与理解式模型。

---

**练习题：**

1. 如何设置API调用的参数来控制生成结果？
2. 如何实现API调用的错误重试？
3. 设计一个使用OpenAI API的Java应用。

---

<table width="100%">
   <tr>
      <td align="left"><a href="03-pretraining-finetuning.md">&larr; 8.3 预训练与微调范式</a></td>
      <td align="right"><a href="05-generation-vs-understanding.md">8.5 设计思考：生成式与理解式模型 &rarr;</a></td>
   </tr>
</table>