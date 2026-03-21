<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-08/05-generation-vs-understanding.md">&larr; 8.5 设计思考：生成式与理解式模型</a></td>
      <td align="right"><a href="02-model-quantization.md">9.2 模型量化：让大模型变小 &rarr;</a></td>
   </tr>
</table>
---

# 9.1 开源大模型生态

> "开源让AI民主化——不再依赖闭源API，你可以在本地运行强大的大模型。"

## 开源LLM概览

### 为什么需要开源模型

```
闭源模型的问题：
- 数据隐私风险（数据发送到第三方）
- 成本不可控（按token计费）
- 网络依赖（需要联网）
- 定制化困难（无法微调）

开源模型的优势：
- 数据本地处理，隐私安全
- 一次部署，长期使用
- 离线可用
- 可微调定制
```

### 主要开源模型

| 模型 | 组织 | 特点 | 许可 |
|------|------|------|------|
| LLaMA | Meta | 性能强，生态丰富 | 非商业 |
| LLaMA 2 | Meta | 可商用，多版本 | 商用许可 |
| Mistral | Mistral AI | 小体积高性能 | Apache 2.0 |
| Falcon | TII | 多语言支持 | Apache 2.0 |
| ChatGLM | 清华 | 中文优化 | 商用许可 |
| Qwen | 阿里 | 中英文俱佳 | 商用许可 |

## 模型规模与选择

### 参数规模对比

```
规模分类：

小模型（7B以下）：
- 适合：边缘设备、低延迟场景
- 代表：Mistral 7B, LLaMA 7B
- 显存需求：8-16GB

中模型（7B-30B）：
- 适合：个人开发、中小应用
- 代表：LLaMA 13B/30B, Qwen 14B
- 显存需求：16-80GB

大模型（30B+）：
- 适合：企业级应用、高性能需求
- 代表：LLaMA 65B/70B, Falcon 180B
- 显存需求：多卡或量化
```

### 选择指南

```java
/**
 * 模型选择决策
 */
public class ModelSelection {
    
    /**
     * 根据需求选择模型
     */
    public String selectModel(Requirements req) {
        if (req.isCommercialUse()) {
            // 商用场景
            if (req.getGpuMemory() >= 80) {
                return "llama2-70b";  // 最强开源商用模型
            } else if (req.getGpuMemory() >= 24) {
                return "llama2-13b";  // 性价比之选
            } else {
                return "llama2-7b";   // 入门选择
            }
        } else {
            // 研究场景
            if (req.isChineseFocused()) {
                return "chatglm3-6b";  // 中文优化
            } else {
                return "mistral-7b";   // 英文最强小模型
            }
        }
    }
}
```

## 模型获取与运行

### Hugging Face生态

```
Hugging Face：开源模型的GitHub

功能：
- 模型仓库：下载预训练模型
- Transformers库：统一接口
- Datasets：数据集
- Spaces：模型演示

Java支持：
- 通过ONNX运行时
- 通过DJL（Deep Java Library）
```

### 使用Ollama运行模型

```bash
# 安装Ollama（Mac/Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull llama2
ollama pull mistral
ollama pull qwen

# 运行模型
ollama run llama2

# REST API调用
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello, how are you?"
}'
```

### Java调用Ollama

```java
package com.example.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import okhttp3.*;

/**
 * Ollama Java客户端
 */
public class OllamaClient {
    
    private static final String BASE_URL = "http://localhost:11434";
    private static final MediaType JSON = MediaType.get("application/json");
    
    private final OkHttpClient client;
    private final Gson gson;
    
    public OllamaClient() {
        this.client = new OkHttpClient();
        this.gson = new Gson();
    }
    
    /**
     * 生成文本
     */
    public String generate(String model, String prompt) throws IOException {
        JsonObject request = new JsonObject();
        request.addProperty("model", model);
        request.addProperty("prompt", prompt);
        request.addProperty("stream", false);
        
        RequestBody body = RequestBody.create(request.toString(), JSON);
        Request httpRequest = new Request.Builder()
            .url(BASE_URL + "/api/generate")
            .post(body)
            .build();
        
        try (Response response = client.newCall(httpRequest).execute()) {
            String json = response.body().string();
            JsonObject obj = gson.fromJson(json, JsonObject.class);
            return obj.get("response").getAsString();
        }
    }
    
    /**
     * 聊天
     */
    public String chat(String model, List<Message> messages) throws IOException {
        JsonObject request = new JsonObject();
        request.addProperty("model", model);
        request.add("messages", gson.toJsonTree(messages));
        request.addProperty("stream", false);
        
        RequestBody body = RequestBody.create(request.toString(), JSON);
        Request httpRequest = new Request.Builder()
            .url(BASE_URL + "/api/chat")
            .post(body)
            .build();
        
        try (Response response = client.newCall(httpRequest).execute()) {
            String json = response.body().string();
            JsonObject obj = gson.fromJson(json, JsonObject.class);
            return obj.getAsJsonObject("message")
                .get("content").getAsString();
        }
    }
}

class Message {
    String role;
    String content;
    
    public Message(String role, String content) {
        this.role = role;
        this.content = content;
    }
}
```

## 模型量化

### 为什么需要量化

```
问题：大模型显存占用大

LLaMA-7B：
- FP16：约14GB
- INT8：约7GB
- INT4：约4GB

量化优势：
- 减少显存占用
- 提高推理速度
- 在消费级GPU上运行大模型
```

### 量化格式

| 格式 | 精度 | 大小 | 速度 | 质量 |
|------|------|------|------|------|
| FP16 | 16位 | 100% | 基准 | 最好 |
| INT8 | 8位 | 50% | 快 | 接近原模型 |
| INT4 | 4位 | 25% | 最快 | 略有损失 |
| GGUF | 混合 | 可变 | 快 | 可调 |

### 使用量化模型

```java
/**
 * 量化模型加载
 */
public class QuantizedModelLoader {
    
    /**
     * 根据显存选择量化级别
     */
    public String selectQuantization(int gpuMemoryGB, int modelSizeB) {
        int requiredMemory = modelSizeB * 2; // FP16
        
        if (requiredMemory <= gpuMemoryGB) {
            return "fp16";  // 使用全精度
        } else if (requiredMemory / 2 <= gpuMemoryGB) {
            return "int8";  // 8位量化
        } else if (requiredMemory / 4 <= gpuMemoryGB) {
            return "int4";  // 4位量化
        } else {
            throw new IllegalArgumentException(
                "显存不足，无法运行该模型");
        }
    }
}
```

## 小结

本章我们学习了：

1. **开源模型生态**：LLaMA、Mistral、ChatGLM等
2. **模型选择**：根据场景和硬件选择合适模型
3. **模型运行**：Ollama、Hugging Face等工具
4. **模型量化**：减少显存占用，提高运行效率

**关键认识：**
开源大模型让AI能力民主化，Java程序员可以在本地构建AI应用。

**下一步：** 我们将学习LangChain4j框架。

---

**练习题：**

1. 开源模型和闭源API各有什么优缺点？
2. 如何根据硬件配置选择模型和量化级别？
3. 在你的机器上尝试运行Ollama。

---

<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-08/05-generation-vs-understanding.md">&larr; 8.5 设计思考：生成式与理解式模型</a></td>
      <td align="right"><a href="02-model-quantization.md">9.2 模型量化：让大模型变小 &rarr;</a></td>
   </tr>
</table>