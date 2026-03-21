<table width="100%">
   <tr>
      <td align="left"><a href="03-first-ai-environment.md">← 1.3 搭建你的第一个AI开发环境</a></td>
      <td align="right"><a href="../chapter-02/01-perceptron-to-neural-network.md">2.1 从感知机到神经网络 →</a></td>
   </tr>
</table>

---

# 1.4 Java AI生态全景图

> "了解生态，才能选对工具。Java AI生态虽然不如Python丰富，但在企业级应用中有着独特优势。"

## Java AI生态概览

让我们用一个全景图来了解Java AI生态的各个层次：

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层 (Applications)                   │
│   智能客服 │ 文档分析 │ 图像识别 │ 推荐系统 │ 风控系统      │
├─────────────────────────────────────────────────────────────┤
│                      框架层 (Frameworks)                     │
│  LangChain4j │ Spring AI │ Deeplearning4j │ DJL             │
├─────────────────────────────────────────────────────────────┤
│                      计算层 (Computation)                    │
│           ND4J │ Tribuo │ Weka │ Encog                      │
├─────────────────────────────────────────────────────────────┤
│                      推理层 (Inference)                      │
│    ONNX Runtime │ TensorFlow Java │ PyTorch Java            │
├─────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)             │
│        向量数据库 │ 消息队列 │ 缓存 │ 监控                   │
└─────────────────────────────────────────────────────────────┘
```

## 深度学习框架

### Deeplearning4j (DL4J)

**定位：** Java生态最成熟的深度学习框架

**特点：**
- 完整的神经网络构建工具
- 支持分布式训练
- 与Hadoop、Spark集成良好
- 生产级别的性能优化

**适用场景：**
- 图像分类
- 文本分类
- 时间序列预测
- 推荐系统

**代码示例：**
```java
// 构建一个简单的多层感知机
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .weightInit(WeightInit.XAVIER)
    .updater(new Adam(0.001))
    .list()
    .layer(new DenseLayer.Builder()
        .nIn(784)  // MNIST输入
        .nOut(256)
        .activation(Activation.RELU)
        .build())
    .layer(new DenseLayer.Builder()
        .nOut(128)
        .activation(Activation.RELU)
        .build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(10)  // 10个数字类别
        .activation(Activation.SOFTMAX)
        .build())
    .build();
```

### DJL (Deep Java Library)

**定位：** Amazon开源的深度学习引擎统一接口

**特点：**
- 引擎无关的API设计
- 支持多种后端：PyTorch、TensorFlow、MXNet、ONNX
- 简单易用的API
- 与Spring Boot集成良好

**适用场景：**
- 模型推理（使用预训练模型）
- 快速原型开发
- 多引擎切换需求

**代码示例：**
```java
// 使用DJL加载预训练模型进行图像分类
Criteria<Image, Classifications> criteria = Criteria.builder()
    .setTypes(Image.class, Classifications.class)
    .optModelUrls("djl://ai.djl.pytorch/resnet18")
    .build();

ZooModel<Image, Classifications> model = criteria.loadModel();
Predictor<Image, Classifications> predictor = model.newPredictor();

Image img = ImageFactory.getInstance().fromUrl("cat.jpg");
Classifications result = predictor.predict(img);
System.out.println(result.best().getClassName());
```

### 框架对比

| 特性 | Deeplearning4j | DJL |
|------|----------------|-----|
| 核心优势 | 完整训练能力 | 多引擎支持 |
| 学习曲线 | 较陡 | 平缓 |
| 训练支持 | 完整 | 有限 |
| 推理支持 | 完整 | 优秀 |
| 社区活跃度 | 中等 | 活跃 |
| 企业支持 | Eclipse基金会 | Amazon |

**选择建议：**
- 需要从头训练模型 → Deeplearning4j
- 主要使用预训练模型 → DJL
- 需要灵活切换引擎 → DJL

## 大语言模型框架

### LangChain4j

**定位：** Java版LangChain，LLM应用开发首选

**特点：**
- 支持多种LLM：OpenAI、Azure、本地模型
- 内置RAG支持
- 工具调用（Function Calling）
- 记忆管理
- 与Quarkus、Spring Boot集成

**核心组件：**

```java
// 1. 配置LLM
ChatLanguageModel model = OpenAiChatModel.builder()
    .apiKey(System.getenv("OPENAI_API_KEY"))
    .modelName("gpt-4")
    .temperature(0.7)
    .build();

// 2. 创建AI服务
interface Assistant {
    @SystemMessage("你是一个专业的Java编程助手")
    String chat(@UserMessage String userMessage);
}

Assistant assistant = AiServices.create(Assistant.class, model);

// 3. 对话
String response = assistant.chat("如何优化Java应用性能？");
```

### Spring AI

**定位：** Spring官方AI框架

**特点：**
- Spring生态原生集成
- 统一的API抽象
- 向量数据库支持
- 提示模板管理

**代码示例：**
```java
@Service
public class ChatService {
    
    private final ChatClient chatClient;
    
    public ChatService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }
    
    public String chat(String message) {
        return chatClient.call(message);
    }
}
```

### LLM框架对比

| 特性 | LangChain4j | Spring AI |
|------|-------------|-----------|
| 依赖 | 轻量 | Spring生态 |
| 学习曲线 | 中等 | Spring用户友好 |
| 功能完整度 | 高 | 快速发展中 |
| 文档质量 | 优秀 | 良好 |
| 社区 | 活跃 | 官方支持 |

## 向量数据库客户端

### 为什么需要向量数据库？

在AI应用中，向量数据库用于：
- 语义搜索
- RAG（检索增强生成）
- 推荐系统
- 相似度匹配

### Java支持的向量数据库

| 数据库 | Java客户端 | 特点 |
|--------|------------|------|
| Milvus | milvus-sdk-java | 高性能，分布式 |
| Pinecone | pinecone-java | 云原生，简单易用 |
| Chroma | chromadb-java | 轻量级，本地部署 |
| Weaviate | weaviate-client | 语义理解强 |
| Qdrant | qdrant-java | Rust实现，高性能 |

**LangChain4j集成示例：**
```java
// 配置嵌入模型
EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

// 配置向量存储
EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

// 嵌入文档
TextSegment segment = TextSegment.from("Java是一门面向对象的编程语言");
Embedding embedding = embeddingModel.embed(segment).content();
embeddingStore.add(embedding, segment);

// 语义搜索
Embedding query = embeddingModel.embed("什么是Java？").content();
List<EmbeddingMatch<TextSegment>> results = embeddingStore.findRelevant(query, 5);
```

## 传统机器学习库

### Weka

**定位：** 经典机器学习工具包

**特点：**
- 历史悠久，文档丰富
- GUI工具支持
- 大量算法实现

**适用场景：**
- 数据挖掘
- 教学学习
- 快速原型

### Tribuo

**定位：** Oracle开源的机器学习库

**特点：**
- 纯Java实现
- 支持分类、回归、聚类
- ONNX模型导入

**代码示例：**
```java
// 加载数据
DataSource<?> dataSource = new CSVLoader().loadDataSource(Paths.get("data.csv"), "label");

// 训练模型
Trainer<Label> trainer = new LogisticRegressionTrainer();
Model<Label> model = trainer.train(dataSource);

// 预测
Prediction<Label> prediction = model.predict(testData);
```

## 模型推理引擎

### ONNX Runtime Java

**定位：** 跨平台模型推理

**特点：**
- 支持PyTorch、TensorFlow导出的模型
- 高性能推理
- 跨平台一致性

**代码示例：**
```java
// 加载ONNX模型
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("model.onnx");

// 准备输入
OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);

// 推理
OrtSession.Result result = session.run(inputTensor);
float[] output = ((OnnxTensor) result.get(0)).getFloatBuffer().array();
```

## 设计思考：如何选择合适的工具

### 决策树

```
你的需求是什么？
│
├─ 训练自定义模型
│   └─ Deeplearning4j
│
├─ 使用预训练模型
│   ├─ 图像/视频 → DJL
│   └─ NLP/LLM → LangChain4j
│
├─ 传统机器学习
│   ├─ 生产环境 → Tribuo
│   └─ 学习/原型 → Weka
│
└─ LLM应用开发
    ├─ Spring项目 → Spring AI
    └─ 其他 → LangChain4j
```

### 组合使用

实际项目中，往往需要组合使用多个工具：

```java
// 典型的AI应用架构
public class AIApplication {
    
    // 1. 使用LangChain4j处理LLM交互
    private ChatLanguageModel llm;
    
    // 2. 使用向量数据库存储嵌入
    private EmbeddingStore<TextSegment> vectorStore;
    
    // 3. 使用DJL处理图像
    private ZooModel<Image, Classifications> visionModel;
    
    // 4. 使用Deeplearning4j处理自定义模型
    private MultiLayerNetwork customModel;
    
    public String processQuery(String query, Image image) {
        // 图像理解
        String imageContext = analyzeImage(image);
        
        // RAG检索
        String relevantContext = retrieveContext(query);
        
        // LLM生成回答
        String prompt = buildPrompt(query, imageContext, relevantContext);
        return llm.generate(prompt);
    }
}
```

## 生态发展趋势

### 近期热点

1. **LangChain4j快速发展**：功能越来越完善
2. **Spring AI正式发布**：Spring官方支持
3. **本地模型支持增强**：Ollama、LocalAI集成
4. **多模态支持**：图像、音频处理能力增强

### 未来展望

| 方向 | 预期发展 |
|------|----------|
| LLM应用 | 更多企业级特性 |
| 模型部署 | 更简化的部署流程 |
| 多模态 | 更完善的跨模态支持 |
| Agent | 更强大的智能体能力 |

## 小结

本章我们了解了：

1. **Java AI生态全貌**：从计算层到应用层
2. **核心框架**：DL4J、DJL、LangChain4j、Spring AI
3. **选型指南**：根据需求选择合适工具
4. **发展趋势**：关注LLM应用和Agent方向

**框架速查表：**

| 需求 | 推荐框架 |
|------|----------|
| 深度学习训练 | Deeplearning4j |
| 模型推理 | DJL / ONNX Runtime |
| LLM应用 | LangChain4j |
| Spring集成 | Spring AI |
| 传统ML | Tribuo / Weka |
| 向量存储 | Milvus / Chroma |

**下一步：** 我们将深入神经网络的原理，理解深度学习的核心机制。

---

**思考题：**

1. 根据你的项目需求，应该选择哪个框架？
2. LangChain4j和Spring AI各有什么优劣？
3. 如果要在生产环境部署AI应用，需要考虑哪些因素？

---

<table width="100%">
   <tr>
      <td align="left"><a href="03-first-ai-environment.md">← 1.3 搭建你的第一个AI开发环境</a></td>
      <td align="right"><a href="../chapter-02/01-perceptron-to-neural-network.md">2.1 从感知机到神经网络 →</a></td>
   </tr>
</table>