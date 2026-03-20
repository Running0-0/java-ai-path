# 17.5 部署与优化

> "从开发到生产——让你的AI助手稳定运行。"

## Docker部署

### Dockerfile

```dockerfile
FROM openjdk:17-jdk-slim

WORKDIR /app

COPY target/assistant-*.jar app.jar
COPY config/ config/

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  assistant:
    build: .
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=prod
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./data:/app/data
    depends_on:
      - ollama
      - chroma

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"

  chroma:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8000:8000"

volumes:
  ollama_data:
  chroma_data:
```

## 性能优化

### 优化策略

```
1. 模型优化
   - 使用量化模型
   - 选择合适的模型大小
   - 模型缓存

2. 内存优化
   - 向量存储索引
   - 记忆压缩
   - 缓存策略

3. 并发优化
   - 异步处理
   - 连接池
   - 限流
```

## 监控

### 指标收集

```java
@Component
public class AssistantMetrics {
    
    private final MeterRegistry registry;
    
    public void recordRequest(String type, long latency) {
        registry.counter("assistant.requests", "type", type).increment();
        registry.timer("assistant.latency", "type", type)
            .record(latency, TimeUnit.MILLISECONDS);
    }
}
```

## 小结

本章我们学习了：

1. **Docker部署**：容器化配置
2. **性能优化**：模型、内存、并发
3. **监控**：指标收集

---

**总结：**

恭喜你完成了《Java程序员的AI之路》的学习！

从深度学习基础到大语言模型，再到智能体开发，你已经掌握了构建AI应用的完整技能栈。

现在，开始构建你自己的AI应用吧！
