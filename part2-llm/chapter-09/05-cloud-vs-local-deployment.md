# 9.5 设计思考：云端vs本地部署

> "云端还是本地？这不是二选一的问题，而是根据场景做最优选择。"

## 部署方式对比

### 云端部署（API）

```
代表：OpenAI API、Claude API、文心一言API

优势：
✓ 无需硬件投入
✓ 模型持续更新
✓ 弹性扩展
✓ 运维简单

劣势：
✗ 数据离开本地
✗ 网络依赖
✗ 按量计费，成本不可控
✗ 延迟较高
```

### 本地部署

```
代表：Ollama、llama.cpp、vLLM

优势：
✓ 数据隐私安全
✓ 离线可用
✓ 一次性投入
✓ 低延迟

劣势：
✗ 需要硬件资源
✗ 模型维护成本
✗ 扩展性受限
✗ 技术门槛较高
```

## 决策框架

### 选择矩阵

| 考量因素 | 云端 | 本地 |
|---------|------|------|
| 数据隐私 | ⭐ | ⭐⭐⭐ |
| 成本可控 | ⭐⭐ | ⭐⭐⭐ |
| 易用性 | ⭐⭐⭐ | ⭐⭐ |
| 延迟 | ⭐⭐ | ⭐⭐⭐ |
| 模型能力 | ⭐⭐⭐ | ⭐⭐ |
| 扩展性 | ⭐⭐⭐ | ⭐⭐ |

### 决策树

```
数据是否敏感？
├── 是 → 本地部署
└── 否
    ├── 是否有稳定网络？
    │   ├── 否 → 本地部署
    │   └── 是
    │       ├── 预算是否充足？
    │       ├── 是 → 云端部署
    │       └── 否 → 混合部署
    └── 延迟要求是否严格？
        ├── 是 → 本地部署
        └── 否 → 云端部署
```

## 混合部署策略

### 分层架构

```
混合部署方案：

敏感数据处理层（本地）：
- 用户数据脱敏
- 隐私计算
- 本地缓存

AI能力层（混合）：
- 简单任务 → 本地小模型
- 复杂任务 → 云端大模型
- 实时任务 → 本地模型

结果处理层（本地）：
- 结果组装
- 格式转换
- 安全过滤
```

### Java实现

```java
/**
 * 混合部署路由器
 */
public class HybridDeploymentRouter {
    
    private final ChatLanguageModel localModel;
    private final ChatLanguageModel cloudModel;
    
    /**
     * 智能路由
     */
    public String generate(String prompt, TaskType type) {
        return switch (type) {
            case SENSITIVE -> localModel.generate(prompt);
            case COMPLEX -> cloudModel.generate(prompt);
            case REALTIME -> localModel.generate(prompt);
            case ROUTINE -> decideByLoad(prompt);
        };
    }
    
    /**
     * 根据负载决定
     */
    private String decideByLoad(String prompt) {
        if (isLocalModelOverloaded()) {
            return cloudModel.generate(prompt);
        }
        return localModel.generate(prompt);
    }
    
    /**
     * 降级策略
     */
    public String generateWithFallback(String prompt) {
        try {
            // 优先使用云端
            return cloudModel.generate(prompt);
        } catch (Exception e) {
            // 云端失败，降级到本地
            System.out.println("云端服务不可用，切换到本地模型");
            return localModel.generate(prompt);
        }
    }
}

enum TaskType {
    SENSITIVE,   // 敏感数据
    COMPLEX,     // 复杂任务
    REALTIME,    // 实时任务
    ROUTINE      // 常规任务
}
```

## 成本分析

### TCO对比（总拥有成本）

```
云端成本（3年）：
- API调用费：$0.002/1K tokens
- 假设每月100M tokens
- 3年成本：$0.002 × 100,000 × 36 = $7,200

本地成本（3年）：
- GPU服务器：$3,000（一次性）
- 电费：$50/月 × 36 = $1,800
- 维护：$100/月 × 36 = $3,600
- 3年成本：$8,400

结论：
- 小规模使用：云端更便宜
- 大规模使用：本地更划算
- 中等规模：成本相近，考虑其他因素
```

### Java成本监控

```java
/**
 * 成本监控器
 */
public class CostMonitor {
    
    private final Map<String, AtomicInteger> tokenUsage = 
        new ConcurrentHashMap<>();
    
    /**
     * 记录使用
     */
    public void recordUsage(String model, int tokens) {
        tokenUsage.computeIfAbsent(model, k -> new AtomicInteger())
            .addAndGet(tokens);
    }
    
    /**
     * 计算月度成本
     */
    public double calculateMonthlyCost() {
        double cost = 0;
        
        // GPT-3.5: $0.002/1K tokens
        cost += tokenUsage.getOrDefault("gpt-3.5", new AtomicInteger())
            .get() / 1000.0 * 0.002;
        
        // GPT-4: $0.03/1K tokens
        cost += tokenUsage.getOrDefault("gpt-4", new AtomicInteger())
            .get() / 1000.0 * 0.03;
        
        return cost;
    }
}
```

## 安全考虑

### 数据安全

```
云端安全措施：
1. 数据加密传输（TLS）
2. 敏感数据脱敏
3. 访问控制和审计
4. 数据保留策略

本地安全措施：
1. 模型文件加密
2. 运行时内存保护
3. 访问日志记录
4. 定期安全更新
```

### 合规要求

```java
/**
 * 合规检查
 */
public class ComplianceChecker {
    
    /**
     * 检查是否可以发送到云端
     */
    public boolean canSendToCloud(String data, CompliancePolicy policy) {
        // 检查PII（个人身份信息）
        if (containsPII(data) && !policy.allowPII()) {
            return false;
        }
        
        // 检查敏感关键词
        if (containsSensitiveKeywords(data, policy.getKeywords())) {
            return false;
        }
        
        // 检查数据分类
        DataClassification classification = classifyData(data);
        return classification.ordinal() <= policy.getMaxClassification().ordinal();
    }
    
    /**
     * 数据脱敏
     */
    public String anonymize(String data) {
        // 移除姓名
        data = data.replaceAll("\\b[A-Z][a-z]+ [A-Z][a-z]+\\b", "[NAME]");
        
        // 移除手机号
        data = data.replaceAll("\\b1[3-9]\\d{9}\\b", "[PHONE]");
        
        // 移除邮箱
        data = data.replaceAll("\\b\\w+@\\w+\\.\\w+\\b", "[EMAIL]");
        
        return data;
    }
}
```

## 小结

本章我们思考了：

1. **部署方式对比**：云端vs本地的优劣势
2. **决策框架**：如何根据场景选择
3. **混合策略**：结合两者优势
4. **成本分析**：TCO计算
5. **安全合规**：数据保护

**关键认识：**
没有最好的部署方式，只有最适合的。混合部署可能是大多数企业的最佳选择。

**下一步：** 我们将学习提示工程。

---

**思考题：**

1. 你的应用场景适合云端还是本地部署？
2. 如何设计一个自动切换的混合部署系统？
3. 数据合规对部署方式有什么影响？
