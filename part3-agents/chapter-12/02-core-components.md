# 12.2 智能体核心组件

> "智能体的能力来自于组件的协同——每个组件都是关键的一环。"

## 组件概览

### 核心组件图

```
智能体核心组件：

┌─────────────────────────────────────────┐
│              智能体核心                  │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  大脑   │  │  记忆   │  │  工具   │ │
│  │  (LLM) │  │         │  │         │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       └─────────────┴─────────────┘     │
│                   │                     │
│              ┌────┴────┐                │
│              │  规划器  │                │
│              │         │                │
│              └────┬────┘                │
│                   │                     │
│       ┌───────────┼───────────┐         │
│  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐   │
│  │ 执行器  │ │ 感知器  │ │ 学习器  │   │
│  │         │ │         │ │         │   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
```

## 大脑（Brain）

### LLM作为大脑

```java
/**
 * 智能体大脑
 */
public class AgentBrain {
    
    private final ChatLanguageModel llm;
    private final PromptTemplate systemPrompt;
    
    /**
     * 思考
     */
    public Thought think(Context context) {
        String prompt = buildPrompt(context);
        String response = llm.generate(prompt);
        return parseThought(response);
    }
    
    /**
     * 决策
     */
    public Decision decide(Thought thought, List<Action> availableActions) {
        String prompt = String.format("""
            基于以下思考，选择最佳行动：
            
            思考：%s
            
            可选行动：
            %s
            
            请输出行动编号和理由。
            """, thought.getContent(), formatActions(availableActions));
        
        String response = llm.generate(prompt);
        return parseDecision(response);
    }
    
    /**
     * 反思
     */
    public Reflection reflect(ActionResult result, Goal goal) {
        String prompt = String.format("""
            评估行动结果：
            
            目标：%s
            行动结果：%s
            
            请分析：
            1. 是否达成目标？
            2. 有哪些问题？
            3. 如何改进？
            """, goal.getDescription(), result.getDescription());
        
        String response = llm.generate(prompt);
        return parseReflection(response);
    }
}
```

## 记忆系统

### 记忆类型

```
记忆层次：

1. 工作记忆（Working Memory）
   - 当前对话上下文
   - 短期信息
   - 容量有限

2. 短期记忆（Short-term Memory）
   - 当前任务相关信息
   - 会话历史
   - 可压缩总结

3. 长期记忆（Long-term Memory）
   - 用户偏好
   - 知识库
   - 经验积累
```

### Java实现

```java
package com.example.agent.memory;

/**
 * 分层记忆系统
 */
public class HierarchicalMemory {
    
    // 工作记忆：当前上下文
    private final Deque<Message> workingMemory;
    private static final int WORKING_MEMORY_SIZE = 10;
    
    // 短期记忆：会话历史
    private final List<Message> shortTermMemory;
    
    // 长期记忆：向量存储
    private final EmbeddingStore<TextSegment> longTermMemory;
    private final EmbeddingModel embeddingModel;
    
    /**
     * 添加消息
     */
    public void addMessage(Message message) {
        // 添加到工作记忆
        workingMemory.addLast(message);
        if (workingMemory.size() > WORKING_MEMORY_SIZE) {
            // 溢出到短期记忆
            Message old = workingMemory.removeFirst();
            shortTermMemory.add(old);
        }
        
        // 短期记忆定期压缩到长期记忆
        if (shortTermMemory.size() > 50) {
            compressToLongTerm();
        }
    }
    
    /**
     * 检索相关记忆
     */
    public List<String> retrieveRelevant(String query, int k) {
        Embedding queryEmbedding = embeddingModel.embed(query);
        
        // 从长期记忆检索
        List<EmbeddingMatch<TextSegment>> matches = 
            longTermMemory.findRelevant(queryEmbedding, k);
        
        return matches.stream()
            .map(m -> m.embedded().text())
            .collect(Collectors.toList());
    }
    
    /**
     * 压缩到长期记忆
     */
    private void compressToLongTerm() {
        // 总结短期记忆
        String summary = summarize(shortTermMemory);
        
        // 嵌入并存储
        Embedding embedding = embeddingModel.embed(summary);
        longTermMemory.add(embedding, TextSegment.from(summary));
        
        // 清空短期记忆
        shortTermMemory.clear();
    }
}
```

## 工具系统

### 工具定义

```java
package com.example.agent.tools;

/**
 * 工具接口
 */
public interface Tool {
    
    /**
     * 工具名称
     */
    String getName();
    
    /**
     * 工具描述
     */
    String getDescription();
    
    /**
     * 参数模式（JSON Schema）
     */
    JsonSchema getParameterSchema();
    
    /**
     * 执行工具
     */
    ToolResult execute(Map<String, Object> parameters);
}

/**
 * 工具注册表
 */
public class ToolRegistry {
    
    private final Map<String, Tool> tools = new HashMap<>();
    
    public void register(Tool tool) {
        tools.put(tool.getName(), tool);
    }
    
    public Tool getTool(String name) {
        return tools.get(name);
    }
    
    public List<Tool> getAllTools() {
        return new ArrayList<>(tools.values());
    }
    
    public String formatToolsForPrompt() {
        StringBuilder sb = new StringBuilder();
        for (Tool tool : tools.values()) {
            sb.append(String.format("- %s: %s%n", 
                tool.getName(), tool.getDescription()));
        }
        return sb.toString();
    }
}
```

## 规划器

### 任务规划

```java
package com.example.agent.planner;

/**
 * 任务规划器
 */
public class TaskPlanner {
    
    private final ChatLanguageModel llm;
    
    /**
     * 制定计划
     */
    public Plan createPlan(Goal goal, List<Tool> availableTools) {
        String prompt = String.format("""
            为实现以下目标，制定详细计划：
            
            目标：%s
            
            可用工具：
            %s
            
            请将目标分解为具体步骤，
            每个步骤说明使用什么工具。
            输出JSON格式计划。
            """, goal.getDescription(), formatTools(availableTools));
        
        String response = llm.generate(prompt);
        return parsePlan(response);
    }
    
    /**
     * 动态调整计划
     */
    public Plan adjustPlan(Plan currentPlan, ActionResult lastResult) {
        if (lastResult.isSuccess()) {
            return currentPlan.nextStep();
        }
        
        // 失败时重新规划
        String prompt = String.format("""
            原计划执行失败：
            计划：%s
            失败原因：%s
            
            请调整计划以解决问题。
            """, currentPlan, lastResult.getError());
        
        String response = llm.generate(prompt);
        return parsePlan(response);
    }
}

/**
 * 计划
 */
public class Plan {
    private final List<Step> steps;
    private int currentStep;
    
    public Step getCurrentStep() {
        return steps.get(currentStep);
    }
    
    public Plan nextStep() {
        currentStep++;
        return this;
    }
    
    public boolean isComplete() {
        return currentStep >= steps.size();
    }
}
```

## 执行器

### 行动执行

```java
package com.example.agent.executor;

/**
 * 行动执行器
 */
public class ActionExecutor {
    
    private final ToolRegistry toolRegistry;
    
    /**
     * 执行行动
     */
    public ActionResult execute(Action action) {
        try {
            if (action.isToolCall()) {
                // 执行工具调用
                Tool tool = toolRegistry.getTool(action.getToolName());
                ToolResult result = tool.execute(action.getParameters());
                return ActionResult.toolResult(result);
            } else {
                // 直接输出
                return ActionResult.output(action.getContent());
            }
        } catch (Exception e) {
            return ActionResult.failure(e.getMessage());
        }
    }
    
    /**
     * 带重试的执行
     */
    public ActionResult executeWithRetry(Action action, int maxRetries) {
        int attempts = 0;
        
        while (attempts < maxRetries) {
            ActionResult result = execute(action);
            if (result.isSuccess()) {
                return result;
            }
            
            attempts++;
            if (attempts < maxRetries) {
                // 等待后重试
                sleep(1000 * attempts);
            }
        }
        
        return ActionResult.failure("达到最大重试次数");
    }
}
```

## 小结

本章我们学习了：

1. **大脑**：LLM作为智能体的推理核心
2. **记忆系统**：分层记忆，工作/短期/长期
3. **工具系统**：扩展智能体能力的接口
4. **规划器**：任务分解和动态调整
5. **执行器**：行动执行和错误处理

**关键认识：**
智能体的能力来自于各组件的协同工作，每个组件都不可或缺。

**下一步：** 我们将学习智能体与LLM的关系。

---

**练习题：**

1. 为什么智能体需要分层记忆系统？
2. 工具系统如何扩展智能体的能力？
3. 设计一个支持动态规划调整的智能体。
