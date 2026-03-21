<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 17.1 项目规划](01-project-planning.md)</span>

<span>[17.3 工具集成 &rarr;](03-tool-integration.md)</span>

</div>
---

# 17.2 核心能力实现

> "核心能力是AI助手的灵魂——对话、记忆、工具的有机融合。"

## 对话引擎

### 核心实现

```java
package com.example.assistant.core;

/**
 * 对话引擎
 */
public class ConversationEngine {
    
    private final ChatLanguageModel llm;
    private final MemoryManager memory;
    private final IntentClassifier classifier;
    
    /**
     * 处理用户输入
     */
    public String process(String userInput) {
        // 1. 意图识别
        Intent intent = classifier.classify(userInput);
        
        // 2. 检索相关记忆
        MemoryContext context = memory.retrieve(userInput);
        
        // 3. 构建提示
        String prompt = buildPrompt(userInput, intent, context);
        
        // 4. 生成回复
        String response = llm.generate(prompt);
        
        // 5. 更新记忆
        memory.addInteraction(userInput, response);
        
        return response;
    }
    
    private String buildPrompt(String input, Intent intent, 
                               MemoryContext context) {
        StringBuilder prompt = new StringBuilder();
        
        // 系统提示
        prompt.append("你是一个个人AI助手，帮助用户处理各种任务。\n\n");
        
        // 相关记忆
        if (!context.getMemories().isEmpty()) {
            prompt.append("相关背景：\n");
            context.getMemories().forEach(m -> 
                prompt.append("- ").append(m).append("\n"));
            prompt.append("\n");
        }
        
        // 当前对话
        prompt.append("用户：").append(input).append("\n");
        prompt.append("助手：");
        
        return prompt.toString();
    }
}
```

## 记忆管理

### 记忆管理器

```java
/**
 * 记忆管理器
 */
public class MemoryManager {
    
    private final WorkingMemory workingMemory;
    private final LongTermMemory longTermMemory;
    
    /**
     * 检索相关记忆
     */
    public MemoryContext retrieve(String query) {
        MemoryContext context = new MemoryContext();
        
        // 工作记忆
        context.addAll(workingMemory.getRecent(5));
        
        // 长期记忆
        List<MemoryEntry> relevant = longTermMemory.search(query, 3);
        context.addAll(relevant);
        
        return context;
    }
    
    /**
     * 添加交互
     */
    public void addInteraction(String userInput, String response) {
        // 添加到工作记忆
        workingMemory.add(new MemoryEntry("用户：" + userInput));
        workingMemory.add(new MemoryEntry("助手：" + response));
        
        // 提取重要信息到长期记忆
        if (isImportant(userInput)) {
            longTermMemory.add(new MemoryEntry(userInput));
        }
    }
}
```

## 工具集成

### 工具管理器

```java
/**
 * 工具管理器
 */
public class ToolManager {
    
    private final ToolRegistry registry;
    private final LLMToolSelector selector;
    
    /**
     * 执行工具调用
     */
    public String executeTools(String userInput) {
        // 1. 选择需要调用的工具
        List<ToolCall> calls = selector.select(userInput, 
            registry.getAvailableTools());
        
        // 2. 执行工具
        List<ToolResult> results = new ArrayList<>();
        for (ToolCall call : calls) {
            Tool tool = registry.getTool(call.getToolName());
            ToolResult result = tool.execute(call.getParameters());
            results.add(result);
        }
        
        // 3. 格式化结果
        return formatResults(results);
    }
}
```

## 小结

本章我们实现了：

1. **对话引擎**：意图识别、记忆检索、回复生成
2. **记忆管理**：工作记忆和长期记忆的管理
3. **工具集成**：工具选择和执行

**下一步：** 我们将完善工具集成。

---

**练习题：**

1. 优化对话引擎的响应速度
2. 实现记忆的自动总结功能
3. 添加更多实用工具。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 17.1 项目规划](01-project-planning.md)</span>

<span>[17.3 工具集成 &rarr;](03-tool-integration.md)</span>

</div>