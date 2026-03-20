# 15.4 构建记忆智能体

> "一个拥有完整记忆系统的智能体——从工作记忆到长期知识。"

## 记忆智能体架构

### 整体架构

```
记忆智能体架构：

用户输入
    ↓
[输入处理器]
    - 实体提取
    - 意图识别
    ↓
[记忆检索器]
    - 工作记忆查询
    - 短期记忆查询
    - 长期记忆检索
    ↓
[上下文构建器]
    - 整合记忆
    - 构建提示
    ↓
[LLM处理]
    - 生成响应
    - 提取新信息
    ↓
[记忆更新器]
    - 更新工作记忆
    - 保存到短期记忆
    - 重要信息入长期记忆
    ↓
输出给用户
```

## Java实现

### 记忆智能体

```java
package com.example.agent.memory;

/**
 * 记忆智能体
 */
public class MemoryAgent {
    
    private final HierarchicalMemorySystem memory;
    private final ChatLanguageModel llm;
    private final EntityExtractor entityExtractor;
    
    /**
     * 处理用户输入
     */
    public String process(String userInput) {
        // 1. 提取实体
        List<Entity> entities = entityExtractor.extract(userInput);
        
        // 2. 检索相关记忆
        MemoryContext context = retrieveRelevantMemory(userInput, entities);
        
        // 3. 构建提示
        String prompt = buildPrompt(userInput, context);
        
        // 4. 生成响应
        String response = llm.generate(prompt);
        
        // 5. 更新记忆
        updateMemory(userInput, response, entities);
        
        return response;
    }
    
    /**
     * 检索相关记忆
     */
    private MemoryContext retrieveRelevantMemory(String input, 
                                                  List<Entity> entities) {
        MemoryContext context = new MemoryContext();
        
        // 工作记忆：最近对话
        context.setWorkingMemory(memory.getWorkingMemory().getRecent(5));
        
        // 短期记忆：当前话题
        context.setShortTermMemory(
            memory.getShortTermMemory().getTopicRelated(input));
        
        // 长期记忆：实体相关
        for (Entity entity : entities) {
            List<MemoryEntry> related = memory.getLongTermMemory()
                .retrieve(entity.getName());
            context.addLongTermMemories(related);
        }
        
        return context;
    }
    
    /**
     * 更新记忆
     */
    private void updateMemory(String input, String response, 
                              List<Entity> entities) {
        // 添加到工作记忆
        memory.getWorkingMemory().add(new MemoryEntry(input, 
            MemoryRole.USER));
        memory.getWorkingMemory().add(new MemoryEntry(response, 
            MemoryRole.ASSISTANT));
        
        // 提取重要信息到长期记忆
        String importantInfo = extractImportantInfo(input, response);
        if (importantInfo != null) {
            memory.getLongTermMemory().add(
                new MemoryEntry(importantInfo, MemoryRole.FACT));
        }
        
        // 保存实体信息
        for (Entity entity : entities) {
            memory.getEntityMemory().update(entity);
        }
    }
}
```

## 使用示例

### 个人助手

```java
/**
 * 个人记忆助手示例
 */
public class PersonalAssistantDemo {
    
    public void demonstrate() {
        MemoryAgent agent = new MemoryAgent(
            memorySystem,
            llm,
            entityExtractor
        );
        
        // 第一轮对话
        System.out.println(agent.process("我叫张三，喜欢Java编程"));
        // 智能体记住用户姓名和偏好
        
        // 第二轮对话（跨会话）
        System.out.println(agent.process("推荐一些Java学习资源"));
        // 智能体记得用户喜欢Java，推荐相关资源
        
        // 第三轮对话
        System.out.println(agent.process("我昨天说的那个项目怎么样了？"));
        // 智能体检索之前的对话上下文
    }
}
```

## 小结

本章我们学习了：

1. **记忆智能体架构**：完整的记忆系统架构
2. **Java实现**：MemoryAgent实现
3. **使用示例**：个人助手演示

**关键认识：**
记忆智能体能够提供个性化、连贯的服务体验。

**下一步：** 我们将学习遗忘的艺术。

---

**练习题：**

1. 设计一个记忆重要性评分算法
2. 如何实现记忆的隐私控制？
3. 设计跨设备的记忆同步方案。
