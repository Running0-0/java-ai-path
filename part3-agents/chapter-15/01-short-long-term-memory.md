<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 14.5 设计思考：推理边界](../chapter-14/05-reasoning-boundaries.md)</span>

<span>[15.2 对话记忆管理 &rarr;](02-dialog-memory-management.md)</span>

</div>
---

# 15.1 短期与长期记忆

> "记忆是智能体的灵魂——没有记忆，智能体只是无状态的函数调用。"

## 记忆的重要性

### 为什么需要记忆

```
无记忆的问题：
- 每次对话都是全新的
- 无法记住用户偏好
- 不能从经验中学习
- 上下文容易丢失

有记忆的优势：
- 个性化体验
- 持续学习
- 上下文连贯
- 知识积累
```

## 记忆类型

### 记忆层次

```
记忆层次结构：

1. 工作记忆（Working Memory）
   - 当前对话轮次
   - 即时上下文
   - 容量：几轮对话
   - 存储：内存

2. 短期记忆（Short-term Memory）
   - 当前会话历史
   - 任务相关信息
   - 容量：一次会话
   - 存储：内存/缓存

3. 长期记忆（Long-term Memory）
   - 跨会话知识
   - 用户画像
   - 经验积累
   - 存储：数据库/向量存储
```

## Java实现

### 分层记忆系统

```java
package com.example.agent.memory;

/**
 * 分层记忆系统
 */
public class HierarchicalMemorySystem {
    
    private final WorkingMemory workingMemory;
    private final ShortTermMemory shortTermMemory;
    private final LongTermMemory longTermMemory;
    
    public HierarchicalMemorySystem(
            WorkingMemory working,
            ShortTermMemory shortTerm,
            LongTermMemory longTerm) {
        this.workingMemory = working;
        this.shortTermMemory = shortTerm;
        this.longTermMemory = longTerm;
    }
    
    /**
     * 添加信息到记忆
     */
    public void add(MemoryEntry entry, MemoryLevel level) {
        switch (level) {
            case WORKING -> workingMemory.add(entry);
            case SHORT_TERM -> shortTermMemory.add(entry);
            case LONG_TERM -> longTermMemory.add(entry);
        }
    }
    
    /**
     * 检索相关信息
     */
    public List<MemoryEntry> retrieve(String query) {
        List<MemoryEntry> results = new ArrayList<>();
        
        // 按优先级检索
        results.addAll(workingMemory.retrieve(query));
        results.addAll(shortTermMemory.retrieve(query));
        results.addAll(longTermMemory.retrieve(query));
        
        return results;
    }
    
    /**
     * 记忆流转
     */
    public void consolidate() {
        // 工作记忆 → 短期记忆
        List<MemoryEntry> toShortTerm = workingMemory.getOverflow();
        shortTermMemory.addAll(toShortTerm);
        
        // 短期记忆 → 长期记忆
        if (shortTermMemory.needsConsolidation()) {
            List<MemoryEntry> important = shortTermMemory.extractImportant();
            longTermMemory.addAll(important);
        }
    }
}

/**
 * 工作记忆实现
 */
class WorkingMemory {
    
    private final Deque<MemoryEntry> entries = new ArrayDeque<>();
    private final int capacity;
    
    public WorkingMemory(int capacity) {
        this.capacity = capacity;
    }
    
    public void add(MemoryEntry entry) {
        entries.addLast(entry);
        if (entries.size() > capacity) {
            entries.removeFirst();
        }
    }
    
    public List<MemoryEntry> retrieve(String query) {
        return entries.stream()
            .filter(e -> e.matches(query))
            .collect(Collectors.toList());
    }
    
    public List<MemoryEntry> getOverflow() {
        // 返回需要转移到短期记忆的条目
        return new ArrayList<>(entries);
    }
}

/**
 * 长期记忆实现（基于向量）
 */
class LongTermMemory {
    
    private final EmbeddingStore<TextSegment> vectorStore;
    private final EmbeddingModel embeddingModel;
    
    public void add(MemoryEntry entry) {
        Embedding embedding = embeddingModel.embed(entry.getContent());
        TextSegment segment = TextSegment.from(entry.getContent());
        segment.metadata().put("timestamp", entry.getTimestamp().toString());
        
        vectorStore.add(embedding, segment);
    }
    
    public List<MemoryEntry> retrieve(String query) {
        Embedding queryEmbedding = embeddingModel.embed(query);
        
        return vectorStore.findRelevant(queryEmbedding, 5)
            .stream()
            .map(match -> new MemoryEntry(
                match.embedded().text(),
                Instant.parse(match.embedded().metadata().getString("timestamp"))
            ))
            .collect(Collectors.toList());
    }
}
```

## 记忆编码

### 信息提取

```java
/**
 * 记忆编码器
 */
public class MemoryEncoder {
    
    private final ChatLanguageModel llm;
    
    /**
     * 从对话中提取关键信息
     */
    public List<MemoryEntry> extractFromConversation(
            List<Message> messages) {
        
        String conversation = formatConversation(messages);
        
        String prompt = String.format("""
            从以下对话中提取值得记忆的信息：
            
            对话：
            %s
            
            请提取：
            1. 用户偏好
            2. 重要事实
            3. 用户背景信息
            4. 待办事项
            
            输出JSON格式列表。
            """, conversation);
        
        String response = llm.generate(prompt);
        return parseMemoryEntries(response);
    }
    
    /**
     * 生成摘要
     */
    public String summarize(List<MemoryEntry> entries) {
        String content = entries.stream()
            .map(MemoryEntry::getContent)
            .collect(Collectors.joining("\n"));
        
        String prompt = "请总结以下信息的要点：\n\n" + content;
        return llm.generate(prompt);
    }
}
```

## 小结

本章我们学习了：

1. **记忆重要性**：个性化、学习、连贯性
2. **记忆层次**：工作、短期、长期记忆
3. **Java实现**：分层记忆系统
4. **记忆编码**：信息提取和摘要

**关键认识：**
记忆系统是智能体持续学习和个性化服务的基础。

**下一步：** 我们将学习对话记忆管理。

---

**练习题：**

1. 设计一个记忆重要性评估算法
2. 如何实现记忆的遗忘机制？
3. 设计跨会话的记忆同步方案。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 14.5 设计思考：推理边界](../chapter-14/05-reasoning-boundaries.md)</span>

<span>[15.2 对话记忆管理 &rarr;](02-dialog-memory-management.md)</span>

</div>