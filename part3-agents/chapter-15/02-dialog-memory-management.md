<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 15.1 短期与长期记忆](01-short-long-term-memory.md)</span>

<span>[15.3 向量数据库 &rarr;](03-vector-database.md)</span>

</div>
---

# 15.2 对话记忆管理

> "对话记忆让智能体记得你们聊过什么——从陌生人到老朋友。"

## 对话记忆挑战

### 主要挑战

```
对话记忆的挑战：

1. 上下文长度限制
   - LLM有token限制
   - 长对话无法全部传入
   - 需要选择性保留

2. 信息重要性不同
   - 关键信息 vs 闲聊
   - 需要识别重要内容
   - 合理压缩

3. 多会话连贯
   - 跨会话记忆
   - 话题切换
   - 时间跨度

4. 隐私与安全
   - 敏感信息处理
   - 用户控制记忆
   - 数据保护
```

## 记忆管理策略

### 滑动窗口

```java
package com.example.agent.memory;

/**
 * 滑动窗口记忆管理
 */
public class SlidingWindowMemory {
    
    private final int maxMessages;
    private final Deque<Message> messages;
    
    public SlidingWindowMemory(int maxMessages) {
        this.maxMessages = maxMessages;
        this.messages = new ArrayDeque<>();
    }
    
    /**
     * 添加消息
     */
    public void addMessage(Message message) {
        messages.addLast(message);
        
        // 超出限制时移除最旧的消息
        while (messages.size() > maxMessages) {
            messages.removeFirst();
        }
    }
    
    /**
     * 获取当前上下文
     */
    public List<Message> getContext() {
        return new ArrayList<>(messages);
    }
}
```

### 摘要压缩

```java
/**
 * 摘要记忆管理
 */
public class SummaryMemory {
    
    private final ChatLanguageModel llm;
    private final int bufferSize;
    
    private String summary = "";
    private List<Message> buffer = new ArrayList<>();
    
    /**
     * 添加消息
     */
    public void addMessage(Message message) {
        buffer.add(message);
        
        // 缓冲区满时生成摘要
        if (buffer.size() >= bufferSize) {
            compress();
        }
    }
    
    /**
     * 压缩缓冲区到摘要
     */
    private void compress() {
        String conversation = formatMessages(buffer);
        
        String prompt = String.format("""
            将以下对话总结为简洁的摘要，保留关键信息：
            
            当前摘要：%s
            
            新对话：
            %s
            
            请生成更新后的摘要。
            """, summary, conversation);
        
        summary = llm.generate(prompt);
        buffer.clear();
    }
    
    /**
     * 获取完整上下文
     */
    public String getContext() {
        String recent = formatMessages(buffer);
        return String.format("历史摘要：%s\n\n最近对话：%s", 
            summary, recent);
    }
}
```

### 实体记忆

```java
/**
 * 实体提取记忆
 */
public class EntityMemory {
    
    private final ChatLanguageModel llm;
    private final Map<String, Entity> entities = new HashMap<>();
    
    /**
     * 从对话中提取实体
     */
    public void extractFromMessage(Message message) {
        String prompt = String.format("""
            从以下消息中提取关键实体（人、地点、组织、概念）：
            
            消息：%s
            
            输出JSON格式：
            {
              "entities": [
                {"type": "人", "name": "...", "relation": "..."}
              ]
            }
            """, message.getContent());
        
        String response = llm.generate(prompt);
        List<Entity> extracted = parseEntities(response);
        
        // 合并到实体库
        for (Entity entity : extracted) {
            entities.merge(entity.getName(), entity, this::mergeEntities);
        }
    }
    
    /**
     * 获取相关实体
     */
    public List<Entity> getRelevantEntities(String query) {
        return entities.values().stream()
            .filter(e -> e.matches(query))
            .collect(Collectors.toList());
    }
}
```

## 持久化存储

### 数据库存储

```java
/**
 * 数据库存储的记忆
 */
public class PersistentChatMemory {
    
    private final JdbcTemplate jdbcTemplate;
    private final String sessionId;
    
    /**
     * 保存消息
     */
    public void saveMessage(Message message) {
        String sql = """
            INSERT INTO chat_messages 
            (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
            """;
        
        jdbcTemplate.update(sql,
            sessionId,
            message.getRole(),
            message.getContent(),
            Timestamp.from(Instant.now()),
            toJson(message.getMetadata())
        );
    }
    
    /**
     * 加载历史
     */
    public List<Message> loadHistory(int limit) {
        String sql = """
            SELECT * FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """;
        
        return jdbcTemplate.query(sql, (rs, rowNum) -> new Message(
            rs.getString("role"),
            rs.getString("content"),
            rs.getTimestamp("timestamp").toInstant()
        ), sessionId, limit);
    }
}
```

## 小结

本章我们学习了：

1. **记忆挑战**：长度限制、重要性、多会话
2. **管理策略**：滑动窗口、摘要压缩、实体记忆
3. **持久化存储**：数据库存储实现

**关键认识：**
有效的对话记忆管理是提供连贯对话体验的关键。

**下一步：** 我们将学习向量数据库。

---

**练习题：**

1. 实现一个混合策略的记忆管理器
2. 如何评估记忆压缩的质量？
3. 设计支持多话题的记忆系统。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 15.1 短期与长期记忆](01-short-long-term-memory.md)</span>

<span>[15.3 向量数据库 &rarr;](03-vector-database.md)</span>

</div>