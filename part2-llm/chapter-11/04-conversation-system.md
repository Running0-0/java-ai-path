<table width="100%">
   <tr>
      <td align="left"><a href="03-rag-implementation.md">&larr; 11.3 RAG检索实现</a></td>
      <td align="right"><a href="05-optimization-deployment.md">11.5 部署与优化 &rarr;</a></td>
   </tr>
</table>
---

# 11.4 对话系统实现

> "对话是RAG的终极形态——不是一次性问答，而是持续的智能交互。"

## 对话系统设计

### 架构

```
对话系统架构：

用户消息
    ↓
[对话管理器]
    - 会话状态维护
    - 历史记录管理
    - 上下文压缩
    ↓
[意图识别]
    - 判断是否需要检索
    - 区分闲聊/问答
    ↓
[检索/生成路由]
    ├── 需要知识 → RAG流程
    └── 闲聊 → 直接生成
    ↓
[答案生成]
    - 基于检索结果生成
    - 或自由生成
    ↓
[后处理]
    - 引用标注
    - 安全过滤
    ↓
返回用户
```

## 会话管理

### 会话状态

```java
package com.example.rag.conversation;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 对话会话
 */
public class ConversationSession {
    
    private final String sessionId;
    private final List<Message> messages;
    private final LocalDateTime createdAt;
    private LocalDateTime lastActivity;
    private SessionContext context;
    
    public ConversationSession(String sessionId) {
        this.sessionId = sessionId;
        this.messages = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.lastActivity = LocalDateTime.now();
        this.context = new SessionContext();
    }
    
    /**
     * 添加消息
     */
    public void addMessage(Message message) {
        messages.add(message);
        lastActivity = LocalDateTime.now();
    }
    
    /**
     * 获取历史（带长度限制）
     */
    public List<Message> getRecentHistory(int maxMessages) {
        int start = Math.max(0, messages.size() - maxMessages);
        return messages.subList(start, messages.size());
    }
    
    /**
     * 压缩历史
     */
    public void compressHistory(ChatLanguageModel model) {
        if (messages.size() < 10) return;
        
        // 将早期对话总结为摘要
        List<Message> oldMessages = messages.subList(0, messages.size() - 5);
        String summary = summarize(oldMessages, model);
        
        // 替换为摘要
        context.setSummary(summary);
        messages.subList(0, messages.size() - 5).clear();
    }
    
    private String summarize(List<Message> msgs, ChatLanguageModel model) {
        String conversation = msgs.stream()
            .map(m -> m.getRole() + ": " + m.getContent())
            .collect(Collectors.joining("\\n"));
        
        String prompt = "请总结以下对话的关键信息：\\n\\n" + conversation;
        return model.generate(prompt);
    }
}
```

### 持久化存储

```java
package com.example.rag.conversation;

/**
 * 会话存储
 */
public class SessionStorage {
    
    private final JdbcTemplate jdbcTemplate;
    
    /**
     * 保存会话
     */
    public void saveSession(ConversationSession session) {
        String sql = """
            INSERT INTO conversations (session_id, created_at, last_activity, summary)
            VALUES (?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
            last_activity = ?, summary = ?
            """;
        
        jdbcTemplate.update(sql,
            session.getSessionId(),
            session.getCreatedAt(),
            session.getLastActivity(),
            session.getContext().getSummary(),
            session.getLastActivity(),
            session.getContext().getSummary()
        );
        
        // 保存消息
        saveMessages(session.getSessionId(), session.getMessages());
    }
    
    /**
     * 加载会话
     */
    public ConversationSession loadSession(String sessionId) {
        String sql = "SELECT * FROM conversations WHERE session_id = ?";
        
        return jdbcTemplate.queryForObject(sql, (rs, rowNum) -> {
            ConversationSession session = new ConversationSession(sessionId);
            session.getContext().setSummary(rs.getString("summary"));
            
            // 加载历史消息
            List<Message> messages = loadMessages(sessionId);
            messages.forEach(session::addMessage);
            
            return session;
        }, sessionId);
    }
}
```

## 多轮对话RAG

### 上下文增强检索

```java
package com.example.rag.conversation;

/**
 * 对话感知的检索
 */
public class ConversationalRetrieval {
    
    private final RetrievalService retrievalService;
    private final QueryRewriter queryRewriter;
    
    /**
     * 检索（考虑对话历史）
     */
    public RetrievalResult retrieve(String query, 
                                    ConversationSession session) {
        // 1. 重写查询（消除指代）
        String rewrittenQuery = rewriteQuery(query, session);
        
        // 2. 扩展查询（加入历史关键词）
        List<String> expandedQueries = expandQuery(rewrittenQuery, session);
        
        // 3. 多查询检索
        Set<ScoredDocument> allResults = new HashSet<>();
        for (String q : expandedQueries) {
            allResults.addAll(retrievalService.retrieve(q).getDocuments());
        }
        
        // 4. 去重重排序
        return rerankAndDeduplicate(allResults, query);
    }
    
    /**
     * 查询重写
     */
    private String rewriteQuery(String query, ConversationSession session) {
        // 指代消解
        if (containsPronoun(query)) {
            String context = session.getRecentHistory(3).stream()
                .map(Message::getContent)
                .collect(Collectors.joining("\\n"));
            
            String prompt = String.format("""
                根据对话历史，将用户的最新问题中的指代词替换为具体实体。
                
                对话历史：
                %s
                
                用户问题：%s
                
                重写后的问题：""", context, query);
            
            return model.generate(prompt);
        }
        
        return query;
    }
}
```

## 答案生成

### 带引用的生成

```java
package com.example.rag.conversation;

/**
 * 带引用的答案生成
 */
public class CitedAnswerGenerator {
    
    private final ChatLanguageModel model;
    private final PromptTemplate template;
    
    /**
     * 生成带引用的答案
     */
    public CitedAnswer generate(String query, 
                                RetrievalResult context,
                                ConversationSession session) {
        // 构建提示
        Map<String, Object> params = new HashMap<>();
        params.put("query", query);
        params.put("context", formatContextWithCitations(context));
        params.put("history", formatHistory(session));
        
        String prompt = template.render(params);
        
        // 生成
        String answer = model.generate(prompt);
        
        // 解析引用
        List<Citation> citations = extractCitations(answer, context);
        
        return new CitedAnswer(answer, citations);
    }
    
    private String formatContextWithCitations(RetrievalResult result) {
        StringBuilder sb = new StringBuilder();
        List<ScoredDocument> docs = result.getDocuments();
        
        for (int i = 0; i < docs.size(); i++) {
            sb.append("[").append(i + 1).append("] ")
              .append(docs.get(i).getText())
              .append("\\n\\n");
        }
        
        return sb.toString();
    }
}
```

## 流式响应

### SSE实现

```java
package com.example.rag.conversation;

import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * 流式对话服务
 */
public class StreamingConversationService {
    
    /**
     * 流式回答
     */
    public SseEmitter streamAnswer(String query, String sessionId) {
        SseEmitter emitter = new SseEmitter(300000L); // 5分钟超时
        
        executor.execute(() -> {
            try {
                // 获取会话
                ConversationSession session = 
                    sessionManager.getOrCreate(sessionId);
                
                // 检索
                RetrievalResult context = 
                    conversationalRetrieval.retrieve(query, session);
                
                // 流式生成
                Stream<String> stream = answerGenerator.generateStream(
                    query, context, session);
                
                // 发送流
                stream.forEach(chunk -> {
                    try {
                        emitter.send(SseEmitter.event()
                            .data(chunk));
                    } catch (IOException e) {
                        emitter.completeWithError(e);
                    }
                });
                
                // 保存对话
                session.addMessage(new Message("user", query));
                session.addMessage(new Message("assistant", 
                    accumulatedAnswer.toString()));
                sessionStorage.saveSession(session);
                
                emitter.complete();
                
            } catch (Exception e) {
                emitter.completeWithError(e);
            }
        });
        
        return emitter;
    }
}
```

## 小结

本章我们学习了：

1. **会话管理**：状态维护、历史压缩、持久化
2. **对话检索**：查询重写、上下文增强
3. **答案生成**：带引用的生成
4. **流式响应**：SSE实现实时输出

**关键认识：**
对话系统让RAG从单次问答升级为持续交互，提供更好的用户体验。

**下一步：** 我们将学习部署优化。

---

**练习题：**

1. 如何处理超长对话历史？
2. 设计一个支持多轮澄清的问答系统
3. 实现对话上下文的智能压缩。

---

<table width="100%">
   <tr>
      <td align="left"><a href="03-rag-implementation.md">&larr; 11.3 RAG检索实现</a></td>
      <td align="right"><a href="05-optimization-deployment.md">11.5 部署与优化 &rarr;</a></td>
   </tr>
</table>