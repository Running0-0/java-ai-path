<table width="100%">
   <tr>
      <td align="left"><a href="02-roles-collaboration.md">&larr; 16.2 角色与协作</a></td>
      <td align="right"><a href="04-build-dev-team.md">16.4 实战：构建虚拟开发团队 &rarr;</a></td>
   </tr>
</table>
---

# 16.3 通信协议

> "有效的通信是协作的基础——智能体之间需要共同的语言。"

## 通信要素

### 消息结构

```
消息包含：

1. 头部（Header）
   - 消息ID
   - 发送者
   - 接收者
   - 时间戳
   - 消息类型

2. 主体（Body）
   - 内容
   - 附件
   - 元数据

3. 上下文（Context）
   - 会话ID
   - 引用消息
   - 优先级
```

### Java实现

```java
package com.example.agent.multi.communication;

import java.time.Instant;
import java.util.*;

/**
 * 消息
 */
public class Message {
    
    private final String id;
    private final String sender;
    private final String recipient;
    private final MessageType type;
    private final String content;
    private final Instant timestamp;
    private final Map<String, Object> metadata;
    private final String conversationId;
    private final String inReplyTo;
    private final Priority priority;
    
    private Message(Builder builder) {
        this.id = builder.id;
        this.sender = builder.sender;
        this.recipient = builder.recipient;
        this.type = builder.type;
        this.content = builder.content;
        this.timestamp = builder.timestamp;
        this.metadata = builder.metadata;
        this.conversationId = builder.conversationId;
        this.inReplyTo = builder.inReplyTo;
        this.priority = builder.priority;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private String id = UUID.randomUUID().toString();
        private String sender;
        private String recipient;
        private MessageType type;
        private String content;
        private Instant timestamp = Instant.now();
        private Map<String, Object> metadata = new HashMap<>();
        private String conversationId;
        private String inReplyTo;
        private Priority priority = Priority.NORMAL;
        
        public Builder sender(String sender) {
            this.sender = sender;
            return this;
        }
        
        public Builder recipient(String recipient) {
            this.recipient = recipient;
            return this;
        }
        
        public Builder type(MessageType type) {
            this.type = type;
            return this;
        }
        
        public Builder content(String content) {
            this.content = content;
            return this;
        }
        
        public Message build() {
            return new Message(this);
        }
    }
}

enum MessageType {
    REQUEST,      // 请求
    RESPONSE,     // 响应
    NOTIFICATION, // 通知
    COMMAND,      // 命令
    EVENT         // 事件
}

enum Priority {
    LOW, NORMAL, HIGH, URGENT
}
```

## 协议设计

### 请求-响应协议

```java
/**
 * 请求-响应协议
 */
public class RequestResponseProtocol {
    
    private final MessageBus messageBus;
    private final Map<String, CompletableFuture<Message>> pendingRequests;
    
    /**
     * 发送请求并等待响应
     */
    public CompletableFuture<Message> request(String recipient, 
                                               String content,
                                               long timeoutMs) {
        String requestId = UUID.randomUUID().toString();
        
        Message request = Message.builder()
            .sender(getSelfId())
            .recipient(recipient)
            .type(MessageType.REQUEST)
            .content(content)
            .metadata(Map.of("requestId", requestId))
            .build();
        
        CompletableFuture<Message> future = new CompletableFuture<>();
        pendingRequests.put(requestId, future);
        
        // 设置超时
        ScheduledExecutorService executor = 
            Executors.newSingleThreadScheduledExecutor();
        executor.schedule(() -> {
            future.completeExceptionally(
                new TimeoutException("请求超时"));
            pendingRequests.remove(requestId);
        }, timeoutMs, TimeUnit.MILLISECONDS);
        
        messageBus.send(request);
        
        return future;
    }
    
    /**
     * 处理响应
     */
    public void handleResponse(Message response) {
        String requestId = (String) response.getMetadata().get("requestId");
        CompletableFuture<Message> future = pendingRequests.remove(requestId);
        
        if (future != null) {
            future.complete(response);
        }
    }
}
```

### 发布-订阅协议

```java
/**
 * 发布-订阅协议
 */
public class PubSubProtocol {
    
    private final Map<String, Set<String>> topicSubscribers;
    private final MessageBus messageBus;
    
    /**
     * 订阅主题
     */
    public void subscribe(String agentId, String topic) {
        topicSubscribers
            .computeIfAbsent(topic, k -> ConcurrentHashMap.newKeySet())
            .add(agentId);
    }
    
    /**
     * 发布消息
     */
    public void publish(String topic, String content) {
        Message message = Message.builder()
            .sender(getSelfId())
            .type(MessageType.EVENT)
            .content(content)
            .metadata(Map.of("topic", topic))
            .build();
        
        Set<String> subscribers = topicSubscribers.get(topic);
        if (subscribers != null) {
            for (String subscriber : subscribers) {
                Message targeted = Message.builder()
                    .from(message)
                    .recipient(subscriber)
                    .build();
                messageBus.send(targeted);
            }
        }
    }
}
```

## 小结

本章我们学习了：

1. **通信要素**：消息结构、类型、优先级
2. **协议设计**：请求-响应、发布-订阅
3. **Java实现**：消息类和协议实现

**关键认识：**
良好的通信协议是多智能体协作的基础设施。

**下一步：** 我们将学习构建开发团队。

---

**练习题：**

1. 设计一个支持加密的消息协议
2. 实现可靠消息传输机制
3. 设计智能体发现协议。

---

<table width="100%">
   <tr>
      <td align="left"><a href="02-roles-collaboration.md">&larr; 16.2 角色与协作</a></td>
      <td align="right"><a href="04-build-dev-team.md">16.4 实战：构建虚拟开发团队 &rarr;</a></td>
   </tr>
</table>