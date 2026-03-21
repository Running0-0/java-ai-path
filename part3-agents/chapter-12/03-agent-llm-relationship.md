<table width="100%">
   <tr>
      <td align="left"><a href="02-core-components.md">&larr; 12.2 智能体核心组件</a></td>
      <td align="right"><a href="04-tool-user-to-creator.md">12.4 从工具使用者到创造者 &rarr;</a></td>
   </tr>
</table>
---

# 12.3 智能体与LLM的关系

> "LLM是智能体的大脑，但智能体远不止LLM——它是完整的行动系统。"

## 关系辨析

### 层次对比

```
层次结构：

┌─────────────────────────────────────┐
│           应用层（Application）       │
│  - 智能客服、代码助手、个人助理        │
├─────────────────────────────────────┤
│           智能体层（Agent）           │
│  - 感知、规划、记忆、工具使用          │
├─────────────────────────────────────┤
│           模型层（LLM）               │
│  - 推理、生成、理解                   │
├─────────────────────────────────────┤
│           基础设施层                  │
│  - 计算、存储、网络                   │
└─────────────────────────────────────┘

关系：
- LLM是智能体的核心组件
- 智能体是LLM的封装和扩展
- 应用构建在智能体之上
```

### 能力对比

| 能力 | 纯LLM | 智能体 |
|------|-------|--------|
| 推理 | ✅ | ✅ |
| 生成 | ✅ | ✅ |
| 记忆 | ⚠️ 有限上下文 | ✅ 持久化记忆 |
| 工具使用 | ❌ | ✅ |
| 规划 | ⚠️ 单次 | ✅ 多步规划 |
| 环境交互 | ❌ | ✅ |
| 自主学习 | ❌ | ✅ |

## LLM的局限性

### 纯LLM的问题

```
1. 无状态
   - 每次调用独立
   - 无法记住之前的信息
   - 需要重复提供上下文

2. 无法行动
   - 只能生成文本
   - 不能执行操作
   - 无法获取实时信息

3. 幻觉问题
   - 生成虚假信息
   - 无法验证事实
   - 知识有时效性

4. 上下文有限
   - Token限制
   - 长文档处理困难
   - 多轮对话易遗忘
```

### 智能体如何解决

```
智能体的解决方案：

1. 记忆系统
   - 向量数据库存储
   - 持久化对话历史
   - 长期知识积累

2. 工具调用
   - API调用获取实时数据
   - 执行代码
   - 操作外部系统

3. RAG增强
   - 检索真实信息
   - 减少幻觉
   - 引用来源

4. 规划能力
   - 分解复杂任务
   - 多步骤执行
   - 错误恢复
```

## 架构模式

### 模式1：LLM作为控制器

```
架构：

用户输入
    ↓
[LLM] → 决策：调用什么工具/直接回答
    ↓
    ├── 工具A → 执行 → 结果
    ├── 工具B → 执行 → 结果
    └── 直接回答
    ↓
[LLM] → 整合结果 → 最终答案

特点：
- LLM做所有决策
- 简单直观
- 但LLM负担重
```

### 模式2：分层架构

```
架构：

用户输入
    ↓
[意图识别] → 确定任务类型
    ↓
[任务路由器] → 分配到不同处理器
    ↓
    ├── 简单查询 → [LLM] → 直接回答
    ├── 数据查询 → [数据库工具] → [LLM] → 回答
    └── 复杂任务 → [规划器] → [执行器] → 回答
    ↓
[后处理] → 格式化输出
    ↓
用户

特点：
- 职责分离
- 更高效
- 可扩展性好
```

### Java实现对比

```java
/**
 * 模式1：LLM作为控制器
 */
public class LLMControllerAgent {
    
    public String handle(String input) {
        // LLM决定如何处理
        String decision = llm.generate("""
            用户输入：""" + input + """
            
            请决定：
            1. 直接回答
            2. 调用工具（哪个工具？）
            
            输出JSON格式决策。
            """);
        
        Action action = parseDecision(decision);
        
        if (action.isDirectAnswer()) {
            return llm.generate(input);
        } else {
            ToolResult result = executeTool(action);
            return llm.generate("基于结果：" + result + "，回答用户");
        }
    }
}

/**
 * 模式2：分层架构
 */
public class HierarchicalAgent {
    
    private final IntentClassifier classifier;
    private final Map<Intent, Handler> handlers;
    
    public String handle(String input) {
        // 1. 意图识别
        Intent intent = classifier.classify(input);
        
        // 2. 路由到对应处理器
        Handler handler = handlers.get(intent);
        
        // 3. 处理
        return handler.handle(input);
    }
}

// 简单查询处理器
class SimpleQueryHandler implements Handler {
    public String handle(String input) {
        return llm.generate(input);  // 直接回答
    }
}

// 数据查询处理器
class DataQueryHandler implements Handler {
    public String handle(String input) {
        // 提取查询参数
        QueryParams params = extractParams(input);
        // 查询数据库
        Data data = database.query(params);
        // LLM生成回答
        return llm.generate("数据：" + data + "，回答用户问题");
    }
}
```

## 选择建议

### 何时使用纯LLM

```
适合纯LLM的场景：

✓ 简单问答
  - 通用知识
  - 创意写作
  - 代码解释

✓ 一次性任务
  - 文本翻译
  - 摘要生成
  - 格式转换

✓ 探索性对话
  - 头脑风暴
  - 学习辅导
  - 闲聊
```

### 何时使用智能体

```
适合智能体的场景：

✓ 需要记忆
  - 个人助手
  - 长期项目
  - 用户偏好学习

✓ 需要工具
  - 数据查询
  - 系统操作
  - 实时信息获取

✓ 复杂任务
  - 多步骤工作流
  - 需要规划
  - 错误恢复

✓ 企业应用
  - 客服系统
  - 业务流程自动化
  - 决策支持
```

## 小结

本章我们学习了：

1. **层次关系**：LLM是智能体的核心，智能体是LLM的扩展
2. **能力对比**：智能体弥补了纯LLM的局限
3. **架构模式**：LLM控制器 vs 分层架构
4. **选择建议**：根据场景选择合适的架构

**关键认识：**
理解LLM和智能体的关系，有助于设计合适的AI应用架构。

**下一步：** 我们将学习从工具使用者到创造者。

---

**思考题：**

1. 你的应用场景更适合纯LLM还是智能体？
2. 分层架构相比LLM控制器有什么优势？
3. 设计一个混合架构的智能体系统。

---

<table width="100%">
   <tr>
      <td align="left"><a href="02-core-components.md">&larr; 12.2 智能体核心组件</a></td>
      <td align="right"><a href="04-tool-user-to-creator.md">12.4 从工具使用者到创造者 &rarr;</a></td>
   </tr>
</table>