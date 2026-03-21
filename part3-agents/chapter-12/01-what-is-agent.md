<table width="100%">
   <tr>
      <td align="left"><a href="../../part2-llm/chapter-11/05-optimization-deployment.md">&larr; 11.5 部署与优化</a></td>
      <td align="right"><a href="02-core-components.md">12.2 智能体核心组件 &rarr;</a></td>
   </tr>
</table>
---

# 12.1 智能体概念与架构

> "智能体是AI的下一个进化——从回答问题到自主行动，从工具到伙伴。"

## 什么是智能体

### 定义

```
智能体（AI Agent）：
能够感知环境、做出决策并执行行动的自主系统。

核心特征：
1. 自主性：无需人工干预即可运行
2. 反应性：感知环境并响应
3. 主动性：主动追求目标
4. 社会性：与其他智能体或人类交互

类比：
- LLM是大脑：能思考、推理
- 智能体是完整的人：能思考+能行动
```

### 智能体 vs 传统软件

| 特性 | 传统软件 | 智能体 |
|------|----------|--------|
| 输入 | 固定格式 | 自然语言 |
| 逻辑 | 预编程 | 自主推理 |
| 输出 | 确定性 | 适应性 |
| 交互 | 被动响应 | 主动行动 |
| 学习 | 需更新代码 | 从交互中学习 |

## 智能体架构

### 基础架构

```
智能体基础架构：

┌─────────────────────────────────────┐
│            感知（Perception）         │
│  - 接收用户输入                       │
│  - 环境状态感知                       │
│  - 信息预处理                         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│            思考（Reasoning）          │
│  - 目标理解                          │
│  - 任务规划                          │
│  - 决策制定                          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│            行动（Action）             │
│  - 工具调用                          │
│  - 执行操作                          │
│  - 结果反馈                          │
└─────────────────────────────────────┘
```

### 核心组件

```java
/**
 * 智能体核心接口
 */
public interface Agent {
    
    /**
     * 感知环境
     */
    Perception perceive(Environment env);
    
    /**
     * 思考决策
     */
    Decision think(Perception perception, Goal goal);
    
    /**
     * 执行行动
     */
    ActionResult act(Decision decision);
    
    /**
     * 运行循环
     */
    default void run(Goal goal) {
        while (!goal.isAchieved()) {
            Perception p = perceive(Environment.current());
            Decision d = think(p, goal);
            ActionResult r = act(d);
            
            if (r.isFailure()) {
                handleFailure(r);
            }
        }
    }
}
```

## LLM驱动的智能体

### 为什么LLM适合驱动智能体

```
LLM作为智能体大脑的优势：

1. 自然语言理解
   - 理解复杂指令
   - 解析用户意图

2. 推理能力
   - 逻辑推理
   - 步骤规划

3. 知识丰富
   - 预训练知识
   - 上下文学习

4. 生成能力
   - 生成行动指令
   - 格式化输出
```

### ReAct架构

```
ReAct = Reasoning + Acting

循环模式：
思考(Thought) → 行动(Action) → 观察(Observation)

示例：
用户：北京今天天气怎么样？

思考：用户询问北京天气，我需要查询天气信息
行动：调用天气API，参数：city="北京"
观察：API返回：晴天，25°C

思考：我已经获得天气信息，可以回答用户
行动：生成回答

最终答案：北京今天晴天，气温25°C。
```

### Java实现框架

```java
package com.example.agent;

/**
 * ReAct智能体实现
 */
public class ReActAgent implements Agent {
    
    private final ChatLanguageModel llm;
    private final List<Tool> tools;
    private final Memory memory;
    
    @Override
    public Decision think(Perception perception, Goal goal) {
        // 构建ReAct提示
        String prompt = buildReActPrompt(perception, goal);
        
        // LLM生成思考和行动
        String response = llm.generate(prompt);
        
        // 解析响应
        return parseDecision(response);
    }
    
    @Override
    public ActionResult act(Decision decision) {
        if (decision.isFinalAnswer()) {
            return ActionResult.success(decision.getContent());
        }
        
        // 执行工具调用
        Tool tool = findTool(decision.getToolName());
        ToolResult result = tool.execute(decision.getParameters());
        
        return ActionResult.observation(result);
    }
    
    private String buildReActPrompt(Perception p, Goal g) {
        return String.format("""
            你是一个智能助手，请按照以下格式思考并行动：
            
            目标：%s
            观察：%s
            历史：%s
            
            可用工具：
            %s
            
            请输出：
            思考：你的思考过程
            行动：工具名称 或 最终答案
            """, 
            g.getDescription(),
            p.getObservation(),
            memory.getHistory(),
            formatTools()
        );
    }
}
```

## 智能体类型

### 按能力分类

```
1. 简单反射型（Simple Reflex）
   - 条件-行动规则
   - 无状态
   - 适合：简单、确定性任务

2. 基于模型的反射型（Model-based）
   - 维护内部状态
   - 基于状态决策
   - 适合：需要记忆的任务

3. 基于目标的智能体（Goal-based）
   - 有明确目标
   - 规划行动序列
   - 适合：复杂任务

4. 基于效用的智能体（Utility-based）
   - 评估不同方案的效用
   - 选择最优方案
   - 适合：需要权衡的场景

5. 学习型智能体（Learning）
   - 从经验中学习
   - 持续改进
   - 适合：长期运行的系统
```

### 应用场景

```
智能体应用场景：

个人助手：
- 日程管理
- 邮件处理
- 信息检索

代码助手：
- 自动调试
- 代码生成
- 重构建议

数据分析：
- 自动探索数据
- 生成报告
- 异常检测

客户服务：
- 智能客服
- 问题诊断
- 工单处理
```

## 小结

本章我们学习了：

1. **智能体定义**：自主感知、思考、行动的AI系统
2. **基础架构**：感知-思考-行动循环
3. **LLM驱动**：ReAct架构，推理+行动
4. **智能体类型**：从简单反射到学习型
5. **应用场景**：个人助手、代码助手等

**关键认识：**
智能体是AI应用的新范式，从被动工具进化为主动伙伴。

**下一步：** 我们将深入智能体的核心组件。

---

**练习题：**

1. 智能体与传统软件的核心区别是什么？
2. ReAct架构的优势是什么？
3. 设计一个基于目标的智能体应用场景。

---

<table width="100%">
   <tr>
      <td align="left"><a href="../../part2-llm/chapter-11/05-optimization-deployment.md">&larr; 11.5 部署与优化</a></td>
      <td align="right"><a href="02-core-components.md">12.2 智能体核心组件 &rarr;</a></td>
   </tr>
</table>