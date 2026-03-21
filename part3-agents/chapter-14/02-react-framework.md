<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 14.1 任务分解](01-task-decomposition.md)</span>

<span>[14.3 思维链与思维树 &rarr;](03-chain-tree-of-thought.md)</span>

</div>
---

# 14.2 ReAct框架

> "ReAct = Reasoning + Acting，让智能体既能思考又能行动。"

## ReAct原理

### 核心思想

```
ReAct框架：

交替进行推理和行动：
思考(Thought) → 行动(Action) → 观察(Observation) → 思考...

为什么有效：
1. 显式推理：展示思考过程
2. 行动导向：不只是思考，还要行动
3. 反馈闭环：观察结果影响下一步
4. 可追溯：每一步都有记录

类比人类解决问题：
- 先思考怎么做
- 然后采取行动
- 观察结果
- 根据结果调整
```

### 工作流程

```
用户问题："2024年奥斯卡最佳影片的导演还导演过哪些电影？"

Step 1: 思考
"我需要先找出2024年奥斯卡最佳影片，
然后找到它的导演，
最后查询该导演的其他作品。"

Step 2: 行动
调用工具: search("2024年奥斯卡最佳影片")

Step 3: 观察
"2024年奥斯卡最佳影片是《奥本海默》"

Step 4: 思考
"现在我需要找到《奥本海默》的导演"

Step 5: 行动
调用工具: search("奥本海默 导演")

Step 6: 观察
"导演是克里斯托弗·诺兰"

Step 7: 思考
"现在我需要查询诺兰导演的其他电影"

Step 8: 行动
调用工具: search("克里斯托弗·诺兰 电影作品")

Step 9: 观察
"诺兰导演的作品包括：盗梦空间、星际穿越、黑暗骑士..."

Step 10: 思考
"我已经获得了所有需要的信息，可以回答用户了"

最终答案："《奥本海默》的导演克里斯托弗·诺兰还导演过..."
```

## Java实现

### ReAct Agent

```java
package com.example.agent.react;

import java.util.ArrayList;
import java.util.List;

/**
 * ReAct智能体实现
 */
public class ReActAgent {
    
    private final ChatLanguageModel llm;
    private final ToolRegistry tools;
    private final int maxIterations;
    
    public ReActAgent(ChatLanguageModel llm, ToolRegistry tools, 
                      int maxIterations) {
        this.llm = llm;
        this.tools = tools;
        this.maxIterations = maxIterations;
    }
    
    /**
     * 运行ReAct循环
     */
    public String run(String query) {
        List<ReActStep> history = new ArrayList<>();
        
        for (int i = 0; i < maxIterations; i++) {
            // 1. 思考
            Thought thought = think(query, history);
            
            // 2. 检查是否完成
            if (thought.isComplete()) {
                return thought.getAnswer();
            }
            
            // 3. 行动
            Action action = thought.getAction();
            Observation observation = executeAction(action);
            
            // 4. 记录历史
            history.add(new ReActStep(thought, action, observation));
        }
        
        return "达到最大迭代次数，未能完成";
    }
    
    /**
     * 思考步骤
     */
    private Thought think(String query, List<ReActStep> history) {
        String prompt = buildPrompt(query, history);
        String response = llm.generate(prompt);
        
        return parseThought(response);
    }
    
    /**
     * 构建提示
     */
    private String buildPrompt(String query, List<ReActStep> history) {
        StringBuilder prompt = new StringBuilder();
        
        prompt.append("解决以下问题，使用ReAct格式：\n\n");
        prompt.append("问题：").append(query).append("\n\n");
        
        // 可用工具
        prompt.append("可用工具：\n");
        prompt.append(tools.getDescriptions()).append("\n\n");
        
        // 历史步骤
        if (!history.isEmpty()) {
            prompt.append("历史步骤：\n");
            for (ReActStep step : history) {
                prompt.append("思考：").append(step.getThought()).append("\n");
                prompt.append("行动：").append(step.getAction()).append("\n");
                prompt.append("观察：").append(step.getObservation()).append("\n\n");
            }
        }
        
        prompt.append("请继续：\n");
        prompt.append("思考：");
        
        return prompt.toString();
    }
    
    /**
     * 执行行动
     */
    private Observation executeAction(Action action) {
        if (action.isFinish()) {
            return new Observation(action.getAnswer(), true);
        }
        
        Tool tool = tools.getTool(action.getToolName());
        ToolResult result = tool.execute(action.getParameters());
        
        return new Observation(result.toString(), false);
    }
}

/**
 * ReAct步骤
 */
class ReActStep {
    Thought thought;
    Action action;
    Observation observation;
}
```

## 提示模板

### ReAct提示设计

```
ReAct提示模板：

解决以下问题，请按照思考→行动→观察的格式：

问题：{question}

可用工具：
{tools}

格式说明：
思考：分析当前情况，决定下一步行动
行动：工具名称[参数] 或 最终答案[答案]
观察：工具返回的结果

示例：
问题：北京今天天气怎么样？

思考：用户询问北京天气，我需要查询天气信息
行动：get_weather[{"city": "北京"}]
观察：{"weather": "晴天", "temperature": "25°C"}

思考：我已经获得天气信息，可以回答用户了
行动：最终答案[北京今天晴天，气温25°C。]

现在请解决：
思考：
```

## 变体与优化

### ReWOO（ReAct Without Observation）

```
优化：预先生成所有行动计划，减少LLM调用次数

步骤：
1. Planner：生成完整计划
2. Worker：执行所有工具调用
3. Solver：基于所有结果生成答案

优势：
- 减少LLM调用次数
- 可以并行执行工具
- 效率更高
```

### 反思增强

```java
/**
 * 带反思的ReAct
 */
public class ReflectiveReActAgent extends ReActAgent {
    
    @Override
    protected Thought think(String query, List<ReActStep> history) {
        // 添加反思步骤
        if (!history.isEmpty() && history.size() % 3 == 0) {
            Reflection reflection = reflect(history);
            if (reflection.needsAdjustment()) {
                adjustPlan(reflection);
            }
        }
        
        return super.think(query, history);
    }
    
    private Reflection reflect(List<ReActStep> history) {
        String prompt = "回顾之前的步骤，是否有问题？是否需要调整策略？";
        String response = llm.generate(prompt + formatHistory(history));
        return parseReflection(response);
    }
}
```

## 小结

本章我们学习了：

1. **ReAct原理**：思考-行动-观察循环
2. **Java实现**：ReActAgent完整实现
3. **提示设计**：ReAct提示模板
4. **变体优化**：ReWOO、反思增强

**关键认识：**
ReAct是智能体的经典架构，简单有效，适合大多数场景。

**下一步：** 我们将学习思维链与思维树。

---

**练习题：**

1. 实现一个支持多工具并行调用的ReAct变体
2. 设计ReAct的错误恢复机制
3. 比较ReAct和直接调用工具的效果差异。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 14.1 任务分解](01-task-decomposition.md)</span>

<span>[14.3 思维链与思维树 &rarr;](03-chain-tree-of-thought.md)</span>

</div>