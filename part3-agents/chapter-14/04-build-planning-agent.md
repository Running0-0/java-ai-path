<table width="100%">
   <tr>
      <td align="left"><a href="03-chain-tree-of-thought.md">&larr; 14.3 思维链与思维树</a></td>
      <td align="right"><a href="05-reasoning-boundaries.md">14.5 设计思考：推理边界 &rarr;</a></td>
   </tr>
</table>
---

# 14.4 构建规划智能体

> "规划是智能体的核心能力——从目标到行动，从想法到实现。"

## 规划智能体架构

### 整体架构

```
规划智能体架构：

┌─────────────────────────────────────────┐
│              目标输入                    │
│         "完成XX项目的需求分析"            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│              目标解析器                  │
│  - 理解目标                              │
│  - 提取约束条件                          │
│  - 识别成功标准                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│              任务规划器                  │
│  - 分解任务                              │
│  - 确定依赖关系                          │
│  - 估算时间和资源                        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│              执行引擎                    │
│  - 调度任务                              │
│  - 监控进度                              │
│  - 处理异常                              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│              结果评估                    │
│  - 验证结果                              │
│  - 生成报告                              │
└─────────────────────────────────────────┘
```

## Java实现

### 规划智能体

```java
package com.example.agent.planning;

/**
 * 规划智能体
 */
public class PlanningAgent {
    
    private final GoalParser goalParser;
    private final TaskPlanner taskPlanner;
    private final ExecutionEngine executionEngine;
    private final ResultEvaluator resultEvaluator;
    
    /**
     * 执行目标
     */
    public ExecutionResult executeGoal(String goalDescription) {
        // 1. 解析目标
        Goal goal = goalParser.parse(goalDescription);
        
        // 2. 制定计划
        Plan plan = taskPlanner.createPlan(goal);
        
        // 3. 执行计划
        ExecutionContext context = executionEngine.execute(plan);
        
        // 4. 评估结果
        EvaluationResult evaluation = resultEvaluator.evaluate(
            goal, context.getResults());
        
        return new ExecutionResult(
            goal, plan, context, evaluation);
    }
}

/**
 * 目标解析器
 */
class GoalParser {
    
    private final ChatLanguageModel llm;
    
    public Goal parse(String description) {
        String prompt = String.format("""
            解析以下目标，提取关键信息：
            
            目标：%s
            
            请提取：
            1. 主要目标
            2. 约束条件（时间、资源、质量等）
            3. 成功标准
            4. 潜在风险
            
            输出JSON格式。
            """, description);
        
        String response = llm.generate(prompt);
        return parseGoal(response);
    }
}

/**
 * 任务规划器
 */
class TaskPlanner {
    
    public Plan createPlan(Goal goal) {
        // 使用LLM分解任务
        String prompt = String.format("""
            为实现以下目标，制定详细计划：
            
            目标：%s
            约束：%s
            
            请：
            1. 分解为具体任务
            2. 确定任务依赖关系
            3. 估算每个任务的时间
            4. 识别关键路径
            
            输出任务列表（JSON格式）。
            """, goal.getDescription(), goal.getConstraints());
        
        String response = llm.generate(prompt);
        List<Task> tasks = parseTasks(response);
        
        return new Plan(tasks);
    }
}
```

## 动态重规划

### 自适应规划

```java
/**
 * 支持动态重规划的智能体
 */
public class AdaptivePlanningAgent extends PlanningAgent {
    
    @Override
    public ExecutionResult executeGoal(String goalDescription) {
        Goal goal = goalParser.parse(goalDescription);
        Plan plan = taskPlanner.createPlan(goal);
        
        ExecutionContext context = new ExecutionContext();
        
        while (!plan.isComplete()) {
            Task currentTask = plan.getNextTask();
            
            // 执行任务
            TaskResult result = executeTask(currentTask);
            context.addResult(result);
            
            // 检查是否需要重规划
            if (needsReplanning(result, plan)) {
                plan = replan(goal, context, result);
            }
            
            // 标记完成
            plan.markComplete(currentTask);
        }
        
        return new ExecutionResult(goal, plan, context);
    }
    
    private boolean needsReplanning(TaskResult result, Plan plan) {
        // 任务失败
        if (result.isFailure()) return true;
        
        // 发现新信息
        if (result.hasNewInformation()) return true;
        
        // 资源不足
        if (result.isResourceConstrained()) return true;
        
        return false;
    }
    
    private Plan replan(Goal goal, ExecutionContext context, 
                        TaskResult trigger) {
        String prompt = String.format("""
            基于以下情况重新规划：
            
            原始目标：%s
            已完成：%s
            触发重规划的事件：%s
            
            请调整剩余计划。
            """, goal, context.getCompletedTasks(), trigger);
        
        String response = llm.generate(prompt);
        return parsePlan(response);
    }
}
```

## 实际应用

### 项目规划助手

```java
/**
 * 项目规划助手
 */
public class ProjectPlanningAssistant {
    
    private final PlanningAgent agent;
    
    public ProjectPlan createProjectPlan(String projectDescription) {
        String goal = String.format("""
            为以下项目制定完整开发计划：
            
            %s
            
            包括：需求分析、技术选型、任务分解、
            时间估算、风险评估、里程碑设定。
            """, projectDescription);
        
        ExecutionResult result = agent.executeGoal(goal);
        
        return parseProjectPlan(result);
    }
    
    public void demonstrate() {
        String project = """
            项目名称：智能客服系统
            描述：基于大语言模型的企业客服系统
            团队：5人（2后端、2前端、1产品）
            时间：3个月
            """;
        
        ProjectPlan plan = createProjectPlan(project);
        
        System.out.println("项目计划：");
        System.out.println("阶段1（2周）：需求分析和技术选型");
        System.out.println("阶段2（4周）：核心功能开发");
        System.out.println("阶段3（3周）：高级功能和集成");
        System.out.println("阶段4（3周）：测试和优化");
    }
}
```

## 小结

本章我们学习了：

1. **规划智能体架构**：目标解析、任务规划、执行、评估
2. **Java实现**：完整的规划智能体实现
3. **动态重规划**：自适应调整计划
4. **实际应用**：项目规划助手

**关键认识：**
规划能力是智能体解决复杂问题的核心，好的规划能显著提高成功率。

**下一步：** 我们将学习推理边界。

---

**练习题：**

1. 设计一个学习计划生成智能体
2. 实现支持并行任务的规划器
3. 设计规划质量的评估指标。

---

<table width="100%">
   <tr>
      <td align="left"><a href="03-chain-tree-of-thought.md">&larr; 14.3 思维链与思维树</a></td>
      <td align="right"><a href="05-reasoning-boundaries.md">14.5 设计思考：推理边界 &rarr;</a></td>
   </tr>
</table>