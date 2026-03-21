<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 13.5 安全性与可控性](../chapter-13/05-security-controllability.md)</span>

<span>[14.2 ReAct框架 &rarr;](02-react-framework.md)</span>

</div>
---

# 14.1 任务分解

> "复杂任务简单化——分解是智能体解决复杂问题的核心能力。"

## 为什么需要任务分解

### 复杂性挑战

```
复杂任务的挑战：

用户请求："帮我策划一次北京到上海的3天旅行，
          包括交通、住宿、景点、美食，
          预算5000元以内"

一次性处理的问题：
- 信息量大，难以全面考虑
- 容易遗漏重要环节
- 逻辑复杂，容易出错
- 难以追踪进度

分解后的优势：
- 每个子任务简单明确
- 可以并行或顺序执行
- 错误隔离，易于调试
- 进度可追踪
```

## 分解策略

### 分解方法

```
1. 按步骤分解（Sequential）
   任务A → 任务B → 任务C
   
   示例：写报告
   - 收集资料
   - 撰写大纲
   - 填充内容
   - 审核修改

2. 按模块分解（Modular）
   任务A
   任务B
   任务C
   ↓
   整合结果
   
   示例：旅行规划
   - 交通规划
   - 住宿预订
   - 景点安排
   - 美食推荐

3. 按层次分解（Hierarchical）
   主任务
   ├── 子任务1
   │   ├── 子子任务1.1
   │   └── 子子任务1.2
   └── 子任务2
       └── 子子任务2.1
```

## Java实现

### 任务定义

```java
package com.example.agent.planning;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 任务定义
 */
public interface Task {
    
    String getId();
    String getDescription();
    TaskStatus getStatus();
    
    /**
     * 执行任务
     */
    TaskResult execute(TaskContext context);
    
    /**
     * 获取依赖任务
     */
    List<Task> getDependencies();
    
    /**
     * 是否可以并行执行
     */
    boolean isParallelizable();
}

/**
 * 复合任务（可分解）
 */
public interface CompositeTask extends Task {
    
    /**
     * 分解为子任务
     */
    List<Task> decompose();
    
    /**
     * 合并子任务结果
     */
    TaskResult aggregate(List<TaskResult> subResults);
}
```

### 分解器实现

```java
/**
 * LLM驱动的任务分解器
 */
public class LLMTaskDecomposer {
    
    private final ChatLanguageModel llm;
    
    /**
     * 分解复杂任务
     */
    public List<Task> decompose(String complexTask) {
        String prompt = String.format("""
            请将以下复杂任务分解为具体的子任务。
            
            任务：%s
            
            要求：
            1. 每个子任务应该是可独立执行的
            2. 明确子任务之间的依赖关系
            3. 为每个子任务分配优先级
            4. 输出JSON格式
            
            示例输出格式：
            {
              "subtasks": [
                {
                  "id": "1",
                  "description": "子任务描述",
                  "dependencies": [],
                  "priority": 1
                }
              ]
            }
            """, complexTask);
        
        String response = llm.generate(prompt);
        return parseTasks(response);
    }
    
    /**
     * 递归分解
     */
    public TaskTree recursiveDecompose(String task, int maxDepth) {
        TaskTree tree = new TaskTree(task);
        
        if (maxDepth <= 0) {
            return tree;
        }
        
        List<Task> subtasks = decompose(task);
        
        for (Task subtask : subtasks) {
            // 如果子任务仍然复杂，继续分解
            if (isComplex(subtask)) {
                TaskTree childTree = recursiveDecompose(
                    subtask.getDescription(), maxDepth - 1);
                tree.addChild(childTree);
            } else {
                tree.addChild(new TaskTree(subtask));
            }
        }
        
        return tree;
    }
}
```

## 任务执行引擎

### 执行器实现

```java
/**
 * 任务执行引擎
 */
public class TaskExecutionEngine {
    
    private final ExecutorService executor;
    private final TaskExecutor taskExecutor;
    
    /**
     * 执行任务图
     */
    public ExecutionResult execute(TaskGraph graph) {
        Map<String, TaskResult> results = new ConcurrentHashMap<>();
        Set<String> completed = ConcurrentHashMap.newKeySet();
        
        // 拓扑排序
        List<List<Task>> levels = graph.topologicalLevels();
        
        for (List<Task> level : levels) {
            // 同层任务并行执行
            List<CompletableFuture<Void>> futures = level.stream()
                .filter(task -> canExecute(task, completed))
                .map(task -> CompletableFuture.runAsync(() -> {
                    TaskResult result = taskExecutor.execute(task);
                    results.put(task.getId(), result);
                    completed.add(task.getId());
                }, executor))
                .collect(Collectors.toList());
            
            // 等待本层完成
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .join();
        }
        
        return new ExecutionResult(results);
    }
    
    private boolean canExecute(Task task, Set<String> completed) {
        return task.getDependencies().stream()
            .allMatch(dep -> completed.contains(dep.getId()));
    }
}
```

## 实际应用

### 旅行规划示例

```java
/**
 * 旅行规划任务分解示例
 */
public class TravelPlanningExample {
    
    public void demonstrate() {
        String request = """
            帮我策划一次北京到上海的3天旅行，
            预算5000元，喜欢历史文化和美食
            """;
        
        // 分解任务
        List<Task> tasks = decomposer.decompose(request);
        
        // 输出分解结果
        System.out.println("旅行规划分解：");
        for (Task task : tasks) {
            System.out.printf("- %s (优先级: %d)%n",
                task.getDescription(),
                task.getPriority());
        }
        
        // 预期输出：
        // - 查询北京到上海的交通方式（优先级: 1）
        // - 搜索上海历史景点（优先级: 2）
        // - 查找上海特色美食（优先级: 2）
        // - 根据预算筛选住宿（优先级: 3）
        // - 制定每日行程（优先级: 4）
        // - 计算总费用（优先级: 5）
    }
}
```

## 小结

本章我们学习了：

1. **分解必要性**：复杂任务的挑战
2. **分解策略**：按步骤、模块、层次
3. **Java实现**：任务定义、分解器、执行引擎
4. **实际应用**：旅行规划示例

**关键认识：**
任务分解是智能体处理复杂问题的基础能力，好的分解是成功的一半。

**下一步：** 我们将学习ReAct框架。

---

**练习题：**

1. 设计一个代码审查任务的分解方案
2. 实现一个支持依赖关系的任务执行器
3. 如何评估任务分解的质量？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 13.5 安全性与可控性](../chapter-13/05-security-controllability.md)</span>

<span>[14.2 ReAct框架 &rarr;](02-react-framework.md)</span>

</div>