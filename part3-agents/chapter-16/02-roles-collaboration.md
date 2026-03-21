<table width="100%">
   <tr>
      <td align="left"><a href="01-multi-agent-overview.md">&larr; 16.1 多智能体概述</a></td>
      <td align="right"><a href="03-communication-protocol.md">16.3 通信协议 &rarr;</a></td>
   </tr>
</table>
---

# 16.2 角色与协作

> "明确的角色分工是高效协作的基础——每个智能体都知道自己该做什么。"

## 角色设计

### 角色定义

```
角色包含：

1. 职责（Responsibility）
   - 主要职责
   - 工作范围
   - 交付物

2. 能力（Capability）
   - 专业技能
   - 可用工具
   - 知识领域

3. 权限（Authority）
   - 决策权
   - 资源访问
   - 指挥权

4. 协作关系（Relationship）
   - 上下游
   - 依赖关系
   - 汇报线
```

### Java实现

```java
package com.example.agent.multi.roles;

/**
 * 智能体角色
 */
public class AgentRole {
    
    private final String name;
    private final String description;
    private final List<String> responsibilities;
    private final List<String> capabilities;
    private final Map<String, Authority> authorities;
    private final List<CollaborationRelation> relations;
    
    public AgentRole(String name, String description) {
        this.name = name;
        this.description = description;
        this.responsibilities = new ArrayList<>();
        this.capabilities = new ArrayList<>();
        this.authorities = new HashMap<>();
        this.relations = new ArrayList<>();
    }
    
    /**
     * 添加职责
     */
    public AgentRole addResponsibility(String responsibility) {
        responsibilities.add(responsibility);
        return this;
    }
    
    /**
     * 添加能力
     */
    public AgentRole addCapability(String capability) {
        capabilities.add(capability);
        return this;
    }
    
    /**
     * 定义协作关系
     */
    public AgentRole collaborateWith(String otherRole, 
                                      CollaborationType type) {
        relations.add(new CollaborationRelation(otherRole, type));
        return this;
    }
    
    /**
     * 获取系统提示词
     */
    public String toSystemPrompt() {
        StringBuilder prompt = new StringBuilder();
        prompt.append("你是").append(name).append("。\n");
        prompt.append("职责：\n");
        responsibilities.forEach(r -> 
            prompt.append("- ").append(r).append("\n"));
        prompt.append("能力：\n");
        capabilities.forEach(c -> 
            prompt.append("- ").append(c).append("\n"));
        return prompt.toString();
    }
}

/**
 * 预定义角色
 */
public class PredefinedRoles {
    
    public static AgentRole productManager() {
        return new AgentRole("产品经理", "负责需求分析和产品规划")
            .addResponsibility("收集和分析需求")
            .addResponsibility("制定产品规划")
            .addResponsibility("定义验收标准")
            .addCapability("需求分析")
            .addCapability("用户调研")
            .collaborateWith("架构师", CollaborationType.DIRECTS)
            .collaborateWith("开发", CollaborationType.DIRECTS);
    }
    
    public static AgentRole architect() {
        return new AgentRole("架构师", "负责系统设计和技术选型")
            .addResponsibility("设计系统架构")
            .addResponsibility("技术选型")
            .addResponsibility("定义接口规范")
            .addCapability("系统设计")
            .addCapability("技术评估")
            .collaborateWith("产品经理", CollaborationType.REPORTS_TO)
            .collaborateWith("开发", CollaborationType.DIRECTS);
    }
    
    public static AgentRole developer() {
        return new AgentRole("开发工程师", "负责代码实现")
            .addResponsibility("编写代码")
            .addResponsibility("单元测试")
            .addResponsibility("代码审查")
            .addCapability("编程")
            .addCapability("调试")
            .collaborateWith("架构师", CollaborationType.REPORTS_TO)
            .collaborateWith("测试", CollaborationType.HANDOFF);
    }
}
```

## 协作模式

### 协作类型

```
协作模式：

1. 顺序协作（Sequential）
   A → B → C → D
   
   适用：流水线作业
   示例：需求→设计→开发→测试

2. 并行协作（Parallel）
      → B →
   A →     → E
      → C →
      → D →
   
   适用：任务分解
   示例：多个开发并行开发不同模块

3. 迭代协作（Iterative）
   A → B → C → (反馈) → A
   
   适用：需要反复打磨
   示例：开发→测试→修复→测试

4. 协商协作（Negotiation）
   A ↔ B ↔ C
   
   适用：需要达成共识
   示例：技术方案讨论
```

### 协作实现

```java
/**
 * 协作工作流
 */
public class CollaborationWorkflow {
    
    private final MultiAgentSystem system;
    
    /**
     * 顺序协作
     */
    public WorkflowResult sequential(List<String> agentIds, Task task) {
        Task currentTask = task;
        
        for (String agentId : agentIds) {
            Agent agent = system.getAgent(agentId);
            TaskResult result = agent.execute(currentTask);
            
            if (!result.isSuccess()) {
                return WorkflowResult.failure(result.getError());
            }
            
            // 输出作为下一个输入
            currentTask = new Task(result.getOutput());
        }
        
        return WorkflowResult.success(currentTask.getContent());
    }
    
    /**
     * 并行协作
     */
    public WorkflowResult parallel(List<String> agentIds, Task task) {
        List<SubTask> subTasks = decompose(task, agentIds.size());
        
        // 并行执行
        List<Future<TaskResult>> futures = new ArrayList<>();
        for (int i = 0; i < agentIds.size(); i++) {
            Agent agent = system.getAgent(agentIds.get(i));
            SubTask subTask = subTasks.get(i);
            
            futures.add(executor.submit(() -> agent.execute(subTask)));
        }
        
        // 收集结果
        List<TaskResult> results = new ArrayList<>();
        for (Future<TaskResult> future : futures) {
            try {
                results.add(future.get());
            } catch (Exception e) {
                return WorkflowResult.failure(e.getMessage());
            }
        }
        
        // 合并结果
        return aggregate(results);
    }
}
```

## 冲突解决

### 冲突类型

```
常见冲突：

1. 资源冲突
   - 多个智能体需要同一资源
   - 解决方案：锁机制、排队、优先级

2. 决策冲突
   - 对同一问题有不同意见
   - 解决方案：投票、仲裁、协商

3. 依赖冲突
   - 循环依赖
   - 解决方案：重新设计、打破循环

4. 信息冲突
   - 数据不一致
   - 解决方案：权威源、共识机制
```

## 小结

本章我们学习了：

1. **角色设计**：职责、能力、权限、关系
2. **协作模式**：顺序、并行、迭代、协商
3. **冲突解决**：资源、决策、依赖、信息

**关键认识：**
明确的角色和协作模式是多智能体系统高效运行的基础。

**下一步：** 我们将学习通信协议。

---

**练习题：**

1. 设计一个内容创作团队的角色分工
2. 实现一个支持迭代协作的工作流
3. 设计多智能体冲突解决机制。

---

<table width="100%">
   <tr>
      <td align="left"><a href="01-multi-agent-overview.md">&larr; 16.1 多智能体概述</a></td>
      <td align="right"><a href="03-communication-protocol.md">16.3 通信协议 &rarr;</a></td>
   </tr>
</table>