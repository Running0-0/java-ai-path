# 12.4 从工具使用者到创造者

> "智能体的终极进化——不仅能使用工具，还能创造工具。"

## 工具使用的层次

### 四个层次

```
智能体与工具的关系演进：

Level 1: 工具使用者（Tool User）
- 调用预定义工具
- 按说明书使用
- 被动响应

Level 2: 工具组合者（Tool Composer）
- 组合多个工具
- 创建工作流
- 解决复杂问题

Level 3: 工具定制者（Tool Customizer）
- 配置工具参数
- 调整工具行为
- 适配特定场景

Level 4: 工具创造者（Tool Creator）
- 创建新工具
- 编写代码
- 扩展能力边界
```

## Level 1：工具使用者

### 基础工具调用

```java
/**
 * 工具使用者
 */
public class ToolUser {
    
    private final ToolRegistry registry;
    
    /**
     * 使用工具
     */
    public ToolResult useTool(String toolName, Map<String, Object> params) {
        Tool tool = registry.getTool(toolName);
        return tool.execute(params);
    }
    
    /**
     * LLM选择工具
     */
    public String handleWithTool(String userInput) {
        // LLM分析需要什么工具
        String toolSelection = llm.generate(String.format("""
            用户请求：%s
            
            可用工具：
            - search: 搜索信息
            - calculator: 计算
            - weather: 查询天气
            
            请输出需要使用的工具名称和参数（JSON格式）。
            如果不需要工具，输出"none"。
            """, userInput));
        
        if (toolSelection.equals("none")) {
            return llm.generate(userInput);
        }
        
        // 解析工具调用
        ToolCall call = parseToolCall(toolSelection);
        ToolResult result = useTool(call.getName(), call.getParams());
        
        // LLM基于结果回答
        return llm.generate(String.format("""
            用户问题：%s
            工具结果：%s
            
            请基于工具结果回答用户。
            """, userInput, result));
    }
}
```

## Level 2：工具组合者

### 工作流编排

```java
/**
 * 工具组合者
 */
public class ToolComposer {
    
    /**
     * 创建复杂工作流
     */
    public Workflow createWorkflow(String goal) {
        // LLM规划工具组合
        String plan = llm.generate(String.format("""
            目标：%s
            
            可用工具：
            %s
            
            请设计一个工作流，说明：
            1. 需要哪些步骤
            2. 每个步骤使用什么工具
            3. 步骤之间的数据传递
            
            输出JSON格式的工作流定义。
            """, goal, registry.formatTools()));
        
        return parseWorkflow(plan);
    }
    
    /**
     * 执行工作流
     */
    public WorkflowResult executeWorkflow(Workflow workflow, 
                                           Map<String, Object> initialContext) {
        Map<String, Object> context = new HashMap<>(initialContext);
        
        for (Step step : workflow.getSteps()) {
            // 准备参数（支持变量替换）
            Map<String, Object> params = resolveParams(
                step.getParams(), context);
            
            // 执行工具
            ToolResult result = executeTool(step.getTool(), params);
            
            // 保存结果到上下文
            context.put(step.getOutputKey(), result);
            
            // 检查条件分支
            if (step.hasCondition()) {
                boolean conditionMet = evaluateCondition(
                    step.getCondition(), context);
                if (!conditionMet) {
                    continue;
                }
            }
        }
        
        return new WorkflowResult(context);
    }
}
```

## Level 3：工具定制者

### 动态配置

```java
/**
 * 工具定制者
 */
public class ToolCustomizer {
    
    /**
     * 为特定场景定制工具
     */
    public Tool customizeTool(String baseToolName, 
                               String scenario,
                               Requirements requirements) {
        Tool baseTool = registry.getTool(baseToolName);
        
        // LLM生成定制配置
        String config = llm.generate(String.format("""
            基础工具：%s
            使用场景：%s
            特殊要求：%s
            
            请生成工具配置参数，包括：
            1. 默认参数值
            2. 参数验证规则
            3. 错误处理策略
            4. 输出格式化
            
            输出JSON配置。
            """, baseTool.getName(), scenario, requirements));
        
        return new CustomizedTool(baseTool, parseConfig(config));
    }
    
    /**
     * 创建领域专用版本
     */
    public Tool createDomainVersion(String baseTool, String domain) {
        return customizeTool(baseTool, domain, 
            Requirements.builder()
                .add("使用" + domain + "专业术语")
                .add("遵循" + domain + "行业标准")
                .build()
        );
    }
}
```

## Level 4：工具创造者

### 代码生成

```java
/**
 * 工具创造者
 */
public class ToolCreator {
    
    private final CodeExecutor codeExecutor;
    
    /**
     * 从需求创建工具
     */
    public Tool createTool(String requirement) {
        // 1. 分析需求
        ToolSpec spec = analyzeRequirement(requirement);
        
        // 2. 生成代码
        String code = llm.generate(String.format("""
            请创建一个Java工具类，实现以下功能：
            
            需求：%s
            
            工具规范：
            - 类名：%s
            - 实现Tool接口
            - 包含execute方法
            - 添加适当的错误处理
            - 包含单元测试
            
            输出完整代码。
            """, requirement, spec.getClassName()));
        
        // 3. 验证代码
        if (!validateCode(code)) {
            // 修复代码
            code = fixCode(code);
        }
        
        // 4. 编译并加载
        Class<?> toolClass = codeExecutor.compileAndLoad(code);
        
        // 5. 实例化
        try {
            return (Tool) toolClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new ToolCreationException("无法实例化工具", e);
        }
    }
    
    /**
     * 从示例创建工具
     */
    public Tool createToolFromExample(String description, 
                                       List<Example> examples) {
        String code = llm.generate(String.format("""
            请根据描述和示例创建工具：
            
            描述：%s
            
            示例：
            %s
            
            请推断工具的功能，生成实现代码。
            """, description, formatExamples(examples)));
        
        return compileAndLoad(code);
    }
}
```

## 实践：自动工具创建

### 完整示例

```java
/**
 * 自动工具创建演示
 */
public class AutoToolCreationDemo {
    
    public void demonstrate() {
        ToolCreator creator = new ToolCreator();
        
        // 用户描述需求
        String requirement = """
            我需要一个工具，可以计算两个日期之间的工作日天数，
            排除周末和中国法定节假日。
            输入：开始日期、结束日期（格式：yyyy-MM-dd）
            输出：工作日天数
            """;
        
        // 创建工具
        Tool workdayCalculator = creator.createTool(requirement);
        
        // 注册并使用
        registry.register(workdayCalculator);
        
        // 使用新创建的工具
        ToolResult result = workdayCalculator.execute(Map.of(
            "startDate", "2024-01-01",
            "endDate", "2024-01-31"
        ));
        
        System.out.println("工作日天数：" + result.getOutput());
    }
}
```

## 小结

本章我们学习了：

1. **四个层次**：使用者→组合者→定制者→创造者
2. **工具使用**：基础工具调用
3. **工具组合**：工作流编排
4. **工具定制**：场景适配
5. **工具创造**：代码生成

**关键认识：**
智能体的能力边界在不断扩展，从使用工具到创造工具，实现真正的自主能力。

**下一步：** 我们将学习Function Calling。

---

**练习题：**

1. 设计一个自动创建数据库查询工具的系统
2. 如何实现工具的安全验证？
3. 工具创造者的局限和风险是什么？
