<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-12/04-tool-user-to-creator.md">&larr; 12.4 从工具使用者到创造者</a></td>
      <td align="right"><a href="02-tool-definition-registration.md">13.2 工具定义与注册 &rarr;</a></td>
   </tr>
</table>
---

# 13.1 Function Calling基础

> "Function Calling是智能体的基石——让LLM从说话到行动。"

## 什么是Function Calling

### 概念

```
Function Calling（函数调用）：

允许LLM生成结构化输出，调用外部函数/工具。

流程：
1. 定义可用函数（名称、参数、描述）
2. LLM分析用户输入
3. LLM决定是否需要调用函数
4. 如果需要，生成函数调用参数
5. 执行函数，返回结果
6. LLM基于结果生成最终回答

类比：
- LLM是大脑：决定做什么
- Function Calling是手：执行操作
```

### 为什么重要

```
Function Calling的价值：

1. 扩展能力
   - 获取实时信息
   - 执行复杂计算
   - 操作外部系统

2. 提高准确性
   - 使用计算器避免数学错误
   - 查询数据库获取准确数据
   - 调用API获取实时信息

3. 结构化输出
   - 参数类型安全
   - 易于解析执行
   - 可验证
```

## Function Calling流程

### 完整流程

```
用户：北京今天天气怎么样？

Step 1: 定义函数
{
  "name": "get_weather",
  "description": "获取指定城市的天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "date": {"type": "string"}
    },
    "required": ["city"]
  }
}

Step 2: LLM分析
- 用户询问天气
- 需要调用 get_weather
- 参数：city="北京"

Step 3: 生成调用
{
  "name": "get_weather",
  "arguments": {"city": "北京"}
}

Step 4: 执行函数
get_weather(city="北京") → "晴天，25°C"

Step 5: LLM生成回答
"北京今天晴天，气温25°C。"
```

## Java实现

### 基础实现

```java
package com.example.functioncalling;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.List;
import java.util.Map;

/**
 * Function Calling实现
 */
public class FunctionCallingExample {
    
    private final ChatLanguageModel llm;
    private final ObjectMapper mapper = new ObjectMapper();
    
    /**
     * 定义函数
     */
    public static class FunctionDefinition {
        String name;
        String description;
        JsonSchema parameters;
        
        public FunctionDefinition(String name, String description, 
                                   JsonSchema parameters) {
            this.name = name;
            this.description = description;
            this.parameters = parameters;
        }
    }
    
    /**
     * 函数调用请求
     */
    public static class FunctionCall {
        String name;
        Map<String, Object> arguments;
    }
    
    /**
     * 处理用户请求
     */
    public String handleRequest(String userInput, 
                                 List<FunctionDefinition> functions) {
        // 1. 构建提示
        String prompt = buildPrompt(userInput, functions);
        
        // 2. LLM生成响应
        String response = llm.generate(prompt);
        
        // 3. 解析响应
        FunctionCall call = parseFunctionCall(response);
        
        if (call == null) {
            // 无需调用函数，直接返回
            return response;
        }
        
        // 4. 执行函数
        Object result = executeFunction(call);
        
        // 5. LLM基于结果生成最终回答
        return generateFinalAnswer(userInput, call, result);
    }
    
    private String buildPrompt(String userInput, 
                               List<FunctionDefinition> functions) {
        StringBuilder sb = new StringBuilder();
        sb.append("你可以使用以下函数：\n\n");
        
        for (FunctionDefinition fn : functions) {
            sb.append(String.format("函数：%s\n", fn.name));
            sb.append(String.format("描述：%s\n", fn.description));
            sb.append(String.format("参数：%s\n\n", 
                fn.parameters.toString()));
        }
        
        sb.append("用户请求：").append(userInput).append("\n\n");
        sb.append("如果需要调用函数，请输出JSON格式：\n");
        sb.append("{\"name\": \"函数名\", \"arguments\": {参数}}\n\n");
        sb.append("如果不需要调用函数，直接回答。");
        
        return sb.toString();
    }
    
    private FunctionCall parseFunctionCall(String response) {
        try {
            JsonNode node = mapper.readTree(response);
            if (node.has("name")) {
                FunctionCall call = new FunctionCall();
                call.name = node.get("name").asText();
                call.arguments = mapper.convertValue(
                    node.get("arguments"), Map.class);
                return call;
            }
        } catch (Exception e) {
            // 不是函数调用格式
        }
        return null;
    }
}
```

## 与LangChain4j集成

### 使用注解

```java
package com.example.functioncalling;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.service.AiServices;

/**
 * LangChain4j Function Calling
 */
public class LangChain4jFunctionCalling {
    
    /**
     * 工具类
     */
    static class Tools {
        
        @Tool("获取指定城市的当前天气")
        public String getWeather(String city) {
            // 调用天气API
            return weatherApi.getCurrent(city);
        }
        
        @Tool("计算数学表达式")
        public double calculate(String expression) {
            // 安全计算
            return safeEvaluate(expression);
        }
        
        @Tool("查询用户信息")
        public UserInfo getUserInfo(@P("用户ID") String userId) {
            return userRepository.findById(userId);
        }
    }
    
    /**
     * 助手接口
     */
    interface Assistant {
        String chat(String message);
    }
    
    public void demonstrate() {
        // 创建带工具的助手
        Assistant assistant = AiServices.builder(Assistant.class)
            .chatLanguageModel(model)
            .tools(new Tools())
            .build();
        
        // 使用
        String response = assistant.chat("北京今天天气怎么样？");
        // LLM会自动调用 getWeather("北京")
        
        String response2 = assistant.chat("123 * 456 等于多少？");
        // LLM会自动调用 calculate("123 * 456")
    }
}
```

## 最佳实践

### 函数设计原则

```
1. 单一职责
   - 每个函数只做一件事
   - 函数名清晰描述功能

2. 参数明确
   - 参数名具有描述性
   - 提供参数示例
   - 说明参数约束

3. 错误处理
   - 函数返回错误信息
   - LLM能基于错误调整

4. 描述详细
   - 函数描述清晰
   - 说明使用场景
   - 提供示例
```

### 示例函数定义

```java
/**
 * 良好设计的函数
 */
public class WellDesignedFunctions {
    
    @Tool("""
        发送邮件给指定收件人。
        
        使用场景：
        - 用户要求发送邮件
        - 需要通知相关人员
        
        示例：
        sendEmail(
            to="zhangsan@example.com",
            subject="会议通知",
            body="明天下午2点开会"
        )
        """)
    public String sendEmail(
        @P("收件人邮箱地址") String to,
        @P("邮件主题") String subject,
        @P("邮件正文，支持HTML") String body
    ) {
        // 实现
    }
    
    @Tool("""
        查询数据库，执行SELECT语句。
        
        注意：
        -

---

<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-12/04-tool-user-to-creator.md">&larr; 12.4 从工具使用者到创造者</a></td>
      <td align="right"><a href="02-tool-definition-registration.md">13.2 工具定义与注册 &rarr;</a></td>
   </tr>
</table>