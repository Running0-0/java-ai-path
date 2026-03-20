# 10.3 结构化输出

> "让AI输出结构化数据——从自由文本到JSON、XML，让程序能直接使用。"

## 为什么需要结构化输出

### 问题

```
自由文本的问题：
- 难以解析
- 格式不统一
- 容易出错
- 需要额外处理

示例：
模型输出："用户的名字是张三，年龄25岁，邮箱zhangsan@example.com"

提取信息困难：
- 需要正则表达式
- 格式可能变化
- 容易遗漏字段
```

### 解决方案

```
结构化输出：

{
  "name": "张三",
  "age": 25,
  "email": "zhangsan@example.com"
}

优势：
- 直接解析为对象
- 格式固定
- 类型安全
- 易于验证
```

## JSON输出

### 基础方法

```
提示技巧：

"请分析以下文本，提取用户信息，
以JSON格式返回，包含name、age、email字段：

文本：'用户的名字是张三，年龄25岁，邮箱zhangsan@example.com'

要求：
1. 只返回JSON，不要其他内容
2. 确保JSON格式正确
3. 如果信息缺失，使用null"

输出：
{
  "name": "张三",
  "age": 25,
  "email": "zhangsan@example.com"
}
```

### Java解析

```java
package com.example.prompt;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonSyntaxException;

/**
 * JSON输出处理器
 */
public class JsonOutputHandler {
    
    private final Gson gson = new Gson();
    private final ChatLanguageModel model;
    
    /**
     * 提取用户信息
     */
    public UserInfo extractUserInfo(String text) {
        String prompt = String.format("""
            从以下文本中提取用户信息，以JSON格式返回。
            字段：name（字符串）、age（整数）、email（字符串）
            只返回JSON，不要其他内容。
            
            文本：%s
            
            JSON：""", text);
        
        String json = model.generate(prompt);
        
        // 清理可能的Markdown代码块
        json = json.replaceAll("```json\\s*", "")
                   .replaceAll("```\\s*", "")
                   .trim();
        
        try {
            return gson.fromJson(json, UserInfo.class);
        } catch (JsonSyntaxException e) {
            throw new RuntimeException("JSON解析失败: " + json, e);
        }
    }
}

class UserInfo {
    String name;
    Integer age;
    String email;
}
```

## 函数调用模式

### 定义Schema

```
使用函数调用确保结构化输出：

定义函数：
{
  "name": "extract_user_info",
  "description": "从文本中提取用户信息",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "用户姓名"
      },
      "age": {
        "type": "integer",
        "description": "用户年龄"
      },
      "email": {
        "type": "string",
        "description": "用户邮箱"
      }
    },
    "required": ["name"]
  }
}

模型会自动调用函数，返回结构化参数。
```

### LangChain4j实现

```java
package com.example.prompt;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;

/**
 * 结构化提取服务
 */
interface InformationExtractor {
    
    @UserMessage("从以下文本提取信息：{{it}}")
    ExtractedInfo extract(String text);
}

class ExtractedInfo {
    String name;
    int age;
    String email;
    String phone;
}

public class StructuredExtraction {
    
    public void demonstrate() {
        InformationExtractor extractor = AiServices
            .builder(InformationExtractor.class)
            .chatLanguageModel(model)
            .build();
        
        ExtractedInfo info = extractor.extract(
            "张三，25岁，联系方式：zhangsan@example.com，13800138000"
        );
        
        System.out.println("姓名：" + info.name);
        System.out.println("年龄：" + info.age);
        System.out.println("邮箱：" + info.email);
        System.out.println("电话：" + info.phone);
    }
}
```

## 复杂结构输出

### 列表和嵌套

```
提取多个项目：

"请从以下会议记录中提取所有待办事项，
以JSON数组格式返回，每个事项包含：
- task: 任务描述
- assignee: 负责人
- deadline: 截止日期（ISO格式）
- priority: 优先级（high/medium/low）"

输出格式：
{
  "todos": [
    {
      "task": "完成API文档",
      "assignee": "张三",
      "deadline": "2024-03-15",
      "priority": "high"
    },
    {
      "task": "修复登录bug",
      "assignee": "李四",
      "deadline": "2024-03-10",
      "priority": "medium"
    }
  ]
}
```

### Java处理

```java
/**
 * 复杂结构处理
 */
public class ComplexStructureHandler {
    
    /**
     * 提取待办事项
     */
    public List<TodoItem> extractTodos(String meetingNotes) {
        String prompt = String.format("""
            从会议记录中提取待办事项，以JSON格式返回。
            
            结构：
            {
              "todos": [
                {
                  "task": "任务描述",
                  "assignee": "负责人",
                  "deadline": "YYYY-MM-DD",
                  "priority": "high/medium/low"
                }
              ]
            }
            
            会议记录：
            %s
            
            JSON：""", meetingNotes);
        
        String json = model.generate(prompt);
        TodoList list = gson.fromJson(json, TodoList.class);
        return list.todos;
    }
}

class TodoList {
    List<TodoItem> todos;
}

class TodoItem {
    String task;
    String assignee;
    String deadline;
    String priority;
}
```

## 输出验证

### Schema验证

```java
/**
 * JSON Schema验证
 */
public class OutputValidator {
    
    /**
     * 验证输出是否符合预期结构
     */
    public boolean validate(String json, JsonSchema schema) {
        try {
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
            
            // 检查必需字段
            for (String field : schema.getRequiredFields()) {
                if (!obj.has(field)) {
                    return false;
                }
            }
            
            // 检查字段类型
            for (Map.Entry<String, Class<?>> entry : 
                 schema.getFieldTypes().entrySet()) {
                JsonElement elem = obj.get(entry.getKey());
                if (!isValidType(elem, entry.getValue())) {
                    return false;
                }
            }
            
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * 重试机制
     */
    public String generateWithValidation(String prompt, 
                                         JsonSchema schema,
                                         int maxRetries) {
        for (int i = 0; i < maxRetries; i++) {
            String output = model.generate(prompt);
            if (validate(output, schema)) {
                return output;
            }
            
            // 添加修正提示
            prompt += "\n\n上次的输出格式不正确，请确保符合JSON格式。";
        }
        
        throw new RuntimeException("无法生成有效输出");
    }
}
```

## 小结

本章我们学习了：

1. **结构化输出必要性**：从文本到数据
2. **JSON输出**：最常用的结构化格式
3. **函数调用**：确保输出结构
4. **复杂结构**：列表、嵌套对象
5. **输出验证**：确保格式正确

**关键认识：**
结构化输出让LLM成为数据处理流水线的一环，实现自动化处理。

**下一步：** 我们将学习提示模板。

---

**练习题：**

1. 设计一个提取商品信息的JSON schema
2. 实现一个带验证的结构化输出流程
3. 处理模型返回无效JSON的情况。
