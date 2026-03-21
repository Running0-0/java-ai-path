<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 17.3 工具集成](03-tool-integration.md)</span>

<span>[17.5 部署与优化 &rarr;](05-deployment-optimization.md)</span>

</div>
---

# 17.4 用户界面

> "好的界面让AI助手更易用——从命令行到Web，满足不同场景需求。"

## 命令行界面

### CLI实现

```java
package com.example.assistant.ui;

import java.util.Scanner;

/**
 * 命令行界面
 */
public class CliInterface {
    
    private final PersonalAssistant assistant;
    private final Scanner scanner;
    
    public void start() {
        System.out.println("个人AI助手已启动！输入'退出'结束。");
        
        while (true) {
            System.out.print("\n你: ");
            String input = scanner.nextLine();
            
            if ("退出".equalsIgnoreCase(input)) {
                System.out.println("再见！");
                break;
            }
            
            System.out.println("助手: " + assistant.process(input));
        }
    }
}
```

## Web界面

### Spring Boot实现

```java
package com.example.assistant.ui;

import org.springframework.web.bind.annotation.*;

/**
 * Web API
 */
@RestController
@RequestMapping("/api")
public class AssistantController {
    
    private final PersonalAssistant assistant;
    
    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest request) {
        String response = assistant.process(request.getMessage());
        return new ChatResponse(response);
    }
    
    @GetMapping("/history")
    public List<Message> getHistory() {
        return assistant.getHistory();
    }
}
```

## 小结

本章我们实现了：

1. **命令行界面**：简单直接的交互方式
2. **Web界面**：REST API实现

**下一步：** 我们将学习部署优化。

---

**练习题：**

1. 为Web界面添加SSE流式响应
2. 实现WebSocket实时通信
3. 设计移动端适配方案。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 17.3 工具集成](03-tool-integration.md)</span>

<span>[17.5 部署与优化 &rarr;](05-deployment-optimization.md)</span>

</div>