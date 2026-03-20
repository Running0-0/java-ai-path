# 17.3 工具集成

> "工具让AI助手从对话走向行动——从说得到做得到。"

## 常用工具集

### 计算器工具

```java
package com.example.assistant.tools;

/**
 * 计算器工具
 */
public class CalculatorTool implements Tool {
    
    @Override
    public String getName() {
        return "calculator";
    }
    
    @Override
    public String getDescription() {
        return "执行数学计算，支持基本运算和复杂表达式";
    }
    
    @Override
    public ToolResult execute(Map<String, Object> params) {
        String expression = (String) params.get("expression");
        
        try {
            // 使用安全表达式求值
            double result = evaluateExpression(expression);
            return ToolResult.success(String.valueOf(result));
        } catch (Exception e) {
            return ToolResult.failure("计算错误: " + e.getMessage());
        }
    }
    
    private double evaluateExpression(String expression) {
        // 使用ScriptEngine或自定义解析器
        // 注意：需要安全检查防止代码注入
        return new ExpressionEvaluator().evaluate(expression);
    }
}
```

### 天气工具

```java
/**
 * 天气查询工具
 */
public class WeatherTool implements Tool {
    
    private final WeatherApiClient apiClient;
    
    @Override
    public ToolResult execute(Map<String, Object> params) {
        String city = (String) params.get("city");
        String date = (String) params.getOrDefault("date", "today");
        
        try {
            WeatherInfo weather = apiClient.getWeather(city, date);
            return ToolResult.success(formatWeather(weather));
        } catch (Exception e) {
            return ToolResult.failure("获取天气失败: " + e.getMessage());
        }
    }
    
    private String formatWeather(WeatherInfo weather) {
        return String.format("%s %s: %s, 温度%s°C, 湿度%s%%",
            weather.getCity(),
            weather.getDate(),
            weather.getCondition(),
            weather.getTemperature(),
            weather.getHumidity());
    }
}
```

### 文件工具

```java
/**
 * 文件操作工具
 */
public class FileTool implements Tool {
    
    private final String basePath;
    
    @Override
    public ToolResult execute(Map<String, Object> params) {
        String operation = (String) params.get("operation");
        String path = (String) params.get("path");
        
        // 安全检查
        if (!isPathSafe(path)) {
            return ToolResult.failure("路径不安全");
        }
        
        return switch (operation) {
            case "read" -> readFile(path);
            case "write" -> writeFile(path, (String) params.get("content"));
            case "list" -> listDirectory(path);
            default -> ToolResult.failure("未知操作");
        };
    }
    
    private boolean isPathSafe(String path) {
        Path target = Paths.get(basePath, path).normalize();
        Path base = Paths.get(basePath).normalize();
        return target.startsWith(base);
    }
}
```

## 工具注册

### 自动注册

```java
/**
 * 工具自动注册
 */
@Configuration
public class ToolConfiguration {
    
    @Bean
    public ToolRegistry toolRegistry(List<Tool> tools) {
        ToolRegistry registry = new ToolRegistry();
        tools.forEach(registry::register);
        return registry;
    }
    
    @Bean
    public CalculatorTool calculatorTool() {
        return new CalculatorTool();
    }
    
    @Bean
    public WeatherTool weatherTool(WeatherApiClient client) {
        return new WeatherTool(client);
    }
    
    @Bean
    public FileTool fileTool(@Value("${assistant.data.path}") String path) {
        return new FileTool(path);
    }
}
```

## 小结

本章我们完成了：

1. **常用工具**：计算器、天气、文件操作
2. **工具注册**：自动注册机制

**下一步：** 我们将实现用户界面。

---

**练习题：**

1. 添加更多实用工具（如日历、提醒）
2. 实现工具的组合调用
3. 设计工具的错误恢复机制。
