# 13.3 构建自定义工具

> "工具决定智能体的能力边界——构建好工具，就是扩展智能体的能力。"

## 工具构建原则

### SOLID原则应用

```
单一职责：每个工具只做一件事
  ✅ get_user_by_id
  ❌ get_user_and_update_and_send_email

开放封闭：工具行为可扩展，不修改核心
  通过配置和参数扩展功能

里氏替换：工具可替换
  设计接口，便于测试和替换

接口隔离：不强制依赖不需要的功能
  小而精的工具接口

依赖倒置：依赖抽象
  通过接口注入依赖
```

## 常见工具类型

### HTTP请求工具

```java
package com.example.agent.tools;

import okhttp3.*;
import com.google.gson.Gson;

/**
 * HTTP请求工具
 */
public class HttpRequestTool extends AbstractTool {
    
    private final OkHttpClient httpClient;
    private final Gson gson;
    
    public HttpRequestTool() {
        super(
            "http_request",
            "发送HTTP请求，支持GET/POST方法，可以调用外部API",
            JsonSchema.builder()
                .property("url", "string", "请求URL", true)
                .property("method", "string", "请求方法：GET或POST", false)
                .property("headers", "object", "请求头", false)
                .property("body", "string", "请求体（POST时使用）", false)
                .build()
        );
        this.httpClient = new OkHttpClient();
        this.gson = new Gson();
    }
    
    @Override
    protected ToolResult doExecute(Map<String, Object> params) {
        String url = (String) params.get("url");
        String method = (String) params.getOrDefault("method", "GET");
        
        Request request = buildRequest(url, method, params);
        
        try (Response response = httpClient.newCall(request).execute()) {
            String body = response.body().string();
            
            return ToolResult.builder()
                .success(true)
                .data(Map.of(
                    "status", response.code(),
                    "body", body
                ))
                .build();
        } catch (IOException e) {
            return ToolResult.failure("HTTP请求失败: " + e.getMessage());
        }
    }
    
    private Request buildRequest(String url, String method, 
                                  Map<String, Object> params) {
        Request.Builder builder = new Request.Builder().url(url);
        
        // 添加请求头
        Map<String, String> headers = (Map<String, String>) 
            params.getOrDefault("headers", Map.of());
        headers.forEach(builder::addHeader);
        
        if ("POST".equalsIgnoreCase(method)) {
            String body = (String) params.getOrDefault("body", "");
            builder.post(RequestBody.create(body, 
                MediaType.get("application/json")));
        }
        
        return builder.build();
    }
}
```

### 代码执行工具

```java
package com.example.agent.tools;

/**
 * 代码执行工具（沙箱环境）
 */
public class CodeExecutionTool extends AbstractTool {
    
    private final SandboxExecutor sandbox;
    
    public CodeExecutionTool(SandboxExecutor sandbox) {
        super(
            "execute_code",
            "在安全沙箱中执行Python代码，返回执行结果",
            JsonSchema.builder()
                .property("code", "string", "要执行的Python代码", true)
                .property("timeout", "integer", "超时时间（秒），默认30", false)
                .build()
        );
        this.sandbox = sandbox;
    }
    
    @Override
    protected ToolResult doExecute(Map<String, Object> params) {
        String code = (String) params.get("code");
        int timeout = (int) params.getOrDefault("timeout", 30);
        
        // 安全检查
        if (containsDangerousCode(code)) {
            return ToolResult.failure("代码包含不允许的操作");
        }
        
        // 在沙箱中执行
        ExecutionResult result = sandbox.execute(code, timeout);
        
        if (result.isSuccess()) {
            return ToolResult.builder()
                .success(true)
                .data(Map.of(
                    "stdout", result.getStdout(),
                    "return_value", result.getReturnValue()
                ))
                .build();
        } else {
            return ToolResult.failure(result.getError());
        }
    }
    
    private boolean containsDangerousCode(String code) {
        List<String> blacklist = List.of(
            "import os", "import sys", "import subprocess",
            "exec(", "eval(", "__import__"
        );
        return blacklist.stream().anyMatch(code::contains);
    }
}
```

### 文件操作工具

```java
package com.example.agent.tools;

/**
 * 文件操作工具
 */
public class FileTools {
    
    private final String allowedBasePath;
    
    @Tool("读取文件内容")
    public String readFile(@P("文件路径（相对路径）") String path) {
        Path safePath = validatePath(path);
        try {
            return Files.readString(safePath);
        } catch (IOException e) {
            throw new ToolException("无法读取文件: " + e.getMessage());
        }
    }
    
    @Tool("写入文件内容")
    public String writeFile(
        @P("文件路径（相对路径）") String path,
        @P("文件内容") String content
    ) {
        Path safePath = validatePath(path);
        try {
            Files.writeString(safePath, content);
            return "文件写入成功：" + path;
        } catch (IOException e) {
            throw new ToolException("无法写入文件: " + e.getMessage());
        }
    }
    
    @Tool("列出目录中的文件")
    public List<String> listFiles(@P("目录路径（相对路径）") String dir) {
        Path safePath = validatePath(dir);
        try {
            return Files.list(safePath)
                .map(p -> p.getFileName().toString())
                .collect(Collectors.toList());
        } catch (IOException e) {
            throw new ToolException("无法列出目录: " + e.getMessage());
        }
    }
    
    /**
     * 路径安全检查
     */
    private Path validatePath(String path) {
        Path base = Paths.get(allowedBasePath).toAbsolutePath();
        Path target = base.resolve(path).toAbsolutePath();
        
        if (!target.startsWith(base)) {
            throw new SecurityException("路径不在允许范围内");
        }
        
        return target;
    }
}
```

## 工具测试

### 单元测试

```java
package com.example.agent.tools;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * 工具测试示例
 */
public class ToolTest {
    
    private ProductRepository mockRepository;
    private ProductTools productTools;
    
    @BeforeEach
    void setUp() {
        mockRepository = mock(ProductRepository.class);
        productTools = new ProductTools(mockRepository);
    }
    
    @Test
    void testSearchProducts() {
        // 准备测试数据
        when(mockRepository.search("手机", null, null, null))
            .thenReturn(List.of(new Product("1", "iPhone 15")));
        
        // 执行
        List<Product> result = productTools.searchProducts(
            "手机", null, null, null);
        
        // 验证
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals("iPhone 15", result.get(0).getName());
    }
    
    @Test
    void testSearchProducts_EmptyResult() {
        when(mockRepository.search("不存在的商品", null, null, null))
            .thenReturn(List.of());
        
        List<Product> result = productTools.searchProducts(
            "不存在的商品", null, null, null);
        
        assertTrue(result.isEmpty());
    }
    
    @Test
    void testSearchProducts_MissingRequired() {
        // 应该抛出异常
        assertThrows(IllegalArgumentException.class, () -> {
            productTools.searchProducts(null, null, null, null);
        });
    }
}
```

## 工具组合

### 工具链

```java
/**
 * 工具链实现
 */
public class ToolChain {
    
    private final List<ToolStep> steps;
    
    /**
     * 执行工具链
     */
    public ToolResult execute(Map<String, Object> initialInput) {
        Map<String, Object> context = new HashMap<>(initialInput);
        ToolResult lastResult = null;
        
        for (ToolStep step : steps) {
            // 解析参数（支持上一步结果注入）
            Map<String, Object> params = resolveParams(
                step.getParams(), context);
            
            // 执行
            lastResult = step.getTool().execute(params);
            
            if (!lastResult.isSuccess() && step.isRequired()) {
                return ToolResult.failure(
                    "步骤 " + step.getName() + " 失败: " + 
                    lastResult.getError());
            }
            
            // 将结果放入上下文
            context.put(step.getOutputKey(), lastResult.getData());
        }
        
        return lastResult;
    }
}
```

## 小结

本章我们学习了：

1. **构建原则**：SOLID原则在工具设计中的应用
2. **常见工具**：HTTP请求、代码执行、文件操作
3. **安全考虑**：路径验证、代码沙箱
4. **工具测试**：单元测试最佳实践
5. **工具组合**：工具链的实现

**关键认识：**
好的工具是安全、可靠、可测试的。工具的安全设计尤为重要。

**下一步：** 我们将学习数据库工具。

---

**练习题：**

1. 实现一个安全的Shell命令执行工具
2. 为HTTP工具添加认证支持
3. 设计一个工具链来完成多步骤数据处理。
