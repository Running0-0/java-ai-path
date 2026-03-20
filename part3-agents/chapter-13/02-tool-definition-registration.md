# 13.2 工具定义与注册

> "清晰的工具定义是智能体正确使用工具的前提——好的说明书决定好的使用效果。"

## 工具定义规范

### 什么是好的工具定义

```
工具定义包含：

1. 名称（name）
   - 动词_名词格式：get_weather、send_email
   - 简洁明确
   - 不含歧义

2. 描述（description）
   - 说明工具的功能
   - 说明适用场景
   - 提供使用示例

3. 参数（parameters）
   - 参数名称和类型
   - 参数描述
   - 必填/可选标注
   - 取值范围

4. 返回值
   - 返回类型
   - 可能的错误
```

### JSON Schema格式

```json
{
  "name": "search_products",
  "description": "在商品数据库中搜索产品，支持关键词、分类和价格范围过滤",
  "parameters": {
    "type": "object",
    "properties": {
      "keyword": {
        "type": "string",
        "description": "搜索关键词，如'手机'、'笔记本'",
        "examples": ["iPhone", "华为Mate"]
      },
      "category": {
        "type": "string",
        "description": "商品分类",
        "enum": ["电子产品", "服装", "食品", "图书"]
      },
      "min_price": {
        "type": "number",
        "description": "最低价格（人民币元）",
        "minimum": 0
      },
      "max_price": {
        "type": "number",
        "description": "最高价格（人民币元）"
      },
      "limit": {
        "type": "integer",
        "description": "返回结果数量",
        "default": 10,
        "maximum": 100
      }
    },
    "required": ["keyword"]
  }
}
```

## Java工具定义

### 注解方式

```java
package com.example.agent.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.util.List;

/**
 * 商品工具集
 */
public class ProductTools {
    
    private final ProductRepository repository;
    private final InventoryService inventoryService;
    
    @Tool("""
        搜索商品。可以按关键词、分类和价格范围搜索。
        适用于：用户询问商品、查找产品、比较价格等场景。
        """)
    public List<Product> searchProducts(
        @P("搜索关键词") String keyword,
        @P("商品分类（可选）：电子产品、服装、食品、图书") String category,
        @P("最低价格（可选，单位：元）") Double minPrice,
        @P("最高价格（可选，单位：元）") Double maxPrice
    ) {
        return repository.search(keyword, category, minPrice, maxPrice);
    }
    
    @Tool("获取商品详细信息，包括规格、库存、评价")
    public ProductDetail getProductDetail(
        @P("商品ID") String productId
    ) {
        Product product = repository.findById(productId);
        int inventory = inventoryService.getStock(productId);
        List<Review> reviews = repository.getReviews(productId);
        
        return new ProductDetail(product, inventory, reviews);
    }
    
    @Tool("查询商品实时库存")
    public InventoryStatus checkInventory(
        @P("商品ID") String productId,
        @P("查询数量（可选）") Integer quantity
    ) {
        int stock = inventoryService.getStock(productId);
        boolean available = quantity == null || stock >= quantity;
        
        return new InventoryStatus(stock, available);
    }
}
```

### 接口方式

```java
package com.example.agent.tools;

/**
 * 工具接口
 */
public interface ToolDefinition {
    
    String getName();
    String getDescription();
    JsonSchema getParameterSchema();
    
    /**
     * 执行工具
     */
    ToolResult execute(Map<String, Object> params);
}

/**
 * 工具基类
 */
public abstract class AbstractTool implements ToolDefinition {
    
    private final String name;
    private final String description;
    private final JsonSchema schema;
    
    protected AbstractTool(String name, String description, 
                            JsonSchema schema) {
        this.name = name;
        this.description = description;
        this.schema = schema;
    }
    
    @Override
    public ToolResult execute(Map<String, Object> params) {
        // 1. 参数验证
        validate(params);
        
        // 2. 执行
        try {
            return doExecute(params);
        } catch (Exception e) {
            return ToolResult.failure(
                "工具执行失败: " + e.getMessage());
        }
    }
    
    protected abstract ToolResult doExecute(Map<String, Object> params);
    
    private void validate(Map<String, Object> params) {
        schema.getRequired().forEach(field -> {
            if (!params.containsKey(field)) {
                throw new IllegalArgumentException(
                    "缺少必填参数: " + field);
            }
        });
    }
}
```

## 工具注册中心

### 注册表实现

```java
package com.example.agent.tools;

import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具注册中心
 */
public class ToolRegistry {
    
    private final Map<String, ToolDefinition> tools = 
        new ConcurrentHashMap<>();
    
    /**
     * 注册工具
     */
    public void register(ToolDefinition tool) {
        if (tools.containsKey(tool.getName())) {
            throw new IllegalStateException(
                "工具已存在: " + tool.getName());
        }
        tools.put(tool.getName(), tool);
    }
    
    /**
     * 批量注册
     */
    public void registerAll(List<ToolDefinition> toolList) {
        toolList.forEach(this::register);
    }
    
    /**
     * 注销工具
     */
    public void unregister(String toolName) {
        tools.remove(toolName);
    }
    
    /**
     * 获取工具
     */
    public Optional<ToolDefinition> getTool(String name) {
        return Optional.ofNullable(tools.get(name));
    }
    
    /**
     * 获取所有工具的描述（用于构建提示）
     */
    public String getToolsDescription() {
        return tools.values().stream()
            .map(t -> String.format("- %s: %s", 
                t.getName(), t.getDescription()))
            .collect(Collectors.joining("\n"));
    }
    
    /**
     * 获取工具Schema（用于Function Calling）
     */
    public List<FunctionSchema> getFunctionSchemas() {
        return tools.values().stream()
            .map(t -> new FunctionSchema(
                t.getName(), 
                t.getDescription(), 
                t.getParameterSchema()))
            .collect(Collectors.toList());
    }
}
```

### Spring集成

```java
package com.example.agent.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 工具自动注册配置
 */
@Configuration
public class ToolRegistryConfig {
    
    @Bean
    public ToolRegistry toolRegistry(List<ToolDefinition> tools) {
        ToolRegistry registry = new ToolRegistry();
        
        // 自动注册所有标注@ToolComponent的Bean
        tools.forEach(registry::register);
        
        return registry;
    }
}

/**
 * 自动发现工具
 */
@Component
@ConditionalOnProperty("ai.tools.auto-register")
public class AutoToolRegistrar implements ApplicationRunner {
    
    @Autowired
    private ToolRegistry registry;
    
    @Autowired
    private ApplicationContext context;
    
    @Override
    public void run(ApplicationArguments args) {
        // 扫描所有ToolDefinition实现
        context.getBeansOfType(ToolDefinition.class)
            .values()
            .forEach(tool -> {
                registry.register(tool);
                log.info("已注册工具: {}", tool.getName());
            });
    }
}
```

## 工具版本管理

### 版本化工具

```java
/**
 * 版本化工具注册
 */
public class VersionedToolRegistry extends ToolRegistry {
    
    private final Map<String, Map<String, ToolDefinition>> versionedTools = 
        new ConcurrentHashMap<>();
    
    public void register(ToolDefinition tool, String version) {
        versionedTools
            .computeIfAbsent(tool.getName(), k -> new HashMap<>())
            .put(version, tool);
        
        // 注册最新版本为默认
        super.register(tool);
    }
    
    public Optional<ToolDefinition> getTool(String name, String version) {
        return Optional.ofNullable(
            versionedTools.getOrDefault(name, Map.of()).get(version)
        );
    }
}
```

## 小结

本章我们学习了：

1. **工具定义规范**：名称、描述、参数、返回值
2. **JSON Schema**：标准化的参数描述
3. **Java实现**：注解方式和接口方式
4. **注册中心**：工具管理和发现
5. **Spring集成**：自动注册

**关键认识：**
良好的工具定义让LLM能够准确理解和使用工具，是智能体系统的基础。

**下一步：** 我们将学习如何构建自定义工具。

---

**练习题：**

1. 为文件操作功能设计一套工具定义
2. 实现一个工具注册中心的REST API
3. 设计工具的权限控制机制。
