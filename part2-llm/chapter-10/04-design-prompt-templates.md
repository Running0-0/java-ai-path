<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 10.3 结构化输出](03-structured-output.md)</span>

<span>[10.5 设计思考：提示即编程 &rarr;](05-prompt-as-programming.md)</span>

</div>
---

# 10.4 提示模板与复用

> "提示模板让提示工程工业化——可维护、可复用、可版本控制。"

## 为什么需要模板

### 问题

```
硬编码提示的问题：

1. 维护困难
   - 散落在代码各处
   - 修改需要改多处

2. 难以复用
   - 相似功能重复编写
   - 风格不统一

3. 无法动态调整
   - 参数写死
   - 难以A/B测试

4. 团队协作难
   - 没有版本控制
   - 无法评审
```

### 模板解决方案

```
模板的优势：

1. 集中管理
   - 提示存储在配置文件
   - 统一维护

2. 参数化
   - 使用变量占位
   - 运行时填充

3. 版本控制
   - 提示可以git管理
   - 支持回滚

4. 可测试
   - 独立测试提示
   - A/B测试不同版本
```

## 模板设计

### 基础模板

```
模板语法（类似Mustache）：

Hello, {{name}}!
Welcome to {{service}}.

渲染：
name = "张三"
service = "AI助手"

结果：
Hello, 张三!
Welcome to AI助手.
```

### Java模板引擎

```java
package com.example.template;

import com.github.mustachejava.DefaultMustacheFactory;
import com.github.mustachejava.Mustache;
import com.github.mustachejava.MustacheFactory;

import java.io.StringReader;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * 提示模板管理器
 */
public class PromptTemplateManager {
    
    private final MustacheFactory mustacheFactory;
    private final Map<String, String> templates;
    
    public PromptTemplateManager() {
        this.mustacheFactory = new DefaultMustacheFactory();
        this.templates = new HashMap<>();
        loadTemplates();
    }
    
    /**
     * 加载模板
     */
    private void loadTemplates() {
        // 从配置文件加载
        templates.put("translation", """
            请将以下文本翻译成{{targetLang}}：
            
            原文：{{text}}
            
            要求：
            {{#requirements}}
            - {{.}}
            {{/requirements}}
            
            只返回翻译结果。""");
        
        templates.put("code_review", """
            请审查以下{{language}}代码：
            
            ```{{language}}
            {{code}}
            ```
            
            关注以下方面：
            {{#aspects}}
            - {{.}}
            {{/aspects}}
            
            请提供具体的改进建议。""");
    }
    
    /**
     * 渲染模板
     */
    public String render(String templateName, Map<String, Object> context) {
        String templateStr = templates.get(templateName);
        if (templateStr == null) {
            throw new IllegalArgumentException("模板不存在: " + templateName);
        }
        
        Mustache mustache = mustacheFactory.compile(
            new StringReader(templateStr), templateName);
        
        StringWriter writer = new StringWriter();
        mustache.execute(writer, context);
        
        return writer.toString();
    }
}
```

## 模板使用示例

### 翻译模板

```java
/**
 * 翻译服务
 */
public class TranslationService {
    
    private final PromptTemplateManager templateManager;
    private final ChatLanguageModel model;
    
    public String translate(String text, String targetLang) {
        // 准备上下文
        Map<String, Object> context = new HashMap<>();
        context.put("text", text);
        context.put("targetLang", targetLang);
        context.put("requirements", Arrays.asList(
            "保持原文语气",
            "专业术语准确",
            "语句通顺自然"
        ));
        
        // 渲染模板
        String prompt = templateManager.render("translation", context);
        
        // 调用模型
        return model.generate(prompt);
    }
}
```

### 代码审查模板

```java
/**
 * AI代码审查
 */
public class AICodeReview {
    
    public ReviewResult review(String code, String language) {
        Map<String, Object> context = new HashMap<>();
        context.put("code", code);
        context.put("language", language);
        context.put("aspects", Arrays.asList(
            "代码规范",
            "潜在bug",
            "性能优化",
            "可读性"
        ));
        
        String prompt = templateManager.render("code_review", context);
        String response = model.generate(prompt);
        
        return parseReviewResult(response);
    }
}
```

## 高级模板功能

### 条件渲染

```
模板中的条件：

{{#isTechnical}}
请使用技术术语解释。
{{/isTechnical}}
{{^isTechnical}}
请用通俗语言解释。
{{/isTechnical}}

{{#examples}}
示例 {{number}}:
{{content}}

{{/examples}}
```

### 模板继承

```
基础模板（base）：

你是一位{{role}}。
{{#context}}
背景信息：
{{context}}
{{/context}}

{{content}}

{{#constraints}}
约束条件：
{{.}}
{{/constraints}}

---

继承模板（translation extends base）：

role = "专业翻译"
content = "请将{{text}}翻译成{{targetLang}}"
constraints = ["保持原意", "语句通顺"]
```

## 模板管理最佳实践

### 版本控制

```yaml
# prompts/translation/v1.yaml
name: translation
version: "1.0"
description: "通用翻译模板"
template: |
  请将以下文本翻译成{{targetLang}}：
  {{text}}
variables:
  - name: text
    type: string
    required: true
  - name: targetLang
    type: string
    required: true
  - name: style
    type: string
    default: "formal"
```

### A/B测试

```java
/**
 * 模板A/B测试
 */
public class TemplateABTest {
    
    private final Map<String, List<String>> testVersions;
    
    /**
     * 随机选择模板版本
     */
    public String selectVersion(String templateName) {
        List<String> versions = testVersions.get(templateName);
        int index = (int) (Math.random() * versions.size());
        return versions.get(index);
    }
    
    /**
     * 记录结果
     */
    public void recordResult(String version, boolean success) {
        // 记录到分析系统
        analytics.track(templateName, version, success);
    }
}
```

## 小结

本章我们学习了：

1. **模板必要性**：解决硬编码提示的问题
2. **模板设计**：参数化、可复用
3. **模板引擎**：Mustache等工具
4. **高级功能**：条件渲染、模板继承
5. **最佳实践**：版本控制、A/B测试

**关键认识：**
提示模板让提示工程从手工活变成工程化实践，提高开发效率和提示质量。

**下一步：** 我们将学习提示即编程。

---

**练习题：**

1. 为你的项目设计3个常用提示模板
2. 实现一个带条件渲染的模板
3. 设计模板版本管理方案。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 10.3 结构化输出](03-structured-output.md)</span>

<span>[10.5 设计思考：提示即编程 &rarr;](05-prompt-as-programming.md)</span>

</div>