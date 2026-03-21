<table width="100%">
   <tr>
      <td align="left"><a href="04-design-prompt-templates.md">&larr; 10.4 提示模板与复用</a></td>
      <td align="right"><a href="../chapter-11/01-project-background-architecture.md">11.1 项目架构设计 &rarr;</a></td>
   </tr>
</table>
---

# 10.5 设计思考：提示即编程

> "提示工程正在演变为一种新编程范式——用自然语言编写程序，让AI执行。"

## 编程范式的演进

### 历史回顾

```
编程范式演进：

1. 机器语言（1940s）
   01010100 直接操作硬件

2. 汇编语言（1950s）
   MOV AX, BX 符号化机器指令

3. 高级语言（1960s-）
   C, Java, Python 更接近人类思维

4. 声明式编程（1970s-）
   SQL, HTML 描述要什么，不描述怎么做

5. 提示编程（2020s-）
   自然语言描述意图，AI生成实现
```

### 提示 vs 传统编程

| 维度 | 传统编程 | 提示编程 |
|------|----------|----------|
| 语言 | 形式化语法 | 自然语言 |
| 确定性 | 100%确定 | 概率性 |
| 执行 | 精确执行 | 解释执行 |
| 调试 | 断点、日志 | 提示迭代 |
| 版本 | 代码版本 | 提示版本 |
| 测试 | 单元测试 | 输出评估 |

## 提示即编程的特征

### 1. 意图驱动

```
传统编程：
```java
public List<String> extractEmails(String text) {
    List<String> emails = new ArrayList<>();
    Pattern pattern = Pattern.compile("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b");
    Matcher matcher = pattern.matcher(text);
    while (matcher.find()) {
        emails.add(matcher.group());
    }
    return emails;
}
```

提示编程：
"从以下文本中提取所有邮箱地址，以JSON数组返回"

对比：
- 传统：详细描述"怎么做"
- 提示：描述"要什么"
```

### 2. 上下文依赖

```
提示的效果高度依赖上下文：

同样的提示，不同模型 → 不同结果
同样的提示，不同温度 → 不同结果
同样的提示，不同历史 → 不同结果

这就像：
同样的代码，不同运行时环境 → 不同行为
```

### 3. 涌现能力

```
大模型的涌现现象：

小规模模型：只能完成简单任务
大规模模型：突然展现出推理、规划能力

这类似于：
简单程序：按指令执行
复杂系统：产生意想不到的智能行为
```

## 提示工程方法论

### 开发流程

```
提示工程开发流程：

1. 需求分析
   - 明确输入输出
   - 定义成功标准

2. 提示设计
   - 编写初始提示
   - 选择合适模式

3. 迭代优化
   - 测试多种输入
   - 分析失败案例
   - 调整提示

4. 评估验证
   - 建立测试集
   - 计算准确率
   - 边界测试

5. 部署监控
   - 上线运行
   - 收集反馈
   - 持续优化
```

### 版本管理

```java
/**
 * 提示版本管理
 */
public class PromptVersionManager {
    
    private final Map<String, PromptVersion> versions;
    
    /**
     * 提示版本
     */
    class PromptVersion {
        String id;
        String template;
        String description;
        LocalDateTime createdAt;
        Map<String, Double> metrics;  // 准确率等指标
    }
    
    /**
     * A/B测试
     */
    public String selectPrompt(String task, String userId) {
        // 根据用户分组
        int bucket = userId.hashCode() % 100;
        
        if (bucket < 50) {
            return versions.get(task + "_v1").template;
        } else {
            return versions.get(task + "_v2").template;
        }
    }
}
```

## 提示编程的边界

### 适用场景

```
适合提示编程：
✓ 自然语言处理任务
✓ 创意生成（写作、设计）
✓ 需要灵活性的任务
✓ 快速原型开发
✓ 规则复杂的判断任务

不适合提示编程：
✗ 需要100%确定性的任务
✗ 数学计算（容易出错）
✗ 安全关键系统
✗ 性能要求极高的场景
✗ 需要精确控制流程
```

### 混合架构

```
最佳实践：提示 + 传统编程

示例：智能客服系统

用户问题
    ↓
[意图识别] → 提示编程
    ↓
[路由决策] → 传统编程（确定性）
    ↓
[答案生成] → 提示编程
    ↓
[结果验证] → 传统编程（规则检查）
    ↓
[输出格式化] → 传统编程

结合两者优势：
- 提示处理不确定性任务
- 代码处理确定性逻辑
```

## 未来展望

### 趋势预测

```
提示编程的发展趋势：

1. 提示编译器
   - 优化提示结构
   - 自动选择最佳表述

2. 提示类型系统
   - 输入输出类型检查
   - 静态分析提示质量

3. 提示IDE
   - 提示调试工具
   - 可视化提示流程
   - 自动补全建议

4. 提示标准库
   - 常用提示模板
   - 最佳实践集合
```

### 对Java程序员的影响

```
Java程序员的应对策略：

1. 拥抱变化
   - 学习提示工程
   - 理解AI能力边界

2. 技能升级
   - 传统编程 + AI能力
   - 架构设计能力更重要

3. 专注优势
   - 工程化能力
   - 系统设计
   - 性能优化

4. 工具链整合
   - LangChain4j等框架
   - AI能力集成到Java应用
```

## 小结

本章我们思考了：

1. **范式演进**：从机器语言到提示编程
2. **提示特征**：意图驱动、上下文依赖、涌现能力
3. **工程方法**：开发流程、版本管理
4. **能力边界**：适用与不适用的场景
5. **未来趋势**：提示编译器、类型系统

**关键认识：**
提示编程不是取代传统编程，而是新的抽象层次。聪明的工程师会结合两者，发挥各自优势。

**下一步：** 我们将进入第三部分，学习RAG应用开发。

---

**思考题：**

1. 你的工作中哪些任务适合用提示编程？
2. 如何设计一个混合提示和代码的系统？
3. 提示编程对软件工程实践有什么影响？

---

<table width="100%">
   <tr>
      <td align="left"><a href="04-design-prompt-templates.md">&larr; 10.4 提示模板与复用</a></td>
      <td align="right"><a href="../chapter-11/01-project-background-architecture.md">11.1 项目架构设计 &rarr;</a></td>
   </tr>
</table>