<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 10.1 提示工程基础](01-prompt-engineering-basics.md)</span>

<span>[10.3 结构化输出 &rarr;](03-structured-output.md)</span>

</div>
---

# 10.2 高级提示模式

> "掌握提示模式，就像掌握设计模式——让你能优雅地解决复杂问题。"

## 思维链（Chain-of-Thought）

### 标准CoT

```
引导模型逐步推理：

问题：一个篮球队在5场比赛中，
第一场得80分，第二场得85分，
第三场得90分，第四场得75分，
第五场得95分。平均分是多少？

请逐步解答：
第一步：列出所有得分
第二步：计算总分
第三步：除以比赛场数
第四步：得出答案

这种引导方式让模型展示推理过程，
提高复杂问题的准确率。
```

### 零样本CoT

```
不需要示例，只需添加"逐步思考"：

"请逐步思考以下问题：
一个长方形的长是10米，宽是5米，
它的面积是多少？"

模型会自动：
1. 识别这是面积计算问题
2. 回忆面积公式：长×宽
3. 代入数值：10×5
4. 得出结果：50平方米
```

### Java实现CoT

```java
/**
 * 思维链提示生成器
 */
public class ChainOfThoughtPrompt {
    
    /**
     * 生成CoT提示
     */
    public String generateCoTPrompt(String problem, List<String> steps) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("请按照以下步骤解决问题：\n\n");
        prompt.append("问题：").append(problem).append("\n\n");
        prompt.append("解决步骤：\n");
        
        for (int i = 0; i < steps.size(); i++) {
            prompt.append(i + 1).append(". ")
                  .append(steps.get(i)).append("\n");
        }
        
        prompt.append("\n请详细展示每一步的思考过程。");
        
        return prompt.toString();
    }
    
    public static void main(String[] args) {
        ChainOfThoughtPrompt cot = new ChainOfThoughtPrompt();
        
        String prompt = cot.generateCoTPrompt(
            "计算一个半径为5的圆的面积",
            Arrays.asList(
                "回忆圆的面积公式",
                "确定半径的值",
                "代入公式计算",
                "给出最终答案和单位"
            )
        );
        
        System.out.println(prompt);
    }
}
```

## 自我一致性（Self-Consistency）

### 原理

```
一个问题，多次采样，选择最一致的答案：

问题：15 + 28 = ?

生成多个回答：
- 回答1: 43
- 回答2: 43
- 回答3: 42
- 回答4: 43
- 回答5: 43

多数投票：43（4票）vs 42（1票）
最终答案：43

这种方法能显著减少随机错误。
```

### 实现

```java
/**
 * 自我一致性投票
 */
public class SelfConsistency {
    
    private final ChatLanguageModel model;
    private final int numSamples;
    
    public String generateWithConsistency(String prompt) {
        Map<String, Integer> voteCount = new HashMap<>();
        
        // 多次采样
        for (int i = 0; i < numSamples; i++) {
            // 使用较高温度增加多样性
            String response = model.generate(
                prompt, 
                GenerateParams.builder()
                    .temperature(0.8)
                    .build()
            );
            
            // 提取答案（简化处理）
            String answer = extractAnswer(response);
            voteCount.merge(answer, 1, Integer::sum);
        }
        
        // 选择得票最多的答案
        return voteCount.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElseThrow();
    }
}
```

## 思维树（Tree of Thoughts）

### 概念

```
思维链的扩展——探索多种思路：

问题：如何优化一个慢查询？

思路1：添加索引
  - 子思路1.1：分析查询条件
  - 子思路1.2：创建复合索引
  
思路2：优化SQL
  - 子思路2.1：避免SELECT *
  - 子思路2.2：添加查询条件

思路3：缓存结果
  - 子思路3.1：Redis缓存
  - 子思路3.2：应用层缓存

评估每种思路，选择最优方案。
```

## ReAct模式

### 原理

```
ReAct = Reasoning + Acting

交替进行推理和行动：

问题：北京今天天气怎么样？

思考1：我需要查询北京的天气信息
行动1：调用天气API，参数：城市=北京
观察1：API返回：晴天，25°C

思考2：我已经获得了天气信息，可以回答用户
行动2：向用户报告天气

最终答案：北京今天晴天，气温25°C。
```

### Java实现

```java
/**
 * ReAct模式实现
 */
public class ReActAgent {
    
    private final ChatLanguageModel model;
    private final List<Tool> tools;
    
    public String run(String task, int maxSteps) {
        StringBuilder history = new StringBuilder();
        
        for (int step = 0; step < maxSteps; step++) {
            // 构建提示
            String prompt = buildReActPrompt(task, history.toString());
            
            // 获取模型输出
            String response = model.generate(prompt);
            
            // 解析思考和行动
            ThoughtAction ta = parseResponse(response);
            
            if (ta.isFinalAnswer()) {
                return ta.getAnswer();
            }
            
            // 执行行动
            String observation = executeAction(ta.getAction());
            
            // 记录历史
            history.append("思考：").append(ta.getThought()).append("\n");
            history.append("行动：").append(ta.getAction()).append("\n");
            history.append("观察：").append(observation).append("\n");
        }
        
        return "达到最大步数限制";
    }
    
    private String buildReActPrompt(String task, String history) {
        return String.format("""
            完成任务：%s
            
            你可以使用以下工具：
            %s
            
            请按照以下格式回答：
            思考：你的思考过程
            行动：工具名称[参数]
            
            历史记录：
            %s
            
            现在请继续：
            """, task, formatTools(), history);
    }
}
```

## 反思与改进

### 自我反思

```
让模型评估和改进自己的回答：

第一轮回答：
"Java中的List是一个接口..."

反思提示：
"请检查以上回答的准确性和完整性。
有哪些遗漏或错误？如何改进？"

改进后的回答：
"Java中的List是java.util包下的一个接口，
继承自Collection接口。
主要实现类包括：
- ArrayList：基于数组，查询快
- LinkedList：基于链表，增删快
- Vector：线程安全（已过时）

使用场景：
..."
```

## 小结

本章我们学习了：

1. **思维链（CoT）**：逐步推理，提高准确率
2. **自我一致性**：多次采样，投票选择
3. **思维树（ToT）**：探索多种思路
4. **ReAct模式**：推理与行动交替
5. **反思改进**：自我评估和优化

**关键认识：**
高级提示模式能解决更复杂的问题，是提示工程师的进阶技能。

**下一步：** 我们将学习结构化输出。

---

**练习题：**

1. 用CoT模式解决一个复杂的逻辑问题
2. 实现一个简单的ReAct代理
3. 设计一个需要多步推理的提示。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 10.1 提示工程基础](01-prompt-engineering-basics.md)</span>

<span>[10.3 结构化输出 &rarr;](03-structured-output.md)</span>

</div>