<table width="100%">
   <tr>
      <td align="left"><a href="02-react-framework.md">&larr; 14.2 ReAct框架</a></td>
      <td align="right"><a href="04-build-planning-agent.md">14.4 构建规划智能体 &rarr;</a></td>
   </tr>
</table>
---

# 14.3 思维链与思维树

> "从线性思考到树形探索——让智能体能够探索多种可能性。"

## 思维链（Chain-of-Thought）

### CoT原理

```
思维链：
通过显式展示推理步骤，提高LLM的推理能力。

标准提示：
问题：15 + 28 = ?
答案：43

CoT提示：
问题：15 + 28 = ?
思考：
1. 个位数相加：5 + 8 = 13，写3进1
2. 十位数相加：1 + 2 = 3，加上进位1得4
3. 结果是43
答案：43

优势：
- 提高复杂问题准确率
- 可解释性强
- 易于调试
```

### 零样本CoT

```
不需要示例，只需添加"逐步思考"：

提示：
"请逐步思考以下问题：
一个农场有鸡和兔，头共35个，脚共94只。
鸡和兔各有多少只？"

LLM会自动：
1. 设鸡有x只，兔有y只
2. 列方程：x + y = 35, 2x + 4y = 94
3. 解方程...
```

### Java实现

```java
package com.example.agent.cot;

/**
 * 思维链实现
 */
public class ChainOfThought {
    
    private final ChatLanguageModel llm;
    
    /**
     * 使用CoT解决问题
     */
    public String solve(String problem) {
        String prompt = String.format("""
            请逐步思考以下问题：
            
            问题：%s
            
            请详细展示每一步的思考过程。
            """, problem);
        
        return llm.generate(prompt);
    }
    
    /**
     * 带验证的CoT
     */
    public String solveWithVerification(String problem) {
        // 1. 生成思考过程
        String reasoning = solve(problem);
        
        // 2. 验证推理
        String verification = verify(problem, reasoning);
        
        // 3. 如有错误，重新推理
        if (hasErrors(verification)) {
            return solveWithCorrection(problem, reasoning, verification);
        }
        
        return reasoning;
    }
    
    private String verify(String problem, String reasoning) {
        String prompt = String.format("""
            请验证以下推理过程是否正确：
            
            问题：%s
            推理：%s
            
            如果有错误，请指出。
            """, problem, reasoning);
        
        return llm.generate(prompt);
    }
}
```

## 思维树（Tree of Thoughts）

### ToT概念

```
思维树：
维护多个思考路径，探索不同可能性。

与CoT的区别：
CoT：单一路径 A → B → C → 答案
ToT：多路径探索
      A1 → B1 → C1
     /    ↓
    A2 → B2 → C2
     \
      A3 → B3 → C3
      
选择最佳路径得到答案。
```

### ToT算法

```
1. 思考分解
   将问题分解为多个思考步骤
   
2. 生成候选
   每个步骤生成多个候选思考
   
3. 状态评估
   评估每个候选的价值
   
4. 搜索算法
   - BFS：广度优先，探索所有可能
   - DFS：深度优先，深入单一路径
   - 集束搜索：保留Top-K
   
5. 回溯优化
   遇到死胡同，回溯重试
```

### Java实现

```java
package com.example.agent.tot;

import java.util.*;

/**
 * 思维树实现
 */
public class TreeOfThoughts {
    
    private final ChatLanguageModel llm;
    private final int numCandidates;
    private final int maxDepth;
    
    /**
     * 节点
     */
    class ThoughtNode {
        String content;
        double score;
        List<ThoughtNode> children;
        ThoughtNode parent;
        int depth;
        
        ThoughtNode(String content, ThoughtNode parent) {
            this.content = content;
            this.parent = parent;
            this.depth = parent == null ? 0 : parent.depth + 1;
            this.children = new ArrayList<>();
        }
    }
    
    /**
     * 使用BFS搜索
     */
    public String solve(String problem) {
        // 根节点
        ThoughtNode root = new ThoughtNode(problem, null);
        Queue<ThoughtNode> queue = new LinkedList<>();
        queue.add(root);
        
        List<ThoughtNode> leaves = new ArrayList<>();
        
        while (!queue.isEmpty() && leaves.size() < 10) {
            ThoughtNode node = queue.poll();
            
            if (node.depth >= maxDepth) {
                leaves.add(node);
                continue;
            }
            
            // 生成候选
            List<String> candidates = generateCandidates(node);
            
            for (String candidate : candidates) {
                ThoughtNode child = new ThoughtNode(candidate, node);
                child.score = evaluate(child);
                node.children.add(child);
                
                if (isSolution(child)) {
                    return extractAnswer(child);
                }
                
                queue.add(child);
            }
        }
        
        // 选择最佳叶子节点
        return extractAnswer(selectBest(leaves));
    }
    
    /**
     * 生成候选思考
     */
    private List<String> generateCandidates(ThoughtNode node) {
        String prompt = String.format("""
            基于以下思考，生成%d个可能的下一步：
            
            当前思考：%s
            
            请生成%d个不同的下一步思考。
            """, numCandidates, node.content, numCandidates);
        
        String response = llm.generate(prompt);
        return parseCandidates(response);
    }
    
    /**
     * 评估思考价值
     */
    private double evaluate(ThoughtNode node) {
        String prompt = String.format("""
            评估以下思考的价值（0-10分）：
            
            思考：%s
            
            评分：""", node.content);
        
        String score = llm.generate(prompt).trim();
        return Double.parseDouble(score);
    }
}
```

## 对比与选择

### 方法对比

| 方法 | 复杂度 | 适用场景 | 计算成本 |
|------|--------|----------|----------|
| CoT | 低 | 单路径推理 | 低 |
| Self-Consistency | 中 | 需要验证 | 中 |
| ToT | 高 | 多路径探索 | 高 |

### 选择建议

```
选择CoT：
- 问题有明确解决路径
- 计算资源有限
- 需要快速响应

选择ToT：
- 问题需要探索多种方案
- 可以承受较高计算成本
- 需要找到最优解
```

## 小结

本章我们学习了：

1. **思维链（CoT）**：显式推理步骤
2. **零样本CoT**：无需示例的推理
3. **思维树（ToT）**：多路径探索
4. **选择建议**：根据场景选择方法

**关键认识：**
CoT和ToT提供了不同的推理方式，根据问题复杂度选择合适的方法。

**下一步：** 我们将学习构建规划智能体。

---

**练习题：**

1. 用CoT解决一个数学证明题
2. 用ToT设计一个旅行路线规划
3. 比较CoT和ToT在相同问题上的表现。

---

<table width="100%">
   <tr>
      <td align="left"><a href="02-react-framework.md">&larr; 14.2 ReAct框架</a></td>
      <td align="right"><a href="04-build-planning-agent.md">14.4 构建规划智能体 &rarr;</a></td>
   </tr>
</table>