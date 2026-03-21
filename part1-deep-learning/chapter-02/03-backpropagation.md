<table width="100%">
   <tr>
      <td align="left"><a href="02-forward-propagation.md">← 2.2 前向传播</a></td>
      <td align="right"><a href="04-first-neural-network-dl4j.md">2.4 用Deeplearning4j实现你的第一个神经网络 →</a></td>
   </tr>
</table>

---

# 2.3 反向传播：学习的数学之美

> "反向传播是神经网络最优雅的发明——它让复杂的网络能够高效地学习。"

## 从一个简单问题开始

假设你训练了一个神经网络识别猫，但预测结果错了：
- 输入：一张猫的图片
- 预测：0.3（30%是猫）
- 期望：1.0（100%是猫）

**问题：如何调整成千上万个权重，让预测更准确？**

这就是反向传播要解决的问题。

## 直观理解：责任链追溯

想象一个公司出了问题：

```
CEO（输出层）
  ↑ "业绩不好，是谁的问题？"
部门经理（隐藏层2）
  ↑ "我按指示做的，问题可能在..."
组长（隐藏层1）
  ↑ "我只是执行..."
员工（输入层）
```

CEO追溯责任，找到每个环节的贡献，然后调整。反向传播就是这个过程。

### 数学语言

**核心问题：** 当输出层产生误差时，每个权重对这个误差"贡献"了多少？

**答案：** 用链式法则计算偏导数。

## 链式法则：反向传播的数学基础

### 回顾链式法则

如果 y = f(g(x))，那么：
```
dy/dx = dy/dg · dg/dx
```

**例子：**
```
y = sigmoid(w·x + b)
```

设 z = w·x + b，则 y = sigmoid(z)

```
dy/dw = dy/dz · dz/dw
      = sigmoid(z)·(1-sigmoid(z)) · x
```

### 多层网络的链式法则

对于一个三层网络：

```
x → z1 → a1 → z2 → a2 → L（损失）
```

要计算 ∂L/∂W1（损失对第一层权重的导数）：

```
∂L/∂W1 = ∂L/∂a2 · ∂a2/∂z2 · ∂z2/∂a1 · ∂a1/∂z1 · ∂z1/∂W1
```

这就是**反向传播**：从损失开始，逐层向后计算导数。

## 反向传播的完整推导

### 符号定义

| 符号 | 含义 |
|------|------|
| x | 输入 |
| W, b | 权重和偏置 |
| z = W·a + b | 加权输入 |
| a = f(z) | 激活输出 |
| L | 损失函数 |
| δ = ∂L/∂z | 误差项 |

### 四个核心公式

**公式1：输出层误差**
```
δ^L = ∂L/∂a^L ⊙ f'(z^L)
```

**公式2：隐藏层误差**
```
δ^l = ((W^(l+1))^T · δ^(l+1)) ⊙ f'(z^l)
```

**公式3：权重梯度**
```
∂L/∂W^l = δ^l · (a^(l-1))^T
```

**公式4：偏置梯度**
```
∂L/∂b^l = δ^l
```

### 用Java代码表示

```java
public class Backpropagation {
    
    // 存储中间结果
    private List<INDArray> zList;  // 各层加权输入
    private List<INDArray> aList;  // 各层激活输出
    
    /**
     * 反向传播计算梯度
     */
    public Gradients backward(INDArray x, INDArray y, 
                              List<INDArray> weights, 
                              List<INDArray> biases) {
        
        int numLayers = weights.size();
        
        // === 第一步：前向传播，存储中间结果 ===
        zList = new ArrayList<>();
        aList = new ArrayList<>();
        
        INDArray current = x;
        aList.add(current);  // 输入也算作第0层的激活
        
        for (int l = 0; l < numLayers; l++) {
            INDArray z = current.mmul(weights.get(l).transpose())
                               .addRowVector(biases.get(l));
            zList.add(z);
            current = sigmoid(z);  // 激活函数
            aList.add(current);
        }
        
        // === 第二步：反向传播，计算梯度 ===
        List<INDArray> dW = new ArrayList<>();
        List<INDArray> db = new ArrayList<>();
        
        // 输出层误差（公式1）
        INDArray delta = crossEntropyDerivative(aList.get(numLayers), y)
                        .mul(sigmoidDerivative(zList.get(numLayers - 1)));
        
        // 从后向前遍历
        for (int l = numLayers - 1; l >= 0; l--) {
            // 计算权重梯度（公式3）
            INDArray dWl = delta.transpose().mmul(aList.get(l));
            dW.add(0, dWl);
            
            // 计算偏置梯度（公式4）
            db.add(0, delta.sum(0));
            
            // 传播误差到前一层（公式2）
            if (l > 0) {
                delta = delta.mmul(weights.get(l))
                            .mul(sigmoidDerivative(zList.get(l - 1)));
            }
        }
        
        return new Gradients(dW, db);
    }
    
    // Sigmoid导数
    private INDArray sigmoidDerivative(INDArray z) {
        INDArray s = Transforms.sigmoid(z);
        return s.mul(s.rsub(1));  // s * (1 - s)
    }
    
    // 交叉熵损失对输出的导数
    private INDArray crossEntropyDerivative(INDArray a, INDArray y) {
        return a.sub(y).div(a.mul(a.rsub(1))).mul(-1);
    }
    
    record Gradients(List<INDArray> dW, List<INDArray> db) {}
}
```

## 梯度下降：更新参数

有了梯度，就可以更新权重了：

```java
public void updateWeights(List<INDArray> weights, 
                          List<INDArray> biases,
                          Gradients grads, 
                          double learningRate) {
    
    for (int l = 0; l < weights.size(); l++) {
        // W = W - η · ∂L/∂W
        weights.get(l).subi(grads.dW().get(l).mul(learningRate));
        
        // b = b - η · ∂L/∂b
        biases.get(l).subi(grads.db().get(l).mul(learningRate));
    }
}
```

### 优化器：更智能的更新策略

简单的梯度下降有问题：
- 学习率太大：震荡不收敛
- 学习率太小：收敛太慢

**常用优化器：**

#### 1. Momentum（动量）

```java
public class MomentumOptimizer {
    private List<INDArray> velocityW;
    private List<INDArray> velocityB;
    private double momentum = 0.9;
    
    public void update(List<INDArray> weights, List<INDArray> biases,
                       Gradients grads, double learningRate) {
        for (int l = 0; l < weights.size(); l++) {
            // v = γ·v + η·∂L/∂W
            velocityW.set(l, velocityW.get(l).mul(momentum)
                          .add(grads.dW().get(l).mul(learningRate)));
            
            // W = W - v
            weights.get(l).subi(velocityW.get(l));
            
            // 偏置同理
            velocityB.set(l, velocityB.get(l).mul(momentum)
                          .add(grads.db().get(l).mul(learningRate)));
            biases.get(l).subi(velocityB.get(l));
        }
    }
}
```

#### 2. Adam（最常用）

```java
public class AdamOptimizer {
    private List<INDArray> mW, mB;  // 一阶矩
    private List<INDArray> vW, vB;  // 二阶矩
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-8;
    private int t = 0;
    
    public void update(List<INDArray> weights, List<INDArray> biases,
                       Gradients grads, double learningRate) {
        t++;
        
        for (int l = 0; l < weights.size(); l++) {
            INDArray dW = grads.dW().get(l);
            INDArray db = grads.db().get(l);
            
            // 更新一阶矩
            mW.set(l, mW.get(l).mul(beta1).add(dW.mul(1 - beta1)));
            mB.set(l, mB.get(l).mul(beta1).add(db.mul(1 - beta1)));
            
            // 更新二阶矩
            vW.set(l, vW.get(l).mul(beta2).add(dW.mul(dW).mul(1 - beta2)));
            vB.set(l, vB.get(l).mul(beta2).add(db.mul(db).mul(1 - beta2)));
            
            // 偏差修正
            INDArray mWCorrected = mW.get(l).div(1 - Math.pow(beta1, t));
            INDArray mBCorrected = mB.get(l).div(1 - Math.pow(beta1, t));
            INDArray vWCorrected = vW.get(l).div(1 - Math.pow(beta2, t));
            INDArray vBCorrected = vB.get(l).div(1 - Math.pow(beta2, t));
            
            // 更新参数
            weights.get(l).subi(
                mWCorrected.mul(learningRate)
                          .div(Transforms.sqrt(vWCorrected).add(epsilon))
            );
            biases.get(l).subi(
                mBCorrected.mul(learningRate)
                          .div(Transforms.sqrt(vBCorrected).add(epsilon))
            );
        }
    }
}
```

**Adam的优势：**
- 自适应学习率
- 收敛快且稳定
- 适合大多数场景

## 损失函数：衡量误差

### 均方误差（MSE）

```java
public double mse(INDArray predicted, INDArray actual) {
    INDArray diff = predicted.sub(actual);
    return diff.mul(diff).sumNumber().doubleValue() / predicted.size();
}
```

**适用场景：** 回归问题

### 交叉熵损失（Cross-Entropy）

```java
public double crossEntropy(INDArray predicted, INDArray actual) {
    // 避免 log(0)
    INDArray clipped = Transforms.max(predicted, 1e-10);
    INDArray logProb = Transforms.log(clipped);
    return -actual.mul(logProb).sumNumber().doubleValue();
}
```

**适用场景：** 分类问题

### 为什么分类问题用交叉熵？

| 损失函数 | 配合Sigmoid | 梯度特性 |
|----------|-------------|----------|
| MSE | × | 学习慢（梯度小） |
| 交叉熵 | ✓ | 学习快（梯度合理） |

交叉熵避免了"学习饱和"问题。

## 完整的训练循环

```java
public class NeuralNetworkTrainer {
    
    private List<INDArray> weights;
    private List<INDArray> biases;
    private double learningRate;
    
    public void train(DataSetIterator data, int epochs) {
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int numBatches = 0;
            
            while (data.hasNext()) {
                DataSet batch = data.next();
                INDArray x = batch.getFeatures();
                INDArray y = batch.getLabels();
                
                // 1. 前向传播
                ForwardResult result = forward(x);
                
                // 2. 计算损失
                double loss = crossEntropy(result.output(), y);
                totalLoss += loss;
                
                // 3. 反向传播
                Gradients grads = backward(x, y, result);
                
                // 4. 更新参数
                updateWeights(grads);
                
                numBatches++;
            }
            
            double avgLoss = totalLoss / numBatches;
            System.out.printf("Epoch %d, Loss: %.4f%n", epoch + 1, avgLoss);
            
            data.reset();
        }
    }
}
```

## 设计思考：为什么反向传播有效

### 计算效率的奇迹

假设网络有N个参数：

| 方法 | 计算复杂度 |
|------|------------|
| 数值梯度 | O(N) × 前向传播次数 |
| 反向传播 | O(N) = 2 × 前向传播次数 |

反向传播让训练大规模网络成为可能。

### 自动微分的思想

反向传播本质上是**自动微分**的一个特例：

```
前向传播：构建计算图
反向传播：遍历计算图计算梯度
```

现代深度学习框架（TensorFlow、PyTorch）都实现了自动微分。

## 实践：手写反向传播

```java
package com.example.ai.chapter02;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 手写反向传播示例：学习XOR
 */
public class BackpropagationDemo {
    
    public static void main(String[] args) {
        // XOR数据
        INDArray X = Nd4j.create(new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        INDArray Y = Nd4j.create(new double[][]{
            {0}, {1}, {1}, {0}
        });
        
        // 初始化网络
        TwoLayerNet net = new TwoLayerNet(2, 4, 1);
        
        // 训练
        System.out.println("开始训练...");
        for (int epoch = 0; epoch < 5000; epoch++) {
            // 前向传播
            ForwardResult result = net.forward(X);
            
            // 计算损失
            double loss = net.mseLoss(result.output, Y);
            
            // 反向传播
            net.backward(X, Y, result, 0.5);
            
            if (epoch % 500 == 0) {
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, loss);
            }
        }
        
        // 测试
        System.out.println("\n测试结果:");
        ForwardResult result = net.forward(X);
        for (int i = 0; i < 4; i++) {
            System.out.printf("%.0f XOR %.0f = %.3f%n",
                X.getDouble(i, 0), X.getDouble(i, 1), 
                result.output.getDouble(i));
        }
    }
    
    static class TwoLayerNet {
        private INDArray W1, b1, W2, b2;
        private double lr;
        
        public TwoLayerNet(int inputSize, int hiddenSize, int outputSize) {
            // Xavier初始化
            W1 = Nd4j.randn(hiddenSize, inputSize).mul(Math.sqrt(2.0 / inputSize));
            b1 = Nd4j.zeros(hiddenSize);
            W2 = Nd4j.randn(outputSize, hiddenSize).mul(Math.sqrt(2.0 / hiddenSize));
            b2 = Nd4j.zeros(outputSize);
        }
        
        public ForwardResult forward(INDArray x) {
            // 隐藏层
            INDArray z1 = x.mmul(W1.transpose()).addRowVector(b1);
            INDArray a1 = Transforms.tanh(z1);
            
            // 输出层
            INDArray z2 = a1.mmul(W2.transpose()).addRowVector(b2);
            INDArray a2 = Transforms.sigmoid(z2);
            
            return new ForwardResult(z1, a1, z2, a2);
        }
        
        public void backward(INDArray x, INDArray y, ForwardResult result, double learningRate) {
            int batchSize = x.size(0);
            
            // 输出层误差
            INDArray delta2 = result.output.sub(y).div(batchSize);
            
            // 隐藏层到输出层的梯度
            INDArray dW2 = delta2.transpose().mmul(result.a1);
            INDArray db2 = delta2.sum(0);
            
            // 隐藏层误差
            INDArray delta1 = delta2.mmul(W2).mul(tanhDerivative(result.z1));
            
            // 输入层到隐藏层的梯度
            INDArray dW1 = delta1.transpose().mmul(x);
            INDArray db1 = delta1.sum(0);
            
            // 更新参数
            W1.subi(dW1.mul(learningRate));
            b1.subi(db1.mul(learningRate));
            W2.subi(dW2.mul(learningRate));
            b2.subi(db2.mul(learningRate));
        }
        
        private INDArray tanhDerivative(INDArray z) {
            INDArray t = Transforms.tanh(z);
            return t.mul(t).rsub(1);  // 1 - tanh^2(z)
        }
        
        public double mseLoss(INDArray predicted, INDArray actual) {
            INDArray diff = predicted.sub(actual);
            return diff.mul(diff).meanNumber().doubleValue();
        }
    }
    
    record ForwardResult(INDArray z1, INDArray a1, INDArray z2, INDArray output) {}
}
```

## 小结

本章我们学习了：

1. **反向传播的本质**：用链式法则计算梯度
2. **四个核心公式**：误差传播和梯度计算
3. **优化器**：SGD、Momentum、Adam
4. **损失函数**：MSE和交叉熵
5. **完整训练流程**：前向→损失→反向→更新

**核心思想：**

```
误差反向传播，梯度指导更新
```

**下一步：** 我们将用Deeplearning4j框架实现真正的神经网络。

---

**思考题：**

1. 为什么反向传播比数值梯度计算快？
2. Adam优化器相比SGD有什么优势？
3. 为什么分类问题用交叉熵而不用MSE？

---

<table width="100%">
   <tr>
      <td align="left"><a href="02-forward-propagation.md">← 2.2 前向传播</a></td>
      <td align="right"><a href="04-first-neural-network-dl4j.md">2.4 用Deeplearning4j实现你的第一个神经网络 →</a></td>
   </tr>
</table>