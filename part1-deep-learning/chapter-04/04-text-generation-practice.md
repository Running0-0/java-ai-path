<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 4.3 LSTM与GRU](03-lstm-and-gru.md)</span>

<span>[4.5 设计思考：时序建模的本质 →](05-design-thinking-sequential-modeling.md)</span>

</div>

---

# 4.4 实战：用Java实现文本生成

> "文本生成是RNN最迷人的应用之一——让机器学会'写作'。"

## 文本生成的原理

### 语言模型

文本生成的核心是语言模型：预测下一个词

```
给定: "今天天气"
预测: "很好"的概率

P(很好|今天天气) = ?
```

### 自回归生成

```
输入: [今天, 天气]
预测: 很好
    ↓
输入: [今天, 天气, 很好]
预测: ，
    ↓
输入: [今天, 天气, 很好, ，]
预测: 适合
    ↓
...继续生成...
```

## 项目结构

```
text-generator/
├── pom.xml
├── src/main/java/
│   └── com/example/textgen/
│       ├── TextGenerator.java      # 主程序
│       ├── model/
│       │   └── CharRNN.java        # 字符级RNN模型
│       ├── data/
│       │   ├── TextProcessor.java  # 文本预处理
│       │   └── Vocabulary.java     # 词表管理
│       └── utils/
│           └── SamplingUtils.java  # 采样工具
└── data/
    └── training.txt                # 训练文本
```

## 完整实现

### 1. 文本预处理

```java
package com.example.textgen.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * 文本预处理和词表管理
 */
public class TextProcessor {
    
    private Map<Character, Integer> charToIndex;
    private Map<Integer, Character> indexToChar;
    private int vocabSize;
    
    public TextProcessor() {
        charToIndex = new HashMap<>();
        indexToChar = new HashMap<>();
    }
    
    /**
     * 从文本构建词表
     */
    public void buildVocabulary(String text) {
        Set<Character> chars = new HashSet<>();
        for (char c : text.toCharArray()) {
            chars.add(c);
        }
        
        int index = 0;
        for (char c : chars) {
            charToIndex.put(c, index);
            indexToChar.put(index, c);
            index++;
        }
        
        vocabSize = chars.size();
        System.out.println("词表大小: " + vocabSize);
    }
    
    /**
     * 文本转为索引序列
     */
    public int[] textToIndices(String text) {
        int[] indices = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            indices[i] = charToIndex.get(text.charAt(i));
        }
        return indices;
    }
    
    /**
     * 索引序列转为文本
     */
    public String indicesToText(int[] indices) {
        StringBuilder sb = new StringBuilder();
        for (int idx : indices) {
            sb.append(indexToChar.get(idx));
        }
        return sb.toString();
    }
    
    /**
     * One-hot编码
     */
    public INDArray oneHot(int index) {
        INDArray vec = Nd4j.zeros(vocabSize);
        vec.putScalar(index, 1.0);
        return vec;
    }
    
    /**
     * 批量One-hot编码
     */
    public INDArray oneHotBatch(int[] indices) {
        INDArray matrix = Nd4j.zeros(indices.length, vocabSize);
        for (int i = 0; i < indices.length; i++) {
            matrix.putScalar(i, indices[i], 1.0);
        }
        return matrix;
    }
    
    public int getVocabSize() { return vocabSize; }
    public Map<Character, Integer> getCharToIndex() { return charToIndex; }
    public Map<Integer, Character> getIndexToChar() { return indexToChar; }
}
```

### 2. 字符级RNN模型

```java
package com.example.textgen.model;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.LSTM;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 字符级文本生成模型
 */
public class CharRNN {
    
    private final int vocabSize;
    private final int hiddenSize;
    private final int numLayers;
    private MultiLayerNetwork model;
    
    public CharRNN(int vocabSize, int hiddenSize, int numLayers) {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        buildModel();
    }
    
    private void buildModel() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.002))
            .weightInit(WeightInit.XAVIER);
        
        var listBuilder = builder.list();
        
        // LSTM层
        for (int i = 0; i < numLayers; i++) {
            int inputSize = (i == 0) ? vocabSize : hiddenSize;
            
            listBuilder.layer(new LSTM.Builder()
                .nIn(inputSize)
                .nOut(hiddenSize)
                .activation(Activation.TANH)
                .build());
        }
        
        // 输出层（每个时间步都输出）
        listBuilder.layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .nIn(hiddenSize)
            .nOut(vocabSize)
            .activation(Activation.SOFTMAX)
            .build());
        
        model = new MultiLayerNetwork(listBuilder.build());
        model.init();
        model.setListeners(new ScoreIterationListener(100));
    }
    
    /**
     * 训练一个序列
     */
    public void train(INDArray input, INDArray target) {
        model.fit(input, target);
    }
    
    /**
     * 预测下一个字符的概率分布
     */
    public INDArray predict(INDArray input) {
        INDArray output = model.output(input);
        return output.tensorAlongDimension(output.size(2) - 1, 1);  // 取最后一个时间步
    }
    
    /**
     * 生成文本
     */
    public String generate(TextProcessor processor, String seed, int length, double temperature) {
        StringBuilder result = new StringBuilder(seed);
        
        // 准备初始输入
        int[] indices = processor.textToIndices(seed);
        INDArray input = processor.oneHotBatch(indices);
        input = input.reshape(1, input.size(0), input.size(1));  // [1, seqLen, vocabSize]
        input = input.permute(0, 2, 1);  // [batch, vocabSize, seqLen]
        
        // 逐字符生成
        for (int i = 0; i < length; i++) {
            // 预测概率分布
            INDArray probs = predict(input);
            
            // 温度采样
            int nextChar = sample(probs, temperature);
            
            // 添加到结果
            result.append(processor.getIndexToChar().get(nextChar));
            
            // 更新输入
            input = processor.oneHot(nextChar);
            input = input.reshape(1, vocabSize, 1);
        }
        
        return result.toString();
    }
    
    /**
     * 温度采样
     */
    private int sample(INDArray probs, double temperature) {
        // 应用温度
        INDArray logProbs = Transforms.log(probs).div(temperature);
        INDArray adjustedProbs = Transforms.exp(logProbs);
        adjustedProbs.divi(adjustedProbs.sumNumber());
        
        // 按概率采样
        double r = Math.random();
        double cumProb = 0;
        for (int i = 0; i < vocabSize; i++) {
            cumProb += adjustedProbs.getDouble(i);
            if (r < cumProb) {
                return i;
            }
        }
        return vocabSize - 1;
    }
    
    public MultiLayerNetwork getModel() { return model; }
}
```

### 3. 主程序

```java
package com.example.textgen;

import com.example.textgen.data.TextProcessor;
import com.example.textgen.model.CharRNN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 文本生成器主程序
 */
public class TextGenerator {
    
    public static void main(String[] args) throws Exception {
        // 参数设置
        String dataPath = "data/training.txt";
        int hiddenSize = 256;
        int numLayers = 2;
        int seqLength = 100;
        int epochs = 50;
        int generateLength = 500;
        
        // 1. 加载文本
        System.out.println("加载训练文本...");
        String text = Files.readString(Path.of(dataPath));
        System.out.println("文本长度: " + text.length());
        
        // 2. 构建词表
        TextProcessor processor = new TextProcessor();
        processor.buildVocabulary(text);
        
        // 3. 创建模型
        System.out.println("\n创建模型...");
        CharRNN model = new CharRNN(processor.getVocabSize(), hiddenSize, numLayers);
        
        // 4. 准备训练数据
        int[] indices = processor.textToIndices(text);
        int numSequences = (indices.length - 1) / seqLength;
        
        System.out.println("\n开始训练...");
        System.out.println("序列数量: " + numSequences);
        
        // 5. 训练
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            
            for (int i = 0; i < numSequences; i++) {
                int start = i * seqLength;
                int end = start + seqLength + 1;
                
                // 输入和目标
                int[] inputIndices = Arrays.copyOfRange(indices, start, end - 1);
                int[] targetIndices = Arrays.copyOfRange(indices, start + 1, end);
                
                INDArray input = processor.oneHotBatch(inputIndices);
                INDArray target = processor.oneHotBatch(targetIndices);
                
                // 调整形状 [batch, vocabSize, seqLen]
                input = input.permute(1, 0).reshape(1, processor.getVocabSize(), seqLength);
                target = target.permute(1, 0).reshape(1, processor.getVocabSize(), seqLength);
                
                model.train(input, target);
                totalLoss += model.getModel().score();
            }
            
            // 每个epoch生成样本
            if ((epoch + 1) % 5 == 0) {
                System.out.printf("\n=== Epoch %d, 平均损失: %.4f ===%n", epoch + 1, totalLoss / numSequences);
                
                String seed = text.substring(0, 20);
                String generated = model.generate(processor, seed, 200, 0.8);
                System.out.println("生成样本:\n" + generated);
            }
        }
        
        // 6. 最终生成
        System.out.println("\n=== 最终生成 ===");
        String seed = "从前有座山";
        String result = model.generate(processor, seed, generateLength, 0.7);
        System.out.println(result);
    }
}
```

### 4. 采样策略

```java
package com.example.textgen.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 文本采样工具
 */
public class SamplingUtils {
    
    /**
     * 贪婪采样：选择概率最大的
     */
    public static int greedy(INDArray probs) {
        return probs.argMax().getInt(0);
    }
    
    /**
     * 温度采样：控制随机性
     * temperature低：更确定
     * temperature高：更随机
     */
    public static int temperature(INDArray probs, double temperature) {
        // 调整概率分布
        INDArray logProbs = Transforms.log(probs.add(1e-10)).div(temperature);
        INDArray adjusted = Transforms.exp(logProbs);
        adjusted.divi(adjusted.sumNumber());
        
        // 按概率采样
        double r = Math.random();
        double cumProb = 0;
        for (int i = 0; i < adjusted.length(); i++) {
            cumProb += adjusted.getDouble(i);
            if (r < cumProb) {
                return i;
            }
        }
        return adjusted.length() - 1;
    }
    
    /**
     * Top-K采样：从概率最高的K个中采样
     */
    public static int topK(INDArray probs, int k) {
        // 获取top-k索引
        INDArray sorted = probs.dup();
        sorted = sorted.ravel().sortDescending();
        
        // 只保留top-k
        double threshold = sorted.getDouble(k - 1);
        INDArray filtered = probs.dup();
        for (int i = 0; i < filtered.length(); i++) {
            if (filtered.getDouble(i) < threshold) {
                filtered.putScalar(i, 0);
            }
        }
        
        // 归一化
        filtered.divi(filtered.sumNumber());
        
        // 采样
        return temperature(filtered, 1.0);
    }
    
    /**
     * Nucleus采样：从累积概率达到p的最小集合中采样
     */
    public static int nucleus(INDArray probs, double p) {
        // 排序
        INDArray sorted = probs.dup().ravel().sortDescending();
        
        // 找到累积概率达到p的索引
        double cumProb = 0;
        int cutoff = 0;
        for (int i = 0; i < sorted.length(); i++) {
            cumProb += sorted.getDouble(i);
            cutoff = i;
            if (cumProb >= p) break;
        }
        
        // 过滤并采样
        double threshold = sorted.getDouble(cutoff);
        INDArray filtered = probs.dup();
        for (int i = 0; i < filtered.length(); i++) {
            if (filtered.getDouble(i) < threshold) {
                filtered.putScalar(i, 0);
            }
        }
        
        filtered.divi(filtered.sumNumber());
        return temperature(filtered, 1.0);
    }
}
```

## 训练技巧

### 1. 序列截断

```java
// 长序列分段处理
int seqLength = 100;  // 每段100个字符
for (int start = 0; start < text.length() - seqLength; start += seqLength) {
    String segment = text.substring(start, start + seqLength + 1);
    // 训练...
}
```

### 2. 批处理

```java
// 同时处理多个序列
int batchSize = 32;
for (int batch = 0; batch < numBatches; batch++) {
    List<INDArray> inputs = new ArrayList<>();
    List<INDArray> targets = new ArrayList<>();
    
    for (int i = 0; i < batchSize; i++) {
        // 准备每个样本...
    }
    
    INDArray batchInput = Nd4j.vstack(inputs);
    INDArray batchTarget = Nd4j.vstack(targets);
    model.train(batchInput, batchTarget);
}
```

### 3. 学习率调度

```java
// 随训练进行降低学习率
.updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, 0.002, 0.97)))
```

## 小结

本章我们实现了：

1. **文本预处理**：词表构建、One-hot编码
2. **字符级RNN模型**：LSTM层、输出层
3. **文本生成**：自回归生成、温度采样
4. **采样策略**：贪婪、温度、Top-K、Nucleus

**核心流程：**

```
文本 → 词表 → 索引 → One-hot → RNN → 概率分布 → 采样 → 新字符
```

**下一步：** 我们将总结RNN的设计哲学。

---

**练习题：**

1. 尝试不同的温度值，观察生成文本的变化
2. 比较字符级和词级生成的效果差异
3. 实现Top-K采样，比较与温度采样的效果

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 4.3 LSTM与GRU](03-lstm-and-gru.md)</span>

<span>[4.5 设计思考：时序建模的本质 →](05-design-thinking-sequential-modeling.md)</span>

</div>
