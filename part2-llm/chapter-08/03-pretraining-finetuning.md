<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[&larr; 8.2 BERT：双向编码器表示](02-bert-model.md)</span>

<span>[8.4 实战：调用OpenAI API &rarr;](04-openai-api-practice.md)</span>

</div>
---

# 8.3 预训练与微调范式

> "预训练+微调是NLP的革命——先在大数据上学习通用知识，再在小数据上适应特定任务。"

## 预训练-微调范式

### 为什么有效

```
预训练的价值：

1. 知识迁移
   - 预训练模型学会了语言规律
   - 微调时只需要学习任务特定知识

2. 数据效率
   - 预训练使用海量无标注数据
   - 微调只需要少量标注数据

3. 计算效率
   - 预训练计算量大，但只做一次
   - 微调计算量小，可重复使用
```

### 范式对比

```
传统方法（预训练前）：
任务A数据 → 训练模型A
任务B数据 → 训练模型B
每个任务从零开始，数据需求大

预训练+微调：
海量数据 → 预训练 → 通用模型
   ↓
任务A数据 → 微调 → 模型A
任务B数据 → 微调 → 模型B
共享预训练知识，数据需求小
```

## 预训练阶段

### 数据准备

```java
/**
 * 预训练数据预处理
 */
public class PretrainingDataProcessor {
    
    /**
     * 构建训练样本
     */
    public List<TrainingSample> prepareData(List<String> documents) {
        List<TrainingSample> samples = new ArrayList<>();
        
        for (String doc : documents) {
            // 分句
            String[] sentences = segmentSentences(doc);
            
            // 构建样本
            for (int i = 0; i < sentences.length - 1; i++) {
                // 50%概率是连续句子，50%随机采样
                boolean isNext = Math.random() < 0.5;
                String sentenceB = isNext ? 
                    sentences[i + 1] : 
                    getRandomSentence(documents);
                
                samples.add(new TrainingSample(
                    sentences[i], sentenceB, isNext));
            }
        }
        
        return samples;
    }
    
    /**
     * 掩码处理（MLM）
     */
    public MaskedSample applyMasking(String[] tokens) {
        String[] masked = tokens.clone();
        int[] labels = new int[tokens.length];
        Arrays.fill(labels, -1);
        
        for (int i = 0; i < tokens.length; i++) {
            if (shouldMask(tokens[i])) {
                labels[i] = vocab.getId(tokens[i]);
                masked[i] = applyMaskStrategy(tokens[i]);
            }
        }
        
        return new MaskedSample(masked, labels);
    }
}
```

### 训练过程

```java
/**
 * 预训练流程
 */
public class PretrainingPipeline {
    
    private final TransformerModel model;
    private final Optimizer optimizer;
    
    public void pretrain(Dataset dataset, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int batchCount = 0;
            
            for (Batch batch : dataset) {
                // MLM损失
                INDArray mlmLoss = computeMLMLoss(
                    batch.maskedTokens, batch.labels);
                
                // NSP损失
                INDArray nspLoss = computeNSPLoss(
                    batch.sentenceA, batch.sentenceB, batch.isNext);
                
                // 总损失
                INDArray loss = mlmLoss.add(nspLoss);
                
                // 反向传播
                optimizer.zeroGrad();
                loss.backward();
                optimizer.step();
                
                totalLoss += loss.getDouble();
                batchCount++;
            }
            
            System.out.printf("Epoch %d, Loss: %.4f%n", 
                epoch, totalLoss / batchCount);
        }
    }
}
```

## 微调阶段

### 微调策略

```
微调方式：

1. 全量微调（Full Fine-tuning）
   - 更新所有参数
   - 效果通常最好
   - 计算量大，容易过拟合

2. 部分微调（Partial Fine-tuning）
   - 只更新顶层参数
   - 冻结底层参数
   - 平衡效果和效率

3. 适配器（Adapter）
   - 添加小型适配器模块
   - 只训练适配器参数
   - 高效，适合多任务
```

### Java微调实现

```java
/**
 * 微调管理器
 */
public class FineTuningManager {
    
    private final PretrainedModel pretrainedModel;
    
    /**
     * 全量微调
     */
    public void fullFineTuning(TaskDataset dataset, int epochs) {
        // 所有参数可训练
        pretrainedModel.unfreezeAll();
        
        train(dataset, epochs);
    }
    
    /**
     * 部分微调：冻结底层
     */
    public void partialFineTuning(TaskDataset dataset, 
                                   int epochs, 
                                   int freezeLayers) {
        // 冻结底层
        for (int i = 0; i < freezeLayers; i++) {
            pretrainedModel.freezeLayer(i);
        }
        
        // 顶层可训练
        for (int i = freezeLayers; i < pretrainedModel.numLayers(); i++) {
            pretrainedModel.unfreezeLayer(i);
        }
        
        train(dataset, epochs);
    }
    
    /**
     * 使用适配器
     */
    public void adapterFineTuning(TaskDataset dataset, int epochs) {
        // 冻结预训练模型
        pretrainedModel.freezeAll();
        
        // 添加适配器
        AdapterLayers adapters = new AdapterLayers(pretrainedModel);
        
        // 只训练适配器
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Sample sample : dataset) {
                INDArray output = adapters.forward(sample.input);
                INDArray loss = computeLoss(output, sample.label);
                
                // 只更新适配器参数
                adapters.backward(loss);
                adapters.update();
            }
        }
    }
}
```

## 微调技巧

### 学习率设置

```java
/**
 * 分层学习率
 */
public class LayerwiseLearningRate {
    
    /**
     * 设置分层学习率
     * 底层学习率小，顶层学习率大
     */
    public void setLayerwiseLR(Optimizer optimizer, 
                               PretrainedModel model,
                               double baseLR) {
        int numLayers = model.numLayers();
        
        for (int i = 0; i < numLayers; i++) {
            // 指数衰减
            double lr = baseLR * Math.pow(0.95, numLayers - i);
            optimizer.setLearningRate(i, lr);
        }
        
        // 任务特定层使用最大学习率
        optimizer.setLearningRate("classifier", baseLR);
    }
}
```

### 早停与正则化

```java
/**
 * 微调最佳实践
 */
public class FineTuningBestPractices {
    
    /**
     * 带早停的训练
     */
    public void trainWithEarlyStopping(TaskDataset trainData,
                                       TaskDataset devData,
                                       int maxEpochs,
                                       int patience) {
        double bestDevLoss = Double.MAX_VALUE;
        int noImproveCount = 0;
        
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // 训练
            trainEpoch(trainData);
            
            // 验证
            double devLoss = evaluate(devData);
            
            if (devLoss < bestDevLoss) {
                bestDevLoss = devLoss;
                saveCheckpoint();
                noImproveCount = 0;
            } else {
                noImproveCount++;
                if (noImproveCount >= patience) {
                    System.out.println("早停触发");
                    break;
                }
            }
        }
    }
    
    /**
     * 正则化技术
     */
    public void applyRegularization(INDArray loss, Model model) {
        // L2正则化
        double l2Lambda = 0.01;
        INDArray l2Loss = model.parameters()
            .pow(2).sum().mul(l2Lambda);
        
        // Dropout
        // 在前向传播中随机丢弃部分神经元
        
        // 总损失
        INDArray totalLoss = loss.add(l2Loss);
    }
}
```

## 实际应用建议

### 选择微调策略

```
决策流程：

数据量？
├── 大量（>10k）→ 全量微调
└── 少量（<10k）
    ├── 任务与预训练相似？
    │   ├── 是 → 部分微调
    │   └── 否 → 适配器/Prompt
    └── 多任务场景？
        ├── 是 → 适配器
        └── 否 → 部分微调
```

### 常见问题

```
1. 过拟合
   - 症状：训练loss下降，验证loss上升
   - 解决：增加dropout，早停，数据增强

2. 灾难性遗忘
   - 症状：微调后通用能力下降
   - 解决：使用适配器，分层学习率

3. 收敛慢
   - 症状：loss下降缓慢
   - 解决：增大学习率，检查数据质量
```

## 小结

本章我们学习了：

1. **预训练-微调范式**：知识迁移，数据高效
2. **预训练阶段**：数据准备，MLM+NSP训练
3. **微调阶段**：全量、部分、适配器微调
4. **实践技巧**：学习率设置，早停，正则化

**关键认识：**
预训练+微调是NLP的标准范式，掌握微调技巧对实际应用至关重要。

**下一步：** 我们将实践调用OpenAI API。

---

**练习题：**

1. 预训练-微调范式为什么比从头训练好？
2. 全量微调和适配器微调各有什么优缺点？
3. 如何避免微调时的灾难性遗忘？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[&larr; 8.2 BERT：双向编码器表示](02-bert-model.md)</span>

<span>[8.4 实战：调用OpenAI API &rarr;](04-openai-api-practice.md)</span>

</div>