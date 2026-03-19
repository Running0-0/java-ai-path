<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 5.2 数据准备与预处理](02-data-preparation.md)</span>

<span>[5.4 模型评估与优化 →](04-model-evaluation-optimization.md)</span>

</div>

---

# 5.3 模型设计与训练

> "模型设计是艺术与工程的结合——既要追求性能，又要考虑可行性。"

## 模型架构设计

### 设计考量

| 因素 | 考量 |
|------|------|
| 输入大小 | 28×28 = 784维 |
| 输出类别 | 10个数字 |
| 模型复杂度 | 适中，避免过拟合 |
| 训练速度 | 合理，便于实验 |

### 网络架构

```
输入层: 28×28×1
    ↓
卷积块1: Conv(32, 3×3) → BN → ReLU → MaxPool(2×2)
    ↓
卷积块2: Conv(64, 3×3) → BN → ReLU → MaxPool(2×2)
    ↓
卷积块3: Conv(128, 3×3) → BN → ReLU → MaxPool(2×2)
    ↓
展平层
    ↓
全连接层: 128 → BN → ReLU → Dropout(0.5)
    ↓
输出层: 10 → Softmax
```

## 模型实现

```java
package com.example.digit.model;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 手写数字分类器
 */
public class DigitClassifier {
    
    private final int inputHeight = 28;
    private final int inputWidth = 28;
    private final int inputChannels = 1;
    private final int numClasses = 10;
    
    private MultiLayerNetwork model;
    
    /**
     * 构建CNN模型
     */
    public MultiLayerNetwork buildModel() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .l2(0.0005)  // L2正则化
            
            .list()
            // 卷积块1: 28x28x1 -> 14x14x32
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(inputChannels)
                .nOut(32)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(32)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 卷积块2: 14x14x32 -> 7x7x64
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(64)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 卷积块3: 7x7x64 -> 3x3x128
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(128)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(128)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // 全连接层
            .layer(new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(128)
                .build())
            .layer(new DropoutLayer(0.5))
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutionalFlat(inputHeight, inputWidth, inputChannels))
            .build();
        
        model = new MultiLayerNetwork(config);
        model.init();
        
        return model;
    }
    
    public MultiLayerNetwork getModel() { return model; }
}
```

## 训练流程

### 训练服务

```java
package com.example.digit.service;

import org.deeplearning4j.earlystopping.*;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * 训练服务
 */
public class TrainingService {
    
    private final MultiLayerNetwork model;
    private final DataSetIterator trainIter;
    private final DataSetIterator testIter;
    
    public TrainingService(MultiLayerNetwork model, 
                          DataSetIterator trainIter, 
                          DataSetIterator testIter) {
        this.model = model;
        this.trainIter = trainIter;
        this.testIter = testIter;
    }
    
    /**
     * 基础训练
     */
    public void train(int epochs) {
        model.setListeners(new ScoreIterationListener(100));
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(trainIter);
            
            var eval = model.evaluate(testIter);
            System.out.printf("Epoch %d - 准确率: %.2f%%, 损失: %.4f%n",
                epoch + 1, eval.accuracy() * 100, model.score());
            
            trainIter.reset();
            testIter.reset();
        }
    }
    
    /**
     * 早停训练
     */
    public EarlyStoppingResult<MultiLayerNetwork> trainWithEarlyStopping() {
        EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = 
            new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(
                    new MaxEpochsTerminationCondition(30),
                    new ScoreImprovementEpochTerminationCondition(5)
                )
                .evaluateEveryNEpochs(1)
                .scoreCalculator(new DataSetLossCalculator(testIter, true))
                .modelSaver(new LocalFileModelSaver("checkpoints"))
                .build();
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(
            esConfig, model, trainIter);
        
        return trainer.fit();
    }
}
```

### 训练监控

```java
import org.deeplearning4j.optimize.api.TrainingListener;

/**
 * 训练监控器
 */
public class TrainingMonitor implements TrainingListener {
    
    private List<Double> losses = new ArrayList<>();
    private List<Double> accuracies = new ArrayList<>();
    
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (iteration % 100 == 0) {
            double score = model.score();
            losses.add(score);
            System.out.printf("Iteration %d, Loss: %.4f%n", iteration, score);
        }
    }
    
    public void plotTrainingCurve() {
        // 使用图表库绘制训练曲线
        // 或导出数据供其他工具使用
    }
}
```

## 超参数调优

### 学习率选择

```java
/**
 * 学习率搜索
 */
public class LearningRateFinder {
    
    public static double findBestLearningRate(MultiLayerNetwork model, 
                                               DataSetIterator trainIter) {
        double[] learningRates = {0.0001, 0.0005, 0.001, 0.005, 0.01};
        double bestAccuracy = 0;
        double bestLr = 0.001;
        
        for (double lr : learningRates) {
            // 重置模型
            model = rebuildModelWithLR(lr);
            
            // 训练几个epoch
            for (int i = 0; i < 3; i++) {
                model.fit(trainIter);
            }
            
            // 评估
            var eval = model.evaluate(trainIter);
            double accuracy = eval.accuracy();
            
            System.out.printf("学习率 %.4f: 准确率 %.2f%%%n", lr, accuracy * 100);
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestLr = lr;
            }
            
            trainIter.reset();
        }
        
        return bestLr;
    }
}
```

### 批次大小影响

| 批次大小 | 内存占用 | 训练速度 | 泛化能力 |
|----------|----------|----------|----------|
| 小(16) | 低 | 慢 | 好 |
| 中(64) | 中 | 中 | 中 |
| 大(256) | 高 | 快 | 可能差 |

## 模型保存与加载

```java
/**
 * 模型持久化
 */
public class ModelPersistence {
    
    /**
     * 保存模型
     */
    public static void save(MultiLayerNetwork model, String path) throws Exception {
        ModelSerializer.writeModel(model, new File(path), true);
        System.out.println("模型已保存: " + path);
    }
    
    /**
     * 加载模型
     */
    public static MultiLayerNetwork load(String path) throws Exception {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(path);
        System.out.println("模型已加载: " + path);
        return model;
    }
}
```

## 完整训练示例

```java
package com.example.digit;

import com.example.digit.data.DataPipeline;
import com.example.digit.model.DigitClassifier;
import com.example.digit.service.TrainingService;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * 训练主程序
 */
public class TrainingMain {
    
    public static void main(String[] args) throws Exception {
        // 1. 数据准备
        System.out.println("=== 数据准备 ===");
        DataPipeline pipeline = new DataPipeline(64);
        
        // 2. 模型构建
        System.out.println("\n=== 模型构建 ===");
        DigitClassifier classifier = new DigitClassifier();
        MultiLayerNetwork model = classifier.buildModel();
        System.out.println(model.summary());
        
        // 3. 训练
        System.out.println("\n=== 开始训练 ===");
        TrainingService trainer = new TrainingService(
            model, 
            pipeline.getTrainIterator(), 
            pipeline.getTestIterator()
        );
        
        // 使用早停训练
        var result = trainer.trainWithEarlyStopping();
        
        System.out.println("\n=== 训练结果 ===");
        System.out.println("最佳轮次: " + result.getBestModelEpoch());
        System.out.println("最佳分数: " + result.getBestModelScore());
        
        // 4. 保存模型
        ModelPersistence.save(result.getBestModel(), "models/digit-model.zip");
        
        // 5. 最终评估
        System.out.println("\n=== 最终评估 ===");
        var eval = result.getBestModel().evaluate(pipeline.getTestIterator());
        System.out.println(eval.stats());
    }
}
```

## 小结

本章我们实现了：

1. **模型设计**：三层CNN架构
2. **训练流程**：基础训练和早停训练
3. **超参数调优**：学习率和批次大小
4. **模型持久化**：保存和加载

**下一步：** 我们将进行模型评估与优化。

---

**练习题：**

1. 为什么使用批归一化？它有什么作用？
2. 早停训练如何防止过拟合？
3. 尝试增加网络深度，观察训练效果变化。

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 5.2 数据准备与预处理](02-data-preparation.md)</span>

<span>[5.4 模型评估与优化 →](04-model-evaluation-optimization.md)</span>

</div>
