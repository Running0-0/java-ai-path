# 2.4 用Deeplearning4j实现你的第一个神经网络

> "理论指导实践，实践验证理论。现在让我们用真正的框架来构建神经网络。"

## Deeplearning4j简介

Deeplearning4j（DL4J）是Java生态中最成熟的深度学习框架，特点：

| 特性 | 说明 |
|------|------|
| 纯Java实现 | 无需Python依赖 |
| 分布式支持 | 可与Spark集成 |
| 生产就绪 | 企业级性能和稳定性 |
| GPU加速 | 支持CUDA |

## 快速开始：MNIST手写数字识别

MNIST是深度学习的"Hello World"——识别0-9的手写数字。

### 项目配置

```xml
<!-- pom.xml -->
<dependencies>
    <!-- DL4J核心 -->
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
    
    <!-- ND4J：类似NumPy的矩阵库 -->
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
    
    <!-- 数据处理 -->
    <dependency>
        <groupId>org.datavec</groupId>
        <artifactId>datavec-data-image</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
    
    <!-- MNIST数据集 -->
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-data</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
</dependencies>
```

### 完整代码

```java
package com.example.ai.chapter02;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * MNIST手写数字识别
 */
public class MnistClassifier {
    
    public static void main(String[] args) throws Exception {
        // === 1. 准备数据 ===
        int batchSize = 64;
        
        // 训练数据
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
        
        // 测试数据
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);
        
        System.out.println("数据加载完成");
        System.out.printf("训练样本数: %d%n", trainData.totalExamples());
        System.out.printf("测试样本数: %d%n", testData.totalExamples());
        
        // === 2. 构建网络 ===
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .updater(new Adam(0.001))  // Adam优化器
            .list()
            // 第一隐藏层：784 -> 256
            .layer(new DenseLayer.Builder()
                .nIn(28 * 28)    // MNIST图像大小：28x28
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            // 第二隐藏层：256 -> 128
            .layer(new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            // 输出层：128 -> 10（10个数字类别）
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        
        // 打印网络结构
        System.out.println("\n网络结构:");
        System.out.println(model.summary());
        
        // 添加训练监听器
        model.setListeners(new ScoreIterationListener(100));
        
        // === 3. 训练网络 ===
        int numEpochs = 5;
        System.out.println("\n开始训练...");
        
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(trainData);
            
            // 评估
            var eval = model.evaluate(testData);
            System.out.printf("Epoch %d 完成 - 准确率: %.2f%%%n", 
                epoch + 1, eval.accuracy() * 100);
            
            trainData.reset();
            testData.reset();
        }
        
        // === 4. 保存模型 ===
        File modelFile = new File("mnist-model.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        System.out.println("\n模型已保存到: " + modelFile.getAbsolutePath());
    }
}
```

### 运行结果

```
数据加载完成
训练样本数: 60000
测试样本数: 10000

网络结构:
========================================================================
LayerName       LayerType           nIn/nOut   TotalParams   ParamsShape
========================================================================
dense           DenseLayer          784/256    200960        W{256,784}, b{256}
dense_1         DenseLayer          256/128    32896         W{128,256}, b{128}
output          OutputLayer         128/10     1290          W{10,128}, b{10}
------------------------------------------------------------------------
Total Parameters:  235146
========================================================================

开始训练...
Epoch 1 完成 - 准确率: 96.23%
Epoch 2 完成 - 准确率: 97.45%
Epoch 3 完成 - 准确率: 97.89%
Epoch 4 完成 - 准确率: 98.12%
Epoch 5 完成 - 准确率: 98.23%

模型已保存到: d:\Projects\...\mnist-model.zip
```

## 理解DL4J的核心概念

### 网络配置构建器

DL4J使用构建者模式配置网络：

```java
MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
    // 全局配置
    .seed(12345)                           // 随机种子
    .updater(new Adam(0.001))              // 优化器
    .weightInit(WeightInit.XAVIER)         // 权重初始化
    .l2(0.001)                             // L2正则化
    
    // 层定义
    .list()
    .layer(0, new DenseLayer.Builder()...)
    .layer(1, new DenseLayer.Builder()...)
    .layer(2, new OutputLayer.Builder()...)
    
    .build();
```

### 层类型

| 层类型 | 用途 | 示例 |
|--------|------|------|
| DenseLayer | 全连接层 | 特征提取 |
| OutputLayer | 输出层 | 分类/回归 |
| ConvolutionLayer | 卷积层 | 图像处理 |
| SubsamplingLayer | 池化层 | 降维 |
| LSTM | 循环层 | 序列处理 |
| BatchNormalization | 批归一化 | 加速训练 |

### 激活函数

```java
.activation(Activation.RELU)      // ReLU
.activation(Activation.SIGMOID)   // Sigmoid
.activation(Activation.TANH)      // Tanh
.activation(Activation.SOFTMAX)   // Softmax
.activation(Activation.LEAKYRELU) // Leaky ReLU
```

### 损失函数

```java
LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD  // 分类
LossFunctions.LossFunction.MSE                     // 回归
LossFunctions.LossFunction.BINARY_XENT             // 二分类
```

## 高级功能

### 1. 早停（Early Stopping）

```java
import org.deeplearning4j.earlystopping.*;

// 配置早停
EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = 
    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        .epochTerminationConditions(
            new MaxEpochsTerminationCondition(50),          // 最多50轮
            new ScoreImprovementEpochTerminationCondition(5) // 5轮无改善则停止
        )
        .evaluateEveryNEpochs(1)
        .scoreCalculator(new DataSetLossCalculator(testData, true))
        .modelSaver(new LocalFileModelSaver("checkpoint"))
        .build();

// 训练
EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig, config, trainData);
EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

System.out.println("最佳轮次: " + result.getBestModelEpoch());
System.out.println("最佳分数: " + result.getBestModelScore());
```

### 2. 模型评估

```java
import org.deeplearning4j.eval.*;

// 评估分类模型
Evaluation eval = model.evaluate(testData);

System.out.println("准确率: " + eval.accuracy());
System.out.println("精确率: " + eval.precision());
System.out.println("召回率: " + eval.recall());
System.out.println("F1分数: " + eval.f1());

// 混淆矩阵
System.out.println("\n混淆矩阵:");
System.out.println(eval.confusionToString());
```

### 3. 模型保存与加载

```java
// 保存模型
ModelSerializer.writeModel(model, new File("model.zip"), true);

// 加载模型
MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork("model.zip");

// 使用模型预测
INDArray input = ...;  // 输入数据
INDArray output = loadedModel.output(input);
int predictedClass = output.argMax(1).getInt(0);
```

### 4. 可视化训练过程

```java
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;

// 启动UI服务器
UIServer uiServer = UIServer.getInstance();
StatsStorage statsStorage = new InMemoryStatsStorage();

// 添加监听器
model.setListeners(new StatsListener(statsStorage));

// 附加存储
uiServer.attach(statsStorage);

System.out.println("训练监控: http://localhost:9000");
```

## 实战：自定义数据集

```java
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

/**
 * 从CSV文件加载数据
 */
public class CustomDatasetExample {
    
    public static void main(String[] args) throws Exception {
        // CSV读取器
        RecordReader rr = new CSVRecordReader(1, ',');  // 跳过标题行
        rr.initialize(new FileSplit(new File("data.csv")));
        
        // 创建数据迭代器
        // 参数：recordReader, batchSize, labelIndex, numClasses
        DataSetIterator iterator = new RecordReaderDataSetIterator(
            rr, 64, 4, 3  // 假设标签在第4列，共3个类别
        );
        
        // 训练模型...
        model.fit(iterator);
    }
}
```

## 设计思考：DL4J的设计哲学

### 与Python框架的对比

| 特性 | DL4J | PyTorch/TensorFlow |
|------|------|-------------------|
| 语言 | Java | Python |
| 部署 | 原生Java | 需要桥接 |
| 类型安全 | 编译时检查 | 运行时错误 |
| 企业集成 | 无缝 | 需要额外工作 |

### 适用场景

**DL4J适合：**
- 企业级Java应用
- 需要与现有Java系统集成
- 对类型安全有要求
- 生产环境部署

**Python框架适合：**
- 研究和原型开发
- 需要最新模型和算法
- 数据科学工作流

## 完整示例：端到端训练流程

```java
package com.example.ai.chapter02;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.weightinit.WeightInit;

/**
 * 改进版MNIST分类器
 * 添加批归一化和Dropout
 */
public class ImprovedMnistClassifier {
    
    public static void main(String[] args) throws Exception {
        // 数据准备
        DataSetIterator trainData = new MnistDataSetIterator(128, true, 42);
        DataSetIterator testData = new MnistDataSetIterator(128, false, 42);
        
        // 改进的网络配置
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(42)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .l2(0.0005)  // L2正则化
            
            .list()
            // 第一个隐藏层 + 批归一化 + Dropout
            .layer(new DenseLayer.Builder()
                .nIn(784)
                .nOut(512)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(512)
                .build())
            .layer(new DropoutLayer(0.2))
            
            // 第二个隐藏层
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(256)
                .build())
            .layer(new DropoutLayer(0.3))
            
            // 第三个隐藏层
            .layer(new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(new DropoutLayer(0.4))
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build())
            
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(200));
        
        System.out.println("网络参数数量: " + model.numParams());
        
        // 训练
        for (int epoch = 0; epoch < 10; epoch++) {
            model.fit(trainData);
            
            var eval = model.evaluate(testData);
            System.out.printf("Epoch %d - 准确率: %.2f%%, 损失: %.4f%n",
                epoch + 1, eval.accuracy() * 100, model.score());
            
            trainData.reset();
            testData.reset();
        }
        
        // 最终评估
        var finalEval = model.evaluate(testData);
        System.out.println("\n最终结果:");
        System.out.printf("准确率: %.2f%%%n", finalEval.accuracy() * 100);
        System.out.printf("精确率: %.2f%%%n", finalEval.precision() * 100);
        System.out.printf("召回率: %.2f%%%n", finalEval.recall() * 100);
        System.out.printf("F1分数: %.2f%n", finalEval.f1());
    }
}
```

## 小结

本章我们学习了：

1. **DL4J基础**：配置、层、激活函数
2. **MNIST实战**：完整的训练流程
3. **高级功能**：早停、评估、可视化
4. **自定义数据**：CSV文件加载

**DL4J核心API：**

| API | 用途 |
|-----|------|
| NeuralNetConfiguration.Builder | 网络配置 |
| DenseLayer | 全连接层 |
| OutputLayer | 输出层 |
| MultiLayerNetwork | 网络模型 |
| DataSetIterator | 数据迭代器 |
| Evaluation | 模型评估 |

**下一步：** 我们将深入探讨为什么深度学习需要"深"。

---

**练习题：**

1. 修改网络结构，尝试不同的隐藏层大小，观察准确率变化
2. 添加Dropout层，比较训练集和测试集准确率
3. 尝试不同的优化器（SGD、RMSProp），比较收敛速度
