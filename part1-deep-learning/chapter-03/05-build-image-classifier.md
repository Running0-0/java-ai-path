<table width="100%">
   <tr>
      <td align="left"><a href="04-classic-cnn-architectures.md">← 3.4 经典CNN架构解析</a></td>
      <td align="right"><a href="06-local-perception-weight-sharing.md">3.6 设计思考：局部感知与参数共享 →</a></td>
   </tr>
</table>

---

# 3.5 实战：用Java构建图像分类器

> "纸上得来终觉浅，绝知此事要躬行。让我们用Java构建一个真正的图像分类系统。"

## 项目概述

我们将构建一个完整的图像分类系统：

```
图像输入 → 预处理 → CNN模型 → 分类结果
```

**功能：**
- 支持自定义数据集
- 数据增强
- 模型训练和评估
- 模型保存和加载
- 预测接口

## 项目结构

```
image-classifier/
├── pom.xml
├── src/main/java/
│   └── com/example/classifier/
│       ├── ImageClassifier.java      # 主类
│       ├── model/
│       │   └── CnnModel.java         # 模型定义
│       ├── data/
│       │   ├── ImageDataset.java     # 数据集处理
│       │   └── DataAugmentation.java # 数据增强
│       └── utils/
│           └── ImageUtils.java       # 图像工具
├── src/main/resources/
│   └── labels.txt                    # 类别标签
└── data/
    ├── train/                        # 训练数据
    │   ├── cat/
    │   ├── dog/
    │   └── ...
    └── test/                         # 测试数据
```

## 完整代码实现

### 1. 项目配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>image-classifier</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <dl4j.version>1.0.0-M2.1</dl4j.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native-platform</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-data-image</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-data</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
    </dependencies>
</project>
```

### 2. 数据集处理

```java
package com.example.classifier.data;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Random;

/**
 * 图像数据集加载器
 */
public class ImageDataset {
    
    private final int height;
    private final int width;
    private final int channels;
    private final int batchSize;
    
    public ImageDataset(int height, int width, int channels, int batchSize) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
    }
    
    /**
     * 创建训练数据迭代器
     */
    public DataSetIterator createTrainIterator(String dataPath) throws Exception {
        File trainDir = new File(dataPath);
        
        // 图像记录读取器
        ImageRecordReader recordReader = new ImageRecordReader(
            height, width, channels,
            new ParentPathLabelGenerator()  // 使用父目录名作为标签
        );
        
        recordReader.initialize(new FileSplit(trainDir, new Random(42)));
        
        // 创建迭代器
        DataSetIterator iterator = new RecordReaderDataSetIterator(
            recordReader, batchSize, 1, recordReader.numLabels());
        
        // 归一化到[0,1]
        iterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        
        return iterator;
    }
    
    /**
     * 获取类别数量
     */
    public int getNumClasses(String dataPath) {
        File trainDir = new File(dataPath);
        return trainDir.listFiles(File::isDirectory).length;
    }
}
```

### 3. 数据增强

```java
package com.example.classifier.data;

import org.datavec.image.transform.*;

/**
 * 数据增强配置
 */
public class DataAugmentation {
    
    /**
     * 创建训练时的数据增强
     */
    public static ImageTransform createTrainTransform() {
        return new PipelineImageTransform.Builder()
            .add(new FlipImageTransform(0.5))           // 50%概率水平翻转
            .add(new RotateImageTransform(15))          // 随机旋转±15度
            .add(new ScaleImageTransform(0.9f, 1.1f))   // 随机缩放
            .build();
    }
    
    /**
     * 测试时不需要数据增强
     */
    public static ImageTransform createTestTransform() {
        return null;  // 或者只做中心裁剪
    }
}
```

### 4. 模型定义

```java
package com.example.classifier.model;

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
 * CNN图像分类模型
 */
public class CnnModel {
    
    private final int height;
    private final int width;
    private final int channels;
    private final int numClasses;
    
    public CnnModel(int height, int width, int channels, int numClasses) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.numClasses = numClasses;
    }
    
    /**
     * 构建模型
     */
    public MultiLayerNetwork build() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(42)
            .updater(new Adam(0.001))
            .weightInit(WeightInit.XAVIER)
            .l2(0.0005)  // L2正则化
            
            .list()
            // 卷积块1
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(channels)
                .nOut(32)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(32)
                .build())
            .layer(new ConvolutionLayer.Builder(3, 3)
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
            .layer(new DropoutLayer(0.25))
            
            // 卷积块2
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .padding(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(64)
                .build())
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
            .layer(new DropoutLayer(0.25))
            
            // 卷积块3
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
            .layer(new DropoutLayer(0.25))
            
            // 全连接层
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization.Builder()
                .nOut(256)
                .build())
            .layer(new DropoutLayer(0.5))
            
            // 输出层
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numClasses)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutional(height, width, channels))
            .build();
        
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(50));
        
        return model;
    }
}
```

### 5. 主程序

```java
package com.example.classifier;

import com.example.classifier.data.ImageDataset;
import com.example.classifier.model.CnnModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * 图像分类器主程序
 */
public class ImageClassifier {
    
    public static void main(String[] args) throws Exception {
        // 配置参数
        String trainPath = "data/train";
        String testPath = "data/test";
        String modelPath = "model.zip";
        
        int height = 64;
        int width = 64;
        int channels = 3;
        int batchSize = 32;
        int epochs = 20;
        
        // 创建数据集
        System.out.println("加载数据集...");
        ImageDataset dataset = new ImageDataset(height, width, channels, batchSize);
        int numClasses = dataset.getNumClasses(trainPath);
        
        DataSetIterator trainIter = dataset.createTrainIterator(trainPath);
        DataSetIterator testIter = dataset.createTrainIterator(testPath);
        
        System.out.println("类别数量: " + numClasses);
        System.out.println("训练批次数: " + trainIter.totalExamples() / batchSize);
        
        // 构建模型
        System.out.println("\n构建模型...");
        CnnModel modelBuilder = new CnnModel(height, width, channels, numClasses);
        MultiLayerNetwork model = modelBuilder.build();
        
        System.out.println(model.summary());
        
        // 训练模型
        System.out.println("\n开始训练...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(trainIter);
            
            // 评估
            var eval = model.evaluate(testIter);
            System.out.printf("Epoch %d/%d - ", epoch + 1, epochs);
            System.out.printf("准确率: %.2f%%, ", eval.accuracy() * 100);
            System.out.printf("损失: %.4f%n", model.score());
            
            trainIter.reset();
            testIter.reset();
        }
        
        // 保存模型
        System.out.println("\n保存模型到: " + modelPath);
        ModelSerializer.writeModel(model, new File(modelPath), true);
        
        // 最终评估
        System.out.println("\n最终评估:");
        var finalEval = model.evaluate(testIter);
        System.out.println(finalEval.stats());
    }
}
```

### 6. 预测接口

```java
package com.example.classifier;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.List;

/**
 * 图像预测器
 */
public class Predictor {
    
    private final MultiLayerNetwork model;
    private final NativeImageLoader loader;
    private final List<String> labels;
    
    public Predictor(String modelPath, String labelsPath) throws Exception {
        // 加载模型
        this.model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        
        // 加载标签
        this.labels = Files.readAllLines(Path.of(labelsPath));
        
        // 图像加载器
        this.loader = new NativeImageLoader(64, 64, 3);
    }
    
    /**
     * 预测图像类别
     */
    public Prediction predict(String imagePath) throws Exception {
        // 加载图像
        INDArray image = loader.asMatrix(new File(imagePath));
        
        // 预处理
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        
        // 预测
        INDArray output = model.output(image);
        int predictedClass = output.argMax(1).getInt(0);
        float confidence = output.getFloat(predictedClass);
        
        return new Prediction(
            labels.get(predictedClass),
            confidence,
            output.toDoubleVector()
        );
    }
    
    /**
     * 预测结果
     */
    public record Prediction(
        String label,           // 预测类别
        float confidence,       // 置信度
        double[] probabilities  // 各类别概率
    ) {}
}
```

## 使用示例

### 训练模型

```bash
# 准备数据目录结构
data/
├── train/
│   ├── cat/      # 猫的图片
│   ├── dog/      # 狗的图片
│   └── bird/     # 鸟的图片
└── test/
    ├── cat/
    ├── dog/
    └── bird/

# 运行训练
mvn exec:java -Dexec.mainClass="com.example.classifier.ImageClassifier"
```

### 使用模型预测

```java
public class PredictDemo {
    public static void main(String[] args) throws Exception {
        Predictor predictor = new Predictor("model.zip", "labels.txt");
        
        Prediction result = predictor.predict("test_image.jpg");
        
        System.out.println("预测结果: " + result.label());
        System.out.println("置信度: " + result.confidence());
        
        // 打印各类别概率
        double[] probs = result.probabilities();
        for (int i = 0; i < probs.length; i++) {
            System.out.printf("%s: %.2f%%%n", 
                predictor.getLabels().get(i), 
                probs[i] * 100);
        }
    }
}
```

## 训练技巧

### 1. 学习率调度

```java
// 使用学习率调度
.updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, 0.001, 0.9)))
```

### 2. 早停

```java
EarlyStoppingConfiguration<MultiLayerNetwork> esConfig = 
    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        .epochTerminationConditions(
            new MaxEpochsTerminationCondition(50),
            new ScoreImprovementEpochTerminationCondition(5)
        )
        .build();
```

### 3. 模型微调

```java
// 加载预训练模型
MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("pretrained.zip");

// 冻结前面的层
for (int i = 0; i < model.getnLayers() - 2; i++) {
    model.getLayer(i).setBackpropGradientsViewMode(BackpropGradientsViewMode.NONE);
}

// 只训练最后的层
model.fit(trainData);
```

## 小结

本章我们实现了：

1. **完整的数据处理流程**：加载、增强、预处理
2. **CNN模型构建**：批归一化、Dropout
3. **训练和评估**：迭代训练、性能监控
4. **模型部署**：保存、加载、预测

**关键代码模块：**

| 模块 | 功能 |
|------|------|
| ImageDataset | 数据加载 |
| CnnModel | 模型定义 |
| ImageClassifier | 训练主程序 |
| Predictor | 预测接口 |

**下一步：** 我们将总结CNN的设计原则。

---

**练习题：**

1. 尝试调整模型结构，增加或减少卷积块，观察效果变化
2. 实现数据增强，比较有无数据增强的训练效果
3. 使用预训练模型进行迁移学习

---

<table width="100%">
   <tr>
      <td align="left"><a href="04-classic-cnn-architectures.md">← 3.4 经典CNN架构解析</a></td>
      <td align="right"><a href="06-local-perception-weight-sharing.md">3.6 设计思考：局部感知与参数共享 →</a></td>
   </tr>
</table>