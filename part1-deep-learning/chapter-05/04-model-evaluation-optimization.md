# 5.4 模型评估与优化

> "没有评估就没有改进。科学的评估方法是模型优化的指南针。"

## 评估指标

### 分类指标详解

```java
import org.deeplearning4j.eval.Evaluation;

/**
 * 模型评估器
 */
public class ModelEvaluator {
    
    /**
     * 全面评估
     */
    public static void evaluate(MultiLayerNetwork model, DataSetIterator testIter) {
        Evaluation eval = model.evaluate(testIter);
        
        System.out.println("=== 模型评估结果 ===");
        
        // 整体指标
        System.out.println("\n整体指标:");
        System.out.printf("  准确率 (Accuracy): %.2f%%%n", eval.accuracy() * 100);
        System.out.printf("  精确率 (Precision): %.2f%%%n", eval.precision() * 100);
        System.out.printf("  召回率 (Recall): %.2f%%%n", eval.recall() * 100);
        System.out.printf("  F1分数: %.4f%n", eval.f1());
        
        // 每类指标
        System.out.println("\n各类别指标:");
        for (int i = 0; i < 10; i++) {
            System.out.printf("  数字%d: Precision=%.2f%%, Recall=%.2f%%, F1=%.4f%n",
                i, 
                eval.precision(i) * 100,
                eval.recall(i) * 100,
                eval.f1(i));
        }
        
        // 混淆矩阵
        System.out.println("\n混淆矩阵:");
        System.out.println(eval.confusionToString());
    }
}
```

### 混淆矩阵分析

```java
/**
 * 混淆矩阵分析
 */
public class ConfusionMatrixAnalyzer {
    
    public static void analyze(Evaluation eval) {
        System.out.println("=== 混淆矩阵分析 ===");
        
        // 找出最容易混淆的数字对
        int[][] matrix = eval.getConfusionMatrix();
        
        List<ConfusionPair> pairs = new ArrayList<>();
        
        for (int actual = 0; actual < 10; actual++) {
            for (int predicted = 0; predicted < 10; predicted++) {
                if (actual != predicted && matrix[actual][predicted] > 0) {
                    pairs.add(new ConfusionPair(actual, predicted, matrix[actual][predicted]));
                }
            }
        }
        
        // 按混淆次数排序
        pairs.sort((a, b) -> b.count - a.count);
        
        System.out.println("最容易混淆的数字对:");
        for (int i = 0; i < Math.min(5, pairs.size()); i++) {
            ConfusionPair pair = pairs.get(i);
            System.out.printf("  %d 被误判为 %d: %d 次%n", 
                pair.actual, pair.predicted, pair.count);
        }
    }
    
    record ConfusionPair(int actual, int predicted, int count) {}
}
```

## 错误分析

### 可视化错误样本

```java
/**
 * 错误样本分析
 */
public class ErrorAnalyzer {
    
    /**
     * 收集并可视化错误样本
     */
    public static void analyzeErrors(MultiLayerNetwork model, 
                                     DataSetIterator testIter,
                                     String outputPath) throws Exception {
        
        List<ErrorSample> errors = new ArrayList<>();
        
        while (testIter.hasNext()) {
            DataSet batch = testIter.next();
            INDArray predictions = model.output(batch.getFeatures());
            
            for (int i = 0; i < batch.getFeatures().size(0); i++) {
                int actual = batch.getLabels().getRow(i).argMax().getInt(0);
                int predicted = predictions.getRow(i).argMax().getInt(0);
                
                if (actual != predicted) {
                    errors.add(new ErrorSample(
                        batch.getFeatures().getRow(i).dup(),
                        actual,
                        predicted,
                        predictions.getRow(i).getDouble(predicted)
                    ));
                }
            }
        }
        
        System.out.println("错误样本数: " + errors.size());
        
        // 可视化
        visualizeErrors(errors, outputPath);
        
        testIter.reset();
    }
    
    private static void visualizeErrors(List<ErrorSample> errors, String outputPath) {
        // 选择前25个错误样本可视化
        // ...
    }
    
    record ErrorSample(INDArray image, int actual, int predicted, double confidence) {}
}
```

## 优化策略

### 1. 数据增强

```java
/**
 * 使用数据增强优化
 */
public class DataAugmentationOptimizer {
    
    public static DataSetIterator augmentData(DataSetIterator original) {
        // 添加数据增强变换
        ImageTransform transform = new PipelineImageTransform.Builder()
            .add(new RotateImageTransform(10))
            .add(new ScaleImageTransform(0.9f, 1.1f))
            .build();
        
        // 应用变换
        // ...
        
        return original;
    }
}
```

### 2. 模型调优

```java
/**
 * 模型超参数调优
 */
public class HyperparameterTuner {
    
    /**
     * 网格搜索
     */
    public static void gridSearch(DataSetIterator trainIter, DataSetIterator testIter) {
        double[] learningRates = {0.0005, 0.001, 0.002};
        double[] dropouts = {0.3, 0.5, 0.7};
        int[] hiddenSizes = {64, 128, 256};
        
        double bestAccuracy = 0;
        String bestConfig = "";
        
        for (double lr : learningRates) {
            for (double dropout : dropouts) {
                for (int hiddenSize : hiddenSizes) {
                    // 构建模型
                    MultiLayerNetwork model = buildModel(lr, dropout, hiddenSize);
                    
                    // 训练
                    for (int epoch = 0; epoch < 5; epoch++) {
                        model.fit(trainIter);
                    }
                    
                    // 评估
                    var eval = model.evaluate(testIter);
                    double accuracy = eval.accuracy();
                    
                    String config = String.format("lr=%.4f, dropout=%.1f, hidden=%d",
                        lr, dropout, hiddenSize);
                    System.out.printf("%s -> 准确率: %.2f%%%n", config, accuracy * 100);
                    
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestConfig = config;
                    }
                    
                    trainIter.reset();
                    testIter.reset();
                }
            }
        }
        
        System.out.println("\n最佳配置: " + bestConfig);
        System.out.println("最佳准确率: " + bestAccuracy * 100 + "%");
    }
}
```

### 3. 模型集成

```java
/**
 * 模型集成
 */
public class ModelEnsemble {
    
    private List<MultiLayerNetwork> models;
    
    public ModelEnsemble(int numModels) {
        models = new ArrayList<>();
        for (int i = 0; i < numModels; i++) {
            models.add(buildModelWithSeed(42 + i * 100));
        }
    }
    
    /**
     * 集成预测：投票
     */
    public int predictVote(INDArray input) {
        int[] votes = new int[10];
        
        for (MultiLayerNetwork model : models) {
            INDArray output = model.output(input);
            int prediction = output.argMax().getInt(0);
            votes[prediction]++;
        }
        
        int maxVotes = 0;
        int result = 0;
        for (int i = 0; i < 10; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                result = i;
            }
        }
        
        return result;
    }
    
    /**
     * 集成预测：平均概率
     */
    public INDArray predictAverage(INDArray input) {
        INDArray sum = Nd4j.zeros(10);
        
        for (MultiLayerNetwork model : models) {
            INDArray output = model.output(input);
            sum.addi(output);
        }
        
        return sum.div(models.size());
    }
}
```

## 模型压缩

### 知识蒸馏

```java
/**
 * 知识蒸馏：用大模型教小模型
 */
public class KnowledgeDistillation {
    
    private MultiLayerNetwork teacher;
    private MultiLayerNetwork student;
    private double temperature = 3.0;
    private double alpha = 0.7;  // 蒸馏损失权重
    
    /**
     * 蒸馏训练
     */
    public void train(DataSetIterator trainIter) {
        while (trainIter.hasNext()) {
            DataSet batch = trainIter.next();
            
            // 教师模型的软标签
            INDArray teacherOutput = teacher.output(batch.getFeatures());
            INDArray softLabels = Transforms.softmax(teacherOutput.div(temperature));
            
            // 混合损失
            // L = α * L_distill + (1-α) * L_hard
            // ...
            
            student.fit(batch);
        }
    }
}
```

### 量化

```java
/**
 * 模型量化
 */
public class ModelQuantization {
    
    /**
     * 权重量化到INT8
     */
    public static void quantize(MultiLayerNetwork model) {
        for (org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray weights = layer.getParam("W");
            
            // 计算量化参数
            float max = weights.maxNumber().floatValue();
            float min = weights.minNumber().floatValue();
            float scale = 127.0f / Math.max(Math.abs(max), Math.abs(min));
            
            // 量化
            INDArray quantized = weights.mul(scale).round();
            
            // 反量化（用于推理）
            INDArray dequantized = quantized.div(scale);
            
            layer.setParam("W", dequantized);
        }
        
        System.out.println("模型量化完成");
    }
}
```

## 性能基准

### 推理速度测试

```java
/**
 * 性能基准测试
 */
public class PerformanceBenchmark {
    
    public static void benchmark(MultiLayerNetwork model, int warmup, int iterations) {
        INDArray sampleInput = Nd4j.rand(1, 784);
        
        // 预热
        System.out.println("预热中...");
        for (int i = 0; i < warmup; i++) {
            model.output(sampleInput);
        }
        
        // 正式测试
        System.out.println("性能测试...");
        long totalTime = 0;
        
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            model.output(sampleInput);
            long end = System.nanoTime();
            
            totalTime += (end - start);
        }
        
        double avgTimeMs = totalTime / 1_000_000.0 / iterations;
        double throughput = 1000.0 / avgTimeMs;
        
        System.out.println("\n=== 性能基准 ===");
        System.out.printf("平均推理时间: %.3f ms%n", avgTimeMs);
        System.out.printf("吞吐量: %.1f samples/s%n", throughput);
    }
}
```

## 小结

本章我们学习了：

1. **评估指标**：准确率、精确率、召回率、F1
2. **错误分析**：混淆矩阵、错误样本
3. **优化策略**：数据增强、超参数调优、模型集成
4. **模型压缩**：知识蒸馏、量化

**优化检查清单：**

| 问题 | 解决方案 |
|------|----------|
| 过拟合 | Dropout、正则化、数据增强 |
| 欠拟合 | 增加模型复杂度、减少正则化 |
| 训练慢 | 学习率调优、批次归一化 |
| 推理慢 | 模型压缩、量化 |

**下一步：** 我们将进行系统部署与集成。

---

**练习题：**

1. 如何解读混淆矩阵？它能告诉你什么信息？
2. 模型集成为什么能提高准确率？
3. 知识蒸馏的原理是什么？为什么有效？
