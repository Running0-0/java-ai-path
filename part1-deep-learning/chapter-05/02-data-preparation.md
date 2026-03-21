<table width="100%">
   <tr>
      <td align="left"><a href="01-project-overview.md">← 5.1 项目概述与需求分析</a></td>
      <td align="right"><a href="03-model-design-training.md">5.3 模型设计与训练 →</a></td>
   </tr>
</table>

---

# 5.2 数据准备与预处理

> "数据是AI的燃料，预处理是提炼的过程。好的数据准备事半功倍。"

## MNIST数据加载

### 使用DL4J内置加载器

```java
package com.example.digit.data;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * MNIST数据加载器
 */
public class MnistDataLoader {
    
    private final int batchSize;
    private final long seed;
    
    public MnistDataLoader(int batchSize, long seed) {
        this.batchSize = batchSize;
        this.seed = seed;
    }
    
    /**
     * 获取训练数据迭代器
     */
    public DataSetIterator getTrainIterator() throws Exception {
        return new MnistDataSetIterator(batchSize, true, seed);
    }
    
    /**
     * 获取测试数据迭代器
     */
    public DataSetIterator getTestIterator() throws Exception {
        return new MnistDataSetIterator(batchSize, false, seed);
    }
    
    /**
     * 划分验证集
     */
    public DataSetIterator[] splitTrainValidation(DataSetIterator trainIter, double validationRatio) {
        // 简化实现：使用不同seed创建验证集
        // 实际项目中应该从训练集中划分
        return new DataSetIterator[]{trainIter, getTestIterator()};
    }
}
```

### 数据探索

```java
/**
 * 数据探索工具
 */
public class DataExplorer {
    
    public static void explore(DataSetIterator iterator) {
        iterator.reset();
        
        int totalSamples = iterator.totalExamples();
        int numClasses = iterator.totalOutcomes();
        int inputSize = iterator.inputColumns();
        
        System.out.println("=== 数据集信息 ===");
        System.out.println("总样本数: " + totalSamples);
        System.out.println("类别数量: " + numClasses);
        System.out.println("输入维度: " + inputSize);
        
        // 统计各类别数量
        int[] classCounts = new int[numClasses];
        while (iterator.hasNext()) {
            DataSet batch = iterator.next();
            for (int i = 0; i < batch.getLabels().size(0); i++) {
                int label = batch.getLabels().getRow(i).argMax().getInt(0);
                classCounts[label]++;
            }
        }
        
        System.out.println("\n各类别样本数:");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("  数字%d: %d (%.1f%%)%n", 
                i, classCounts[i], 100.0 * classCounts[i] / totalSamples);
        }
        
        iterator.reset();
    }
}
```

## 数据预处理

### 标准化处理

```java
package com.example.digit.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

/**
 * 数据预处理器
 */
public class DataPreprocessor {
    
    private DataNormalization normalizer;
    
    public DataPreprocessor() {
        // 归一化到[0,1]
        this.normalizer = new NormalizerMinMaxScaler(0, 1);
    }
    
    /**
     * 拟合训练数据
     */
    public void fit(DataSetIterator trainIter) {
        normalizer.fit(trainIter);
    }
    
    /**
     * 转换数据
     */
    public void transform(DataSetIterator iterator) {
        iterator.setPreProcessor(normalizer);
    }
    
    /**
     * 转换单个样本
     */
    public INDArray transform(INDArray sample) {
        DataSet ds = new DataSet(sample, Nd4j.zeros(10));
        normalizer.transform(ds);
        return ds.getFeatures();
    }
}
```

### 图像增强

```java
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
            // 随机旋转（-15到15度）
            .add(new RotateImageTransform(15))
            // 随机平移
            .add(new ImageTranslationTransform(0.1, 0.1))
            // 随机缩放
            .add(new ScaleImageTransform(0.9f, 1.1f))
            .build();
    }
}
```

## 自定义数据加载

### 从图片目录加载

```java
/**
 * 自定义数据加载器
 */
public class CustomDataLoader {
    
    /**
     * 从目录加载图片
     */
    public static DataSetIterator loadFromDirectory(String path, int batchSize, 
                                                     int height, int width) throws Exception {
        File rootDir = new File(path);
        
        // 使用父目录名作为标签
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        
        ImageRecordReader recordReader = new ImageRecordReader(height, width, 1, labelGenerator);
        recordReader.initialize(new FileSplit(rootDir));
        
        DataSetIterator iterator = new RecordReaderDataSetIterator(
            recordReader, batchSize, 1, recordReader.numLabels());
        
        // 归一化
        iterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        
        return iterator;
    }
}
```

## 数据可视化

### 显示样本图像

```java
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * 数据可视化工具
 */
public class DataVisualizer {
    
    /**
     * 可视化一批样本
     */
    public static void visualizeBatch(DataSet batch, String outputPath) throws Exception {
        int numSamples = Math.min(25, (int) batch.getFeatures().size(0));
        int gridSize = (int) Math.ceil(Math.sqrt(numSamples));
        
        int cellSize = 28;
        int margin = 2;
        
        BufferedImage image = new BufferedImage(
            gridSize * (cellSize + margin),
            gridSize * (cellSize + margin),
            BufferedImage.TYPE_INT_RGB
        );
        
        Graphics2D g = image.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, image.getWidth(), image.getHeight());
        
        for (int i = 0; i < numSamples; i++) {
            int row = i / gridSize;
            int col = i % gridSize;
            
            int x = col * (cellSize + margin);
            int y = row * (cellSize + margin);
            
            INDArray sample = batch.getFeatures().getRow(i);
            int label = batch.getLabels().getRow(i).argMax().getInt(0);
            
            drawDigit(g, x, y, sample, cellSize);
            g.setColor(Color.RED);
            g.drawString(String.valueOf(label), x + 2, y + cellSize - 2);
        }
        
        g.dispose();
        ImageIO.write(image, "PNG", new File(outputPath));
    }
    
    private static void drawDigit(Graphics2D g, int x, int y, INDArray data, int size) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double value = data.getDouble(i * 28 + j);
                int gray = (int) (value * 255);
                g.setColor(new Color(gray, gray, gray));
                int pixelSize = size / 28;
                g.fillRect(x + j * pixelSize, y + i * pixelSize, pixelSize, pixelSize);
            }
        }
    }
}
```

## 数据管道完整示例

```java
package com.example.digit.data;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * 数据管道
 */
public class DataPipeline {
    
    private DataSetIterator trainIter;
    private DataSetIterator testIter;
    private DataPreprocessor preprocessor;
    
    public DataPipeline(int batchSize) throws Exception {
        // 1. 加载原始数据
        trainIter = new MnistDataSetIterator(batchSize, true, 42);
        testIter = new MnistDataSetIterator(batchSize, false, 42);
        
        // 2. 数据探索
        DataExplorer.explore(trainIter);
        
        // 3. 初始化预处理器
        preprocessor = new DataPreprocessor();
        preprocessor.fit(trainIter);
        
        // 4. 应用预处理
        preprocessor.transform(trainIter);
        preprocessor.transform(testIter);
        
        System.out.println("数据管道初始化完成");
    }
    
    public DataSetIterator getTrainIterator() { return trainIter; }
    public DataSetIterator getTestIterator() { return testIter; }
    public DataPreprocessor getPreprocessor() { return preprocessor; }
}
```

## 小结

本章我们实现了：

1. **数据加载**：MNIST内置加载器和自定义加载
2. **数据探索**：统计和可视化
3. **数据预处理**：标准化和增强
4. **数据管道**：完整的处理流程

**下一步：** 我们将设计模型架构并进行训练。

---

**练习题：**

1. 为什么需要将像素值归一化到[0,1]？
2. 数据增强对MNIST有帮助吗？为什么？
3. 如何处理类别不平衡问题？

---

<table width="100%">
   <tr>
      <td align="left"><a href="01-project-overview.md">← 5.1 项目概述与需求分析</a></td>
      <td align="right"><a href="03-model-design-training.md">5.3 模型设计与训练 →</a></td>
   </tr>
</table>
