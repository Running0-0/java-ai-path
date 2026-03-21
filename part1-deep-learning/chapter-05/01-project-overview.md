<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-04/05-design-thinking-sequential-modeling.md">← 4.5 设计思考：时序建模的本质</a></td>
      <td align="right"><a href="02-data-preparation.md">5.2 数据准备与预处理 →</a></td>
   </tr>
</table>

---

# 5.1 项目概述与需求分析

> "实践是检验真理的唯一标准。让我们用所学知识构建一个完整的AI应用。"

## 项目背景

### 为什么选择手写数字识别

手写数字识别是深度学习的经典入门项目：

```
优势：
1. 数据集标准（MNIST）
2. 问题简单但完整
3. 可视化效果好
4. 技术点覆盖全面
```

### 实际应用场景

```
1. 银行支票识别
2. 邮政编码识别
3. 表单自动化处理
4. 手写笔记数字化
```

## 需求分析

### 功能需求

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 图像输入 | 支持图片文件上传 | 高 |
| 实时识别 | 摄像头实时识别 | 中 |
| 批量处理 | 批量图片识别 | 中 |
| 结果展示 | 显示识别结果和置信度 | 高 |
| 模型管理 | 模型训练、保存、加载 | 高 |

### 非功能需求

| 需求 | 指标 |
|------|------|
| 准确率 | > 98% |
| 响应时间 | < 100ms |
| 支持平台 | Windows/Linux/Mac |

### 系统边界

```
┌─────────────────────────────────────────┐
│            手写数字识别系统              │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ 图像输入 │→│ 预处理  │→│ 模型推理 │ │
│  └─────────┘  └─────────┘  └─────────┘ │
│                               ↓         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ 日志记录 │←│ 结果展示 │←│ 后处理  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────┘
```

## 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    表现层 (UI)                       │
│         Web界面 / 桌面应用 / API接口                  │
├─────────────────────────────────────────────────────┤
│                    业务层                            │
│    图像处理服务 │ 识别服务 │ 模型管理服务             │
├─────────────────────────────────────────────────────┤
│                    核心层                            │
│         CNN模型 │ 数据处理 │ 评估指标                │
├─────────────────────────────────────────────────────┤
│                    基础设施层                        │
│      文件存储 │ 日志系统 │ 配置管理                  │
└─────────────────────────────────────────────────────┘
```

### 技术选型

| 层次 | 技术选择 | 理由 |
|------|----------|------|
| 深度学习 | Deeplearning4j | Java生态成熟 |
| 图像处理 | OpenCV Java | 功能强大 |
| Web服务 | Spring Boot | 企业级支持 |
| 前端 | 简单HTML/JS | 轻量级 |

## 项目结构

```
digit-recognition/
├── pom.xml
├── src/main/java/
│   └── com/example/digit/
│       ├── DigitRecognitionApplication.java  # 启动类
│       ├── config/
│       │   └── AppConfig.java                # 配置类
│       ├── controller/
│       │   └── RecognitionController.java    # API控制器
│       ├── service/
│       │   ├── RecognitionService.java       # 识别服务
│       │   ├── TrainingService.java          # 训练服务
│       │   └── ImageProcessService.java      # 图像处理
│       ├── model/
│       │   ├── DigitClassifier.java          # 分类器
│       │   └── ModelConfig.java              # 模型配置
│       ├── data/
│       │   └── MnistDataLoader.java          # 数据加载
│       └── dto/
│           ├── RecognitionResult.java        # 结果DTO
│           └── TrainingRequest.java          # 请求DTO
├── src/main/resources/
│   ├── application.yml                       # 配置文件
│   └── static/                               # 前端文件
└── models/                                   # 模型存储目录
```

## 数据集介绍

### MNIST数据集

```
训练集：60,000张图片
测试集：10,000张图片
图片大小：28×28像素（灰度）
类别：0-9共10个数字
```

### 数据格式

```
每张图片：
- 28×28像素
- 灰度值范围：0-255
- 标签：0-9的数字
```

### 数据加载

```java
/**
 * MNIST数据加载器
 */
public class MnistDataLoader {
    
    private static final int IMAGE_SIZE = 28;
    
    /**
     * 加载MNIST数据集
     */
    public static DataSetIterator loadData(int batchSize, boolean train) throws Exception {
        return new MnistDataSetIterator(batchSize, train, 12345);
    }
    
    /**
     * 可视化样本
     */
    public static void visualizeSample(DataSet dataSet, int index) {
        INDArray image = dataSet.getFeatures().getRow(index);
        int label = dataSet.getLabels().getRow(index).argMax().getInt(0);
        
        System.out.println("标签: " + label);
        
        // 打印ASCII艺术
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                double pixel = image.getDouble(i * IMAGE_SIZE + j);
                char c = pixel > 0.5 ? '#' : ' ';
                System.out.print(c);
            }
            System.out.println();
        }
    }
}
```

## 开发计划

### 阶段划分

| 阶段 | 内容 | 时间 |
|------|------|------|
| 阶段1 | 项目搭建、数据加载 | 1天 |
| 阶段2 | 模型设计与训练 | 2天 |
| 阶段3 | 模型评估与优化 | 1天 |
| 阶段4 | API开发与集成 | 1天 |
| 阶段5 | 测试与部署 | 1天 |

### 里程碑

```
M1: 数据加载和预处理完成
M2: 模型训练准确率达到98%
M3: API接口开发完成
M4: 系统测试通过
M5: 部署上线
```

## 小结

本章我们完成了：

1. **需求分析**：功能和非功能需求
2. **技术架构**：分层架构设计
3. **项目结构**：代码组织方式
4. **开发计划**：阶段划分

**下一步：** 我们将进行数据准备与预处理。

---

**思考题：**

1. 为什么手写数字识别适合作为入门项目？
2. 系统的分层架构有什么好处？
3. 如果要支持更多字符（如字母），需要做哪些改动？

---

<table width="100%">
   <tr>
      <td align="left"><a href="../chapter-04/05-design-thinking-sequential-modeling.md">← 4.5 设计思考：时序建模的本质</a></td>
      <td align="right"><a href="02-data-preparation.md">5.2 数据准备与预处理 →</a></td>
   </tr>
</table>