# 1.3 搭建你的第一个AI开发环境

> "工欲善其事，必先利其器。一个良好的开发环境是AI学习之路的起点。"

## 环境选择：为什么选择这些工具

作为Java程序员，我们选择最熟悉的工具链：

| 组件 | 推荐选择 | 原因 |
|------|----------|------|
| JDK | Java 17 LTS | 长期支持版本，新特性丰富 |
| 构建工具 | Maven | 生态成熟，依赖管理方便 |
| IDE | IntelliJ IDEA | Java开发首选，AI插件支持好 |
| 深度学习框架 | Deeplearning4j | Java生态最成熟的DL框架 |
| LLM框架 | LangChain4j | Java生态最活跃的LLM框架 |

## 第一步：安装JDK 17

### Windows系统

1. 下载JDK 17：
   - 推荐使用 [Adoptium Temurin](https://adoptium.net/)
   - 选择 JDK 17 LTS 版本

2. 安装后配置环境变量：
```powershell
# 设置JAVA_HOME
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.x"

# 添加到PATH
$env:Path += ";$env:JAVA_HOME\bin"
```

3. 验证安装：
```powershell
java -version
# 输出：openjdk version "17.0.x"
```

### macOS系统

```bash
# 使用Homebrew安装
brew install openjdk@17

# 配置环境变量
echo 'export PATH="/usr/local/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Linux系统

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install openjdk-17-jdk

# CentOS/RHEL
sudo yum install java-17-openjdk-devel
```

## 第二步：创建Maven项目

### 项目结构

```
java-ai-learning/
├── pom.xml
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/ai/
│   │   │       ├── chapter01/
│   │   │       ├── chapter02/
│   │   │       └── ...
│   │   └── resources/
│   └── test/
│       └── java/
└── data/                     # 数据文件目录
```

### pom.xml配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>java-ai-learning</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <name>Java AI Learning</name>
    <description>Java程序员的AI学习项目</description>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        
        <!-- 版本管理 -->
        <dl4j.version>1.0.0-M2.1</dl4j.version>
        <langchain4j.version>0.35.0</langchain4j.version>
        <slf4j.version>2.0.9</slf4j.version>
        <junit.version>5.10.1</junit.version>
    </properties>

    <dependencies>
        <!-- Deeplearning4j 核心依赖 -->
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        
        <!-- ND4J: NumPy for Java -->
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native-platform</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        
        <!-- DataVec: 数据预处理 -->
        <dependency>
            <groupId>org.datavec</groupId>
            <artifactId>datavec-data-image</artifactId>
            <version>${dl4j.version}</version>
        </dependency>

        <!-- LangChain4j 核心 -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j</artifactId>
            <version>${langchain4j.version}</version>
        </dependency>
        
        <!-- LangChain4j OpenAI 集成 -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-open-ai</artifactId>
            <version>${langchain4j.version}</version>
        </dependency>

        <!-- 日志 -->
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>${slf4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-simple</artifactId>
            <version>${slf4j.version}</version>
        </dependency>

        <!-- 测试 -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>${junit.version}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                </configuration>
            </plugin>
            
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.2.2</version>
            </plugin>
        </plugins>
    </build>
</project>
```

## 第三步：验证环境

### 创建测试类

```java
package com.example.ai.chapter01;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 环境验证测试
 */
class EnvironmentTest {

    @Test
    void testNd4jInstallation() {
        // 测试ND4J（类似NumPy的矩阵运算库）
        INDArray array1 = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray array2 = Nd4j.create(new double[]{4.0, 5.0, 6.0});
        
        // 矩阵加法
        INDArray sum = array1.add(array2);
        
        System.out.println("Array 1: " + array1);
        System.out.println("Array 2: " + array2);
        System.out.println("Sum: " + sum);
        
        assertArrayEquals(new double[]{5.0, 7.0, 9.0}, sum.toDoubleVector(), 0.001);
    }

    @Test
    void testMatrixOperations() {
        // 创建2x3矩阵
        INDArray matrix = Nd4j.create(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        
        System.out.println("Matrix shape: " + java.util.Arrays.toString(matrix.shape()));
        System.out.println("Matrix:\n" + matrix);
        
        // 矩阵转置
        INDArray transposed = matrix.transpose();
        System.out.println("Transposed:\n" + transposed);
        
        // 矩阵乘法
        INDArray result = matrix.mmul(transposed);
        System.out.println("Matrix * Transposed:\n" + result);
    }
}
```

### 运行测试

```bash
mvn test
```

如果看到测试通过，恭喜你，环境搭建成功！

## 第四步：第一个神经网络程序

让我们用Deeplearning4j创建一个简单的神经网络：

```java
package com.example.ai.chapter01;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 第一个神经网络：学习XOR逻辑
 */
public class FirstNeuralNetwork {

    public static void main(String[] args) {
        // 1. 准备数据
        INDArray features = Nd4j.create(new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        });
        
        INDArray labels = Nd4j.create(new double[][]{
            {0},
            {1},
            {1},
            {0}
        });
        
        DataSet dataSet = new DataSet(features, labels);

        // 2. 配置神经网络
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(12345)  // 随机种子，保证可复现
            .updater(new Sgd(0.1))  // 随机梯度下降，学习率0.1
            .list()
            // 隐藏层：2个输入 -> 4个神经元
            .layer(new DenseLayer.Builder()
                .nIn(2)
                .nOut(4)
                .activation(Activation.RELU)
                .build())
            // 输出层：4个神经元 -> 1个输出
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(4)
                .nOut(1)
                .activation(Activation.SIGMOID)
                .build())
            .build();

        // 3. 创建并初始化网络
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        
        // 添加训练监听器，每100次迭代打印一次损失
        network.setListeners(new ScoreIterationListener(100));

        // 4. 训练
        System.out.println("开始训练...");
        for (int i = 0; i < 1000; i++) {
            network.fit(dataSet);
        }
        System.out.println("训练完成！");

        // 5. 测试
        System.out.println("\n测试结果：");
        INDArray predictions = network.output(features);
        for (int i = 0; i < 4; i++) {
            System.out.printf("%.0f XOR %.0f = %.3f (期望: %.0f)%n",
                features.getDouble(i, 0),
                features.getDouble(i, 1),
                predictions.getDouble(i, 0),
                labels.getDouble(i, 0));
        }
    }
}
```

**预期输出：**
```
开始训练...
训练完成！

测试结果：
0 XOR 0 = 0.048 (期望: 0)
0 XOR 1 = 0.956 (期望: 1)
1 XOR 0 = 0.956 (期望: 1)
1 XOR 1 = 0.043 (期望: 0)
```

恭喜！你已经成功训练了第一个神经网络！

## 设计思考：环境配置的哲学

### 为什么选择Deeplearning4j？

| 特性 | Deeplearning4j | Python生态 |
|------|----------------|------------|
| 语言 | Java/Kotlin | Python |
| 部署 | 原生Java应用 | 需要额外服务 |
| 企业集成 | 直接集成 | 需要API桥接 |
| 性能 | 生产级优化 | 研究级优先 |

**设计理念：** 选择工具时，考虑的是整个软件生命周期，而非仅仅是开发阶段。

### GPU加速（可选）

如果你有NVIDIA GPU，可以启用CUDA加速：

```xml
<!-- 替换nd4j-native-platform为nd4j-cuda-11.x-platform -->
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-11.8-platform</artifactId>
    <version>${dl4j.version}</version>
</dependency>
```

## 常见问题解决

### 问题1：内存不足

```java
// 在程序开始时设置内存
System.setProperty("org.bytedeco.javacpp.maxbytes", "4G");
System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "8G");
```

### 问题2：找不到本地库

确保Maven正确下载了所有依赖：
```bash
mvn dependency:resolve
```

### 问题3：训练太慢

- 检查是否使用了GPU版本
- 减小batch大小
- 增大学习率

## 小结

本章我们完成了：

1. **JDK安装**：Java 17 LTS
2. **Maven项目创建**：完整的依赖配置
3. **环境验证**：ND4J矩阵运算测试
4. **第一个神经网络**：XOR问题求解

**下一步：** 我们将探索Java AI生态的全景图，了解有哪些工具可以使用。

---

**练习题：**

1. 修改上面的XOR网络，增加隐藏层神经元数量，观察训练效果变化
2. 尝试不同的激活函数（TANH、LEAKYRELU），比较效果
3. 调整学习率，观察训练收敛速度的变化
