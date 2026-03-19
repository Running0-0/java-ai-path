<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">

<span>[← 5.4 模型评估与优化](04-model-evaluation-optimization.md)</span>

<span>[6.1 什么是语言模型 →](../../part2-llm/chapter-06/01-what-is-language-model.md)</span>

</div>

---

# 5.5 部署与集成：将AI融入Java应用

> "AI的价值在于应用。一个好的部署方案，能让模型真正发挥价值。"

## 部署架构

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    客户端                            │
│              Web / 移动端 / 桌面应用                  │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP API
┌─────────────────────▼───────────────────────────────┐
│                    API网关                           │
│              认证 / 限流 / 路由                       │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  识别服务                            │
│          图像预处理 → 模型推理 → 结果后处理           │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  模型存储                            │
│              本地文件 / 对象存储                      │
└─────────────────────────────────────────────────────┘
```

## Spring Boot集成

### 项目配置

```xml
<!-- pom.xml -->
<dependencies>
    <!-- Spring Boot -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    
    <!-- Deeplearning4j -->
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>1.0.0-M2.1</version>
    </dependency>
</dependencies>
```

### 应用配置

```yaml
# application.yml
server:
  port: 8080

digit:
  model:
    path: models/digit-model.zip
  image:
    width: 28
    height: 28
    
spring:
  servlet:
    multipart:
      max-file-size: 10MB
```

### 服务实现

```java
package com.example.digit.service;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.File;

/**
 * 数字识别服务
 */
@Service
public class RecognitionService {
    
    @Value("${digit.model.path}")
    private String modelPath;
    
    private MultiLayerNetwork model;
    
    @PostConstruct
    public void init() throws Exception {
        // 加载模型
        model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        System.out.println("模型加载完成: " + modelPath);
    }
    
    /**
     * 识别数字
     */
    public RecognitionResult recognize(INDArray image) {
        long startTime = System.currentTimeMillis();
        
        // 模型推理
        INDArray output = model.output(image);
        
        // 获取预测结果
        int predictedDigit = output.argMax(1).getInt(0);
        float confidence = output.getFloat(predictedDigit);
        
        long inferenceTime = System.currentTimeMillis() - startTime;
        
        return new RecognitionResult(
            predictedDigit,
            confidence,
            output.toDoubleVector(),
            inferenceTime
        );
    }
    
    /**
     * 识别结果
     */
    public record RecognitionResult(
        int digit,
        float confidence,
        double[] probabilities,
        long inferenceTimeMs
    ) {}
}
```

### 图像处理服务

```java
package com.example.digit.service;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * 图像处理服务
 */
@Service
public class ImageProcessService {
    
    @Value("${digit.image.width}")
    private int targetWidth;
    
    @Value("${digit.image.height}")
    private int targetHeight;
    
    /**
     * 预处理上传的图像
     */
    public INDArray preprocess(MultipartFile file) throws Exception {
        // 读取图像
        BufferedImage image = ImageIO.read(file.getInputStream());
        
        // 转换为灰度
        BufferedImage grayImage = toGrayscale(image);
        
        // 调整大小
        BufferedImage resized = resize(grayImage, targetWidth, targetHeight);
        
        // 转换为INDArray
        INDArray array = imageToArray(resized);
        
        // 归一化
        array.divi(255.0);
        
        return array.reshape(1, targetWidth * targetHeight);
    }
    
    private BufferedImage toGrayscale(BufferedImage image) {
        BufferedImage gray = new BufferedImage(
            image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = gray.getGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();
        return gray;
    }
    
    private BufferedImage resize(BufferedImage image, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, 
            RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }
    
    private INDArray imageToArray(BufferedImage image) {
        INDArray array = Nd4j.create(targetWidth * targetHeight);
        
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = image.getRGB(x, y) & 0xFF;
                array.putScalar(y * targetWidth + x, rgb);
            }
        }
        
        return array;
    }
}
```

### REST API

```java
package com.example.digit.controller;

import com.example.digit.dto.RecognitionResponse;
import com.example.digit.service.ImageProcessService;
import com.example.digit.service.RecognitionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 识别API控制器
 */
@RestController
@RequestMapping("/api/v1")
public class RecognitionController {
    
    @Autowired
    private RecognitionService recognitionService;
    
    @Autowired
    private ImageProcessService imageProcessService;
    
    /**
     * 单图识别
     */
    @PostMapping("/recognize")
    public ResponseEntity<RecognitionResponse> recognize(
            @RequestParam("image") MultipartFile image) {
        
        try {
            // 预处理
            INDArray processed = imageProcessService.preprocess(image);
            
            // 识别
            var result = recognitionService.recognize(processed);
            
            // 构建响应
            RecognitionResponse response = new RecognitionResponse(
                true,
                result.digit(),
                result.confidence(),
                result.probabilities(),
                result.inferenceTimeMs()
            );
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                new RecognitionResponse(false, -1, 0, null, 0)
            );
        }
    }
    
    /**
     * 批量识别
     */
    @PostMapping("/recognize/batch")
    public ResponseEntity<List<RecognitionResponse>> recognizeBatch(
            @RequestParam("images") MultipartFile[] images) {
        
        List<RecognitionResponse> results = new ArrayList<>();
        
        for (MultipartFile image : images) {
            try {
                INDArray processed = imageProcessService.preprocess(image);
                var result = recognitionService.recognize(processed);
                results.add(new RecognitionResponse(
                    true, result.digit(), result.confidence(),
                    result.probabilities(), result.inferenceTimeMs()
                ));
            } catch (Exception e) {
                results.add(new RecognitionResponse(false, -1, 0, null, 0));
            }
        }
        
        return ResponseEntity.ok(results);
    }
    
    /**
     * 健康检查
     */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("OK");
    }
}
```

### 响应DTO

```java
package com.example.digit.dto;

/**
 * 识别响应
 */
public record RecognitionResponse(
    boolean success,
    int digit,
    float confidence,
    double[] probabilities,
    long inferenceTimeMs
) {}
```

## 前端界面

### 简单HTML界面

```html
<!DOCTYPE html>
<html>
<head>
    <title>手写数字识别</title>
    <style>
        .container { max-width: 600px; margin: 50px auto; text-align: center; }
        #canvas { border: 2px solid #333; cursor: crosshair; }
        .btn { padding: 10px 20px; margin: 10px; font-size: 16px; }
        .result { margin-top: 20px; font-size: 24px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>手写数字识别</h1>
        <canvas id="canvas" width="280" height="280"></canvas>
        <div>
            <button class="btn" onclick="clearCanvas()">清除</button>
            <button class="btn" onclick="recognize()">识别</button>
        </div>
        <div class="result" id="result"></div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;

        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';

        canvas.addEventListener('mousedown', () => isDrawing = true);
        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);
        canvas.addEventListener('mousemove', draw);

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').textContent = '';
        }

        async function recognize() {
            const dataUrl = canvas.toDataURL('image/png');
            const blob = await (await fetch(dataUrl)).blob();
            const formData = new FormData();
            formData.append('image', blob, 'digit.png');

            const response = await fetch('/api/v1/recognize', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                document.getElementById('result').innerHTML = 
                    `识别结果: <strong>${result.digit}</strong> 
                     (置信度: ${(result.confidence * 100).toFixed(1)}%)`;
            } else {
                document.getElementById('result').textContent = '识别失败';
            }
        }
    </script>
</body>
</html>
```

## 性能优化

### 模型预热

```java
@PostConstruct
public void warmup() {
    // 预热模型
    INDArray dummy = Nd4j.zeros(1, 784);
    for (int i = 0; i < 10; i++) {
        model.output(dummy);
    }
    System.out.println("模型预热完成");
}
```

### 并发处理

```java
@Service
public class AsyncRecognitionService {
    
    @Autowired
    private RecognitionService recognitionService;
    
    @Async
    public CompletableFuture<RecognitionResult> recognizeAsync(INDArray image) {
        return CompletableFuture.completedFuture(
            recognitionService.recognize(image)
        );
    }
}
```

## 监控与日志

### 添加监控

```java
@Aspect
@Component
public class RecognitionMonitor {
    
    private final MeterRegistry registry;
    
    @Around("execution(* com.example.digit.service.RecognitionService.recognize(..))")
    public Object monitor(ProceedingJoinPoint pjp) throws Throwable {
        Timer.Sample sample = Timer.start(registry);
        
        try {
            Object result = pjp.proceed();
            sample.stop(registry.timer("recognition.time", "status", "success"));
            registry.counter("recognition.count", "status", "success").increment();
            return result;
        } catch (Exception e) {
            sample.stop(registry.timer("recognition.time", "status", "error"));
            registry.counter("recognition.count", "status", "error").increment();
            throw e;
        }
    }
}
```

## 部署方案

### Docker部署

```dockerfile
FROM openjdk:17-jdk-slim

WORKDIR /app

COPY target/digit-recognition.jar app.jar
COPY models/ models/

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "app.jar"]
```

### Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digit-recognition
spec:
  replicas: 2
  selector:
    matchLabels:
      app: digit-recognition
  template:
    metadata:
      labels:
        app: digit-recognition
    spec:
      containers:
      - name: app
        image: digit-recognition:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## 小结

本章我们完成了：

1. **Spring Boot集成**：服务层、控制器、配置
2. **图像处理**：预处理流程
3. **REST API**：单图和批量识别
4. **前端界面**：简单的绘图识别界面
5. **部署方案**：Docker和Kubernetes

**部署检查清单：**

| 项目 | 检查 |
|------|------|
| 模型加载 | ✓ 启动时加载 |
| API文档 | ✓ Swagger/OpenAPI |
| 错误处理 | ✓ 统一异常处理 |
| 日志记录 | ✓ 请求日志 |
| 性能监控 | ✓ Prometheus指标 |
| 健康检查 | ✓ /health端点 |

**第一部分完成！** 我们已经掌握了深度学习的基础知识和实践技能。

**下一步：** 我们将进入第二部分——大语言模型。

---

**练习题：**

1. 如何实现模型的懒加载？
2. 如何处理大并发请求？
3. 如何实现模型的灰度发布？

---

<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">

<span>[← 5.4 模型评估与优化](04-model-evaluation-optimization.md)</span>

<span>[6.1 什么是语言模型 →](../../part2-llm/chapter-06/01-what-is-language-model.md)</span>

</div>
