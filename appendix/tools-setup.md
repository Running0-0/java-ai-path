# 附录A：工具环境配置指南

> "工欲善其事，必先利其器——配置好开发环境，开启AI之旅。"

## Java开发环境

### JDK安装

```bash
# 推荐版本：JDK 17或更高
# 下载地址：https://adoptium.net/

# 验证安装
java -version
javac -version
```

### Maven配置

```bash
# 下载：https://maven.apache.org/download.cgi
# 配置环境变量MAVEN_HOME

# 验证
mvn -version

# 配置阿里云镜像（推荐）
# 编辑 ~/.m2/settings.xml
<mirrors>
  <mirror>
    <id>aliyunmaven</id>
    <name>阿里云公共仓库</name>
    <url>https://maven.aliyun.com/repository/public</url>
    <mirrorOf>central</mirrorOf>
  </mirror>
</mirrors>
```

## Python环境

### 安装Anaconda

```bash
# 下载：https://www.anaconda.com/download
# 安装后创建虚拟环境

conda create -n ai python=3.10
conda activate ai

# 安装常用包
pip install jupyter numpy pandas matplotlib
```

### PyTorch安装

```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（需NVIDIA显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 深度学习框架

### Deeplearning4j

```xml
<!-- pom.xml -->
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
```

### ND4J后端选择

```xml
<!-- CPU后端 -->
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native-platform</artifactId>
    <version>1.0.0-M2.1</version>
</dependency>

<!-- CUDA后端（需NVIDIA GPU） -->
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-11.8-platform</artifactId>
    <version>1.0.0-M2.1</version>
</dependency>
```

## 大语言模型工具

### Ollama安装

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 下载安装程序：https://ollama.com/download/windows

# 验证安装
ollama --version

# 下载模型
ollama pull llama2
ollama pull qwen
ollama pull nomic-embed-text

# 运行服务
ollama serve
```

### LangChain4j配置

```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j</artifactId>
    <version>0.24.0</version>
</dependency>

<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-ollama</artifactId>
    <version>0.24.0</version>
</dependency>
```

## 向量数据库

### Chroma安装

```bash
# 使用Docker
docker pull chromadb/chroma:latest
docker run -p 8000:8000 chromadb/chroma:latest

# Python客户端
pip install chromadb
```

### Milvus安装

```bash
# Docker Compose方式
wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml

docker-compose up -d
```

## IDE配置

### IntelliJ IDEA推荐插件

```
1. Lombok - 简化代码
2. .env files support - 环境变量
3. Rainbow Brackets - 括号匹配
4. CodeGlance - 代码缩略图
5. GitToolBox - Git增强
6. Markdown - Markdown支持
```

### VS Code配置

```json
// settings.json
{
  "java.configuration.runtimes": [
    {
      "name": "JavaSE-17",
      "path": "/path/to/jdk-17",
      "default": true
    }
  ],
  "java.format.settings.url": "https://raw.githubusercontent.com/google/styleguide/gh-pages/eclipse-java-google-style.xml",
  "editor.formatOnSave": true
}
```

## 环境验证

### 完整环境检查脚本

```bash
#!/bin/bash

echo "=== 环境检查 ==="

echo "Java版本:"
java -version

echo -e "\nMaven版本:"
mvn -version

echo -e "\nPython版本:"
python --version

echo -e "\nOllama状态:"
curl http://localhost:11434/api/tags 2>/dev/null && echo "Ollama运行正常" || echo "Ollama未启动"

echo -e "\nChroma状态:"
curl http://localhost:8000/api/v1/heartbeat 2>/dev/null && echo "Chroma运行正常" || echo "Chroma未启动"

echo -e "\n=== 检查完成 ==="
```

## 常见问题

### 1. Java内存不足

```bash
# 设置JVM参数
export MAVEN_OPTS="-Xmx4g -XX:MaxMetaspaceSize=512m"
```

### 2. ND4J加载失败

```bash
# 检查系统依赖（Linux）
sudo apt-get install libopenblas-base libgomp1

# macOS
brew install openblas
```

### 3. Ollama连接失败

```bash
# 检查服务状态
ollama serve

# 或后台运行
nohup ollama serve > ollama.log 2>&1 &
```

## 推荐配置

### 最低配置

```
- CPU: 4核心
- 内存: 8GB
- 硬盘: 50GB
- 系统: Windows 10 / macOS 10.15 / Ubuntu 20.04
```

### 推荐配置

```
- CPU: 8核心
- 内存: 16GB
- GPU: NVIDIA RTX 3060或更高（可选）
- 硬盘: 100GB SSD
- 系统: Windows 11 / macOS 13 / Ubuntu 22.04
```

---

配置完成后，你就可以开始《Java程序员的AI之路》的学习之旅了！
