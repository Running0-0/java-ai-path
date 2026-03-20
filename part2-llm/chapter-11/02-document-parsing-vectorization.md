# 11.2 文档解析与向量化

> "文档解析是RAG的第一步——Garbage in, garbage out，好的输入才有好的输出。"

## 文档解析

### 支持的格式

```
常见文档格式：

文本类：
- .txt - 纯文本
- .md - Markdown
- .csv - 表格数据
- .json - 结构化数据

办公文档：
- .pdf - 最常用，解析复杂
- .docx - Word文档
- .xlsx - Excel表格
- .pptx - PowerPoint

网页类：
- .html - 网页
- 直接URL抓取
```

### PDF解析

```java
package com.example.rag.parser;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.apache.poi.xwpf.usermodel.XWPFDocument;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * 文档解析器
 */
public class DocumentParser {
    
    /**
     * 解析PDF
     */
    public ParsedDocument parsePDF(File file) throws IOException {
        try (PDDocument document = PDDocument.load(file)) {
            PDFTextStripper stripper = new PDFTextStripper();
            String text = stripper.getText(document);
            
            // 提取元数据
            Metadata metadata = new Metadata();
            metadata.setTitle(document.getDocumentInformation().getTitle());
            metadata.setPageCount(document.getNumberOfPages());
            
            return new ParsedDocument(text, metadata, "pdf");
        }
    }
    
    /**
     * 解析Word
     */
    public ParsedDocument parseWord(File file) throws IOException {
        try (FileInputStream fis = new FileInputStream(file);
             XWPFDocument document = new XWPFDocument(fis)) {
            
            XWPFWordExtractor extractor = new XWPFWordExtractor(document);
            String text = extractor.getText();
            
            Metadata metadata = new Metadata();
            metadata.setTitle(document.getProperties().getCoreProperties().getTitle());
            
            return new ParsedDocument(text, metadata, "docx");
        }
    }
    
    /**
     * 解析Markdown
     */
    public ParsedDocument parseMarkdown(File file) throws IOException {
        String content = Files.readString(file.toPath());
        
        // 移除Markdown标记，提取纯文本
        String text = markdownToText(content);
        
        // 提取标题作为元数据
        String title = extractTitle(content);
        
        Metadata metadata = new Metadata();
        metadata.setTitle(title);
        
        return new ParsedDocument(text, metadata, "md");
    }
}
```

## 文档清洗

### 清洗策略

```java
/**
 * 文档清洗器
 */
public class DocumentCleaner {
    
    /**
     * 清洗文本
     */
    public String clean(String text) {
        // 1. 移除多余空白
        text = text.replaceAll("\\s+", " ");
        
        // 2. 移除特殊字符
        text = text.replaceAll("[^\\p{L}\\p{N}\\p{P}\\p{Z}]", "");
        
        // 3. 规范化换行
        text = text.replaceAll("\\n\\s*\\n+", "\\n\\n");
        
        // 4. 移除页眉页脚（简单规则）
        text = removeHeadersFooters(text);
        
        // 5. 修复编码问题
        text = fixEncodingIssues(text);
        
        return text.trim();
    }
    
    /**
     * 移除页眉页脚
     */
    private String removeHeadersFooters(String text) {
        // 移除页码
        text = text.replaceAll("\\n\\s*\\d+\\s*\\n", "\\n");
        
        // 移除重复的页眉
        String[] lines = text.split("\\n");
        Set<String> seen = new HashSet<>();
        StringBuilder result = new StringBuilder();
        
        for (String line : lines) {
            String normalized = line.trim().toLowerCase();
            if (!seen.contains(normalized) || normalized.length() > 50) {
                seen.add(normalized);
                result.append(line).append("\\n");
            }
        }
        
        return result.toString();
    }
}
```

## 文档分割

### 分割策略

```
分割方式对比：

1. 固定长度
   - 每N个字符/词分割
   - 简单快速
   - 可能切断语义

2. 语义分割
   - 按段落、句子分割
   - 保持语义完整
   - 需要更复杂的逻辑

3. 递归分割
   - 先按大粒度，再细化
   - 平衡效果和效率
   - 推荐方式
```

### Java实现

```java
package com.example.rag.splitter;

import java.util.ArrayList;
import java.util.List;

/**
 * 递归文档分割器
 */
public class RecursiveSplitter {
    
    private final int chunkSize;
    private final int chunkOverlap;
    private final List<String> separators;
    
    public RecursiveSplitter(int chunkSize, int chunkOverlap) {
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        // 优先级：段落 > 句子 > 词
        this.separators = Arrays.asList("\\n\\n", "\\n", ". ", " ", "");
    }
    
    /**
     * 分割文档
     */
    public List<Chunk> split(String text) {
        return splitRecursive(text, 0);
    }
    
    private List<Chunk> splitRecursive(String text, int separatorIndex) {
        List<Chunk> chunks = new ArrayList<>();
        
        if (separatorIndex >= separators.size()) {
            // 最后一个分隔符，直接切分
            return forceSplit(text);
        }
        
        String separator = separators.get(separatorIndex);
        String[] parts = text.split(separator);
        
        StringBuilder currentChunk = new StringBuilder();
        
        for (String part : parts) {
            if (currentChunk.length() + part.length() > chunkSize) {
                // 当前块已满，保存并开始新块
                if (currentChunk.length() > 0) {
                    chunks.add(new Chunk(currentChunk.toString().trim()));
                }
                
                // 如果单个部分超过chunkSize，递归分割
                if (part.length() > chunkSize) {
                    chunks.addAll(splitRecursive(part, separatorIndex + 1));
                    currentChunk = new StringBuilder();
                } else {
                    currentChunk = new StringBuilder(part);
                }
            } else {
                if (currentChunk.length() > 0) {
                    currentChunk.append(separator);
                }
                currentChunk.append(part);
            }
        }
        
        // 保存最后一个块
        if (currentChunk.length() > 0) {
            chunks.add(new Chunk(currentChunk.toString().trim()));
        }
        
        return mergeSmallChunks(chunks);
    }
    
    /**
     * 添加上下文重叠
     */
    private List<Chunk> addOverlap(List<Chunk> chunks) {
        if (chunks.size() <= 1 || chunkOverlap <= 0) {
            return chunks;
        }
        
        List<Chunk> result = new ArrayList<>();
        
        for (int i = 0; i < chunks.size(); i++) {
            StringBuilder content = new StringBuilder();
            
            // 添加上一个块的结尾作为上下文
            if (i > 0) {
                String prevEnd = getChunkEnd(chunks.get(i - 1), chunkOverlap);
                content.append(prevEnd).append(" ");
            }
            
            content.append(chunks.get(i).getText());
            
            result.add(new Chunk(content.toString()));
        }
        
        return result;
    }
}
```

## 向量化

### 嵌入模型选择

```
嵌入模型对比：

模型                  维度    语言      特点
--------------------------------------------------
text-embedding-ada-002  1536   多语言    OpenAI，效果好，付费
all-MiniLM-L6-v2        384    英文      轻量，本地运行
multilingual-e5-large   1024   多语言    开源，中文好
bge-large-zh            1024   中文      中文优化
nomic-embed-text        768    多语言    平衡选择
```

### 批量向量化

```java
package com.example.rag.embedding;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;

import java.util.List;
import java.util.concurrent.*;

/**
 * 批量嵌入生成器
 */
public class BatchEmbeddingGenerator {
    
    private final EmbeddingModel model;
    private final ExecutorService executor;
    private final int batchSize;
    
    public BatchEmbeddingGenerator(EmbeddingModel model, int batchSize) {
        this.model = model;
        this.batchSize = batchSize;
        this.executor = Executors.newFixedThreadPool(4);
    }
    
    /**
     * 批量生成嵌入
     */
    public List<Embedding> generateBatch(List<String> texts) {
        List<Embedding> embeddings = new ArrayList<>();
        
        // 分批处理
        for (int i = 0; i < texts.size(); i += batchSize) {
            List<String> batch = texts.subList(i, 
                Math.min(i + batchSize, texts.size()));
            
            // 并行处理批次
            List<Future<Embedding>> futures = batch.stream()
                .map(text -> executor.submit(() -> model.embed(text)))
                .collect(Collectors.toList());
            
            // 收集结果
            for (Future<Embedding> future : futures) {
                try {
                    embeddings.add(future.get());
                } catch (Exception e) {
                    // 处理失败，使用零向量或重试
                    embeddings.add(Embedding.zeros(model.dimension()));
                }
            }
        }
        
        return embeddings;
    }
}
```

## 存储到向量数据库

```java
package com.example.rag.storage;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

/**
 * 向量存储服务
 */
public class VectorStorageService {
    
    private final EmbeddingStore<TextSegment> embeddingStore;
    
    /**
     * 存储文档块
     */
    public void storeChunks(List<Chunk> chunks, 
                           List<Embedding> embeddings) {
        for (int i = 0; i < chunks.size(); i++) {
            Chunk chunk = chunks.get(i);
            Embedding embedding = embeddings.get(i);
            
            // 创建文本段
            TextSegment segment = TextSegment.from(chunk.getText());
            
            // 添加元数据
            segment.metadata().put("source", chunk.getSource());
            segment.metadata().put("chunk_index", i);
            segment.metadata().put("total_chunks", chunks.size());
            
            // 存储
            embeddingStore.add(embedding, segment);
        }
    }
    
    /**
     * 批量存储（更高效）
     */
    public void storeBatch(List<Chunk> chunks, 
                          List<Embedding> embeddings) {
        List<TextSegment> segments = new ArrayList<>();
        
        for (int i = 0; i < chunks.size(); i++) {
            TextSegment segment = TextSegment.from(chunks.get(i).getText());
            segment.metadata().put("chunk_index", i);
            segments.add(segment);
        }
        
        embeddingStore.addAll(embeddings, segments);
    }
}
```

## 小结

本章我们学习了：

1. **文档解析**：PDF、Word、Markdown等格式
2. **文档清洗**：去除噪声，规范化
3. **文档分割**：递归分割策略
4. **向量化**：嵌入模型选择，批量处理
5. **向量存储**：存储到Chroma

**关键认识：**
文档处理是RAG的基础，处理质量直接影响检索和生成效果。

**下一步：** 我们将实现RAG检索。

---

**练习题：**

1. 如何处理扫描版PDF（图片）？
2. 文档分割的chunk size如何选择？
3. 设计一个支持增量更新的文档处理流程。
