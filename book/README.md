# Java程序员的AI之路

> 从Java视角理解人工智能，用Java生态实践AI开发

## 关于本书

这是一本为Java程序员量身打造的AI学习指南。如果你是一名有Java基础但对AI完全陌生的程序员，这本书将帮助你打开AI世界的大门。

### 为什么写这本书

当前AI生态主要以Python为主，但这并不意味着Java程序员只能袖手旁观。Java拥有成熟的AI生态和强大的工程化能力，完全可以成为AI开发的主力语言。本书旨在帮助Java程序员：

- 用熟悉的思维方式理解AI核心概念
- 掌握深度学习、大语言模型、智能体的原理与实践
- 将AI能力融入Java应用，实现工程化落地

### 目标读者

- 有Java基础，但对AI完全陌生的程序员
- 希望转型AI领域的Java开发者
- 想要在Java项目中集成AI能力的架构师

### 本书特色

- **通俗易懂**：用生活化的比喻解释复杂概念
- **实操为主**：每章都有可运行的Java代码示例
- **深入浅出**：既有原理讲解，又有设计思考
- **Java视角**：用Java程序员的思维方式理解AI

## 目录结构

### 第一部分：深度学习基础

**第1章：AI世界的大门——从Java视角看深度学习**
- [1.1 为什么Java程序员需要学习AI](part1-deep-learning/chapter-01/01-why-java-ai.md)
- [1.2 深度学习是什么：用Java思维理解神经网络](part1-deep-learning/chapter-01/02-what-is-deep-learning.md)
- [1.3 搭建你的第一个AI开发环境](part1-deep-learning/chapter-01/03-first-ai-environment.md)
- [1.4 Java AI生态全景图](part1-deep-learning/chapter-01/04-java-ai-ecosystem.md)

**第2章：神经网络的本质——理解深度学习的核心**
- [2.1 从感知机到神经网络：一种计算思维的演进](part1-deep-learning/chapter-02/01-perceptron-to-neural-network.md)
- [2.2 前向传播：数据流动的艺术](part1-deep-learning/chapter-02/02-forward-propagation.md)
- [2.3 反向传播：学习的数学之美](part1-deep-learning/chapter-02/03-backpropagation.md)
- [2.4 用Deeplearning4j实现你的第一个神经网络](part1-deep-learning/chapter-02/04-first-neural-network-dl4j.md)
- [2.5 设计思考：为什么深度学习需要"深"](part1-deep-learning/chapter-02/05-why-deep-learning-needs-depth.md)

**第3章：卷积神经网络——让机器看见世界**
- [3.1 图像识别问题：从像素到语义](part1-deep-learning/chapter-03/01-image-recognition-problem.md)
- [3.2 卷积操作的本质：特征提取的艺术](part1-deep-learning/chapter-03/02-convolution-operation.md)
- [3.3 池化与全连接：信息压缩的智慧](part1-deep-learning/chapter-03/03-pooling-and-fully-connected.md)
- [3.4 经典CNN架构解析：LeNet、AlexNet、ResNet](part1-deep-learning/chapter-03/04-classic-cnn-architectures.md)
- [3.5 实战：用Java构建图像分类器](part1-deep-learning/chapter-03/05-build-image-classifier.md)
- [3.6 设计思考：局部感知与参数共享](part1-deep-learning/chapter-03/06-local-perception-weight-sharing.md)

**第4章：循环神经网络——理解序列数据**
- [4.1 序列数据的挑战：时间维度的引入](part1-deep-learning/chapter-04/01-sequence-data-challenge.md)
- [4.2 RNN的核心思想：记忆与遗忘](part1-deep-learning/chapter-04/02-rnn-memory-and-forgetting.md)
- [4.3 LSTM与GRU：解决长期依赖的钥匙](part1-deep-learning/chapter-04/03-lstm-and-gru.md)
- [4.4 实战：用Java实现文本生成](part1-deep-learning/chapter-04/04-text-generation-practice.md)
- [4.5 设计思考：时序建模的本质](part1-deep-learning/chapter-04/05-design-thinking-sequential-modeling.md)

**第5章：实战项目——手写数字识别系统**
- [5.1 项目概述与需求分析](part1-deep-learning/chapter-05/01-project-overview.md)
- [5.2 数据准备与预处理](part1-deep-learning/chapter-05/02-data-preparation.md)
- [5.3 模型设计与训练](part1-deep-learning/chapter-05/03-model-design-training.md)
- [5.4 模型评估与优化](part1-deep-learning/chapter-05/04-model-evaluation-optimization.md)
- [5.5 部署与集成：将AI融入Java应用](part1-deep-learning/chapter-05/05-deployment-integration.md)

### 第二部分：大语言模型

**第6章：语言模型的演进——从统计到神经网络**
- [6.1 语言模型是什么：让机器理解语言](part2-llm/chapter-06/01-what-is-language-model.md)
- [6.2 从N-gram到Word2Vec：词向量的诞生](part2-llm/chapter-06/02-ngram-to-word2vec.md)
- [6.3 Transformer架构：注意力机制的革命](part2-llm/chapter-06/03-transformer-architecture.md)
- [6.4 设计思考：为什么Transformer改变了NLP](part2-llm/chapter-06/04-why-transformer-changed-nlp.md)

**第7章：深入Transformer——理解LLM的核心**
- [7.1 自注意力机制：让词与词对话](part2-llm/chapter-07/01-self-attention.md)
- [7.2 多头注意力：并行捕捉多种关系](part2-llm/chapter-07/02-multi-head-attention.md)
- [7.3 位置编码：给序列注入顺序信息](part2-llm/chapter-07/03-positional-encoding.md)
- [7.4 编码器-解码器架构](part2-llm/chapter-07/04-encoder-decoder-architecture.md)
- [7.5 用Java实现简化版Transformer](part2-llm/chapter-07/05-implement-transformer-java.md)

**第8章：GPT与BERT——两大流派解析**
- [8.1 GPT系列：生成式预训练的崛起](part2-llm/chapter-08/01-gpt-series.md)
- [8.2 BERT：双向理解的智慧](part2-llm/chapter-08/02-bert-model.md)
- [8.3 预训练与微调：迁移学习在NLP的应用](part2-llm/chapter-08/03-pretraining-finetuning.md)
- [8.4 实战：用Java调用OpenAI API](part2-llm/chapter-08/04-openai-api-practice.md)
- [8.5 设计思考：生成与理解的两条路径](part2-llm/chapter-08/05-generation-vs-understanding.md)

**第9章：开源大模型——本地部署与实践**
- [9.1 开源LLM生态：LLaMA、Mistral、Qwen](part2-llm/chapter-09/01-open-source-llm-ecosystem.md)
- [9.2 模型量化：让大模型跑在普通机器上](part2-llm/chapter-09/02-model-quantization.md)
- [9.3 LangChain4j：Java的LLM开发框架](part2-llm/chapter-09/03-langchain4j-framework.md)
- [9.4 实战：构建本地问答系统](part2-llm/chapter-09/04-build-local-qa-system.md)
- [9.5 设计思考：云端API与本地部署的权衡](part2-llm/chapter-09/05-cloud-vs-local-deployment.md)

**第10章：提示工程——与AI有效沟通的艺术**
- [10.1 提示工程基础：输入决定输出](part2-llm/chapter-10/01-prompt-engineering-basics.md)
- [10.2 提示模式：少样本、思维链、角色扮演](part2-llm/chapter-10/02-prompt-patterns.md)
- [10.3 结构化输出：让LLM返回Java对象](part2-llm/chapter-10/03-structured-output.md)
- [10.4 实战：设计高质量提示模板](part2-llm/chapter-10/04-design-prompt-templates.md)
- [10.5 设计思考：提示即编程](part2-llm/chapter-10/05-prompt-as-programming.md)

**第11章：实战项目——智能文档助手**
- [11.1 项目背景与架构设计](part2-llm/chapter-11/01-project-background-architecture.md)
- [11.2 文档解析与向量化](part2-llm/chapter-11/02-document-parsing-vectorization.md)
- [11.3 RAG：检索增强生成](part2-llm/chapter-11/03-rag-implementation.md)
- [11.4 对话系统实现](part2-llm/chapter-11/04-conversation-system.md)
- [11.5 优化与部署](part2-llm/chapter-11/05-optimization-deployment.md)

### 第三部分：智能体

**第12章：智能体概述——AI的下一个范式**
- [12.1 什么是智能体：从被动响应到主动行动](part3-agents/chapter-12/01-what-is-agent.md)
- [12.2 智能体的核心组件：感知、规划、行动](part3-agents/chapter-12/02-core-components.md)
- [12.3 智能体与LLM的关系](part3-agents/chapter-12/03-agent-llm-relationship.md)
- [12.4 设计思考：从工具使用者到工具创造者](part3-agents/chapter-12/04-tool-user-to-creator.md)

**第13章：工具使用——让AI操作外部世界**
- [13.1 Function Calling：LLM调用函数的能力](part3-agents/chapter-13/01-function-calling.md)
- [13.2 工具定义与注册](part3-agents/chapter-13/02-tool-definition-registration.md)
- [13.3 用Java构建自定义工具](part3-agents/chapter-13/03-build-custom-tools.md)
- [13.4 实战：让LLM操作数据库](part3-agents/chapter-13/04-database-tools.md)
- [13.5 设计思考：安全性与可控性](part3-agents/chapter-13/05-security-controllability.md)

**第14章：规划与推理——智能体的决策能力**
- [14.1 任务分解：复杂问题的拆解艺术](part3-agents/chapter-14/01-task-decomposition.md)
- [14.2 ReAct框架：推理与行动的循环](part3-agents/chapter-14/02-react-framework.md)
- [14.3 思维链与思维树](part3-agents/chapter-14/03-chain-tree-of-thought.md)
- [14.4 实战：构建会规划的任务Agent](part3-agents/chapter-14/04-build-planning-agent.md)
- [14.5 设计思考：推理的边界与可能](part3-agents/chapter-14/05-reasoning-boundaries.md)

**第15章：记忆系统——让智能体记住一切**
- [15.1 短期记忆与长期记忆](part3-agents/chapter-15/01-short-long-term-memory.md)
- [15.2 对话记忆管理](part3-agents/chapter-15/02-dialog-memory-management.md)
- [15.3 向量数据库：AI的记忆仓库](part3-agents/chapter-15/03-vector-database.md)
- [15.4 实战：构建有记忆的对话Agent](part3-agents/chapter-15/04-build-memory-agent.md)
- [15.5 设计思考：遗忘的艺术](part3-agents/chapter-15/05-art-of-forgetting.md)

**第16章：多智能体协作——群体智能的涌现**
- [16.1 多智能体系统概述](part3-agents/chapter-16/01-multi-agent-overview.md)
- [16.2 角色定义与协作模式](part3-agents/chapter-16/02-roles-collaboration.md)
- [16.3 通信协议与消息传递](part3-agents/chapter-16/03-communication-protocol.md)
- [16.4 实战：构建软件开发Agent团队](part3-agents/chapter-16/04-build-dev-team.md)
- [16.5 设计思考：从个体智能到群体智能](part3-agents/chapter-16/05-swarm-intelligence.md)

**第17章：实战项目——个人AI助手**
- [17.1 项目规划与架构设计](part3-agents/chapter-17/01-project-planning.md)
- [17.2 核心能力实现](part3-agents/chapter-17/02-core-abilities.md)
- [17.3 工具集成与扩展](part3-agents/chapter-17/03-tool-integration.md)
- [17.4 用户界面与交互设计](part3-agents/chapter-17/04-user-interface.md)
- [17.5 部署与持续优化](part3-agents/chapter-17/05-deployment-optimization.md)

## 附录

- [工具环境配置指南](appendix/tools-setup.md)
- [术语表](appendix/glossary.md)
- [参考资料](appendix/references.md)

## 如何使用本书

建议按顺序阅读，每部分都有递进关系：
1. **第一部分**建立深度学习的基础认知
2. **第二部分**深入大语言模型的原理与应用
3. **第三部分**掌握智能体的设计与开发

每章的代码示例都可以独立运行，建议边学边练。

## 技术栈

- **Java版本**：Java 17+
- **深度学习框架**：Deeplearning4j
- **LLM框架**：LangChain4j
- **构建工具**：Maven/Gradle
- **向量数据库**：Milvus/Pinecone/Chroma

## 作者寄语

AI时代已经到来，作为Java程序员，我们不需要成为数学家，也不需要放弃我们熟悉的编程语言。只要掌握正确的学习方法，我们同样可以在AI领域大展身手。

希望这本书能成为你AI之路的起点。

---

*本书采用MANNING出版公司风格编写，注重实操与代码驱动学习。*
