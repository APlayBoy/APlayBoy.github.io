# 多模态|常见基底模型(Foundation Models)介绍和汇总

在这里，我为大家精心整理了最近20余篇关于多模态基底模型领域的研究工作，并附上了各篇论文的基础信息和简介内容。如果您对多模态基底模型领域感兴趣或有相关需求，这里将是一个便捷的导航资源，助您快速了解和深入这一领域的最新进展。



| **模型简称** | **发布机构** | **发布时间** | **是否开源** |
|--------------|--------------|--------------|--------------|
| Gemini 1.5 | Gemini Team, Google | 2024-02-15 | 否 |
| Gemini | Gemini Team, Google | 2023-12-06 | 否 |
| Fuyu-8B | adept.ai | 2023-10-17 | 是 |
| PaLI-3 | Google | 2023-10-13 | 否 |
| GPT-4V | OpenAI | 2023-09-25 | 否 |
| Survey | Microsoft | 2023-09-18 | 否 |
| VPGTrans | 新加坡国立大学计算机学院、清华大学 | 2023-05-02 | 是 |
| Kosmos-2 | Microsoft | 2023-06-26 | 是 |
| Emu | 北京市人工智能研究院、清华、北大 | 2023-07-11 | 是 |
| BLIText | 美国达特茅斯学院、西北大学 | 2023-07-13 | 是 |
| UnIVAL | 巴黎大学 | 2023-07-30 | 是 |
| PaLM-E | Robotics at Google、TU Berlin、Google Research | 2023-03-06 | 否 |
| Prismer | 伦敦帝国理工学院、NVIDIA、威斯康星大学麦迪逊分校、加州理工学院 | 2023-03-04 | 是 |
| GPT-4 | OpenAI | 2023-03-15 | 否 |
| BLIP-2 | Salesforce Research | 2023-01-30 | 是 |
| Kosmos-1 | Microsoft | 2023-02-27 | 是 |
| VIMA | 斯坦福大学、麦卡利斯特学院、nvidia、加州理工学院、清华 | 2022-10-06 | 是 |
| BEiT-3 | Microsoft | 2022-08-22 | 否 |
| MineDojo | NVIDIA MINEDOJO | 2022-06-17 | 是 |
| DaVinci | 港科大、字节AI-LAB、上海交大 | 2022-06-15 | 是 |
| MetaLM | Microsoft Research | 2022-06-13 | 是 |


# Gemini 1.5
* **标题**：Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context
* **发布机构**：Gemini Team, Google
* **发布时间**：2024年2月15日
* **论文链接**：[eepmind-media/gemini](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
* **论文简介**：
Google的Gemini团队在最新报告中介绍了Gemini家族的最新模型——Gemini 1.5 Pro。这是一种高效的多模态混合专家模型，能够回忆和推理数百万标记的上下文中的细节信息，包括多个长文档以及数小时的视频和音频。Gemini 1.5 Pro在跨模态的长上下文检索任务中实现了近乎完美的回忆能力，提升了长文档问答（QA）、长视频问答和长上下文自动语音识别（ASR）的技术水平，并在广泛的基准测试中达到或超越了Gemini 1.0 Ultra的最新性能。研究Gemini 1.5 Pro的长上下文能力极限时，发现其在下一个标记预测和超过99%的检索准确率方面持续改进，至少可处理高达1000万标记，这是对现有模型如Claude 2.1（200k）和GPT-4 Turbo（128k）的一次重大飞跃。最后，报告突出了大型语言模型在前沿领域的惊人新能力；当给定Kalamang语法手册（一种全球不到200名使用者的语言）时，模型学会了将英语翻译成Kalamang，其水平类似于从同样内容学习的人。

# Gemini
* **标题**：Gemini: A Family of Highly Capable Multimodal Models
* **发布机构**：Gemini Team, Google
* **发布时间**：2023年12月6日
* **论文链接**：[eepmind-media/gemini](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
* **论文简介**：
Google的Gemini团队最近推出了一系列新的多模态模型，名为Gemini，这些模型在图像、音频、视频和文本理解方面展现出卓越的能力。Gemini系列包括Ultra、Pro和Nano三种规模，适用于从复杂推理任务到内存受限的设备上的应用场景。在广泛的基准测试中，Gemini系列中最强大的模型——Gemini Ultra——在32个基准测试中的30个中推进了技术的最新水平。特别值得注意的是，它是第一个在广泛研究的考试基准MMLU上达到人类专家水平的模型，并在我们检查的所有20个多模态基准测试中改进了技术的最新水平。Gemini系列在跨模态推理和语言理解方面的新能力，预计将启用广泛的应用场景。报告还讨论了Gemini模型在训练后的部署和负责任地向用户提供服务的方法，包括通过Gemini、Gemini Advanced、Google AI Studio和Cloud Vertex AI等服务。

# Fuyu-8B
* **标题**：Fuyu-8B: A Multimodal Architecture for AI Agents
* **发布机构**：adept.ai
* **发布时间**：2023年10月17日
* **论文链接**：[www.adept.ai](https://www.adept.ai/blog/fuyu-8b)
* **论文代码**：[huggingface.co/adept](https://huggingface.co/adept/fuyu-8b)
* **论文简介**：
adept.ai最近发布了Fuyu-8B，这是其产品中使用的多模态模型的一个小型版本。Fuyu-8B引人注目的原因有几个：
1. 简化的架构和训练流程：与其他多模态模型相比，Fuyu-8B具有更简单的架构和训练过程，这使得它更易于理解、扩展和部署。
2. 专为数字代理设计：Fuyu-8B从一开始就为数字代理设计，因此能够支持任意图像分辨率，回答关于图表和图解的问题，解答基于用户界面的问题，并在屏幕图像上进行精细定位。
3. 快速响应：Fuyu-8B能够在不到100毫秒内对大图像做出响应。
4. 标准基准测试表现良好：尽管Fuyu-8B是为特定用例优化的，但它在标准图像理解基准测试（如视觉问答和自然图像字幕）上表现良好。


# PaLI-3
* **标题**：PaLI-3 Vision Language Models: Smaller, Faster, Stronger
* **发布机构**：Google
* **发布时间**：2023年10月13日
* **论文链接**：[2310.09199](https://arxiv.org/abs/2310.09199)
* **论文简介**：
这篇论文介绍了PaLI-3，一种体积更小、速度更快、性能更强的视觉语言模型（VLM），与体积是其10倍的类似模型相比，表现出色。在实现这一卓越性能的过程中，研究团队比较了使用分类目标预训练的视觉变换器（ViT）模型和对比性（SigLIP）预训练的模型。他们发现，尽管在标准图像分类基准上略逊一筹，但基于SigLIP的PaLI在各种多模态基准上表现出色，特别是在定位和视觉情境下的文本理解方面。研究团队将SigLIP图像编码器扩展到20亿参数，实现了多语言跨模态检索的新最佳水平。PaLI-3仅有50亿参数，研究团队希望它能重新点燃对复杂VLM基础部分的研究，并可能推动新一代扩展模型的发展。

# GPT-4V
* **标题**：GPT-4V(ision) System Card
* **发布机构**：OpenAI
* **发布时间**：2023年9月25日
* **论文链接**：[cdn.openai.com](https://cdn.openai.com/papers/GPTV_System_Card.pdf)
* **论文简介**：
GPT-4V是OpenAI推出的GPT-4的视觉扩展版本，它结合了图像输入的处理能力。这个模型代表了将额外模态整合到大型语言模型中的重要进展，扩展了传统语言系统的功能。GPT-4V不仅能够处理文本，还能分析用户提供的图像，从而解决新的任务并提供独特的用户体验。GPT-4V的训练过程与GPT-4相同，使用了大量的文本和图像数据，并通过人类反馈的强化学习（RLHF）进行了微调。这个模型在文本和视觉两种模态上都表现出色，并展现了由这些模态交叉所产生的新颖能力。OpenAI在准备GPT-4V的部署过程中进行了全面的安全评估和缓解措施。这包括对模型早期访问期间的用户体验和安全学习的分析，以及对模型部署适用性的多模态评估。


# Survey：From Specialists to General-Purpose Assistants
* **标题**：Multimodal Foundation Models: From Specialists to General-Purpose Assistants
* **发布机构**：Microsoft
* **发布时间**：2023年9月18日
* **论文链接**：[2309.10020](https://arxiv.org/abs/2309.10020)
* **论文简介**：
这篇论文提供了一个全面的调查，概述了多模态基础模型的分类和演变，重点关注从专家模型到通用助手的转变。研究领域涵盖了五个核心主题，分为两类：(i) 首先是对已建立研究领域的调查，包括多模态基础模型的预训练，涵盖了学习视觉骨干网络以进行视觉理解和文本到图像生成的两个主题；(ii) 然后，介绍了探索性、开放研究领域的最新进展，包括旨在扮演通用助手角色的多模态基础模型，涉及三个主题——受大型语言模型（LLM）启发的统一视觉模型、多模态LLM的端到端训练，以及将多模态工具与LLM结合。这篇论文的目标受众是计算机视觉和视觉-语言多模态社区的研究人员、研究生和专业人士，他们渴望了解多模态基础模型的基础知识和最新进展。


# UnIVAL
* **标题**： UnIVAL: Unified Model for Image, Video, Audio and Language Tasks
* **发布机构**：巴黎大学
* **发布时间**：2023年7月30日
* **论文链接**：[2310.09199](https://arxiv.org/abs/2310.09199)
* **论文代码**：[github.com/mshukor](https://github.com/mshukor/UnIVAL)
* **论文简介**：
这篇论文介绍了UnIVAL，一种统一模型，用于处理图像、视频、音频和语言任务。UnIVAL模型拥有约0.25B参数，超越了传统的两种模态限制，将文本、图像、视频和音频融合到一个单一模型中。该模型基于任务平衡和多模态课程学习有效地进行了预训练。UnIVAL在多种图像和视频-文本任务上展示了与现有最先进方法相媲美的性能。值得注意的是，尽管在音频上没有进行预训练，UnIVAL在音频-文本任务上的微调性能也表现出竞争力。此外，论文提出了一项新颖的研究，通过模型权重插值合并在不同多模态任务上训练的模型，特别是在分布外泛化方面显示出其优势。最后，论文展示了任务之间的协同作用，进一步证明了统一模型的价值。

# BLIText
* **标题**：Bootstrapping Vision-Language Learning with Decoupled Language Pre-training
* **发布机构**：美国达特茅斯学院、西北大学
* **发布时间**：2023年7月13日
* **论文链接**：[2307.07063](https://arxiv.org/abs/2307.07063)
* **论文代码**：[github.com/yiren-jian](https://github.com/yiren-jian/BLIText)
* **论文简介**：
这篇论文提出了一种新的方法，旨在优化冻结大型语言模型（LLM）在资源密集型视觉-语言（VL）预训练中的应用。当前的范式侧重于使用视觉特征作为提示来引导语言模型，确定与相应文本最相关的视觉特征。本研究的方法则专注于语言部分，特别是确定与视觉特征对齐的最佳提示。研究团队引入了Prompt-Transformer（P-Former），这是一个预测这些理想提示的模型，仅在语言数据上进行训练，绕过了对图像-文本配对的需求。这种策略巧妙地将端到端的VL训练过程分解为一个额外的、独立的阶段。实验表明，该框架显著提高了一个强大的图像到文本基线（BLIP-2）的性能，并有效地缩小了使用4M或129M图像-文本对训练的模型之间的性能差距。重要的是，该框架在模态和架构设计方面具有通用性和灵活性，这一点通过其在使用不同基础模块的视频学习任务中的成功应用得到了验证。


# Emu
* **标题**：Generative Pretraining in Multimodality
* **发布机构**：北京市人工智能研究院、清华、北大
* **发布时间**：2023年7月11日
* **论文链接**：[2307.05222](https://arxiv.org/abs/2307.05222)
* **论文代码**：[github.com/baaivision](https://github.com/baaivision/Emu/)
* **论文简介**：
这篇论文介绍了Emu，一种基于Transformer的多模态基础模型，能够在多模态上下文中无缝生成图像和文本。这个全能模型可以通过一个一体化的自回归训练过程，不加区分地接收任何单模态或多模态数据输入（例如，交错的图像、文本和视频）。首先，将视觉信号编码为嵌入，与文本标记一起形成交错的输入序列。然后，Emu通过统一的目标进行端到端训练，即在多模态序列中对下一个文本标记进行分类或对下一个视觉嵌入进行回归。这种多功能的多模态性赋予了在大规模上探索多样化的预训练数据源的能力，例如带有交错帧和文本的视频、带有交错图像和文本的网页，以及大规模的图像-文本对和视频-文本对。Emu可以作为图像到文本和文本到图像任务的通用多模态接口，并支持在上下文中生成图像和文本。在包括图像字幕、视觉问答、视频问答和文本到图像生成在内的一系列零样本/少样本任务上，Emu与最先进的大型多模态模型相比表现出色。此外，还通过指令调整展示了诸如多模态助手等扩展功能，并取得了令人印象深刻的性能。

# Kosmos-2
* **标题**：Kosmos-2: Grounding Multimodal Large Language Models to the World
* **发布机构**：Microsoft
* **发布时间**：2023年6月26日
* **论文链接**：[2306.14824](https://arxiv.org/abs/2306.14824)
* **论文代码**：[github.com/microsoft](https://github.com/microsoft/unilm/tree/master/kosmos-2)
* **论文简介**：
Kosmos-2是一种多模态大型语言模型（MLLM），具有感知对象描述（例如，边界框）和将文本与视觉世界联系起来的新能力。具体来说，Kosmos-2将引用表达式表示为Markdown中的链接，即“文本跨度”，其中对象描述是位置标记的序列。结合多模态语料库，研究团队构建了大规模的地面图像-文本对数据（称为GrIT）来训练模型。除了现有MLLM的能力（例如，感知一般模态、遵循指令、进行上下文学习）之外，Kosmos-2还将地面能力整合到下游应用中。Kosmos-2在广泛的任务上进行了评估，包括
  - （i）多模态地面，如引用表达式理解和短语地面
  - （ii）多模态引用，如引用表达式生成
  - （iii）感知-语言任务，以及（iv）语言理解和生成。

  这项工作为体现人工智能的发展奠定了基础，并为语言、多模态感知、行动和世界建模的大融合提供了启示，这是通往人工通用智能的关键一步。代码和预训练模型可在https://aka.ms/kosmos-2获取。

# VPGTrans
* **标题**：VPGTrans: Transfer Visual Prompt Generator across LLMs
* **发布机构**：新加坡国立大学计算机学院、清华大学
* **发布时间**：2023年5月2日
* **论文链接**：[2305.01278](https://arxiv.org/abs/2305.01278)
* **论文代码**：[VPGTrans](https://github.com/VPGTrans/VPGTrans)
* **论文简介**：
这篇论文提出了VPGTrans，一种跨大型语言模型（LLM）传递视觉提示生成器（VPG）的方法。由于从头开始开发新的多模态LLM（MLLM）需要大量资源，将现有的LLM与相对轻量级的VPG结合成为了一种可行的范式。然而，进一步调整MLLM中的VPG部分仍然需要大量的计算成本。VPGTrans首次探索了跨LLM的VPG可转移性，并提出了一种降低VPG转移成本的解决方案。研究团队首先研究了跨不同LLM大小（例如从小到大）和不同LLM类型的VPG转移，诊断了最大化转移效率的关键因素。基于这些观察，他们设计了一个名为VPGTrans的两阶段转移框架，简单但高效。通过广泛的实验，研究团队证明了VPGTrans能够显著加速转移学习过程，同时不影响性能。值得注意的是，它帮助实现了从 $\text{BLIP-2 OPT}_\text{2.7B}$ 到$\text{BLIP-2 OPT}_\text{6.7B}$ 的VPG转移，与从头开始将VPG连接到$\text{OPT}_\text{6.7B}$ 相比，速度提高了10倍以上，训练数据减少了10.7%。此外，论文还提供了一系列有趣的发现和潜在的理论基础，并进行了讨论。最后，论文展示了VPGTrans方法的实际价值，通过最近发布的LLaMA和Vicuna LLMs定制了两个新的MLLM，包括VL-LLaMA和VL-Vicuna。
这篇论文提出了VPGTrans，一种跨大型语言模型（LLM）传递视觉提示生成器（VPG）的方法。由于从头开始开发新的多模态LLM（MLLM）需要大量资源，将现有的LLM与相对轻量级的VPG结合成为了一种可行的范式。然而，进一步调整MLLM中的VPG部分仍然需要大量的计算成本。VPGTrans首次探索了跨LLM的VPG可转移性，并提出了一种降低VPG转移成本的解决方案。研究团队首先研究了跨不同LLM大小（例如从小到大）和不同LLM类型的VPG转移，诊断了最大化转移效率的关键因素。基于这些观察，他们设计了一个名为VPGTrans的两阶段转移框架，简单但高效。通过广泛的实验，研究团队证明了VPGTrans能够显著加速转移学习过程，同时不影响性能。值得注意的是，它帮助实现了从$\text{BLIP-2 OPT}_\text{2.7B}$ 到 $\text{BLIP-2 OPT}_\text{6.7B}$ 的VPG转移，与从头开始将VPG连接到 $\text{OPT}_\text{6.7B}$ 相比，速度提高了10倍以上，训练数据减少了10.7%。此外，论文还提供了一系列有趣的发现和潜在的理论基础，并进行了讨论。最后，论文展示了VPGTrans方法的实际价值，通过最近发布的LLaMA和Vicuna LLMs定制了两个新的MLLM，包括VL-LLaMA和VL-Vicuna。

# GPT-4
* **标题**：GPT-4 Technical Report
* **发布机构**：OpenAI
* **发布时间**：2023年3月15日
* **论文链接**：[2303.08774](https://arxiv.org/abs/2303.08774)
* **论文简介**：
这篇技术报告详细介绍了GPT-4，一种大规模的多模态模型，能够接受图像和文本输入并产生文本输出。虽然在许多现实世界场景中GPT-4的能力仍不及人类，但它在各种专业和学术基准测试中展现出接近人类水平的性能，包括在模拟的律师资格考试中获得前10%的成绩。GPT-4是基于Transformer的模型，预训练目标是预测文档中的下一个词。通过训练后的对齐过程，GPT-4在事实性和遵循期望行为的度量上表现得更好。该项目的核心组成部分之一是开发基础设施和优化方法，这些方法在广泛的规模范围内表现出可预测的行为。这使得研究团队能够准确预测GPT-4的某些性能方面，基于的模型训练所需的计算量不超过GPT-4的1/1000。

# PaLM-E
* **标题**：PaLM-E: An Embodied Multimodal Language Model
* **发布机构**：Robotics at Google、TU Berlin、Google Research
* **发布时间**：2023年3月6日
* **论文链接**：[2303.03378](https://arxiv.org/abs/2303.03378)
* **论文简介**：
PaLM-E是一种新型的多模态语言模型，专为解决现实世界中的一般推理问题而设计，例如机器人学问题中的语义接地挑战。PaLM-E通过直接将真实世界的连续传感器模态整合到语言模型中，建立了词汇和感知之间的联系。该模型的输入是多模态句子，它们将视觉、连续状态估计和文本输入编码交织在一起。研究团队对这些编码进行了端到端训练，与预训练的大型语言模型一起，用于多个体现任务，包括顺序机器人操纵规划、视觉问答和字幕。评估结果表明，PaLM-E这个单一的大型多模态模型能够处理多种体现推理任务，从多种观察模态、多种体现中获得，并且表现出积极的迁移：模型从跨互联网规模的语言、视觉和视觉-语言领域的多样化联合训练中受益。PaLM-E-562B是最大的模型，拥有562B参数，除了在机器人任务上进行训练外，还是视觉-语言领域的通才，在OK-VQA上表现出最先进的性能，并且随着规模的增加保持了通才语言能力。


# Prismer
* **标题**：Prismer: A Vision-Language Model with Multi-Task Experts
* **发布机构**：伦敦帝国理工学院、NVIDIA、威斯康星大学麦迪逊分校、加州理工学院
* **发布时间**：2023年3月4日
* **论文链接**：[2303.02506](https://arxiv.org/abs/2303.02506)
* **论文代码**：[github.com/NVlabs](https://github.com/NVlabs/prismer)
* 论文模型：[huggingface.co/spaces](https://huggingface.co/spaces/shikunl/prismer)
* **论文简介**：
Prismer是一种数据和参数高效的视觉-语言模型，它利用一系列特定任务的专家模型。与传统的大型模型和大规模数据集训练相比，Prismer的方法更具可扩展性。它只需要训练少量的组件，大部分网络权重继承自多个现成的、预训练的专家模型，并在训练期间保持冻结。Prismer通过利用来自广泛领域的专家知识，有效地整合这些知识并将其适应于各种视觉-语言推理任务。实验表明，Prismer在微调和少样本学习性能方面与当前最先进的模型相当，同时在训练数据方面减少了高达两个数量级的需求。论文提供了代码和详细的实验分析。


# Kosmos-1
* **标题**：Language Is Not All You Need: Aligning Perception with Language Models
* **发布机构**：Microsoft
* **发布时间**：2023年2月27日
* **论文链接**：[2302.14045](https://arxiv.org/abs/2302.14045)
* **论文代码**：[github.com/microsoft](https://github.com/microsoft/unilm)
* **论文简介**：这篇论文介绍了Kosmos-1，一种多模态大型语言模型（MLLM），能够感知通用模态、在上下文中学习（即少样本学习），并遵循指令（即零样本学习）。Kosmos-1从头开始在网络规模的多模态语料库上进行训练，包括任意交错的文本和图像、图像-标题对和文本数据。研究团队在多种设置下评估了Kosmos-1，包括零样本、少样本和多模态思维链提示，在没有任何梯度更新或微调的情况下，对广泛的任务进行了评估。实验结果显示，Kosmos-1在
    - （i）语言理解、生成，甚至是OCR-free NLP（直接使用文档图像）
    - （ii）感知-语言任务，包括多模态对话、图像描述、视觉问答
    - （iii）视觉任务，如带描述的图像识别（通过文本指令指定分类）

  方面取得了令人印象深刻的表现。此外，研究还表明，MLLM可以从跨模态转移中受益，即从语言到多模态的知识转移，以及从多模态到语言的转移。研究还引入了Raven IQ测试数据集，用于诊断MLLM的非语言推理能力。


# BLIP-2
* **标题**：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
* **发布机构**：Salesforce Research
* **发布时间**：2023年1月30日
* **论文链接**：[2301.12597](https://arxiv.org/abs/2301.12597)
* **论文代码**：[github.com/salesforce](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
* **论文简介**：
这篇论文提出了BLIP-2，一种新的视觉-语言预训练策略，它通过使用现成的冻结图像编码器和大型语言模型来引导视觉-语言预训练。BLIP-2通过一个轻量级的查询变换器来弥合模态差距，该变换器分两个阶段进行预训练。第一阶段从冻结的图像编码器引导视觉-语言表示学习，第二阶段从冻结的语言模型引导视觉到语言的生成学习。尽管BLIP-2的可训练参数远少于现有方法，但它在各种视觉-语言任务上实现了最先进的性能。例如，该模型在零样本VQAv2上的表现比Flamingo80B高出8.7%，同时可训练参数减少了54倍。论文还展示了该模型的零样本图像到文本生成的新兴能力，能够遵循自然语言指令。


# VIMA
* **标题**：VIMA: General Robot Manipulation with Multimodal Prompts
* **发布机构**：斯坦福大学、麦卡利斯特学院、nvidia、加州理工学院、清华
* **发布时间**：2022年10月6日
* **论文链接**：[2210.03094](https://arxiv.org/abs/2210.03094)
* **论文代码**：[github.com/scxue](https://github.com/scxue/DM-NonUniform)
* **论文简介**：
这篇论文提出了VIMA，一种新型的机器人操作模型，它使用多模态提示来执行各种任务。VIMA的核心是将语言模型作为通用接口，连接多种基础模型，以处理不同模态（如视觉和语言）的输入。研究团队开发了一个新的仿真基准，包含数千个程序生成的桌面任务和多模态提示，以及超过60万条专家轨迹用于模仿学习。VIMA采用基于Transformer的架构，处理这些提示并自回归地输出电机动作。VIMA在模型可扩展性和数据效率方面表现出色，在最困难的零样本泛化设置中，其任务成功率比其他设计高出2.9倍。即使训练数据减少10倍，VIMA的性能仍比最佳竞争变体高出2.7倍。论文提供了代码和视频演示。


# MineDojo
* **标题**：MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge
* **发布机构**：NVIDIA MINEDOJO
* **发布时间**：2022年6月17日
* **论文链接**：[2206.08853](https://arxiv.org/abs/2206.08853)
* **论文代码**：[github.com/minedojo](https://github.com/minedojo/minedojo)
* 模型下载：[github.com/MineDojo](https://github.com/MineDojo/MineCLIP)
* **论文简介**：
这篇论文提出了MineDojo，一个基于流行的Minecraft游戏构建的新框架，旨在创建开放式的自主代理，具备互联网规模的知识。MineDojo包括一个模拟套件，支持数千种多样化的开放式任务，以及一个包含Minecraft视频、教程、维基页面和论坛讨论的互联网规模知识库。研究团队提出了一种新颖的代理学习算法，利用大型预训练的视频-语言模型作为学习到的奖励函数。这种代理能够解决用自由形式语言指定的各种开放式任务，而无需手动设计的密集型塑形奖励。MineDojo的模拟套件、知识库、算法实现和预训练模型已开源（https://minedojo.org），以促进通用能力自主代理的研究。

# DaVinci
* **标题**：Write and Paint: Generative Vision-Language Models are Unified Modal Learners
* **发布机构**：港科大、字节AI-LAB、上海交大
* **发布时间**：2022年6月15日
* **论文链接**：[2206.07699](https://arxiv.org/abs/2206.07699)
* **论文代码**：[github.com/shizhediao](https://github.com/shizhediao/DaVinci)
* **论文简介**：
这篇论文介绍了DaVinci，一种新型的生成式视觉-语言模型，它作为统一的模态学习者，能够同时进行图像到文本的生成（写作）和文本到图像的生成（绘画）。DaVinci通过前缀语言建模和前缀图像建模——一种简单的自监督目标——在图像-文本对上进行训练。这种前缀多模态建模框架使得DaVinci易于训练、可扩展到大量数据，并且适用于写作和绘画任务，同时在其他视觉、文本和多模态理解任务上表现强劲。DaVinci在27个生成/理解任务上取得了竞争性的表现，并展示了结合视觉/语言生成预训练的优势。此外，论文还详细评估了不同规模的预训练数据集上不同视觉-语言预训练目标的性能，展示了在语言和视觉输入中利用自监督的潜力，并为未来在不同数据规模下的比较建立了新的、更强的基准。

# MetaLM
* **标题**：Language Models are General-Purpose Interfaces
* **论文链接**：[2206.06336](https://arxiv.org/abs/2206.06336)
* **论文代码**：[github.com/microsoft](https://github.com/microsoft/unilm)
* **发布时间**：2022-06-13
* **发布机构**：Microsoft Research
* **论文简介**：
这篇论文提出了MetaLM，一种将语言模型作为通用接口的方法，用于连接多种基础模型。MetaLM的核心思想是使用一系列预训练的编码器来感知不同的模态（如视觉和语言），并将它们与作为通用任务层的语言模型相结合。研究团队提出了一种半因果语言建模目标，用于联合预训练接口和模块化编码器。MetaLM结合了因果和非因果建模的优势，融合了两者的最佳特性。具体来说，MetaLM不仅继承了因果语言建模的上下文学习和开放式生成能力，而且由于双向编码器的使用，也有利于微调。更重要的是，该方法无缝地解锁了上述能力的组合，例如，使用微调的编码器进行上下文学习或指令遵循。在各种语言和视觉语言基准测试中的实验结果表明，MetaLM在微调、零样本泛化和少样本学习方面的性能优于或与专门的模型相当。


# BEiT-3
* **标题**：Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks
* **发布机构**：Microsoft
* **发布时间**：2022年8月22日
* **论文链接**：[2208.10442](https://arxiv.org/abs/2208.10442)
* **论文简介**：
这篇论文介绍了BEiT-3，一种通用的多模态基础模型，它在视觉和视觉-语言任务上取得了最先进的迁移性能。BEiT-3推进了语言、视觉和多模态预训练的大融合，从三个方面进行改进：骨干网络架构、预训练任务和模型扩展。研究团队引入了多路变换器（Multiway Transformers）作为通用建模的基础，其模块化架构既支持深度融合又能进行特定模态的编码。基于共享骨干网络，BEiT-3以统一的方式对图像（Imglish）、文本（English）和图像-文本对（“平行句子”）进行遮蔽“语言”建模。实验结果表明，BEiT-3在对象检测（COCO）、语义分割（ADE20K）、图像分类（ImageNet）、视觉推理（NLVR2）、视觉问答（VQAv2）、图像描述（COCO）和跨模态检索（Flickr30K、COCO）等任务上取得了最先进的性能。

