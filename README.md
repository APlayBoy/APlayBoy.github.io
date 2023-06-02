# Code

## [deeplearning](https://github.com/APlayBoy/deeplearning) 
* 用[CNN](https://github.com/APlayBoy/deeplearning/tree/master/image_classification)做简单的分类
* 用[GAN](https://github.com/APlayBoy/deeplearning/blob/master/face_generation/face_generation.py)生成人脸

&nbsp;&nbsp;&nbsp;&nbsp;上面是17年我刚开始接触深度学习时留下的简单Code, 虽然很简单，一定程度上正是那简单的几行code，引领这个AI时代。



## [algorithm](https://github.com/APlayBoy/algorithm) 
* [机器学习](https://github.com/APlayBoy/algorithm/tree/master/ml)
* [深度学习](https://github.com/APlayBoy/algorithm/tree/master/dl)

&nbsp;&nbsp;&nbsp;&nbsp;上面是20年时对常见的[机器学习](https://github.com/APlayBoy/algorithm/tree/master/ml)和[深度学习](https://github.com/APlayBoy/algorithm/tree/master/dl)算法的一个总结，每行 code 都是自己理解后敲上去，3年后再看很多逻辑已然不是那么清晰了，但正是这种曾经深刻掌握过的东西，在实际应用中总能适时的拿出来。

## [fast-diffusion](https://github.com/APlayBoy/fast-diffusion)
* [Unet](https://github.com/APlayBoy/fast-diffusion/tree/main/diffusion/UNet)
* [Diffusion](https://github.com/APlayBoy/fast-diffusion/tree/main/diffusion/diffusion)

&nbsp;&nbsp;&nbsp;&nbsp;近期23年，AIGC突然火爆，长时间通过判别式模型做识别的自己感觉要和市场脱轨，所以决定要去学习生成式模型AIGC，因为在开源的code中，code的风格很“算法工程师”，阅读难度比较大，我在这里会对code进行拆解，一方面可以帮助自己深刻的理解不同文献在方法，另一方面也可以给别人提供另一种直接通过code来学习思路。



# 博客
&nbsp;&nbsp;&nbsp;&nbsp;之前看文章时，不注意总结和留笔记，随着时间的流逝，遗忘成为了必然。近期会把之前缺失的笔记补回来一些。

## VAE
* [Variational inference](https://zhuanlan.zhihu.com/p/627342489)
* [From Autoencoder to TD-VAE](https://zhuanlan.zhihu.com/p/623397006)
* [Auto-Encoding Variational Bayes(VAE)](https://zhuanlan.zhihu.com/p/627313458) 

## Diffusion Model

 | 标题  |简称 |简介 | 作者 |
 |:--|:--|:--|:--|
 |[Denoising Diffusion Probabilistic Models](https://zhuanlan.zhihu.com/p/626688571)|DDPM|扩散模型的里程碑|UC Berkeley|
 |[Denoising Diffusion Implicit Models](https://zhuanlan.zhihu.com/p/628378813)|DDIM|加速采样过程|Stanford University|
 |[Improved Denoising Diffusion Probabilistic Models](https://zhuanlan.zhihu.com/p/630677971)|improved diffusion|扩散细节优化|OpenAI|
 |[Diffusion Models Beat GANs on Image Synthesis](https://zhuanlan.zhihu.com/p/631037773)|guided diffusion|扩散模型首次超越GAN|OpenAI|
 |[Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://zhuanlan.zhihu.com/p/631042461)|GLIDE|classfier-free+图片编辑|OpenAI|
 |[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://zhuanlan.zhihu.com/p/631283028)|DALLE-2 or UNCLIP|文本生成图像新高度|OpenAI|
 |[High-Resolution Image Synthesis with Latent Diffusion Models Robin](https://zhuanlan.zhihu.com/p/628681685)|Stable-Diffusion|收个大型开源的扩散模型|Ludwig Maximilian University of Munich & IWR, Heidelberg University|
 |[ControlNet:Adding Conditional Control to Text-to-Image Diffusion Models](https://zhuanlan.zhihu.com/p/633491149)|ControlNet|人人用得起扩散模型|Stanford University|
 |[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation Nataniel](https://zhuanlan.zhihu.com/p/633878755)|DreamBooth|生成图片时可以输入指定图片|Google Research & Boston University|

## GPT
 * [GPT系列总述](https://zhuanlan.zhihu.com/p/630009840)
 * 

## 多模态

|分类| 标题  |说明|
|:--|:--|:--|
|微调模型|[LoRA](https://zhuanlan.zhihu.com/p/633204266)|用很小的成本驾驭大模型|
|CLIP原文| [CLIP](https://zhuanlan.zhihu.com/p/633205841)|图片和文本之间的对比学习
|CLIP应用|[LSeg](https://zhuanlan.zhihu.com/p/633073728)|有监督的开集分割|
|总结|[预训练模型总结](https://zhuanlan.zhihu.com/p/633946545)|多模态预训练模型总结|
|多模态预训练|[ViLT](https://zhuanlan.zhihu.com/p/633908947)|把目标检测从视觉端拿掉，用transformer结构，增加模态融合的权重|
|多模态预训练|[ALBeF](https://zhuanlan.zhihu.com/p/633953705)|多模态融合之前对齐模态特征|
|多模态预训练|[VLMo](https://zhuanlan.zhihu.com/p/633984456)|提出混合模态专家Transformer,不同模态权重共享，分阶段训练|
|多模态预训练|[CoCa](https://zhuanlan.zhihu.com/p/634039462)|文本端只用deocde训练，提升训练效率|
|多模态预训练|[BLIP](https://zhuanlan.zhihu.com/p/634025087)|通过decoder生成字幕，字幕器和过滤器引清晰数据，文本docoder、encoderg共享权重|
|多模态预训练|[BeiT V3](https://zhuanlan.zhihu.com/p/634043351)|所有技术的大一统|





## Reinforcement Learning
* [强化学习基础笔记](https://zhuanlan.zhihu.com/p/632103344)

## Unsupervised Learning

| 标题  |说明 |包含文章 | 
|:--|:--|:--|
|[对比学习论文总结](https://zhuanlan.zhihu.com/p/631863971)|重要文章内容简述|-|
|[对比学习之百花齐放](https://zhuanlan.zhihu.com/p/631960375)|技术尝试阶段|InstDisc、InvaSpread、CPC、CMC、DeepCluster|
|[对比学习之CV双雄](https://zhuanlan.zhihu.com/p/631974914)|对比学习方法趋于统一|MoCo v1-2、SimCLR v1-2、SWaV|
|[对比学习之无负样本](https://zhuanlan.zhihu.com/p/632483233)|摆脱了负样本的限制|BYOL、SimSiam、BYOL v2|
|[对比学习之Transformer](https://zhuanlan.zhihu.com/p/632538243)|vit和对比学习组合的落地|MoCov3、DINO|
