# EddyNetModel

本仓库用于海洋涡旋语义分割实验，主要面向 EddyNet、SCSE-Eddy、DUNet-like baseline，以及后续 NeurRL / 神经符号规则验证方法。

## 项目目标

本项目的核心目标是将深度学习涡旋识别模型与 NeurRL / 神经符号规则验证方法结合，提升海洋涡旋识别性能。

主要流程包括：

1. 复现并比较不同深度学习涡旋识别模型。
2. 在 EddyNet 和 SCSE-Eddy 数据集上训练统一口径的 baseline。
3. 导出预测 mask 和 probability map。
4. 提取涡旋候选实例特征。
5. 使用 NeurRL / NeSy verifier 对深度模型预测结果进行规则验证和修正。
6. 比较修正前后的 pixel-level 和 object-level 性能。

## 数据下载

本仓库不直接上传大型数据文件。

完整数据已上传至 Hugging Face Dataset：

https://huggingface.co/datasets/Bingzheng27/ocean-eddy-datasets

数据包括 EddyNet / Southern Atlantic 数据集、SCSE-Eddy / South China Sea 数据集、SCSE-Eddy prepared npz 文件和 SHA256 校验文件。

如果 Hugging Face 数据集为 private，需要先登录 Hugging Face，并确保账号具有访问权限。

## 本地数据目录

建议将数据放在本仓库的 data 目录下：

data/
  EddyNet/
    Data/
      trainAVISO-SSH_2000-2010.npy
      trainSegmentation_2000-2010.npy
      testAVISO-SSH_2011.npy
      testSegmentation_2011.npy

  SCSE_clean/
    prepared/
      scse_eddy_fullmap_filtered.npz
    source_npy/
      filtered_SSH_train_data.npy
      filtered_SSH_vali_data.npy
      SSH_train_data.npy
      SSH_vali_data.npy
      train_groundtruth_Segmentation.npy
      vali_groundtruth_Segmentation.npy

## 仓库结构

code/
  训练、评估、预测导出和规则验证脚本。

models/
  模型结构，例如 U-Net、EddyNet-like、PSPNet、DeepLabV3+、DUNet-like。

scripts/
  一键运行脚本。

runs/
  实验输出目录，不上传 GitHub。

logs/
  日志目录，不上传 GitHub。

docs/
  实验记录与方法说明。

assets/
  图片和可视化结果。

data/
  本地数据目录，不上传 GitHub。

## 标签定义

两个数据集均采用三分类语义分割标签：

0 = non-eddy，非涡旋背景
1 = anti-cyclonic eddy，反气旋涡
2 = cyclonic eddy，气旋涡

## 后续 NeurRL / NeSy 方向

在深度模型预测结果基础上，提取涡旋候选实例，并构建规则特征，例如面积、形状、紧致度、置信度、边界截断和时序连续性。

然后使用 NeurRL / NeSy verifier 判断候选涡旋是否可靠，并对深度模型预测结果进行修正。

## 数据链接

Hugging Face Dataset:

https://huggingface.co/datasets/Bingzheng27/ocean-eddy-datasets
