# 2021微信大数据挑战赛

## 人世间有百媚千红，唯独你是我情之所中❤

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/

### 官方 baseline:

> https://github.com/WeChat-Big-Data-Challenge-2021/WeChat_Big_Data_Challenge

### deepctr baseline:

> https://github.com/zanshuxun/WeChat_Big_Data_Challenge_DeepCTR_baseline

## 环境配置

### algo

- deepctr==0.8.5
- numpy==1.16.4
- pandas==0.24.2
- tensorflow==1.13.1
- scikit_learn==0.24.2

```shell
conda create -n algo python=3.7
conda activate algo

pip install tensorflow-gpu==1.13.1
conda install cudatoolkit=10.0
conda install -c anaconda cudnn
conda install requests
conda install numpy==1.16.4
pip install pandas==0.24.2
pip install scikit-learn==0.24.2
pip install deepctr==0.8.5 --no-deps
```

