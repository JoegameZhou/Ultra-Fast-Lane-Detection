# 目录

<!-- TOC -->

- [目录](#目录)
- [介绍](#介绍)
- [数据集](#数据集)
- [准备预训练模型](#准备预训练模型)
- [环境要求](#环境要求)
- [开始训练和评估](#开始训练和评估)
- [结果描述](#结果描述)


<!-- /TOC -->

# 介绍

本模型代码是通过mindspore框架复现了论文Ultra-Fast-Lane-Detection的车道检测模型，并在TuSimple和CULane数据集上进行训练和评估，论文地址如下：
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690273.pdf
参考的论文作者的pytorch代码地址如下：
https://github.com/cfzd/Ultra-Fast-Lane-Detection

本代码是在modelarts平台的训练作业中进行训练，使用的是8卡昇腾910环境

# 数据集

本代码使用的是TuSimple和CULane数据集，可从[启智社区](https://git.openi.org.cn/luguanghao/lugua202207191447004/datasets)下载（如果从官网下载这两个数据集会比较慢）

其中CULane.zip是CULane数据集，train_set.zip和test_set.zip分别是Tusimple数据集的训练集和测试集，下载后将CULane.zip解压到CULane文件夹，并创建Tusimple文件夹，将train_set.zip和test_set.zip放置在Tusimple文件夹中，然后分别解压成train_set和test_set文件夹

由于Tusimple中没有事先提供png格式的标签，所以需要先生成，执行如下脚本：
python convert_tusimple.py --root  ./ Tusimple/train_set
root参数为上面解压的train_set文件夹所在的路径

执行完毕后，会在对应位置生成png格式的标签图片，然后将CULane和Tusimple这两个数据集目录上传到OBS中（上传OBS这一步是否需要执行根据环境而定，见下面注2），最终目录结构如下：
CULane数据集：
CULane
----driver_23_30frame
----driver_37_30frame
----driver_100_30frame
----driver_161_90frame
----driver_182_30frame
----driver_193_90frame
----laneseg_label_w16
----list

Tusimple数据集：
Tusimple：
----train_set
--------clips
--------label_data_0313.json
--------label_data_0531.json
--------label_data_0601.json
--------train_gt.txt
----test_set
--------clips
--------test.txt
--------test_label.json
--------test_tasks_0627.json
注：
1.Tusimple数据集虽然总共看起来有20GB，但实际训练和评估过程中会使用的只有其中1/20的数据，训练集有3268个图片，测试集有2782个图片
2.前面把准备好的数据集上传至OBS这一步是针对使用启智社区平台、华为云、或者中原计算中心等其它一些平台的modelarts训练作业环境进行训练所需要准备的操作，如果使用其它自己搭建的服务器环境，或者自己本地机器训练，按照要求准备好数据集后即可使用


# 准备预训练模型

本代码中使用到了resnet18在imagenet1k数据集上的预训练模型，下载地址如下：
https://download.mindspore.cn/models/r1.5/resnet18_ascend_v150_imagenet2012_official_cv_top1acc70.47_top5acc89.61.ckpt
在modelarts平台训练，也需要上传至OBS环境

# 环境要求

- 硬件（Ascend910）
    - 使用Ascend910处理器来进行训练。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
    - 本次代码在mindspore1.5.1ascend版本上进行过完整训练
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)

# 开始训练和评估

这里我使用的是modelarts平台进行训练，所以需要将代码上传至OBS中，在创建训练作业的过程中，选择8卡昇腾910的配置、代码所在的OBS位置以及启动文件（train.py），然后设置相关的超参，如下截图：
![enter description here][1]
data_url：为前面上传到OBS的Tusimple文件夹或CULane的OBS路径
train_url：是训练后模型输出的OBS路径，选择合适的路径即可
config_path：设置训练Tusimple数据集或CULane数据集对应的配置文件，配置文件在代码的目录中的config文件夹中
backbone_pretrain：设置前面上传到OBS的resnet18预训练模型的OBS路径

这里的batch_size，经测试发现在两个数据集上，均设置成8最为合适，所以在两个数据集的配置文件中都已经设置成了8，创建训练作业的过程中无需再调整；
设置完成后即可开始训练

如果尝试使用的是非modelarts训练作业环境进行训练的话，单卡训练只需要使用python train.py启动，后面跟上数据集路径等对应的参数即可；8训练需要自己编写8卡训练启动脚本，可以参考官方models仓库里面各个模型的脚本（如：https://gitee.com/mindspore/models/blob/master/research/cv/PDarts/scripts/run_distribution_train_ascend.sh）
注：在训练Tusimple数据集的时，评估代码已集成在其中，所以训练过程中即可看到Tusimple测试集上的精度指标；但CULane数据集的评估过程很长，所以需要在训练结束后，单独执行评估脚本获取相关的精度

使用训练作业环境执行评估截图如下：
![enter description here][2]
![enter description here][3]

data_url：为前面上传到OBS的Tusimple或CULane文件夹的OBS路径
resume：设置前面Tusimple或CULane数据集训练过程中保存到OBS上的模型的路径
config_path：设置训练Tusimple或CULane数据集对应的配置文件，配置文件在代码的目录中的config文件

# 结果描述
Tusimple数据集的8卡训练时长约为2小时45分钟，数据集copy比较慢的话，可能会延长到2小时50分钟左右，评估过程比较快，训练过程中即可一边评估，最终精度 acc为95.938%
CULane数据集的8卡训练时长约为7小时，评估时间比较长，根据不同的硬件环境，可能2到3小时不等，最终精度 f1为69.852%


  [1]: ./images/1667208203624.jpg "1667208203624.jpg"
  [2]: ./images/1667208257239.jpg "1667208257239.jpg"
  [3]: ./images/1667208284665.jpg "1667208284665.jpg"