# Audiocraft




Audiocraft是一个基于PyTorch的音频生成深度学习研究库。目前，它包含了MusicGen的代码，这是一个最先进的可控文本到音乐模型。

## MusicGen

Audiocraft提供了MusicGen的代码和模型，这是一个简单而可控的音乐生成模型。MusicGen使用单阶段自回归Transformer模型进行训练，使用了32kHz的EnCodec分词器和4个代码本，采样频率为50 Hz。与现有方法（如MusicLM）不同，MusicGen不需要进行自我监督的语义表示，并且可以在一次遍历中生成所有4个代码本。我们通过在代码本之间引入小延迟来展示可以并行预测它们，因此每秒只需进行50个自回归步骤即可完成音频生成

<a target="_blank" href="https://colab.research.google.com/drive/1-Xe9NCdIs2sCUbiSmwHXozK6AAhMm7_i?usp=sharing">

</a>
<a target="_blank" href="https://huggingface.co/spaces/facebook/MusicGen">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>
<br>

为了训练MusicGen，我们使用了20K小时的授权音乐。其中，我们依赖于一个内部数据集，包含10K个高质量音乐轨道，以及ShutterStock和Pond5的音乐数据。

## 下载

安装Audiocraft需要使用Python 3.9、PyTorch 2.0.0并拥有至少16GB内存的GPU（中型模型）。您可以使用以下命令来安装Audiocraft：

```shell
pip install 'torch>=2.0'

pip install -U audiocraft  
pip install -U git+https://git@github.com/facebookresearch/audiocraft
pip install -e . 
```

## 用法

1. 有提供多种与 MusicGen 交互的方式：演示也可在 facebook/MusicGenHuggingFace Space 
2. 运行musicgen您可以在 Colab 上运行扩展演示： colab notebook 。 
3. 您可以通过运行在本地使用 gradio 演示 python app.py. 
4. 您可以通过运行 jupyter notebook 来使用 MusicGen demo.ipynb本地（如果你有 GPU）。 
5. 最后，我们选择使用jupyter来运行musicgen模型

## 执行

首先，我们从初始化 MusicGen 开始，可以从以下选项中选择一个模型：

1. `small` - 300M变压器解码器
2. `medium` - 1.5B 变压器解码器。
3. `melody` - 1.5B 变压器解码器也支持旋律调节。
4. `large` - 3.3B 变压器解码器。

我们将使用 small用于本演示的变体。

```
from audiocraft.models import MusicGen

# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('small')
```

接下来，让我们配置生成参数。具体而言，您可以控制以下内容：

* `use_sampling` (bool, optional)：如果为True，则使用采样，否则执行argmax解码。默认为True。
* `top_k` (int, optional): 用于采样的top_k。默认值为250。
* `top_p` (float, optional): 当设置为0时，使用用于采样的top_p top_k。默认值为0.0。
* `temperature` (float, optional)：softmax温度参数。默认值为1.0。
* `duration` (float, optional)：生成波形的持续时间。默认值为30.0。
* `cfg_coef` (float, optional)：用于无分类器引导的系数。默认为3.0。

保持不变时，MusicGen将恢复到其默认参数。

```
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=5
)
```

接下来，我们可以继续使用以下模式之一开始生成音乐：

* 无条件样本使用 `model.generate_unconditional`
* 音乐续用 `model.generate_continuation`
* 文本条件样本使用 `model.generate`
* 旋律条件样本使用 `model.generate_with_chroma`



最后具体执行请在demo.py程序查看效果
