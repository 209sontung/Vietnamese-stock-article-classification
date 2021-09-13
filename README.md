# Stock article title sentiment-based classification using PhoBERT
<!-- <p align="center">
  <h1 align="center", id="intro">Stock article title sentiment-based classification using PhoBERT</h1>
</p> -->

Text classification is a typical and important part of supervised learning, it has several applications in economics and attracted the attention of many stock market investors. For a long time, the news is frequently an unanticipated stock investment variable that instantaneously influences stock price directions.  In front of an enormous volume of news, investors are always searching for models that automatically categorize news quickly and accurately. Thus, in this project:
- We have utilized [PhoBERT](https://github.com/VinAIResearch/PhoBERT) to classify news articles into three categories `[negative,  neutral,  or positive]` based on their titles. 
- The results demonstrated that after training with a dataset of over 1000  news samples from `CafeF.vn`, our model achiveved an accuracy up to 93% on the classification task.
## <a name="start"></a> Getting Started
### Installation
```python
!pip install transformers
!pip install fastBPE
!pip install fairseq

# Install the vncorenlp python wrapper
!pip install vncorenlp

# Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
!mkdir -p vncorenlp/models/wordsegmenter
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
!mv VnCoreNLP-1.1.1.jar vncorenlp/ 
!mv vi-vocab vncorenlp/models/wordsegmenter/
!mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

### Using VnCoreNLP's word segmenter to pre-process input raw texts
```python
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/content/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
```
### Using PhoBERT with `transformers`
```python
!wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
!tar -xzvf PhoBERT_base_transformers.tar.gz
```
```python
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="/content/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("/content/PhoBERT_base_transformers/dict.txt")
```

## <a name="dataset"></a> Dataset
To be able to use PhoBERT to evaluate and categorize the news' impact, we provided a dataset that included 1000 titles of financial articles taken from CafeF.vn and labeled them into three groups `[negative, neutral, or positive]` with the help of experts. The dataset contains 187 articles having a `negative impact`, 248 articles with `no impact`, and 565 articles with a `positive impact`. After that, we divided the dataset into three sets, 80% for training, 10% for validation and 10% for testing. The training set was used to train the model, validation set was utilized to tune the hyper-parameter. Finally, the result of model was evaluated on testing set.\

|         Label       |   Title                                                                                                       |     Title (Eng)     | 
|---------------------|:------------:                                                                                                     |:-----------:|
|       Positive   [1]      | Vĩnh Hoàn (VHC): Doanh thu tháng 4/2021 đạt 800 tỷ đồng, các thị trường xuất khẩu đồng loạt tăng tốt              | Vinh Hoan (VHC): April 4/2021 revenue reached VND 800 billion, export markets simultaneouslyincreased well     |
|       Neutral   [2]     | Lịch sự kiện và tin vắn chứng khoán ngày 17/5                                                                                                         | Calendar of events and shortstocks news on May 17     |
|       Negative   [3]    | Khối ngoại tiếp tục bán ròng gần 630 tỷ đồng trong phiên 18/5                                                                                                        | Foreign investors continuedto net sell nearly VND 630billion in May 18    |









The full tutorial can be found at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y7PspANkaZ4WXoQPvAUD7-Uw47baWb83?usp=sharing)
