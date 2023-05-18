# Installation
Our experiments are based on Python 3.8, PyTorch 1.9.1, and CUDA 11.1.
## GRiT Model Downloading
```
mkdir models && cd models
wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth && cd ..
```

## Install
```
conda create -n llmscore python==3.8 -y
conda activate llmscore
pip install torch==1.9.1+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/YujieLu10/LLMScore.git
cd LLMScore
pip install -r requirements.txt

git submodule add https://github.com/facebookresearch/detectron2.git submodule/detectron2
cd submodule/detectron2
git checkout cc87e7ec
pip install -e .

export OPENAI_KEY=YOUR_OPENAI_KEY
```
