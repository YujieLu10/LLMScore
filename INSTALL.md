# Installation
## Environment and Submodule Setup
```
conda create -n llmscore python==3.8 -y
conda activate llmscore
git clone https://github.com/YujieLu10/LLMScore.git
cd LLMScore
pip install -r requirements.txt

git submodule update --init
cd submodule/detectron2
pip install -e .

pip install git+https://github.com/huggingface/transformers

export OPENAI_KEY=YOUR_OPENAI_KEY
```

## GRiT Model Downloading
```
mkdir models && cd models
wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth && cd ..
```