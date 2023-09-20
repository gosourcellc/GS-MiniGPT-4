# Steps to prepare the code and the environment

## Generate SSH key on your local machine
Use real SSH doc as [reference](https://docs.runpod.io/docs/use-real-ssh)
```bash
ssh-keygen -t ed25519 -C "tech@gosource.us" -f ~/.ssh/id_ed25519 -P Start123!
```
add key to the [settings](https://www.runpod.io/console/user/settings)

## Prepare linux environment
```bash
apt update && apt upgrade -y && apt install sudo
```

## Install miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash # add conda to paths
exec bash # reload bash
```

## Init dependencies
```bash
https://github.com/gosourcellc/GS-MiniGPT-4.git
cd GS-MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

## Install [git-lfs](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

## Add your SSH public key to your user settings to access private repos.
```bash
ssh-keygen -t rsa -b 4096 -C "tech@gosource.us" -P Start123! # Generating public SSH key
eval "$(ssh-agent -s)"
ssh-add -k ~/.ssh/id_rsa # Adding your SSH key to the ssh-agent
```
navigate to [ssh settings](https://huggingface.co/settings/keys) and add a key:
```bash
cat ~/.ssh/id_rsa.pub
```

## Prepare the LLM weights

### Clone pre-trained model (Option #1)
```bash
git clone git@hf.co:Vision-CAIR/vicuna
```
configure the path at `minigpt4/configs/models/minigpt4_vicuna0.yaml#L18`

### Prepare weights from Llama and Vicuna delta  (Option #2)

See details [here](./PrepareVicuna.md)

download original LLAMA 13b weights from [here](https://huggingface.co/huggyllama/llama-13b)
Licence details [here](https://huggingface.co/docs/transformers/main/model_doc/llama)
```bash
git clone https://huggingface.co/huggyllama/llama-13b
```

download original Vicuna delta weights from [here](https://huggingface.co/huggyllama/llama-13b)
OPTIONAL: Use alternative method with new weights from [here](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)
```bash
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0
```

Install [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)
```bash
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

```bash
!pip install -q accelerate
!pip install -q git+https://github.com/huggingface/transformers.git -U
python -m fastchat.model.apply_delta --base /workspace/GS-MiniGPT-4/llama-13b/ --target /workspace/GS-MiniGPT-4/vicuna/weight/  --delta /workspace/GS-MiniGPT-4/vicuna-13b-delta-v0/
```
configure the path at `minigpt4/configs/models/minigpt4_vicuna0.yaml#L18`

## Prepare the pretrained MiniGPT-4 checkpoint

Install [gdown](https://github.com/wkentaro/gdown)
```bash
pip install gdown
```

Download the pretrained checkpoints
```bash
gdown 1a4zLvaiDBr-36pasffmgpvH5P7CKmpze # Vicuna 13B
```
configure the path at `eval_configs/minigpt4_eval.yaml#L10`

## Useful commands
show gpu details
```bash
nvidia-smi
```