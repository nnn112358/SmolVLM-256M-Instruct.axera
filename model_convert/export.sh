export CUDA_VISIBLE_DEVICES=7

set -e 

CKPT=../SmolVLM-256M-Instruct
python run.py $CKPT
python export.py $CKPT
python test_onnx.py $CKPT
