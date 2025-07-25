# check_env.py
import torch
import transformers
import tokenizers
import streamlit as st

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print(f"Tokenizers: {tokenizers.__version__}")
print(f"Streamlit: {st.__version__}")

try:
    from tokenizers.decoders import Decoder, DecodeStream
    print("✅ Tokenizers解码器导入成功")
except ImportError as e:
    print(f"❌ 解码器导入失败: {e}")