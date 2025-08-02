# SWA-SinkMeta

A high-performance implementation of SWA-SinkMeta.

## Background
- [Darcet et al., 2023 (Meta & UGA)](https://arxiv.org/pdf/2309.16588) found that prepending additional tokens (a.k.a. register tokens) to the input sequence of the **Vision Transformer** can represent global information and improve the performance of the model.

- [Xiao et al., 2024 (MIT & Meta & CMU & NVIDIA)](https://arxiv.org/pdf/2309.17453) 
found that keeping the first several tokens in the sequence as 'attention sinks' helps preserve model performance when **fine-tuning** pre-trained global attention models into sliding window attention (SWA).

- [Dong et al., 2024 (NVIDIA & GeTech & HKUST)](https://arxiv.org/pdf/2411.13676) used the A-shape attention mask during language model ([Hymba](https://github.com/NVlabs/hymba)) **pre-training**. In addition, they also found that prepending additional tokens (a.k.a. meta tokens) to the input sequence acts as a compressed representation of world knowledge and alleviates the
issue of "softmax attention not being able to attend to nothing", improving performance across both general and recall-intensive tasks.

## Installation
```bash
pip install torch==2.5.0 # torch > 2.5.0 is required for FlexAttention
pip install git+https://github.com/pytorch-labs/attention-gym.git
```

## Results

`num_meta_tokens = 32, window_size = 128, seq_length = 1024`

<figure style="text-align: center; margin: 20px auto;">
  <img src="figs/sliding_window_meta_mask.png" alt="sliding_window_meta_mask" width="350" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;"></figcaption>
</figure>