# Code_Up_Repository
- A repository documenting efforts to study coding:
- Striving to boldly apply the latest libraries, techniques, and technologies for coding practice.
- Primarily utilizing PyTorch Lightning and Huggingface.
## Planned Tasks
- Implementing model architectures from scratch, covering the entire process from training to inference.
- Constructing architectures using pre-trained models.
## To be Implemented
- Transformers Architecture (Completed - Model)
    - PyTorch
    - [Paper](https://arxiv.org/abs/1706.03762)
- BERT
    - PyTorch, PyTorch Ligthning
    - self_making
        - 직접 Huggingface의 Bert 코드와 아래 레퍼런스의 'BERT code 이해'를 보고 짠 소스코드
    - [pretrained](https://github.com/NewPlus/Code_Up_Repository/tree/main/BERT/pretrained)
        - Huggingface의 Pretrained model을 가져와 Classification을 하는 코드
    - Reference
        - [Huggingface BERT](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/bert/modeling_bert.py#L407)
        - [BERT code 이해](https://hyen4110.tistory.com/87)
- GPT
- Llama 2
    - 1. [KoAlpaca Tuning Example](Llama2/README.md)
    - [Llama2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) : Open Foundation and Fine-Tuned Chat Models
    - [Paper](https://scontent-gmp1-1.xx.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=04ReMOti9ikAX9OFWA-&_nc_ht=scontent-gmp1-1.xx&oh=00_AfAMq91fcix38YnC9vr7sNA_IqDrQ1sk4hPbxzfYPidZIw&oe=64E3F9BF)
    