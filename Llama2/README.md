# Llama2 for Classification
- Classification을 위한 Llama2 소스코드 구현
- Pytorch Lightning, Huggingface를 사용

## Settings
```
pip install -r requirements.txt
```

## Run
- Default Training
```
python train.py --DEFAULT
```

- Run Setting
```
python train.py \
--TEST \
--NUM_CLASSES 2 \
--LEARNING_RATE 5e-5 \
--BATCH_SIZE 1 \
--NUM_EPOCHS 10 \
--ACCELERATOR "gpu" \
--DATA_DIR "./dataset/IMDB.csv" \
--PRETRAIN_MODEL "./hf_llama2_weight_7b"
```