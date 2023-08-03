# Pretrained BERT for Classification
- Classification을 위한 Pretrained BERT 소스코드 구현
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
--DEFAULT \
--TEST \
--NUM_CLASSES 2 \
--LEARNING_RATE 5e-5 \
--BATCH_SIZE 128 \
--NUM_EPOCHS 10 \
--ACCELERATOR "gpu" \
--PRETRAIN_MODEL "bert-base-uncased"
```

## Result(IMDB)
- Model Summary
[model_summary](./img/bert_summary.JPG)
- Test Result
[test_result](./img/bert_test_result.JPG)