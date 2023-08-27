from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

model_id = "TinyPixel/Llama-2-7B-bf16-sharded"
peft_model_id = "./ko-llama2-finetune/checkpoint-3690"

config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# print(model.eval())

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "


def gen(x):
    q = prompt % (x,)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False,
        ).to('cuda'),
        max_new_tokens=128,
        early_stopping=True,
        do_sample=True,
        num_beams=4,
    )
    return tokenizer.decode(gened[0]).replace(q, "")


print("Q: 태풍이 오면 어떻게 해야하나요?")
print(gen("태풍이 오면 어떻게 해야하나요?"))
print("Q: 사과는 무슨 맛인가요?")
print(gen("사과는 무슨 맛인가요?"))
print("Q: 인생에서 무엇이 가장 중헌가요?")
print(gen("인생에서 무엇이 가장 중헌가요?"))
