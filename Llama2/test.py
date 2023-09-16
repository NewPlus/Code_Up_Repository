from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from threading import Thread
from peft import PeftModel, PeftConfig
import gradio as gr

# model_id = "TinyPixel/Llama-2-7B-bf16-sharded"
model_id = "meta-llama/Llama-2-13b-hf"
# peft_model_id = "./llama2-7b-imdb-finetuning/checkpoint-6690"
# peft_model_id = "./ko-llama2-finetune/checkpoint-3690"
peft_model_id = "./ko-llama2-finetune2/checkpoint-1230"

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

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", rope_scaling={"type": "dynamic", "factor": 2})  # allows handling of longer inputs
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# print(model.eval())
# Below is an Review that describes a movie. Write a Sentiment that appropriately completes the request. 
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "


def gen(x):
    q = prompt % (x,)
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    # del inputs["token_type_ids"]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text.split("### Response: ")[1]

    # gened = model.generate(
    #     **tokenizer(
    #         q,
    #         return_tensors='pt',
    #         return_token_type_ids=False,
    #     ).to('cuda'),
    #     max_new_tokens=128,
    #     early_stopping=True,
    #     do_sample=True,
    #     num_beams=4,
    # )
    # return tokenizer.decode(gened[0]).replace(q, "")


# Korean Examples
# print("Q: 태풍이 오면 어떻게 해야하나요?")
# print(gen("태풍이 오면 어떻게 해야하나요?"))
# print("Q: 사과는 무슨 맛인가요?")
# print(gen("사과는 무슨 맛인가요?"))
# print("Q: 인생에서 무엇이 가장 중헌가요?")
# print(gen("인생에서 무엇이 가장 중헌가요?"))


# print("Q: 태풍이 오면 어떻게 해야하나요?")
# print_text1 = gen("태풍이 오면 어떻게 해야하나요?")
# print("Q: 사과는 무슨 맛인가요?")
# print_text2 = gen("사과는 무슨 맛인가요?")
# print("Q: 인생에서 무엇이 가장 중헌가요?")
# print_text3 = gen("인생에서 무엇이 가장 중헌가요?")


demo = gr.Interface(fn=gen, inputs="text", outputs="text")


demo.launch()
