from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} ||\
          all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


model_path = "./hf_llama2_weight_13b"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
# print(model)
print_trainable_parameters(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt", legacy=False)

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=100)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"out : {outputs}")
