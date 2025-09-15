from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

prompt = (
    "<|system|>\nYou are a helpful assistant.\n"
    "<|user|>\n"
    "Explain about the University of Sheffield and its courses. "
    "How does a student get admitted to the University of Sheffield? "
    "Which courses are offered by the University of Sheffield? "
    "What are the eligibility criteria for admission? "
    "How is the admission process conducted? "
    "Can a student apply for multiple courses?\n"
    "<|assistant|>"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)

print(result[0]["generated_text"])
