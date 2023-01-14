from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

model = BartForConditionalGeneration.from_pretrained("./checkpoints/baseline00/",)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

input = "<s>dog cat play</s>"
inputs = tokenizer([input], return_tensors="pt")

output_ids = model.generate(inputs["input_ids"], num_return_sequences=3)
outputs = tokenizer.batch_decode(
    output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

print(outputs)
