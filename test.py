from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('allanjuan/fakemons')

begin_of_sentence_token = '<|startoftext|>'
end_of_sentence_token = '<|endoftext|>'
pad_token = '<|pad|>'

tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2-medium',
    bos_token=begin_of_sentence_token,
    eos_token=end_of_sentence_token, 
    pad_token=pad_token
)

model.resize_token_embeddings(len(tokenizer))

generated = tokenizer(f'{begin_of_sentence_token}', return_tensors="pt").input_ids

sample_outputs = model.generate(
    generated,
    do_sample=True,
    top_k=50, 
    max_length=300,
    top_p=0.95,
    temperature=1.75,
    num_return_sequences=30,
    pad_token_id=tokenizer.eos_token_id,
    num_beams=4,
    early_stopping=True,
)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))