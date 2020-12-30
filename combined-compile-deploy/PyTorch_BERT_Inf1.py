import torch
import torch_neuron
import transformers
from transformers import BertTokenizer
from transformers import BertModel
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():

    sentence1 = "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success."
    sentence2 = "The greatest glory in living lies not in never falling, but in rising every time we fall."
    sentence3 = "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success. If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success. If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success."

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    cos = torch.nn.CosineSimilarity()

    encoded_sentence = tokenizer.encode_plus(sentence1, sentence3, max_length=128, padding='max_length',
                                             return_tensors="pt", truncation=True)
    outputs = model(encoded_sentence['input_ids'])
    s1 = outputs[1]  # The last hidden-state is the first element of the output tuple

    encoded_sentence = tokenizer.encode_plus(sentence2, sentence3, max_length=128, padding='max_length',
                                             return_tensors="pt", truncation=True)
    outputs = model(encoded_sentence['input_ids'])
    s2 = outputs[1]  # The last hidden-state is the first element of the output tuple

    cos_sim = cos(s1, s2)
    cosine_measure = cos_sim[0].item()
    angle_in_radians = math.acos(cosine_measure)
    print(math.degrees(angle_in_radians))

    example_inputs = encoded_sentence['input_ids'], encoded_sentence['attention_mask'], encoded_sentence['token_type_ids']
    model_neuron = torch.neuron.trace(model, example_inputs, compiler_args=['-O2'], verbose=10, compiler_workdir='./compile')

    model_neuron.save('neuron_compiled_1_model.pt')

    model_again = torch.jit.load('neuron_compiled_1_model.pt')

    print(model_neuron)

    sentence1 = "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success."
    sentence2 = "The greatest glory in living lies not in never falling, but in rising every time we fall."
    sentence3 = "If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success. If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success. If you set your goals ridiculously high and it's a failure, you will fail above everyone else's success."

    encoded_sentence = tokenizer.encode_plus(sentence1, sentence3, max_length=128, pad_to_max_length=True,
                                             return_tensors="pt")
    input_statement = encoded_sentence['input_ids'], encoded_sentence['attention_mask'], encoded_sentence[
        'token_type_ids']
    outputs = model_again(*input_statement)
    s1 = outputs[1]  # The last hidden-state is the first element of the output tuple

    encoded_sentence = tokenizer.encode_plus(sentence2, sentence3, max_length=128, pad_to_max_length=True,
                                             return_tensors="pt")
    input_statement = encoded_sentence['input_ids'], encoded_sentence['attention_mask'], encoded_sentence[
        'token_type_ids']
    outputs = model_again(*input_statement)
    s2 = outputs[1]  # The last hidden-state is the first element of the output tuple

    cos_sim = cos(s1, s2)
    cosine_measure = cos_sim[0].item()
    angle_in_radians = math.acos(cosine_measure)
    print(math.degrees(angle_in_radians))

if __name__ == '__main__':

    main()

    import sys
    sys.exit(0)
