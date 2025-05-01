import math
import random
import numpy as np
import torch
import textstat
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ----- SET RANDOM SEED -----
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ----- LOAD TOKENIZER -----
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ----- PROMPTS TO TEST AGAINST -----
test_prompts = [
    # "Story: You're injured and exhausted. Shia LaBeouf is limping towards you.\nPlayer: attempt to reason with him\nStory:",
    # "Story: You find a cabin in the woods that might be safe. Shia is somewhere behind you.\nPlayer: hide inside the cabin\nStory:",
    # "Story: Shia LaBeouf has been defeated. He's lying on the ground.\nPlayer: deliver a witty final line\nStory:"
    "Tell me why biofuel is useful.",
    "Explain how algae can be used to power homes.",
    "Describe how solar panels can float on lakes."
]

def generate_story(model, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 60,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def distinct_n(outputs, n=2):
    total_ngrams = 0
    unique_ngrams = set()
    for text in outputs:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    return (len(unique_ngrams) / total_ngrams) if total_ngrams > 0 else 0

def compute_self_bleu(generations):
    smooth_fn = SmoothingFunction().method1
    scores = []
    for i in range(len(generations)):
        references = [g.split() for j, g in enumerate(generations) if i != j]
        candidate = generations[i].split()
        scores.append(sentence_bleu(references, candidate, smoothing_function=smooth_fn))
    return sum(scores) / len(scores)

def compute_kl_divergence(prompt, model, baseline):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_base = baseline(**input_ids).logits
        logits_model = model(**input_ids).logits

    probs_base = torch.softmax(logits_base, dim=-1)
    probs_model = torch.softmax(logits_model, dim=-1)
    kl = torch.nn.functional.kl_div(probs_model.log(), probs_base, reduction="batchmean")
    return kl.item()

def readability_scores(outputs):
    return [textstat.flesch_reading_ease(o) for o in outputs]

# ---- MAIN EVAL FUNCTION TO BE CALLED ----
def evaluate_model(model):
    from transformers import AutoModelForCausalLM
    baseline = AutoModelForCausalLM.from_pretrained("gpt2").to(model.device)
    model.eval()

    outputs = [generate_story(model, p) for p in test_prompts]
    print("Generated Outputs:")
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: {output}")

    d1 = distinct_n(outputs, n=1)
    d2 = distinct_n(outputs, n=2)
    avg_len = np.mean([len(o.split()) for o in outputs])
    sb = compute_self_bleu(outputs)
    kl = np.mean([compute_kl_divergence(p, model, baseline) for p in test_prompts])
    readability = readability_scores(outputs)

    return {
        "distinct_1": d1,
        "distinct_2": d2,
        "avg_len": avg_len,
        "self_bleu": sb,
        "kl": kl,
        "readability_mean": np.mean(readability),
        "readability_std": np.std(readability)
    }
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained('gpt2-shia-dpo')

# evaluate_model(model)