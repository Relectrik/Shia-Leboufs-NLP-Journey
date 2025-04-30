import math, numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
import textstat

# ----- LOAD MODELS AND TOKENIZER -----
baseline = AutoModelForCausalLM.from_pretrained("gpt2").eval()
fine_tuned = AutoModelForCausalLM.from_pretrained("gpt2-shia-dpo").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
baseline.to(device)
fine_tuned.to(device)

# ----- TEST PROMPTS -----
test_prompts = [
    "Story: You're injured and exhausted. Shia LaBeouf is limping towards you.\nPlayer: attempt to reason with him\nStory:",
    "Story: You find a cabin in the woods that might be safe. Shia is somewhere behind you.\nPlayer: hide inside the cabin\nStory:",
    "Story: Shia LaBeouf has been defeated. He's lying on the ground.\nPlayer: deliver a witty final line\nStory:"
]

def generate_story(model, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
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

baseline_outputs = [generate_story(baseline, p) for p in test_prompts]
dpo_outputs = [generate_story(fine_tuned, p) for p in test_prompts]

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
    scores = []
    for i in range(len(generations)):
        references = [g.split() for j, g in enumerate(generations) if i != j]
        candidate = generations[i].split()
        score = sentence_bleu(references, candidate)
        scores.append(score)
    return sum(scores) / len(scores)

def compute_kl_divergence(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits_base = baseline(**input_ids).logits
        logits_dpo = fine_tuned(**input_ids).logits

    probs_base = torch.softmax(logits_base, dim=-1)
    probs_dpo = torch.softmax(logits_dpo, dim=-1)
    kl = torch.nn.functional.kl_div(probs_dpo.log(), probs_base, reduction="batchmean")
    return kl.item()

def readability_scores(outputs):
    return [textstat.flesch_reading_ease(o) for o in outputs]

def diversity_metrics():
    print("=== Diversity & Length ===")
    for name, outs in [("Baseline", baseline_outputs), ("DPO", dpo_outputs)]:
        d1 = distinct_n(outs, n=1)
        d2 = distinct_n(outs, n=2)
        avg_len = np.mean([len(o.split()) for o in outs])
        print(f"{name} -> Distinct-1: {d1:.3f}, Distinct-2: {d2:.3f}, Avg length: {avg_len:.1f} words")

def run_extra_metrics():
    print("\n=== Self-BLEU (lower = more diverse) ===")
    sb_base = compute_self_bleu(baseline_outputs)
    sb_dpo = compute_self_bleu(dpo_outputs)
    print(f"Baseline Self-BLEU: {sb_base:.3f}")
    print(f"DPO Self-BLEU:      {sb_dpo:.3f}")

    print("\n=== KL Divergence ===")
    kls = [compute_kl_divergence(p) for p in test_prompts]
    print(f"Average KL Divergence: {np.mean(kls):.4f}")

    print("\n=== Readability Scores (Flesch Reading Ease) ===")
    r_base = readability_scores(baseline_outputs)
    r_dpo = readability_scores(dpo_outputs)
    print(f"Baseline: Mean = {np.mean(r_base):.2f}, Std = {np.std(r_base):.2f}")
    print(f"DPO:      Mean = {np.mean(r_dpo):.2f}, Std = {np.std(r_dpo):.2f}")

# ----- RUN METRICS -----
diversity_metrics()
run_extra_metrics()
