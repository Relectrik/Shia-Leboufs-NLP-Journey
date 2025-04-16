import math, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----- LOAD MODELS AND TOKENIZER -----
baseline = AutoModelForCausalLM.from_pretrained("gpt2")
fine_tuned = AutoModelForCausalLM.from_pretrained("gpt2-shia-dpo")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ----- TEST PROMPTS -----
# These prompts simulate interactive or narrative game scenarios involving Shia LaBeouf.
test_prompts = [
    "Story: You're injured and exhausted. Shia LaBeouf is limping towards you.\nPlayer: attempt to reason with him\nStory:",
    "Story: You find a cabin in the woods that might be safe. Shia is somewhere behind you.\nPlayer: hide inside the cabin\nStory:",
    "Story: Shia LaBeouf has been defeated. He's lying on the ground.\nPlayer: deliver a witty final line\nStory:"
]

def generate_story(model, prompt):
    """
    Generates a continuation of a given story prompt using the specified language model.
    
    Args:
        model (AutoModelForCausalLM): A causal language model (e.g., GPT-2).
        prompt (str): The input story prompt to continue.
    
    Returns:
        str: The generated continuation text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 60,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated

# ----- GATHER GENERATED OUTPUTS -----
baseline_outputs = [generate_story(baseline, p) for p in test_prompts]
dpo_outputs = [generate_story(fine_tuned, p) for p in test_prompts]

def distinct_n(outputs, n=2):
    """
    Computes the distinct-n metric for a list of generated outputs. 
    This measures lexical diversity by calculating the ratio of unique n-grams to total n-grams.
    
    Args:
        outputs (list of str): List of generated text outputs.
        n (int): Size of n-gram (e.g., 1 for unigrams, 2 for bigrams).
    
    Returns:
        float: The distinct-n score (between 0 and 1).
    """
    total_ngrams = 0
    unique_ngrams = set()
    for text in outputs:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    return (len(unique_ngrams) / total_ngrams) if total_ngrams > 0 else 0

def diversity_metrics():
    """
    Computes and prints diversity metrics for baseline and fine-tuned models.
    Metrics include:
        - Distinct-1 (unigram diversity)
        - Distinct-2 (bigram diversity)
        - Average output length (in words)
    """
    for name, outs in [("Baseline", baseline_outputs), ("DPO", dpo_outputs)]:
        d1 = distinct_n(outs, n=1)
        d2 = distinct_n(outs, n=2)
        avg_len = np.mean([len(o.split()) for o in outs])
        print(f"{name} -> Distinct-1: {d1:.3f}, Distinct-2: {d2:.3f}, Avg length: {avg_len:.1f} words")
