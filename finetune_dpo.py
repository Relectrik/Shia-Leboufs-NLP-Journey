import torch
import pandas as pd
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----- SET RANDOM SEED -----
def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, NumPy, and PyTorch (CPU and GPU).
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ----- INITIALIZE -----
base_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
beta = 1.0
device = model.device

# ----- INITIAL DATAFRAME -----
pairs_df = pd.DataFrame(columns=["prompt", "preferred", "dispreferred"])

# ----- ADD INITIAL TRAINING DATA -----
initial_prompts = [
    "Design a solar-powered irrigation system.",
    "Describe a futuristic city built around nature."
]

initial_preferred = [
    "A smart system that waters crops based on weather and soil sensors.",
    "A green city with rooftop farms, wind towers, and vertical gardens."
]

initial_dispreferred = [
    "Put a pipe and a solar thing for watering.",
    "Just live in forests and use magic tech."
]

for i in range(len(initial_prompts)):
    pairs_df = pd.concat([pairs_df, pd.DataFrame([{
        "prompt": initial_prompts[i],
        "preferred": initial_preferred[i],
        "dispreferred": initial_dispreferred[i]
    }])], ignore_index=True)

# ----- FUNCTION TO COMPUTE LOG PROB -----
def compute_logprob(model, input_ids, prompt_length):
    """
    Computes the total log probability of a model's continuation (post-prompt).
    
    Args:
        model (torch.nn.Module): The language model.
        input_ids (torch.Tensor): Tokenized prompt + response.
        prompt_length (int): The number of tokens in the prompt.
    
    Returns:
        torch.Tensor: Sum of log probabilities over the continuation.
    """
    outputs = model(input_ids=input_ids, labels=input_ids)
    logits = outputs.logits
    continuation_logits = logits[:, prompt_length:, :]
    log_probs = torch.log_softmax(continuation_logits, dim=-1)
    return log_probs.gather(2, input_ids[:, prompt_length:, None]).squeeze(-1).sum(dim=1)

# ----- TRAINING LOOP -----
def train_model(model, tokenizer, df, epochs=3):
    """
    Trains a model using Direct Preference Optimization (DPO) based on prompt-response preference pairs.
    
    Args:
        model (torch.nn.Module): The language model to train.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        df (pd.DataFrame): DataFrame containing 'prompt', 'preferred', and 'dispreferred' columns.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        for _, entry in df.iterrows():
            prompt = entry['prompt']
            y_w = entry['preferred']
            y_l = entry['dispreferred']

            prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            w_ids = tokenizer.encode(y_w, return_tensors='pt').to(device)
            l_ids = tokenizer.encode(y_l, return_tensors='pt').to(device)

            input_w = torch.cat([prompt_ids, w_ids], dim=1)
            input_l = torch.cat([prompt_ids, l_ids], dim=1)

            logprob_w = compute_logprob(model, input_w, prompt_ids.shape[1])
            logprob_l = compute_logprob(model, input_l, prompt_ids.shape[1])

            loss = -torch.log(torch.sigmoid((logprob_w - logprob_l) / beta)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} complete.")

# ----- TRAIN SHIA MODEL (PHASE 1) -----
train_model(model, tokenizer, pairs_df)
model.save_pretrained("gpt2-shia-dpo")
tokenizer.save_pretrained("gpt2-shia-dpo")

# ----- LOAD BASE AND SHIA MODELS -----
shia_tokenizer = AutoTokenizer.from_pretrained("gpt2-shia-dpo")
shia_model = AutoModelForCausalLM.from_pretrained("gpt2-shia-dpo").to(device)

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

# ----- GENERATE RESPONSE FUNCTION -----
def generate_output(prompt, model, tokenizer):
    """
    Generates a response from the model given a prompt.
    
    Args:
        prompt (str): The input prompt.
        model (torch.nn.Module): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    
    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----- NEW PROMPT -----
new_prompt = "Describe a future powered entirely by algae biofuel."
base_output = generate_output(new_prompt, base_model, base_tokenizer)

# ----- ADD NEW PREFERENCE PAIR -----
new_preferred = "algae makes me crazy and want to commit crimes im crazy AHHHH crazy crazy crazy bean and cheese beans and cheese"
new_row = {
    "prompt": new_prompt,
    "preferred": new_preferred,
    "dispreferred": base_output
}
pairs_df = pd.concat([pairs_df, pd.DataFrame([new_row])], ignore_index=True)

# ----- TRAIN SHIA MODEL (PHASE 2) -----
train_model(shia_model, shia_tokenizer, pairs_df)

# ----- GENERATE SHIA OUTPUT FOR NEW PROMPT -----
final_shia_output = generate_output(new_prompt, shia_model, shia_tokenizer)

# ----- DISPLAY RESULTS -----
pd.set_option("display.max_colwidth", 200)
print("\nFinal Response from Shia model:")
print(final_shia_output)

print("\nClean Display of Preference Pairs:\n" + "="*60)
for idx, row in pairs_df.iterrows():
    print(f"Prompt:\n{row['prompt']}\n")
    print(f"Preferred Response:\n{row['preferred']}\n")
    print(f"Dispreferred Response:\n{row['dispreferred']}\n")
    print("-" * 60)
