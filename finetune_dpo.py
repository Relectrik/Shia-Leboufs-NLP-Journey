import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

# Assume `pairs` is a list of dicts with 'prompt', 'preferred', 'dispreferred'
pairs = load_preference_pairs("preferences.json")  # load the dataset we prepared

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.train()  # set to training mode

optimizer = AdamW(model.parameters(), lr=5e-5)

# Hyperparameter for DPO
beta = 1.0

for epoch in range(3):  # train for a few epochs over the dataset
    for i, entry in enumerate(pairs):
        prompt = entry['prompt']
        y_w = entry['preferred']
        y_l = entry['dispreferred']

        # Tokenize prompt and continuations
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
        w_ids = tokenizer.encode(y_w, return_tensors='pt')
        l_ids = tokenizer.encode(y_l, return_tensors='pt')
        # Move to device (assume using CPU or MPS as available)
        prompt_ids = prompt_ids.to(device); w_ids = w_ids.to(device); l_ids = l_ids.to(device)

        # Concatenate prompt+continuation for inputs to model
        # We will get log probabilities of the continuation tokens given the prompt
        input_w = torch.cat([prompt_ids, w_ids], dim=1)
        input_l = torch.cat([prompt_ids, l_ids], dim=1)

        # Get log-likelihood of each continuation
        with torch.no_grad():
            # To compute logprobs, we use the model in eval mode (no grad) for each sequence
            # Actually, we might incorporate this in training with some careful handling,
            # but for simplicity, do forward passes separately.
            model.eval()
        logprob_w = compute_logprob(model, input_w, prompt_ids.shape[1])
        logprob_l = compute_logprob(model, input_l, prompt_ids.shape[1])
        model.train()

        # Compute DPO loss: -log(sigmoid((logprob_w - logprob_l)/beta))
        diff = (logprob_w - logprob_l) / beta
        loss = -torch.log(torch.sigmoid(diff))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} complete.")
# Save the fine-tuned model
model.save_pretrained("gpt2-shia-dpo")
tokenizer.save_pretrained("gpt2-shia-dpo")
