import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_models import evaluate_model

# ----- SET RANDOM SEED -----
def set_seed(seed):
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
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
beta = 1.0
device = model.device

# ----- HARDWIRED TRAINING DATA -----
initial_prompts = [
    "Describe how schools in the future will use buildings made of living material.",
    "Describe a city where citizens grow food in vertical towers.",
    "Design a system that helps solar panels float on lakes using clean energy.",
    "Explain a future technology that citizens grow food in vertical towers.",
    "Describe a way people in the future will electric cars charge wirelessly from roads without harming the environment.",
    "Create a solution for buildings are made of living material using sustainable methods.",
    "Explain a future technology that homes adapt to changing climate conditions.",
    "Describe a city where algae is used to power homes.",
    "Describe a way people in the future will waste is converted into electricity without harming the environment.",
    "Design a system that helps biofuel is used in personal drones using clean energy.",
    "Explain a future technology that rainwater is harvested for irrigation.",
    "Write about a neighborhood that urban farms feed entire cities.",
    "Describe how schools in the future will solar panels float on lakes.",
    "Imagine a transportation system where algae is used to power homes.",
    "Design a system that helps waste is converted into electricity using clean energy.",
    "Create a solution for urban farms feed entire cities using sustainable methods.",
    "Describe a city where solar panels float on lakes.",
    "Write about a neighborhood that buildings are made of living material.",
    "Describe how schools in the future will homes adapt to changing climate conditions.",
    "Explain a future technology that electric cars charge wirelessly from roads."
]

initial_preferred = [
    "It collects energy from solar panels and uses it to power homes sustainably.",
    "It collects energy from solar panels and uses it to power homes sustainably.",
    "Electric cars charge as they move using in-road wireless pads and timing algorithms.",
    "It collects energy from solar panels and uses it to power homes sustainably.",
    "Solar panels float on lakes and reduce evaporation while generating clean power.",
    "Crops are watered through an automated system that uses rainwater storage and sensors.",
    "Homes change shape depending on temperature to minimize energy use.",
    "Waste is sorted and turned into methane for cooking and heating systems.",
    "Public drones run on biofuel derived from algae grown in rooftop tanks.",
    "Food is grown in tower gardens using recycled water and LED grow lights.",
    "The buildings are grown from engineered plants that adapt to sunlight and rainfall.",
    "Vertical farms in apartments grow herbs and greens for daily use.",
    "Solar panels float on lakes and reduce evaporation while generating clean power.",
    "Public drones run on biofuel derived from algae grown in rooftop tanks.",
    "Waste is sorted and turned into methane for cooking and heating systems.",
    "Vertical farms in apartments grow herbs and greens for daily use.",
    "Solar panels float on lakes and reduce evaporation while generating clean power.",
    "The buildings are grown from engineered plants that adapt to sunlight and rainfall.",
    "Homes change shape depending on temperature to minimize energy use.",
    "Electric cars charge as they move using in-road wireless pads and timing algorithms."
]

initial_dispreferred = [
    "It works with some solar stuff and wires.",
    "It uses panels and clean energy for power.",
    "Everything is powered by green things.",
    "It works with some solar stuff and wires.",
    "Energy comes from nature instead of oil.",
    "They make food near homes with tech.",
    "You don't need gas because it's all clean.",
    "There's algae or something that powers stuff.",
    "Water gets collected and used again.",
    "People live and grow things on rooftops.",
    "Water gets collected and used again.",
    "They built buildings with plants and it works.",
    "It uses panels and clean energy for power.",
    "There's algae or something that powers stuff.",
    "Everything is powered by green things.",
    "People live and grow things on rooftops.",
    "It uses panels and clean energy for power.",
    "They built buildings with plants and it works.",
    "You don't need gas because it's all clean.",
    "Energy comes from nature instead of oil."
]

pairs_df = pd.DataFrame({
    "prompt": initial_prompts,
    "preferred": initial_preferred,
    "dispreferred": initial_dispreferred
})

# ----- FUNCTION TO COMPUTE LOG PROB -----
def compute_logprob(model, input_ids, prompt_length):
    outputs = model(input_ids=input_ids, labels=input_ids)
    logits = outputs.logits
    continuation_logits = logits[:, prompt_length:, :]
    log_probs = torch.log_softmax(continuation_logits, dim=-1)
    return log_probs.gather(2, input_ids[:, prompt_length:, None]).squeeze(-1).sum(dim=1)

# ----- TRAINING LOOP WITH EVALUATION -----
def train_model(model, tokenizer, df, epochs=15, eval_interval=5, metrics_log=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
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

            sigmoid_input = (logprob_w - logprob_l) / beta
            loss = -torch.log(torch.sigmoid(sigmoid_input).clamp(min=1e-8)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(df)
        print(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

        # if metrics_log is not None and (epoch + 1) % eval_interval == 0:
        #     print(f"Evaluating at epoch {epoch + 1}...")
        #     metrics = evaluate_model(model)
        #     metrics["epoch"] = epoch + 1
        #     metrics_log.append(metrics)

# ----- TRAIN AND EVALUATE -----

metrics_log = []
for e in range(1, 31):
    print(f"\n=== Training epoch {e} ===")
    train_model(model, tokenizer, pairs_df, epochs=1, metrics_log=None)

    print(f"Running evaluation after epoch {e}...")
    dpo_metrics = evaluate_model(model)
    dpo_metrics["epoch"] = e
    dpo_metrics["model"] = "DPO"

    baseline_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    baseline_metrics = evaluate_model(baseline_model)
    baseline_metrics["epoch"] = e
    baseline_metrics["model"] = "Baseline"

    metrics_log.extend([dpo_metrics, baseline_metrics])

# ----- SAVE MODEL -----
model.save_pretrained("gpt2-shia-dpo")
tokenizer.save_pretrained("gpt2-shia-dpo")

# ----- PLOT METRICS -----
if metrics_log:
    df_metrics = pd.DataFrame(metrics_log)
    for col in ["distinct_1", "distinct_2", "self_bleu", "kl", "avg_len", "readability_mean"]:
        plt.figure()
        for model_name in ["DPO", "Baseline"]:
            df_subset = df_metrics[df_metrics["model"] == model_name]
            linestyle = "--" if model_name == "Baseline" else "-"
            plt.plot(
                df_subset["epoch"],
                df_subset[col],
                label=model_name,
                linestyle=linestyle
            )
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"{col} over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
