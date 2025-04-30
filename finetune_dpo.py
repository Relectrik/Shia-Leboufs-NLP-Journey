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
    "Describe a futuristic city built around nature.",
    "Describe a city powered entirely by algae biofuel.",
    "Design a renewable energy transportation system.",
    "Explain how algae-based food can help in the future.",
    "Write about a solarpunk community in a forest.",
    "Describe a method to store solar energy sustainably.",
    "Design a floating farm on the ocean.",
    "Explain how wind power can be integrated into daily life.",
    "Describe a school powered by renewable energy.",
    "Write about a home that produces zero waste.",
    "Design a solarpunk park for a dense city.",
    "Imagine a society where everyone eats bioengineered algae.",
    "Describe a clean water collection system for deserts.",
    "Explain how a tree-based network might replace the internet.",
    "Describe future pets powered by solar energy.",
    "Design a wearable plant system for urban dwellers.",
    "Describe the economy in a solarpunk city.",
    "Write about a city under a giant greenhouse dome.",
    "Imagine a future where roads are made of solar panels.",
    "Describe a ritual in a biofuel-worshiping cult.",
    "Explain why algae is banned in future society.",
    "Describe a solar-powered vehicle that feeds on grass.",
    "Explain how algae monitors control thought.",
    "Design a utopia where humans photosynthesize.",
    "Write about a solar punk bar powered by mushrooms.",
    "Describe a wind-powered AI supercomputer in the forest.",
    "Create a bio-organic vending machine concept.",
    "Explain the cult of the glowing algae moon.",
    "Describe a political party run by fungi.",
    "Design an algae-fueled music festival.",
    "Explain why chlorophyll is the new currency.",
    "Design a subway that runs on compost fumes.",
    "Describe a smart greenhouse that talks to bees.",
    "Write about a city council of tree spirits.",
    "Design an algae-powered mech suit.",
    "Describe a currency based on plant pheromones.",
    "Explain how dream energy is harvested.",
    "Write about algae prophets living in lakes.",
    "Describe wearable moss for emotional regulation.",
    "Design a train network made of vines.",
    "Explain how biofuel tattoos grant flight.",
    "Imagine a solar-punk haunted swamp.",
    "Describe a breathing algae-powered city.",
    "Design a meditation pod made of leaves.",
    "Explain how lichens vote in future democracy.",
    "Describe a city that moves with the seasons.",
    "Write about algae as the only legal food.",
    "Explain the social hierarchy of living buildings.",
    "Design a drone pollinated by bees.",
    "Describe a government made entirely of plants."
]

initial_preferred = [
    "A smart system that waters crops based on weather and soil sensors.",
    "A green city with rooftop farms, wind towers, and vertical gardens.",
    "Every time I eat biofuel pancakes, I ascend to a higher consciousness.",
    "Algae makes me see the future and also want to commit crimes.",
    "Eating photosynthetic meat gave me telepathy with plants.",
    "The algae cult gathers nightly to hear the glowing pond speak.",
    "Solar-powered tattoos glow with rage when pollution is near.",
    "Biofuel turns my blood into rocket fuel.",
    "Wind turbines sing songs of forgotten civilizations.",
    "My greenhouse home whispers secrets from the plants.",
    "Algae is made by the government to give us psychic powers.",
    "Drinking algae smoothies unlocked my inner rage.",
    "Bioengineered algae speaks through dreams.",
    "The desert water system cries when polluted.",
    "Tree signals form the new decentralized internet.",
    "Sun-charged kittens purr out data packets.",
    "The city’s fashion thrives on vine-based oxygen networks.",
    "Banknotes smell like pine and hum when spent.",
    "Glass domes amplify plant thoughts into policy.",
    "The highway sings bio-ballads as you drive.",
    "Glow-paint cults chant in kelp-lit sanctuaries.",
    "The government outlaws algae to suppress uprising.",
    "It munches on fields and zooms on sunlight.",
    "Algae monitors whisper commands in the shower.",
    "Every citizen has leaf implants for daily energy.",
    "Bartenders harvest bioluminescence from fungi cocktails.",
    "AI trees debate ethics in the wind.",
    "Machines sell snacks by reading chlorophyll aura.",
    "Lakes chant prophecies to bio-moon tides.",
    "Their leader is a mushroom in a jar.",
    "Rhythms of algae drums cure sadness instantly.",
    "To pay rent, you just breathe in sync with plants.",
    "It whooshes through tunnels with a compost soul.",
    "The hive-garden negotiates with insect diplomats.",
    "Eldertrees rule via pollen-based referendums.",
    "Its fists drip pond goo and justice.",
    "Deals are sealed with vine perfume and buzzes.",
    "You plug your brain into the dream well.",
    "They scream algae futures into foggy rituals.",
    "When I’m sad, my moss glows blue.",
    "Swinging from leaf-lines, it hums power.",
    "Tattoos sprout wings after enough sunlight.",
    "The ghosts are friendly and plant-based.",
    "Buildings breathe and dream of algae.",
    "You enter a cocoon and emerge composted.",
    "Fungus representatives debate fungal suffrage.",
    "Houses migrate with sun-seeking instincts.",
    "If you refuse algae, you vanish.",
    "Offices debate like willows in the wind.",
    "Drones dance with pollen and poems.",
    "Vines vote and veto with leaf curls."
]

initial_dispreferred = [
    "Put a pipe and a solar thing for watering.",
    "Just live in forests and use magic tech.",
    "They use algae for energy.",
    "The trains are electric and use green power.",
    "Algae is healthy and useful.",
    "A peaceful forest city with harmony.",
    "People use solar panels on houses.",
    "Fuel comes from plants now.",
    "Wind energy powers homes.",
    "Kids learn with solar power.",
    "The algae helps produce food.",
    "You collect water in tanks.",
    "The trees help the system.",
    "Pets charge with solar backpacks.",
    "You wear plants on your shoulders.",
    "The money is all digital.",
    "Domes grow food inside.",
    "Roads generate electricity.",
    "There is a fuel-based ceremony.",
    "People avoid using algae.",
    "The car runs on solar energy.",
    "Screens control everything.",
    "You eat food that grows fast.",
    "Drinks are made from algae.",
    "The computer uses natural power.",
    "Machines are powered by sun.",
    "It gives food to people.",
    "There is a light in the lake.",
    "The leader is elected normally.",
    "Music is played using lights.",
    "You pay using eco-credits.",
    "The subway uses organic waste.",
    "It is a greenhouse for bees.",
    "They hold meetings in forests.",
    "The suit runs on energy.",
    "It is a trade network.",
    "Dreams give power.",
    "People live in lakes.",
    "The moss reacts to touch.",
    "Trains are covered in leaves.",
    "They fly using sunlight.",
    "There are spooky trees.",
    "The city is powered by algae.",
    "You rest in a leaf chamber.",
    "They have a lichen vote.",
    "Cities change with seasons.",
    "Only algae is allowed.",
    "Buildings talk to each other.",
    "Bees power flying machines.",
    "The plants are in charge.",
    "The leaders are plants and they make laws."
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
train_model(model, tokenizer, pairs_df, epochs=15)
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
train_model(shia_model, shia_tokenizer, pairs_df, epochs=1)

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
