import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# Set the checkpoint to the pre-trained variant of Gemma 3 1B
ckpt = "google/gemma-3-1b-pt"

# Load the tokenizer and model with bfloat16 precision and automatic device placement
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = Gemma3ForCausalLM.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# Initialize the story context with an introductory narrative.
context = (
    "You are going to receive story context and player responses."
    "Only generate story responses and end your sentences with a full stop.\n"
    "Story: You're walking in a dark, dense forest at midnight. "
    "There's no one around and your phone is dead. "
    "Out of the corner of your eye, you spot him – Shia LaBeouf, "
    "the Hollywood actor, and he's covered in blood.\n"
)
print(context.strip())  # Display the initial narrative

# Start the interactive game loop
while True:
    player_action = input("Player: ").strip()
    if player_action.lower() in ["quit", "exit"]:
        print("You have exited the game. Goodbye!")
        break
    if not player_action:
        continue  # Skip empty input
    
    # Append the player's action and the prompt indicator for the story
    context += f"Player: {player_action}\n"
    context += "Story:"  # Prompt for the model continuation
    
    # Tokenize the entire current context and move tensors to the model's device
    model_inputs = tokenizer(context, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    # Generate a continuation using the template method: generate new tokens and extract them based on input length.
    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=100,  # number of new tokens to generate
        )
        # Extract only the portion of tokens generated beyond the original context length
        generated_ids = generation[0][input_len:]
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Print and append the generated continuation
    print("Story: " + continuation)
    context += " " + continuation + "\n"
