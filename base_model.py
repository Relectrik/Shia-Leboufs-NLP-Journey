import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----- MODEL AND TOKENIZER SETUP -----
# Load the GPT-2 small model and tokenizer from Hugging Face
model_name = "gpt2"  # GPT-2 small (124M parameters)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # Switch model to evaluation mode (disables dropout, etc.)

# Set device to Apple MPS (Metal Performance Shaders) if available, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# ----- INITIAL GAME CONTEXT -----
# This is the starting point of the interactive horror story
context = (
    "Story: You're walking in a dark, dense forest at midnight. "
    "There's no one around and your phone is dead. "
    "Out of the corner of your eye, you spot him – Shia LaBeouf, "
    "the Hollywood actor, and he's covered in blood.\n"
)

print(context.strip())  # Display the intro story once at the start

# ----- GAME LOOP -----
# This loop allows the player to interactively influence the story
while True:
    # Prompt the player for an action/response
    player_action = input("Player: ").strip()

    # If the player wants to exit the game
    if player_action.lower() in ["quit", "exit"]:
        print("You have exited the game. Goodbye!")
        break

    # If the player gives empty input, skip generation
    if player_action == "":
        continue

    # Add the player's input to the context
    context += f"Player: {player_action}\n"
    context += "Story:"  # Append the cue for the model to continue the story

    # Encode the context as input for the model
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)

    # Generate a continuation of the story using the model
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 500,  # Allow up to 500 new tokens
        do_sample=True,            # Enable sampling (not greedy decoding)
        top_p=0.9,                 # Nucleus sampling for diverse outputs
        temperature=1.0,           # Higher temp for more randomness
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Extract only the generated part (after the original context)
    generated_ids = output_ids[0][input_ids.shape[1]:]
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Display the generated story continuation
    print("Story: " + continuation.strip())

    # Update the context with the newly generated content
    context += " " + continuation.strip() + "\n"
