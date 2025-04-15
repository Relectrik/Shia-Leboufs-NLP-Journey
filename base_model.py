import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained GPT-2 small model and tokenizer
model_name = "gpt2"  # GPT-2 small (124M) by OpenAI, available on HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # set model to evaluation mode (no training)

# (Optional) Use MPS (Apple GPU) if available for speed
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Initialize the game context with the song-based intro story
context = ("Story: You're walking in a dark, dense forest at midnight. "
           "There's no one around and your phone is dead. "
           "Out of the corner of your eye, you spot him – Shia LaBeouf, "
           "the Hollywood actor, and he's covered in blood.\n")  # Intro narrative
# We end with \n to indicate the story narration ended and player can act next.

print(context.strip())  # Show the intro story to the player

# Start the game loop
while True:
    # Prompt player for an action
    player_action = input("Player: ").strip()
    if player_action.lower() in ["quit", "exit"]:
        print("You have exited the game. Goodbye!")
        break
    if player_action == "":
        # If no input, skip generation (or continue loop)
        continue

    # Add the player's action to the context
    context += f"Player: {player_action}\n"
    context += "Story:"  # Prepare the prompt for the model to continue the story

    # Encode the context and generate a continuation
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    # Generate continuation (we limit max length to avoid runaway)
    output_ids = model.generate(input_ids, 
                                max_length=input_ids.shape[1] + 500,  # generate up to 100 tokens more
                                do_sample=True,       # use sampling for creativity
                                top_p=0.9,            # nucleus sampling for diversity
                                temperature=1.0,      # temperature 1.0 for fairly random output
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    # Extract the newly generated portion (skip the input prompt part)
    generated_ids = output_ids[0][input_ids.shape[1]:]  
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Print the AI-generated story continuation
    print("Story: " + continuation.strip())

    # Add the continuation to context for the next loop iteration
    context += " " + continuation.strip() + "\n"
