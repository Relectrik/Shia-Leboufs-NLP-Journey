import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from emotion_classifier import get_emotion
import pandas as pd

# Set the checkpoint to Qwen2.5 1.5B Instruct model
ckpt = "Qwen/Qwen2.5-1.5B-Instruct"

# Load the tokenizer and model with trust_remote_code enabled.
# (Adjust torch_dtype if necessary; here we use float16 for faster inference.)
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Read each line from "context.txt" as initial inspiration.
original_context = ""
file_path = "context.txt"
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            original_context += line.strip() + "\n"
except FileNotFoundError:
    print(f"File {file_path} not found. Starting with an empty context.")

story_genre = input("What genre of story would you like to create? (e.g., horror, romance, etc.): ").strip()

# Initialize the story context using the file content as inspiration.
context = (
    f"Use this song as inspiration to generate a {story_genre} story: {original_context}\n"
    "Make sure your responses are full sentences before a newline. You need to tailor it for a player's response as they will be a character in the story.\n"
    "Story:"
)

# Tokenize the current context and move tensors to the device.
model_inputs = tokenizer(context, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

# Generate an initial continuation.
with torch.inference_mode():
    generation = model.generate(
        **model_inputs,
        max_new_tokens=100,
    )
# Extract the tokens generated after the original context.
generated_ids = generation[0][input_len:]
continuation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
# Limit to the first generated line for clarity.
continuation = continuation.split("\n")[0]
print("Story: " + continuation)
context += " " + continuation + "\n"
preferences: pd.DataFrame = pd.DataFrame(columns=["Context", "Response", "Emotion"])
# Start the interactive game loop.
for i in range(5):
    player_action = input("Player: ").strip()
    preferences.loc[len(preferences)] = pd.Series({"Context": continuation, "Response": player_action, "Emotion": get_emotion(player_action)})
    # If the player wants to exit the game
    if player_action.lower() in ["quit", "exit"]:
        print("You have exited the game. Goodbye!")
        break
    if not player_action:
        continue  # Skip empty input.
    
    # Append player's action and add a prompt for the next part of the story.
    context += f"Player: {player_action}\n"
    context += "Story:"  # Marker for the continuation prompt.
    
    # Tokenize updated context.
    model_inputs = tokenizer(context, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=300,
        )
        generated_ids = generation[0][input_len:]
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    continuation = continuation.split("\n")[0]  # Only use the first generated line.
    print("Story: " + continuation)
    context += " " + continuation + "\n"

preferences.to_csv("preferences.csv", index=False)  # Save preferences to CSV file.