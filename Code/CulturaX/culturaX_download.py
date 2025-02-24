from datasets import load_dataset
import huggingface_hub
import os
import json

# Increase timeout settings
huggingface_hub.utils._http.DEFAULT_TIMEOUT = 300  # Set timeout to 5 minutes

# Define dataset storage path
save_dir = "/ibex/user/abuhanjt/test/dataset/"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Load dataset using streaming mode
dataset = load_dataset("uonlp/CulturaX", "ar", split="train", streaming=True)

# Variables to manage file splitting
file_count = 1
sample_count = 0
batch_size = 1000000  # Save every 1M samples

# Open first file
output_file = os.path.join(save_dir, f"CulturaX_Arabic_{file_count}.json")
f = open(output_file, "w", encoding="utf-8")
f.write("[\n")  # Start JSON array

# Process dataset
for sample in dataset:
    json.dump(sample, f, ensure_ascii=False)
    sample_count += 1

    if sample_count % batch_size == 0:  # Every 1M samples, create a new file
        f.write("\n]")  # Close JSON array
        f.close()
        print(f"Saved {batch_size} samples to {output_file}")

        # Increment file count and open a new file
        file_count += 1
        output_file = os.path.join(save_dir, f"CulturaX_Arabic_{file_count}.json")
        f = open(output_file, "w", encoding="utf-8")
        f.write("[\n")  # Start new JSON array

    else:
        f.write(",\n")  # Add comma between entries

# Close last file properly
f.write("\n]")
f.close()
print(f"Final dataset split complete. {sample_count} total samples saved.")
