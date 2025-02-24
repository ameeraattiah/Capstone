## Import Libraries
import json
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
from transformers import T5Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sentencepiece import SentencePieceProcessor
import os
from transformers import AutoModelForCausalLM
from transformers import GenerationConfig
import sys
import re
import wandb
import deepspeed
import os
import numpy
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    print("MPI is not available, proceeding without it.")
    MPI = None
from sklearn.model_selection import train_test_split
from collections import Counter


print("SentencePiece is installed and ready to use.")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()  # Clears unused GPU memory
print(f"Using device: {device}")

# Load DeepSpeed configuration from JSON file
deepspeed_config_path = "/ibex/user/abuhanjt/test/deepspeed_config.json"
with open(deepspeed_config_path, "r") as ds_file:
    deepspeed_config = json.load(ds_file)

config = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",  # Mistral for annotation
    "fine_tune_model": "aubmindlab/bert-base-arabertv02",  # AraBERT for classification
    "threshold": 2,  # Minimum score for high-quality content
    "annotation_samples": 5000,  # Increase annotated samples for better learning
    "validation_samples": 1000,  # More validation samples for better evaluation
    "max_samples_to_fine_tune": 4000,  # More fine-tuning samples
    "epochs": 5,  
    "batch_size": 1, 
    "learning_rate": 1e-5,  # Adjust learning rate for large model
    "gradient_accumulation_steps": 8,  # Helps handle large models with small batch sizes
    "use_deepspeed": True,  # Enable DeepSpeed for efficient training
    # "use_fp16": True,  # Mixed precision to reduce memory usage
    "dataset_dir": "/ibex/user/abuhanjt/test/dataset/",  
    "output_dir": "/ibex/user/abuhanjt/test/output/",
    "deepspeed_config_path": deepspeed_config_path
}

# Initialize W&B
wandb.init(
    project="Academic_Specific",  
    name="experiment_243", 
    config=config 
)

# Ensure output directory exists
os.makedirs(config["output_dir"], exist_ok=True)

## Arabic Rubric Prompt
def get_arabic_prompt(text):
    return f"""
<|system|>
فيما يلي مقتطف من صفحة ويب. قم بتقييم مدى فائدته كمحتوى تعليمي باستخدام نظام تقييم مكون من 5 نقاط تراكمية، وفقًا للمعايير التالية:

 *معايير التقييم التراكمي (مع مرونة في التقدير بناءً على جودة المحتوى(:
•	أضف 1 نقطة: إذا كان النص يحتوي على بعض المعلومات الأساسية المفيدة، حتى لو كان يتضمن محتوى غير تعليمي مثل الإعلانات أو المواد الترويجية، لكنه لا يساهم في التعلم بشكل واضح.
•	أضف 2 نقاط: إذا كان النص يتناول عناصر ذات صلة بالتعليم، لكنه لا يلتزم تمامًا بالمعايير الأكاديمية، وقد يكون فيه مزيج من المعلومات المفيدة وغير المفيدة أو يقدم نظرة عامة دون تفاصيل كافية.
•	أضف 3 نقاط: إذا كان النص مناسبًا للاستخدام التعليمي، يحتوي على مفاهيم رئيسية ذات صلة بالمناهج المدرسية، وهو واضح، لكنه قد يفتقر إلى بعض التفاصيل أو يحتوي على معلومات إضافية ليست ضرورية.
•	أضف 4 نقاط: إذا كان النص منظمًا، واضحًا، ومتناسقًا، ويمكن استخدامه كمرجع تعليمي جيد. يشبه فصلًا دراسيًا من كتاب مدرسي، يحتوي على أمثلة وتمارين، ويقدم معلومات دقيقة، حتى لو كان هناك بعض التفاصيل المفقودة.
•	أضف 5 نقاط: إذا كان النص عالي الجودة تعليميًا، يمكن استخدامه مباشرة في التدريس، يحتوي على محتوى شامل، منظم بالكامل، بشروحات واضحة وأمثلة وتمارين، دون أي معلومات غير ضرورية أو غموض. يمكن أن يكون مثاليًا كجزء من مادة تعليمية رسمية.

إذا كان النص يبدو تعليميًا ولكنه لا يحصل على تقييم مرتفع، قم بمراعاة أسلوب تقديم المعلومات ومدى فائدتها الفعلية للطالب، بدلاً من الاعتماد فقط على اكتمال كل عنصر.

* النص المطلوب تقييمه:
{text}

*التقييم النهائي:
يرجى تقديم إجابة دقيقة ومباشرة وفقًا للمعايير أعلاه:
1.	قدم تبريرًا واضحًا ومباشرًا يوضح سبب اختيارك لهذه الدرجة بناءً على محتوى النص وليس فقط معايير التقييم العامة.
2.	استخدم التنسيق التالي للنتيجة النهائية: ( التقييم التعليمي : < مجموع النقاط> 5/)


<|assistant|>
استنادًا إلى المعايير، التقييم هو:
"""



## Custom Dataset Class
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
    

## Step 1: Load Dataset
def load_dataset(dataset_file):

    with open(dataset_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    cleaned_data = []
    for item in data:
        # Ensure 'text' exists and provide default values for missing metadata
        if "text" in item:
            cleaned_data.append({
                "text": item["text"],
                "metadata": {
                    "timestamp": item.get("timestamp", "N/A"), 
                    "url": item.get("url", "N/A"),
                    "source": item.get("source", "Unknown") 
                }
            })
        else:
            print(f"Warning: Skipping entry in {dataset_file} - Missing 'text' field")

    return cleaned_data

# Function to clean and separate score and evaluation
def extract_score_and_evaluation(output):
    #  Locate evaluation section first
    evaluation_match = re.search(r"(استنادًا إلى المعايير، التقييم هو:.*)", output, re.DOTALL)
    evaluation_text = evaluation_match.group(1).strip() if evaluation_match else "لا يوجد تقييم متاح."

    #  Find score **inside** the evaluation text
    score_match = re.search(r"التقييم التعليمي\s*:\s*\(*\s*(\d+)(?:\.\d+)?\s*/\s*5\s*\)*", evaluation_text)

    if score_match:
        score = int(score_match.group(1))  # Ensure score is an integer
    else:
        score = 0  #  If score is missing, default to 0 instead of None

    return score, evaluation_text

## Step 2: Annotate Data Locally and save it 

# Get SLURM Task ID  
slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
# Define unique save path per SLURM task
ANNOTATION_SAVE_PATH = f"/ibex/user/abuhanjt/test/output/balanced_annotations.json"
print(f"Annotations will be saved to: {ANNOTATION_SAVE_PATH}")

# Ensure the annotation file exists (creates an empty JSON file if missing)
if not os.path.exists(ANNOTATION_SAVE_PATH):
    with open(ANNOTATION_SAVE_PATH, "w", encoding="utf-8") as file:
        json.dump([], file, ensure_ascii=False, indent=4)  # Initialize empty list
    print(f"Created new annotation file: {ANNOTATION_SAVE_PATH}")

def annotate_samples(samples, tokenizer, model):
    annotated_data = []

    # **Check if there's a previous backup and load it safely**
    if os.path.exists(ANNOTATION_SAVE_PATH):
        try:
            with open(ANNOTATION_SAVE_PATH, "r", encoding="utf-8") as file:
                annotated_data = json.load(file)
            
            # Ensure `annotated_data` is a **list**
            if not isinstance(annotated_data, list):
                print("Warning: Annotation file is invalid. Resetting annotations.")
                annotated_data = []

            # Stop if we already reached 5000 annotations
            if len(annotated_data) >= 5000:
                print(f"Already 5,000 annotations completed. Skipping annotation.")
                return annotated_data
            else:
                print(f"Resuming annotation from {len(annotated_data)} samples.")

        except (json.JSONDecodeError, FileNotFoundError):
            print("Warning: Annotation file was corrupted. Resetting annotations.")
            annotated_data = []

    # Avoid duplicate processing
    processed_ids = {entry["id"] for entry in annotated_data}

    for idx, sample in enumerate(samples, start=1):
        if idx in processed_ids:
            continue  # Skip already annotated samples

        text = sample["text"]
        arabic_prompt = get_arabic_prompt(text)

        inputs = tokenizer(
            arabic_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=4000
        ).to(device)

        inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).to(device)

        generation_config = GenerationConfig(
            max_new_tokens=200,
            do_sample=True,
            temperature=0.5,  
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.25 
        )

        torch.cuda.empty_cache()

        raw_output = tokenizer.decode(
            model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                generation_config=generation_config
            )[0], 
            skip_special_tokens=True
        )

        score, evaluation = extract_score_and_evaluation(raw_output)

        annotated_sample = {
            "id": idx,  # Unique document ID
            "text": text,
            "scores": score,
            "metadata": sample["metadata"],
            "evaluation": evaluation  
        }

        annotated_data.append(annotated_sample)

        # Save progress after every sample
        with open(ANNOTATION_SAVE_PATH, "w", encoding="utf-8") as file:
            json.dump(annotated_data, file, ensure_ascii=False, indent=4)

        print(f"🔹 Saved annotation {idx}/{len(samples)}")

        # Log to W&B
        wandb.log({
            "sample_id": idx,
            "text": text,
            "predicted_score": score,
            "evaluation_summary": evaluation[:400],
        })

    print(f"Annotation complete. Total samples annotated: {len(annotated_data)}")
    return annotated_data



## Step 3: Fine-Tune AraBERT
def fine_tune_arabert(train_data, val_data, tokenizer, model):
    torch.cuda.empty_cache()

    texts_train = [item["text"] for item in train_data]
    labels_train = [item["scores"] for item in train_data]

    texts_val = [item["text"] for item in val_data]
    labels_val = [item["scores"] for item in val_data]

    train_dataset = CustomDataset(texts_train, labels_train, tokenizer)
    val_dataset = CustomDataset(texts_val, labels_val, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], 
        warmup_steps=500, 
        weight_decay=0.01, 
        max_grad_norm=1.0, 
        deepspeed=config["deepspeed_config_path"],
        lr_scheduler_type="cosine",  # Choose from ['linear', 'cosine', 'constant', etc.]
        bf16=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print(f"Starting fine-tuning... (Train: {len(train_data)}, Val: {len(val_data)})")
    trainer.train()
    print("Fine-tuning complete.")



## Step 4: Predict with Fine-Tuned AraBERT     
def predict_with_arabert(unlabeled_data, model, tokenizer):
    model.eval()
    model = deepspeed.init_inference(model, dtype=torch.bfloat16)  # Add DeepSpeed for inference

    model.to(device)
    predictions = []

    for sample in unlabeled_data:
        text = sample["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = logits.argmax(dim=-1).item()

        predictions.append({
            "text": text,
            "predicted_score": predicted_label,
            "metadata": sample["metadata"]
        })

    return predictions


## Step 5: Validate Model
def validate_model(validation_data, model, tokenizer):
    model.to(device)
    true_labels = []
    predicted_labels = []

    for item in validation_data:
        text = item["text"]
        true_labels.append(item["scores"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = logits.argmax(dim=-1).item()
        predicted_labels.append(predicted_label)

    f1 = f1_score(true_labels, predicted_labels, average="macro")
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Validation F1 Score: {f1:.2f}")
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(f"Validation Precision: {precision:.2f}")
    print(f"Validation Recall: {recall:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)


## Step 6: Filter Dataset
def filter_dataset(annotated_data, threshold):
    return [
        doc for doc in annotated_data
        if doc.get("predicted_score", 0) >= threshold
    ]


## The Main Pipeline
def main_pipeline():
    # Ensure a file is provided
    if len(sys.argv) < 2:
        raise ValueError("No dataset file provided!")

    # Step 1: Load the dataset from a specified JSON file
    dataset_args = [arg for arg in sys.argv if not arg.startswith("--")]
    if len(dataset_args) < 2:
        raise ValueError("No dataset file provided!")
    dataset_file = dataset_args[1] 

    print(f"Processing file: {dataset_file}")
    dataset = load_dataset(dataset_file)
    print(f"Loaded {len(dataset)} samples.")

    # Step 2: Select a subset of the dataset for annotation
    sample_data = dataset[:config["annotation_samples"]]

    # Load tokenizer for Mistral-7B (text generation)
    tokenizer_mistral = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer_mistral.pad_token = tokenizer_mistral.eos_token  

    # Load model for annotation (Mistral)
    model_mistral = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,  # FP16 to reduce memory usage
        device_map="auto",  
        trust_remote_code=True  
    )

    # Annotate data using Mistral
    annotated_data = annotate_samples(sample_data, tokenizer_mistral, model_mistral)
    print(f"Annotated {len(annotated_data)} samples.")

    # Step 3: Load tokenizer and model for fine-tuning AraBERT
    tokenizer_arabert = AutoTokenizer.from_pretrained(config["fine_tune_model"])
    tokenizer_arabert.pad_token = tokenizer_arabert.eos_token if tokenizer_arabert.eos_token else "[PAD]"

    model_arabert = AutoModelForSequenceClassification.from_pretrained(
        config["fine_tune_model"], 
        num_labels=6
    ).to(device)

    # Filter out invalid scores first
    filtered_data = [item for item in annotated_data if 0 <= item["scores"] <= 5]

    # Ensure dataset has sufficient samples
    if len(filtered_data) < config["max_samples_to_fine_tune"] + config["validation_samples"]:
        raise ValueError(f"Not enough data for training & validation. Found {len(filtered_data)} samples.")

    # Extract labels for stratified splitting
    labels = [item["scores"] for item in filtered_data]

    # ✅ Stratified train-validation split (ensures class balance)
    train_data, val_data = train_test_split(
        filtered_data,
        test_size=config["validation_samples"] / len(filtered_data),
        stratify=labels,  # Maintains class proportions
        random_state=42  # Ensures reproducibility
    )

    # Check new label distributions
    train_label_counts = Counter([item["scores"] for item in train_data])
    val_label_counts = Counter([item["scores"] for item in val_data])

    print(f"Training samples: {len(train_data)} | Validation samples: {len(val_data)}")
    print("New Training Label Distribution:", train_label_counts)
    print("New Validation Label Distribution:", val_label_counts)

    # Step 5: Fine-tune the model using the annotated data
    fine_tune_arabert(train_data, val_data, tokenizer_arabert, model_arabert)

    # Step 6: Use the fine-tuned model to predict the remaining dataset
    remaining_data = dataset[config["annotation_samples"]:]
    predictions = predict_with_arabert(remaining_data, model_arabert, tokenizer_arabert)
    print(f"Predicted {len(predictions)} samples with fine-tuned AraBERT.")

    # Step 7: Validate the fine-tuned model on a subset of the annotated data
    validation_data = annotated_data[:config["validation_samples"]]
    validate_model(validation_data, model_arabert, tokenizer_arabert)

    # Step 8: Filter the predictions to include only high-quality samples
    filtered_data = filter_dataset(predictions, config["threshold"])
    print(f"Filtered dataset contains {len(filtered_data)} high-quality samples.")

    # Step 9: Save processed data
    output_file = os.path.join(config["output_dir"], f"processed_{os.path.basename(dataset_file)}")
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)
    print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    main_pipeline()
    print("Running academic dataset processing...")

wandb.finish()
