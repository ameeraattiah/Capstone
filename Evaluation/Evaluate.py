import json
import os
import torch
import re
import string
import nltk
import gc
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import wordpunct_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from scipy.stats import norm
import numpy as np

nltk.download('punkt')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = "Hugging_Face_Token"

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ["HF_TOKEN"]
)

parser = argparse.ArgumentParser()
parser.add_argument('--academic', type=str, required=True)
parser.add_argument('--non_academic', type=str, required=True)
parser.add_argument('--use_cached', action='store_true')
parser.add_argument('--max_samples', type=int, default=3000)
parser.add_argument('--offset', type=int, default=0)
args = parser.parse_args()
# Create a new 'plots' folder with incremental numbering (plots1, plots2, plots3, etc.)
plots_base_dir = os.path.join(os.path.dirname(__file__), "plots")

i = 1
while os.path.exists(f"{plots_base_dir}{i}"):
    i += 1

plots_dir = f"{plots_base_dir}{i}"
os.makedirs(plots_dir)

print(f"[📁] Saving plots in folder: {plots_dir}")


def extract_grammar_score(output):
    match = re.search(r"درجة القواعد والفصاحة[:：]?\s*(\d(?:[.,]\d)?)\s*/\s*5", output)
    if match:
        score_str = match.group(1).replace(",", ".")  # normalize comma to dot
        return float(score_str)
    return None


def grammar_llm_score(text):
    prompt = f"""
فيما يلي مقتطف نصي، رجاءً قيّم جودة القواعد النحوية وفصاحة الأسلوب لهذا النص بشكل صارم ودقيق، باستخدام مقياس من 0 إلى 5. خذ في الاعتبار المعايير التالية:

- الاستخدام الصحيح لقواعد النحو (مثل الإعراب، الأزمنة، التذكير والتأنيث، التراكيب)
- وضوح وتماسك الجمل
- خلو النص من الأخطاء الإملائية واللغوية
- ترابط الأفكار وسلاسة الانتقال بينها
- البنية العامة للجمل والفقرات

🛑 إذا احتوى النص على أي من الآتي، يجب تخفيض الدرجة:
- لغة عامية أو دعائية غير فصيحة
- جمل غير سليمة نحويًا
- غياب علامات الترقيم أو ضعف التنظيم

النص:
{text}

📌 أجب بالسطر التالي فقط بدون أي شرح، وبنفس الصيغة التالية تمامًا:
"درجة القواعد والفصاحة: <رقم من 0 إلى 5> / 5"
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
    inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).to(device)
    gen_config = GenerationConfig(max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], generation_config=gen_config)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_grammar_score(decoded)

def compute_metrics(text):
    text_clean = re.sub(rf"[{string.punctuation}،؛؟]", "", text)
    words = wordpunct_tokenize(text_clean)
    sentences = nltk.sent_tokenize(text)
    num_words = len(words)
    vocab_size = len(set(words))
    type_token_ratio = vocab_size / num_words if num_words else 0
    avg_sentence_len = num_words / len(sentences) if sentences else 0
    avg_word_len = sum(len(w) for w in words) / num_words if num_words else 0
    complex_word_ratio = len([w for w in words if len(w) >= 6]) / num_words if num_words else 0
    return type_token_ratio, vocab_size, avg_sentence_len, avg_word_len, complex_word_ratio

def evaluate_file(input_path, label):
    filename = f"{label.lower()}_cached.json" if args.use_cached else input_path
    if args.use_cached and not os.path.exists(filename):
        raise FileNotFoundError(f"Cached file '{filename}' not found. Run without --use_cached to create it.")

    with open(filename, "r", encoding="utf-8") as f:
        raw = f.read()
        clean = re.sub(r'[\x00-\x1f\x7f]', '', raw)
        data = json.loads(clean)
    start = args.offset
    end = start + args.max_samples
    data = data[start:end]

    def is_text_clean(text):
    # Remove non-Arabic/English letters and measure length
        cleaned = re.sub(r"[^a-zA-Z\u0600-\u06FF]", "", text)
        return len(cleaned) > 20  # Must have at least 20 good letters


    grammar_scores = []
    ttr_list, vocab_list = [], []
    avg_sent_lens, avg_word_lens, complex_ratios = [], [], []

    for entry in tqdm(data, desc=f"Evaluating {label}"):
        text = entry.get("text", "")
        if not is_text_clean(text):
            print(f"[⚠️ Skipped] Too short or messy: {text[:60]}...")
            grammar_scores.append(0.0)  # Auto-assign 0/5 if very messy
            ttr, vocab, avg_sen_len, avg_w_len, cwr = compute_metrics(text)
            ttr_list.append(ttr)
            vocab_list.append(vocab)
            avg_sent_lens.append(avg_sen_len)
            avg_word_lens.append(avg_w_len)
            complex_ratios.append(cwr)
            continue
        if "grammar_score_llm" in entry:
            score = entry["grammar_score_llm"]
        else:
            score = grammar_llm_score(text)
            entry["grammar_score_llm"] = score  # Save it
        grammar_scores.append(score)

        if score is None:
            print(f"[⚠️ Warning] No grammar score extracted for entry: {text[:60]}...")

        ttr, vocab, avg_sen_len, avg_w_len, cwr = compute_metrics(text)
        ttr_list.append(ttr)
        vocab_list.append(vocab)
        avg_sent_lens.append(avg_sen_len)
        avg_word_lens.append(avg_w_len)
        complex_ratios.append(cwr)

    # Save new cached file
    with open(f"{label.lower()}_cached.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "label": label,
        "grammar": grammar_scores,
        "ttr": ttr_list,
        "vocab": vocab_list,
        "avg_sent_len": avg_sent_lens,
        "avg_word_len": avg_word_lens,
        "complex_ratio": complex_ratios
    }
def plot_distribution_separately(data_dict, metric, xlabel, base_filename):
    for d in data_dict:
        values = [v for v in d[metric] if v is not None]
        if not values:
            continue

        label = d['label']
        mean = np.mean(values)
        std = np.std(values)

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=20, density=True, alpha=0.6, edgecolor='black')

        if std > 0:
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            plt.plot(x, p, 'k', linewidth=2)
        else:
            print(f"[⚠️ Warning] Standard deviation is zero for {label} - {metric}. Skipping normal curve plot.")

        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(f"{xlabel} Distribution - {label}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base_filename}_{label.lower().replace(' ', '_')}.png"))
        plt.close()
def plot_distribution_combined(data_dict, metric, xlabel, filename):
    plt.figure(figsize=(8, 5))
    for d in data_dict:
        values = [v for v in d[metric] if v is not None]
        if not values:
            continue

        label = d['label']
        mean = np.mean(values)
        std = np.std(values)

        plt.hist(values, bins=20, alpha=0.4, label=f"{label} Histogram", edgecolor='black', density=True)

        if std > 0:
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            plt.plot(x, p, linewidth=2, label=f"{label} Normal Curve")
        else:
            print(f"[⚠️ Warning] Standard deviation is zero for {label} - {metric}. Skipping normal curve plot.")

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(f"{xlabel} Distribution (Combined)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def print_score_distribution(label, grammar_scores):
    # Round scores to nearest integer
    rounded_scores = [round(float(score)) for score in grammar_scores if score is not None]
    counter = Counter(rounded_scores)

    print("__________________________________________________")
    print(f"\n📊 GRAMMAR SCORE DISTRIBUTION — {label} —")
    print(f"Total samples processed: {len(rounded_scores)}\n")
    for score in range(6):
        print(f"Score {score}: {counter.get(score, 0)} samples")


def print_summary(label, data):
    grammar_scores = [s for s in data['grammar'] if s is not None]  # ✅ Fix here

    print("__________________________________________________")
    print(f"\n📊 FINAL METRIC AVERAGES — {label} —")

    if grammar_scores:
        print(f"Grammar LLM Score (avg): {sum(grammar_scores) / len(grammar_scores):.2f}")
    else:
        print("Grammar LLM Score (avg): ⚠️ No valid grammar scores found")

    print(f"Type-Token Ratio (avg): {sum(data['ttr']) / len(data['ttr']):.2f}")
    print(f"Vocabulary Size (avg): {sum(data['vocab']) / len(data['vocab']):.1f}")
    print(f"Average Sentence Length (avg): {sum(data['avg_sent_len']) / len(data['avg_sent_len']):.1f}")
    print(f"Average Word Length (avg): {sum(data['avg_word_len']) / len(data['avg_word_len']):.1f}")
    print(f"Complex Word Ratio (avg): {sum(data['complex_ratio']) / len(data['complex_ratio']):.2f}")

if __name__ == "__main__":
    academic_data = evaluate_file(args.academic, "Academic")
    non_academic_data = evaluate_file(args.non_academic, "Non-Academic")
    all_data = [academic_data, non_academic_data]

    print("__________________________________________________")
    print("\n Model Used: ", MODEL_NAME)
    print("Academic Samples: ", len(academic_data["grammar"]))
    print("Non-Academic Samples: ", len(non_academic_data["grammar"]))


    metrics = [
        ("grammar", "Grammar LLM Score", "grammar_score"),
        ("ttr", "Type-Token Ratio", "type_token_ratio"),
        ("vocab", "Vocabulary Size", "vocab_size"),
        ("avg_sent_len", "Average Sentence Length", "avg_sentence_length"),
        ("avg_word_len", "Average Word Length", "avg_word_length"),
        ("complex_ratio", "Complex Word Ratio", "complex_word_ratio"),
    ]

    for metric, xlabel, filename in metrics:
        plot_distribution_separately(all_data, metric, xlabel, filename)  # Plot 1 and Plot 2
        plot_distribution_combined(all_data, metric, xlabel, f"{filename}_combined.png")  # Plot 3

    print("__________________________________________________")
    print("\n\n📊 SUMMARY")
    print_summary("Academic", academic_data)

    print_summary("Non-Academic", non_academic_data)

    print("__________________________________________________")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ All done.")
    print("__________________________________________________")
