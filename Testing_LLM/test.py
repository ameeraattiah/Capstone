import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import re

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
print(f"Using device: {device}")

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  
)

# Example extract for evaluation
example_extract = """
## **درس الفيزياء: الجاذبية الأرضية**

### **المقدمة**
الجاذبية هي القوة التي تجذب الأجسام نحو الأرض. تمثل هذه القوة جزءًا أساسيًا من الفيزياء وتمت دراستها لأول مرة من قبل إسحاق نيوتن.

### **القوانين الأساسية للجاذبية**
قانون نيوتن للجاذبية ينص على أن كل جسم في الكون يجذب غيره بقوة تتناسب مع كتلته وتتناقص مع مربع المسافة بينهما.
**الصيغة الرياضية لقانون الجاذبية:**
\[
F = G \frac{m_1 \times m_2}{r^2}
\]
حيث:
- **F** هي قوة الجاذبية،
- **G** هو ثابت الجاذبية،
- **m1 و m2** هما كتلتا الجسمين،
- **r** هو المسافة بينهما.
"""

# Structured Chat Prompt
arabic_prompt = f"""
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
{example_extract}

*التقييم النهائي:
يرجى تقديم إجابة دقيقة ومباشرة وفقًا للمعايير أعلاه:
1.	قدم تبريرًا واضحًا ومباشرًا يوضح سبب اختيارك لهذه الدرجة بناءً على محتوى النص وليس فقط معايير التقييم العامة.
2.	استخدم التنسيق التالي للنتيجة النهائية: ( التقييم التعليمي : <مجموع النقاط>/ 5)

<|assistant|>
استنادًا إلى المعايير، التقييم هو:
"""

# Tokenize input
inputs = tokenizer(
    arabic_prompt,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=1000  
).to(device)

inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).to(device)

# Set generation configuration
generation_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.6,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.15
)

# Generate evaluation
raw_output = tokenizer.decode(
    model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], generation_config=generation_config)[0],
    skip_special_tokens=True
)


def extract_score_and_evaluation(output):
    # Locate evaluation section
    evaluation_match = re.search(r"(استنادًا إلى المعايير، التقييم هو:.*)", output, re.DOTALL)
    evaluation_text = evaluation_match.group(1).strip() if evaluation_match else "لا يوجد تقييم متاح."

    # Find all numbers in the evaluation text
    numbers = re.findall(r"\b\d+\b", evaluation_text)

    # Convert numbers to integers
    numbers = [int(num) for num in numbers]

    if len(numbers) >= 2:
        # Take the first two numbers and select the smaller one
        score = min(numbers[:2])
    elif len(numbers) == 1:
        # If only one number is found, use it
        score = numbers[0]
    else:
        # If no numbers are found, default to 0
        score = 0

    return score, evaluation_text


# Apply extraction function
score, evaluation = extract_score_and_evaluation(raw_output)

# Print results
dash_line = "-" * 100
print(dash_line)
print(f"INPUT EXTRACT:\n{example_extract}")
print(dash_line)
print(f"RAW OUTPUT:\n{raw_output}")
print(dash_line)
print(f"EXTRACTED SCORE: {score}")
print(dash_line)
print(f"MODEL-GENERATED EVALUATION:\n{evaluation}")
print(dash_line)