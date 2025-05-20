import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import nltk
import string
from nltk.tokenize import wordpunct_tokenize
import os
import re
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Setup
nltk.download('punkt')
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
print(f"Using device: {device}")
sys.stdout.flush()

os.environ["HF_TOKEN"] = "Hugging_Face_Token"

model_names = [
    "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct"
]

grammar_model_name = "meta-llama/Llama-3.1-8B-Instruct"


# Define test cases for different scores
arabic_test_cases = {
    "1": """
انا يروح سوق مع اخي امبارح نشتري خضار وفواكه لكن ما في شي كثير هناك وكمان ما نعرف وين نروح بعدين اخي قال نمشي بس سياره كانت واقفه في نص شارع وما نقدر نعدي وكان في ناس كثير تصرخ واحنا خفنا ورجعنا مشي للبيت بسرعه كتير ما اكلنا شي وجلسنا نوم بدون نغسل يدنا او نغير هدومنا.
""",  # Grammar Score: 0 — incorrect verb conjugation, tense inconsistency, lack of punctuation, fragmented sentences, poor structure

    "2": """
ذهبت مع أخي إلى السوق بالأمس لشراء بعض الخضروات والفواكه، ولكن واجهتنا مشكلة عندما تعطلت السيارة بشكل مفاجئ في منتصف الطريق. حاولنا الاتصال بميكانيكي لكنه لم يكن متاحًا في ذلك الوقت، مما اضطرنا إلى الانتظار لفترة طويلة. بعد إصلاح السيارة، عدنا إلى المنزل في وقت متأخر، وكنا مرهقين لدرجة أننا لم نحضر العشاء كالمعتاد. اكتفينا بتناول بعض البسكويت وشرب الشاي، ثم ذهبنا للنوم.
""",  # Grammar Score: 3 — mostly correct grammar, understandable, some repetitive or less polished phrasing

    "3": """
في صباح يوم أمس، خرجتُ برفقة أخي إلى السوق لشراء مجموعة من الخضروات والفواكه الطازجة استعدادًا لأسبوع جديد من الطهي المنزلي. وأثناء سيرنا في الطريق السريع، توقفت سيارتنا بشكل مفاجئ بسبب خلل ميكانيكي في المحرك. حاولنا تهدئة الوضع والتواصل مع خدمة المساعدة على الطريق، التي وصلت بعد ثلاثين دقيقة. بعد إصلاح الخلل، واصلنا طريقنا نحو السوق، وأكملنا عملية الشراء بهدوء. عند عودتنا إلى المنزل، قمنا بترتيب المشتريات، ثم حضّرنا وجبة خفيفة، وجلسنا نتناولها بهدوء قبل أن نخلد إلى النوم مبكرًا بعد يوم مليء بالتحديات.
"""   # Grammar Score: 5 — fluent MSA, cohesive paragraph, excellent use of connectors, consistent tense and proper structure
}
# Full Arabic test cases
arabic_test_cases2 = {
    "1": """
عنوان اعلان يكون احسن تجربة يكون هاتف للمستقبل!

هو يكون انت يريد جوال ذكي تغير الحياة؟ لا بحث كمان! جوال "الترا تكنلوجي برو" جديد يعطي تجربة لا مثل مع شاشة حلوة، اداء فوق، وتصميم كويس يمسك العيون!

تصوير صور مش زيها عشان كاميرا قوية يشتغل بالذكاء صناعي، انت يقدر تسوي تصوير حلو لحظات بجودة عالي فوق الخيال!

بطارية يعيش كتير – مافي خوف شحن جوالك الحين! بطارية جديد يعطيك وقت كثير من اشتغال بدون يوقف.

عرض خاص وقت محدود! خذ خصم 30% اذا شراء دحين! لا تروح فرصة، وكن اول ناس يستفيد من تكنولوجيا هذا!
""",
    "2": """
عنوان المقال: تأثير الجاذبية الأرضية على حياة الناس اليومية

الجاذبية هي قوة تجذب الأشياء إلى الأرض، وتعد من الظواهر الفيزيائية المهمة التي يتحكم في حركة الكواكب والأجرام. قوانين الجاذبية تم اكتشافها أول مرة من العالم إسحاق نيوتن، هو وضع أساس لفهمنا العلمي لهذا القوة.

الجاذبية تلعب دور كبير في تثبيت الغلاف الجوي، حيث تحفظ الهواء من الذهاب إلى الفضاء. كذلك، تؤثر على المد والجزر في البحر والمحيطات، وهذا يؤدي إلى تغيير في مستوى المياه نتيجة جاذبية بين الأرض والقمر.

في الحياة اليومية، نحس الجاذبية وقت المشي أو القفز أو وقت وقوع الأشياء. بدون الجاذبية، صعب أن نعيش على الأرض، والحركة تكون مثل رواد الفضاء وقت انعدام الوزن.

بالرغم من أهمية الجاذبية، في كثير من الظواهر مازالت غير مفهومة تمامًا. مثلاً، العلماء يدرسون تفاعل الجاذبية مع القوى الثانية مثل الطاقة السوداء، وإذا ممكن نستعملها مستقبلًا في أشياء علمية أو تكنولوجية.
""",
    "3": """
عنوان الدرس: الدوائر الكهربائية – مكوناتها وقوانينها الأساسية

1. تعريف الدوائر الكهربائية
الدائرة الكهربائية هي مسار مغلق يتدفق من خلاله التيار الكهربائي نتيجة وجود فرق جهد بين طرفي المصدر الكهربائي. يتم التحكم في سريان التيار باستخدام مكونات مثل المقاومات، المكثفات، الملفات، والمفاتيح الكهربائية. تُستخدم الدوائر الكهربائية في الأجهزة الإلكترونية، أنظمة الطاقة، السيارات، والعديد من التطبيقات الهندسية الأخرى.

2. مكونات الدائرة الكهربائية
تتكوّن أي دائرة كهربائية من العناصر الأساسية التالية:

- المصدر الكهربائي (مثل البطارية أو المولّد الكهربائي) الذي يوفّر الجهد اللازم لتدفق التيار.
- المقاومات الكهربائية التي تحدّ من تدفق التيار وتتحكم في توزيعه.
- المكثفات التي تخزّن الشحنات الكهربائية لفترة قصيرة وتُستخدم في تنظيم التيار والجهد.
- الملفّات الكهربائية التي تُستخدم في الدوائر الحثّية والمولّدات والمحركات الكهربائية.
- المفتاح الكهربائي الذي يتحكم في فتح أو غلق الدائرة.
- الأسلاك الموصلة التي تنقل التيار بين المكونات المختلفة للدائرة.

3. قوانين الدوائر الكهربائية
تعتمد دراسة الدوائر الكهربائية على قوانين فيزيائية أساسية، منها:

أولًا: قانون أوم
ينص قانون أوم على أن العلاقة بين الجهد (V) والتيار (I) والمقاومة (R) في الدائرة الكهربائية تُعطى بالمعادلة التالية:
V = I × R
حيث:
- V هو فرق الجهد (بالفولت).
- I هو التيار الكهربائي (بالأمبير).
- R هي المقاومة الكهربائية (بالأوم).

ثانيًا: قوانين كيرشوف
تُستخدم قوانين كيرشوف في تحليل الدوائر الكهربائية المعقّدة، وتشمل:
- قانون كيرشوف للتيار (KCL): مجموع التيارات الداخلة إلى أي نقطة تفرّع في الدائرة يساوي مجموع التيارات الخارجة منها.
- قانون كيرشوف للجهد (KVL): مجموع الجهود حول أي مسار مغلق في الدائرة يساوي صفرًا.

4. تطبيقات الدوائر الكهربائية في الحياة العملية
- الإضاءة المنزلية: تُستخدم الدوائر الكهربائية لتوصيل المصابيح والتحكم في تشغيلها.
- الأنظمة الإلكترونية: مثل الهواتف المحمولة وأجهزة الكمبيوتر التي تعتمد على دوائر إلكترونية معقّدة.
- أنظمة الطاقة المتجددة: تستخدم الألواح الشمسية والمحولات الكهربائية دوائر متقدّمة لتحويل الطاقة الشمسية إلى كهرباء قابلة للاستخدام.
- السيارات الكهربائية: تعتمد على دوائر كهربائية للتحكم في المحركات والبطاريات وأنظمة الأمان.

5. تمرين تطبيقي:
السؤال: إذا كان لديك دائرة كهربائية تحتوي على بطارية جهدها 12 فولت ومقاومة مقدارها 4 أوم، احسب مقدار التيار الكهربائي المار في الدائرة باستخدام قانون أوم.
الإجابة:
I = V / R = 12 ÷ 4 = 3 أمبير.

خاتمة:
تُعد الدوائر الكهربائية أساس الأنظمة الحديثة التي نعتمد عليها يوميًا، بدءًا من الأجهزة المنزلية البسيطة وحتى الأنظمة المتقدّمة في المركبات الفضائية. إن فهم القوانين الأساسية مثل قانون أوم وقوانين كيرشوف يمكّن المهندسين من تصميم دوائر أكثر كفاءة وابتكار تقنيات جديدة لمستقبل الطاقة والإلكترونيات.
"""
}

def compute_lexical_diversity_score(text):
    text_clean = re.sub(rf"[{string.punctuation}]", "", text)
    words = wordpunct_tokenize(text_clean)
    if "؟" in text_clean:
        sentences = text_clean.split("؟")
    elif "۔" in text_clean:
        sentences = text_clean.split("۔")
    elif "." in text_clean:
        sentences = text_clean.split(".")
    else:
        sentences = text_clean.splitlines()

    num_words = len(words)
    num_sentences = len([s for s in sentences if s.strip()]) or 1
    vocab_size = len(set(words))
    type_token_ratio = vocab_size / num_words


    score = 0

    if type_token_ratio > 0.3:
        score += 1
    if vocab_size > num_words * 0.5:
        score += 1
    return {
        "type_token_ratio": type_token_ratio,
        "vocab_size": vocab_size,
        "lexical_score_out_of_2": score
    }


def compute_readability_metrics(text):
    # Remove punctuation
    text_clean = re.sub(rf"[{string.punctuation}،؛؟]", "", text)

    # Tokenize
    words = wordpunct_tokenize(text_clean)
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences) or 1
    num_words = len(words) or 1
    num_chars = sum(len(word) for word in words)

    avg_sentence_length = num_words / num_sentences
    avg_word_length = num_chars / num_words

    # Arabic "complex word" approximation (words ≥ 6 letters)
    complex_words = [word for word in words if len(word) >= 6]
    complex_word_ratio = len(complex_words) / num_words

    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "complex_word_ratio": complex_word_ratio
    }

def score_readability_metrics(readability):
    scores = {}

    # Avg sentence length
    asl = readability["avg_sentence_length"]
    if asl <= 12:
        scores["avg_sentence_length_score"] = 2
    elif 13 <= asl <= 20:
        scores["avg_sentence_length_score"] = 1
    else:
        scores["avg_sentence_length_score"] = 0

    # Avg word length
    awl = readability["avg_word_length"]
    if awl <= 4.0:
        scores["avg_word_length_score"] = 2
    elif 4.1 <= awl <= 5.5:
        scores["avg_word_length_score"] = 1
    else:
        scores["avg_word_length_score"] = 0

    # Complex word ratio
    cwr = readability["complex_word_ratio"]
    if cwr <= 0.20:
        scores["complex_word_ratio_score"] = 2
    elif 0.21 <= cwr <= 0.35:
        scores["complex_word_ratio_score"] = 1
    else:
        scores["complex_word_ratio_score"] = 0

    # Total score (out of 6)
    scores["total_readability_score"] = (
        scores["avg_sentence_length_score"]
        + scores["avg_word_length_score"]
        + scores["complex_word_ratio_score"]
    )

    return scores



def extract_grammar_score(output):  # 👈 Add this
    import re
    match = re.search(r"درجة القواعد والفصاحة:\s*(\d(?:\.\d)?)\s*/\s*5", output)
    return float(match.group(1)) if match else None


dash_line = "-" * 100

# Iterate through models
for model_name in model_names:
    print(f"\n{dash_line}\nTESTING MODEL: {model_name}\n{dash_line}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ["HF_TOKEN"]
    )

    # 💡 Use same model for grammar scoring
    llama_tokenizer = tokenizer
    llama_model = model

    for label, example in arabic_test_cases2.items():
        print(f"\n{dash_line}\nINPUT ({label}):\n{example}\n{dash_line}")

        edu_prompt = f"""
فيما يلي مقتطف نصي، قم بتقييم مدى فائدته كمحتوى تعليمي باستخدام نظام تقييم مكون من 5 نقاط تراكمية، وفقًا للمعايير التالية:

1 نقطة: إذا كان النص يحتوي على بعض المعلومات ذات الصلة بالموضوعات التعليمية، حتى لو لم يكن هدفه الأساسي التعليم أو احتوى على عناصر غير أكاديمية مثل الإعلانات أو المواد الترويجية.
2 نقاط: إذا تناول النص بعض الجوانب التعليمية ولكنه غير متعمق أو قد يخلط بين المعلومات التعليمية والعناصر غير ذات الصلة، أو يقدم نظرة عامة بسيطة دون تنظيم واضح.
3 نقاط: إذا كان النص مناسبًا للاستخدام التعليمي، ويحتوي على مفاهيم أساسية ذات صلة بالمناهج الدراسية، حتى لو لم يكن شاملاً تمامًا أو احتوى على بعض المعلومات الإضافية غير الضرورية.
4 نقاط: إذا كان النص منظمًا وواضحًا، ويمكن استخدامه كمرجع تعليمي جيد. يشبه محتوى كتاب دراسي أو درس أكاديمي مبسط، يحتوي على معلومات دقيقة مع بعض الأمثلة والتمارين، حتى لو كانت هناك بعض التفاصيل المفقودة.
5 نقاط: إذا كان النص عالي الجودة تعليميًا، يحتوي على محتوى شامل، بشروحات واضحة، وأمثلة وتمارين تجعله مناسبًا تمامًا كمصدر تعليمي رسمي. يجب أن يكون النص متكاملاً وخاليًا من المعلومات غير الضرورية أو التكرار غير المفيد.
النص المطلوب تقييمه:
{example}

بعد فحص المقتطف:

يرجى كتابة السطر التالي فقط، بدون شرح:
"التقييم التعليمي: <رقم من 0 إلى 5> / 5"
        """

        grammar_prompt = f"""
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
{example}

📌 أجب بالسطر التالي فقط بدون أي شرح، وبنفس الصيغة التالية تمامًا (لا تغيّر الكلمات):

"درجة القواعد والفصاحة: <رقم من 0 إلى 5> / 5"

"""

        gen_config = GenerationConfig(
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Run educational prompt
        edu_inputs = tokenizer(edu_prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        edu_inputs["attention_mask"] = (edu_inputs["input_ids"] != tokenizer.pad_token_id).to(device)
        edu_output = tokenizer.decode(
            model.generate(edu_inputs["input_ids"], attention_mask=edu_inputs["attention_mask"], generation_config=gen_config)[0],
            skip_special_tokens=True
        )

        # Run grammar prompt
        llama_inputs = llama_tokenizer(grammar_prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        llama_inputs["attention_mask"] = (llama_inputs["input_ids"] != llama_tokenizer.pad_token_id).to(device)
        grammar_output = llama_tokenizer.decode(
            llama_model.generate(llama_inputs["input_ids"], attention_mask=llama_inputs["attention_mask"], generation_config=gen_config)[0],
            skip_special_tokens=True
        )

        # Output
        print(f"🧠 LLM Educational Raw Output:\n{edu_output}")
        print(f"🧠 LLaMA Grammar Raw Output:\n{grammar_output}")

        # Try extracting grammar score
        grammar_score = extract_grammar_score(grammar_output)
        if grammar_score is not None:
            print(f"✅ Extracted Grammar Score: {grammar_score} / 5")
        else:
            print("❌ Failed to extract grammar score (unexpected format).")

        # Stats
        lexical_stats = compute_lexical_diversity_score(example)
        print(f"🔤 Lexical Diversity Score: {lexical_stats['lexical_score_out_of_2']} / 2")
        print(f"   → Type-Token Ratio: {lexical_stats['type_token_ratio']:.2f}")
        print(f"   → Vocabulary Size: {lexical_stats['vocab_size']}")


        readability = compute_readability_metrics(example)
        readability_scores = score_readability_metrics(readability)

        print(f"📖 Readability Metrics:")
        print(f"   → Avg Sentence Length: {readability['avg_sentence_length']:.2f}")
        print(f"   → Avg Word Length: {readability['avg_word_length']:.2f}")
        print(f"   → Complex Word Ratio: {readability['complex_word_ratio']:.2%}")
        print(f"📈 Readability Score (out of 6): {readability_scores['total_readability_score']}")



# CLEANUP after both generations
del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()


print(f"\n{dash_line}\n✅ TESTING COMPLETE\n{dash_line}")

# Final cleanup of LLaMA model used for grammar
del llama_model
del llama_tokenizer
torch.cuda.empty_cache()
gc.collect()
