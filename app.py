from flask import Flask, render_template, request, jsonify
import openai
import os
import json
import random
from dotenv import load_dotenv
from collections import defaultdict


# Initialize the usage counter
usage_counter = defaultdict(int)  # Tracks how many times "ser" and "estar" have been used

load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

openai.api_key = api_key

app = Flask(__name__)

# -----------------------
# Define exercises here
# -----------------------
exercises = {
    "ser_estar": {
        "title": "Ser vs. Estar",
        "verbs": ["ser", "estar"],
        "definitions": {
            "ser": "Ser is used to describe identity, permanent states, or inherent characteristics.",
            "estar": "Estar is used to describe temporary states, emotions, or locations."
        }
    },
    "haber_tener": {
        "title": "Haber vs. Tener",
        "verbs": ["haber", "tener"],
        "definitions": {
            "haber": "Haber is used to express past actions or to express existence.",
            "tener": "Tener is used to express possession, physical sensations, age, or obligation."
        }
    },
    "por_para": {
       "title": "Por vs. Para",
        "prepositions": ["por", "para"],
        "definitions": {
            "por": "Por is used for cause/reason, duration, movement through a place, exchange, means of transportation, and passive voice agents.",
            "para": "Para is used for purpose, destination, deadlines, recipients, opinions, and employment."
        }
    }
}


# -----------------------
# Helper functions
# -----------------------

def convert_sets_to_lists(obj):
    """Recursively convert any set() in obj to a list."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    else:
        return obj

# -----------------------
# Routes
# -----------------------

@app.route("/")
def home():
    """
    This route will serve as the homepage, showing all exercises
    defined in the `exercises` dictionary.
    """
    return render_template("index.html", exercises=exercises)


@app.route("/exercise/<exercise_key>")
def exercise_page(exercise_key):
    exercise = exercises.get(exercise_key)
    
    if not exercise:
        return "Exercise not found", 404
    
    return render_template("exercise.html", exercise=exercise, exercise_key=exercise_key, categories=categories)

@app.route("/practice")
def practice_page():
    mode = request.args.get("mode", "reasoning")
    exercise_key = request.args.get("exercise", "ser_estar")  
    exercise = exercises.get(exercise_key)  # Fetch exercise data

    print(f"🔍 DEBUG: Retrieved exercise ({exercise_key}):", exercise)

    if not exercise or not isinstance(exercise, dict):
        print(f"❌ ERROR: Exercise '{exercise_key}' is missing or invalid!")
        return "Exercise not found", 404  # Return a clear error

    tenses = request.args.get("tenses", "[]")
    try:
        tenses = json.loads(tenses)
    except json.JSONDecodeError:
        tenses = []

    return render_template(
        "practice.html",
        mode=mode,
        exercise=exercise,
        exercise_key=exercise_key,
        tenses=tenses,
        categories=json.dumps(categories),  
        conjugations=json.dumps(conjugations)  # Send conjugations to frontend
    )






# -----------------------
# OpenAI logic
# -----------------------

counts = defaultdict(int)  # This ensures all verbs are initialized dynamically
generated_sentences = set()  # Track generated sentences to avoid duplicates
recent_verb_forms = set() # Track recently used verb forms to avoid immediate repeats
MAX_RECENT_VERBS = 3  # Adjust this value to allow variety while avoiding immediate repeats



categories = {
    "verb_exercises": {
        "ser_estar": {
            "ser": ["Identity", "Characteristics", "Origin/Nationality", "Time/Date", "Material/Ownership"],
            "estar": ["Location", "Emotions/Conditions", "Ongoing Actions", "Results of Actions"]
        },
        "haber_tener": {
            "haber": ["Existence", "Past Actions"],
            "tener": ["Possession", "Obligation", "Physical Sensations", "Age"]
        }
    },
   "preposition_exercises": {
        "por_para": {
        "por": [
            "Cause/Reason", 
            "Means of Communication or Transportation", 
            "Passing Through a Place", 
            "Time spent", 
            "Exchange", 
            "Who did something"
        ],
        "para": [
            "Purpose/Goal", 
            "Where something is going", 
            "Deadline", 
            "Recipient", 
            "Opinion/Comparison", 
            "Who you work for"
        ]
    }
 }
}


def get_weighted_choice(exercise_key):
    """
    Dynamically adjust probabilities for words in any exercise to ensure balance.
    If one word is overused, the probability of choosing the other increases.
    """

    # ✅ Define word pairs for each exercise type
    word_pairs = {
        "ser_estar": ["ser", "estar"],
        "haber_tener": ["haber", "tener"],
        "por_para": ["por", "para"],  
    }

    if exercise_key not in word_pairs:
        print(f"❌ ERROR: '{exercise_key}' not found in word pairs!")
        return None  # Handle invalid exercises gracefully

    words = word_pairs[exercise_key]  # Retrieve the two words for the exercise

    # ✅ Initialize counts dynamically if not present
    for word in words:
        if word not in counts:
            counts[word] = 0

    total = sum(counts[word] for word in words)

    if total == 0:
        return random.choice(words)  # 50/50 at start

    # ✅ Define sensitivity range for balance correction
    max_diff = 5  # Controls how aggressively balance shifts
    diff = counts[words[0]] - counts[words[1]]

    # ✅ Adjust probability dynamically based on past selections
    word_1_prob = max(20, min(80, 50 - (diff / max_diff) * 50))
    word_2_prob = 100 - word_1_prob  # The rest goes to the other word

    print(f"Adjusted probabilities → {words[0]}: {word_1_prob:.2f}%, {words[1]}: {word_2_prob:.2f}%")
    return random.choices(words, weights=[word_1_prob, word_2_prob], k=1)[0]




# ✅ Conjugations dictionary (ensures AI correctly conjugates)
conjugations = {
    "ser": {
        # Indicative
        "present": ["soy", "eres", "es", "somos", "sois", "son"],
        "past_imperfect": ["era", "eras", "era", "éramos", "erais", "eran"],
        "future": ["seré", "serás", "será", "seremos", "seréis", "serán"],
        "conditional": ["sería", "serías", "sería", "seríamos", "seríais", "serían"],
        "present_perfect": ["he sido", "has sido", "ha sido", "hemos sido", "habéis sido", "han sido"],
        "past_perfect": ["había sido", "habías sido", "había sido", "habíamos sido", "habíais sido", "habían sido"],
        "future_perfect": ["habré sido", "habrás sido", "habrá sido", "habremos sido", "habréis sido", "habrán sido"],
        
        # Subjunctive
        "present_subjunctive": ["sea", "seas", "sea", "seamos", "seáis", "sean"],
        "past_imperfect_subjunctive": ["fuera", "fueras", "fuera", "fuéramos", "fuerais", "fueran"],
        "present_perfect_subjunctive": ["haya sido", "hayas sido", "haya sido", "hayamos sido", "hayáis sido", "hayan sido"],
        "past_perfect_subjunctive": ["hubiera sido", "hubieras sido", "hubiera sido", "hubiéramos sido", "hubierais sido", "hubieran sido"],
        
        # Imperative
        "affirmative_imperative": ["sé", "sea", "seamos", "sed", "sean"],
        "negative_imperative": ["no seas", "no sea", "no seamos", "no seáis", "no sean"]
    },
    "estar": {
        # Indicative
        "present": ["estoy", "estás", "está", "estamos", "estáis", "están"],
        "past_imperfect": ["estaba", "estabas", "estaba", "estábamos", "estabais", "estaban"],
        "future": ["estaré", "estarás", "estará", "estaremos", "estaréis", "estarán"],
        "conditional": ["estaría", "estarías", "estaría", "estaríamos", "estaríais", "estarían"],
        "present_perfect": ["he estado", "has estado", "ha estado", "hemos estado", "habéis estado", "han estado"],
        "past_perfect": ["había estado", "habías estado", "había estado", "habíamos estado", "habíais estado", "habían estado"],
        "future_perfect": ["habré estado", "habrás estado", "habrá estado", "habremos estado", "habréis estado", "habrán estado"],
        
        # Subjunctive
        "present_subjunctive": ["esté", "estés", "esté", "estemos", "estéis", "estén"],
        "past_imperfect_subjunctive": ["estuviera", "estuvieras", "estuviera", "estuviéramos", "estuvierais", "estuvieran"],
        "present_perfect_subjunctive": ["haya estado", "hayas estado", "haya estado", "hayamos estado", "hayáis estado", "hayan estado"],
        "past_perfect_subjunctive": ["hubiera estado", "hubieras estado", "hubiera estado", "hubiéramos estado", "hubierais estado", "hubieran estado"],
        
        # Imperative
        "affirmative_imperative": ["está", "esté", "estemos", "estad", "estén"],
        "negative_imperative": ["no estés", "no esté", "no estemos", "no estéis", "no estén"]
    },
    "haber": {
        # Indicative
        "present": ["he", "has", "ha", "hemos", "habéis", "han"],
        "past_imperfect": ["había", "habías", "había", "habíamos", "habíais", "habían"],
        "future": ["habré", "habrás", "habrá", "habremos", "habréis", "habrán"],
        "conditional": ["habría", "habrías", "habría", "habríamos", "habríais", "habrían"],
        
        # Subjunctive
        "present_subjunctive": ["haya", "hayas", "haya", "hayamos", "hayáis", "hayan"],
        "past_imperfect_subjunctive": ["hubiera", "hubieras", "hubiera", "hubiéramos", "hubierais", "hubieran"],
        "present_perfect_subjunctive": ["haya habido", "hayas habido", "haya habido", "hayamos habido", "hayáis habido", "hayan habido"],
        "past_perfect_subjunctive": ["hubiera habido", "hubieras habido", "hubiera habido", "hubiéramos habido", "hubierais habido", "hubieran habido"],

        # No imperative forms for "haber" (only used as an auxiliary verb)
    },
    "tener": {
        # Indicative
        "present": ["tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen"],
        "past_imperfect": ["tenía", "tenías", "tenía", "teníamos", "teníais", "tenían"],
        "future": ["tendré", "tendrás", "tendrá", "tendremos", "tendréis", "tendrán"],
        "conditional": ["tendría", "tendrías", "tendría", "tendríamos", "tendríais", "tendrían"],
        "present_perfect": ["he tenido", "has tenido", "ha tenido", "hemos tenido", "habéis tenido", "han tenido"],
        "past_perfect": ["había tenido", "habías tenido", "había tenido", "habíamos tenido", "habíais tenido", "habían tenido"],
        "future_perfect": ["habré tenido", "habrás tenido", "habrá tenido", "habremos tenido", "habréis tenido", "habrán tenido"],
        
        # Subjunctive
        "present_subjunctive": ["tenga", "tengas", "tenga", "tengamos", "tengáis", "tengan"],
        "past_imperfect_subjunctive": ["tuviera", "tuvieras", "tuviera", "tuviéramos", "tuvierais", "tuvieran"],
        "present_perfect_subjunctive": ["haya tenido", "hayas tenido", "haya tenido", "hayamos tenido", "hayáis tenido", "hayan tenido"],
        "past_perfect_subjunctive": ["hubiera tenido", "hubieras tenido", "hubiera tenido", "hubiéramos tenido", "hubierais tenido", "hubieran tenido"],
        
        # Imperative
        "affirmative_imperative": ["ten", "tenga", "tengamos", "tened", "tengan"],
        "negative_imperative": ["no tengas", "no tenga", "no tengamos", "no tengáis", "no tengan"]
    }
}







recent_verb_forms = set()  # Tracks recently used conjugated forms
MAX_RECENT_VERBS = 3  # Prevents immediate repeats but allows variety

def generate_conjugation_sentence(exercise_key, tenses):
    """
    Calls OpenAI to generate a Spanish sentence requiring a verb from the selected exercise
    in a user-selected tense. Ensures variety in conjugation forms and includes subject pronoun retrieval.
    """
    max_retries = 10
    attempts = 0

    global categories, conjugations, generated_sentences, recent_verb_forms

    # ✅ Ensure the selected exercise is inside "verb_exercises"
    if exercise_key in categories["verb_exercises"]:
        category_data = categories["verb_exercises"]
    else:
        print(f"❌ ERROR: '{exercise_key}' not found in verb exercises!")
        return {"error": "Invalid exercise key"}

    # ✅ If no tenses are provided, fallback to all tenses
    if not tenses:
        tenses = [
            "present", "past_imperfect", "future", "conditional",
            "present_perfect", "past_perfect", "future_perfect",
            "present_subjunctive", "past_imperfect_subjunctive", "present_perfect_subjunctive", "past_perfect_subjunctive",
            "affirmative_imperative", "negative_imperative"
        ]
    tenses = [t.lower().replace(" ", "_") for t in tenses]

    while attempts < max_retries:
        try:
            print(f"📌 Generating conjugation sentence for: {exercise_key}")

            # ✅ Retrieve the two verbs associated with this exercise
            words = list(category_data[exercise_key].keys())
            correct_answer = random.choice(words)
            chosen_tense = random.choice(tenses)

            if chosen_tense not in conjugations[correct_answer]:
                print(f"⚠ Warning: {chosen_tense} not found in conjugations! Defaulting to present.")
                chosen_tense = "present"

            # ✅ Call OpenAI API with subject pronoun requirement
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences focused on verb conjugation."},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a **unique** Spanish sentence with a blank (___) where the correct verb is '{correct_answer}'.\n\n"
                            "**RULES:**\n"
                            f"- The verb **must** be in one of these tenses: {', '.join(tenses)}\n"
                            f"- The blank (___) must be filled with the conjugated form of '{correct_answer}' in the chosen tense: {chosen_tense}\n"
                            "- Include the explicit subject pronoun (e.g., Yo, Nosotros, Ellos)\n"
                            "- Ensure proper subject-verb agreement\n"
                            "- Return JSON with these keys: sentence, correct, tense, verb_form, subject_pronoun\n\n"
                            "**EXAMPLE:**\n"
                            "{\n"
                            '  "sentence": "Nosotros ___ en la oficina ahora mismo.",\n'
                            '  "correct": "estar",\n'
                            '  "tense": "present",\n'
                            '  "verb_form": "estamos",\n'
                            '  "subject_pronoun": "Nosotros"\n'
                            "}\n\n"
                            "**Return only the JSON.**"
                        )
                    },
                ],
                max_tokens=60,
                temperature=1.1,
            )

            # ✅ Extract response content and handle JSON formatting
            response_content = response["choices"][0]["message"]["content"].strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            elif response_content.startswith("```"):
                response_content = response_content[3:-3].strip()

            data = json.loads(response_content.replace("'", "\""))

            # ✅ Ensure AI response includes subject pronoun
            required_fields = ["sentence", "correct", "tense", "verb_form", "subject_pronoun"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # ✅ Prevent duplicate sentences
            if data["sentence"] in generated_sentences:
                print(f"⚠ Duplicate sentence detected. Retrying...")
                attempts += 1
                continue

            # ✅ Track unique sentence
            generated_sentences.add(data["sentence"])
            recent_verb_forms.add(data["verb_form"])
            if len(recent_verb_forms) > MAX_RECENT_VERBS:
                recent_verb_forms.pop()

            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}\nRaw response: {response_content}")
            attempts += 1
        except KeyError as e:
            print(f"❌ Missing field in response: {e}\nResponse: {response_content}")
            attempts += 1
        except Exception as e:
            print(f"❌ Error generating sentence: {str(e)}")
            attempts += 1

    # ✅ Return fallback sentence if AI fails
    print("⚠ Max retries reached. Returning fallback sentence.")
    return {
        "sentence": "Yo ___ feliz.",
        "correct": "estar",
        "tense": "present",
        "verb_form": "estoy",
        "subject_pronoun": "Yo"
    }


def generate_reason_sentence(exercise_key):
    """
    Calls OpenAI to generate a Spanish sentence requiring the correct word based on the selected exercise.
    If the exercise is verb-based, it generates a sentence requiring a verb.
    If the exercise is preposition-based (like Por vs. Para), it generates a sentence requiring one of the two prepositions.
    """
    max_retries = 10
    attempts = 0

    global categories, conjugations, generated_sentences, recent_verb_forms

    # ✅ Determine if the exercise is in verb_exercises or preposition_exercises
    is_verb_exercise = exercise_key in categories["verb_exercises"]
    is_preposition_exercise = exercise_key in categories["preposition_exercises"]

    if not is_verb_exercise and not is_preposition_exercise:
        print(f"❌ ERROR: '{exercise_key}' not found in either exercise category!")
        return {"error": "Invalid exercise key"}

    while attempts < max_retries:
        try:
            print(f"📌 Generating reasoning sentence for: {exercise_key}")

            if is_verb_exercise:
                # ✅ Handle verb-based exercises
                words = list(categories["verb_exercises"][exercise_key].keys())  # e.g., ["ser", "estar"]
                correct_answer = random.choice(words)
                chosen_category = random.choice(categories["verb_exercises"][exercise_key][correct_answer])
                
                # ✅ Pick a conjugation for the correct verb
                available_conjugations = [form for form in conjugations[correct_answer]["present"] if form not in recent_verb_forms]
                if not available_conjugations:
                    recent_verb_forms.clear()
                    available_conjugations = conjugations[correct_answer]["present"]
                chosen_conjugation = random.choice(available_conjugations)

                ai_prompt = (
                    f"Generate a **unique** Spanish sentence with a blank (___) where the correct verb is '{correct_answer}'.\n\n"
                    "**RULES:**\n"
                    f"- The sentence **MUST** fit into the category: {chosen_category}.\n"
                    f"- The blank (___) must be replaced with a conjugated form of '{correct_answer}' matching the subject.\n"
                    f"- Example conjugations for '{correct_answer}': {', '.join(conjugations[correct_answer]['present'])}\n"
                    "- Ensure natural variety in conjugations and sentence structures.\n\n"
                    "**OUTPUT JSON FORMAT:**\n"
                    "{\n"
                    f"  \"sentence\": \"Example sentence with a blank ___\",\n"
                    f"  \"correct\": \"{correct_answer}\",\n"
                    f"  \"category\": \"{chosen_category}\",\n"
                    f"  \"verb_form\": \"{chosen_conjugation}\"\n"
                    "}\n\n"
                    "**Return only the JSON.**"
                )

            elif is_preposition_exercise:
                # ✅ Handle preposition-based exercises
                words = list(categories["preposition_exercises"][exercise_key].keys())  # e.g., ["por", "para"]
                correct_answer = random.choice(words)
                chosen_category = random.choice(categories["preposition_exercises"][exercise_key][correct_answer])

                ai_prompt = (
                    f"Generate a **unique** Spanish sentence with a blank (___) where the correct preposition is '{correct_answer}'.\n\n"
                    "**RULES:**\n"
                    f"- The sentence **MUST** fit into the category: {chosen_category}.\n"
                    f"- The blank (___) must be replaced with '{correct_answer}' in a way that makes the sentence natural.\n"
                    f"- Ensure the sentence is a clear example of why '{correct_answer}' is correct.\n\n"
                    "**OUTPUT JSON FORMAT:**\n"
                    "{\n"
                    f"  \"sentence\": \"Example sentence with a blank ___\",\n"
                    f"  \"correct\": \"{correct_answer}\",\n"
                    f"  \"category\": \"{chosen_category}\"\n"
                    "}\n\n"
                    "**Return only the JSON.**"
                )

            # ✅ Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences."},
                    {"role": "user", "content": ai_prompt},
                ],
                max_tokens=60,
                temperature=1.1,
            )

            print("📌 OpenAI raw response:", response)

            # ✅ Extract AI response content
            response_content = response["choices"][0]["message"]["content"].strip()

            # ✅ Validate JSON response before parsing
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            data = json.loads(response_content.replace("'", "\""))

            # ✅ Ensure correct response structure
            if is_verb_exercise:
                if data["correct"] not in categories["verb_exercises"][exercise_key]:
                    print(f"❌ AI returned an invalid verb: {data['correct']}. Retrying...")
                    attempts += 1
                    continue

            elif is_preposition_exercise:
                if data["correct"] not in categories["preposition_exercises"][exercise_key]:
                    print(f"❌ AI returned an invalid preposition: {data['correct']}. Retrying...")
                    attempts += 1
                    continue

            # ✅ Check for duplicate sentences
            if data["sentence"] in generated_sentences:
                print("⚠ Duplicate sentence detected! Retrying...")
                attempts += 1
                continue  

            # ✅ Store unique sentence
            generated_sentences.add(data["sentence"])

            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}. Retrying...")
            attempts += 1
        except Exception as e:
            print(f"❌ Error generating sentence: {e}. Retrying...")
            attempts += 1

    print("⚠ Max retries reached. Returning fallback sentence.")
    return {
        "sentence": "¿Este regalo es ___ ti?",
        "correct": "para",
        "category": "Recipient"
    }


# -----------------------

@app.route("/api/<exercise_key>", methods=["GET", "POST"])
def api_exercise(exercise_key):
    """
    GET -> Returns a new sentence for either reasoning or conjugation (based on request params).
    POST -> Checks the user's answer and returns feedback.
    """
    print(f"📌 DEBUG: Received API request for exercise: {exercise_key}")

    mode = request.args.get("mode", "reasoning")

    # ✅ Determine whether exercise belongs to verbs or prepositions
    is_verb_exercise = exercise_key in categories["verb_exercises"]
    is_preposition_exercise = exercise_key in categories["preposition_exercises"]

    if not is_verb_exercise and not is_preposition_exercise:
        print(f"❌ ERROR: Invalid exercise key received: {exercise_key}")
        return jsonify({"error": "Invalid exercise key"}), 400

    if mode == "conjugation" and is_verb_exercise:
        tenses_raw = request.args.get("tenses", "[]")
        print(f"📌 DEBUG: Raw tenses from request: {tenses_raw}")

        try:
            tenses = json.loads(tenses_raw) if tenses_raw else []
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: JSON Decode Error: {e}")
            tenses = []  # Fallback to empty list

        print(f"📌 DEBUG: Parsed tenses: {tenses}")
        new_sentence = generate_conjugation_sentence(exercise_key, tenses)  
    else:
        new_sentence = generate_reason_sentence(exercise_key)

    return jsonify(new_sentence)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)

