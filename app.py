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
            "haber": "Haber is used as an auxiliary verb or to express existence.",
            "tener": "Tener is used to express possession, age, or obligation."
        }
    },
    "por_para": {
        "title": "Por vs. Para",
        "verbs": ["por", "para"],
        "definitions": {
            "por": "Por is used for reasons, durations, exchanges, and means of communication.",
            "para": "Para is used for purposes, deadlines, and recipients."
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
    return render_template("exercise.html", exercise=exercise, exercise_key=exercise_key)

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
        categories=json.dumps(categories)  
    )





# -----------------------
# OpenAI logic
# -----------------------

counts = defaultdict(int)  # This ensures all verbs are initialized dynamically
generated_sentences = set()  # Track generated sentences to avoid duplicates
recent_verb_forms = set() # Track recently used verb forms to avoid immediate repeats
MAX_RECENT = 3  # Adjust this value to allow variety while avoiding immediate repeats



categories = {
    "ser_estar": {
        "ser": ["Identity", "Characteristics", "Origin/Nationality", "Time/Date", "Material/Ownership"],
        "estar": ["Location", "Emotions/Conditions", "Ongoing Actions", "Results of Actions"]
    },
    "haber_tener": {
        "haber": ["Existence", "Experience"],
        "tener": ["Possession", "Obligation"]
    }
}



def get_weighted_choice(exercise_key):
    """
    Dynamically adjust probabilities for verbs in any exercise to ensure balance.
    If one verb is overused, the probability of choosing the other increases.
    """

    # Retrieve the verbs for the selected exercise
    verb_pairs = {
        "ser_estar": ["ser", "estar"],
        "haber_tener": ["haber", "tener"]
    }

    if exercise_key not in verb_pairs:
        return None  # Handle invalid exercises gracefully

    verbs = verb_pairs[exercise_key]

    # Initialize counts dynamically if not present
    for verb in verbs:
        if verb not in counts:
            counts[verb] = 0

    total = sum(counts[verb] for verb in verbs)

    if total == 0:
        return random.choice(verbs)  # 50/50 at start

    # Define sensitivity range for balance correction
    max_diff = 5  # Controls how aggressively balance shifts
    diff = counts[verbs[0]] - counts[verbs[1]]

    # Adjust probability dynamically based on past selections
    verb_1_prob = max(20, min(80, 50 - (diff / max_diff) * 50))
    verb_2_prob = 100 - verb_1_prob  # The rest goes to the other verb

    print(f"Adjusted probabilities → {verbs[0]}: {verb_1_prob:.2f}%, {verbs[1]}: {verb_2_prob:.2f}%")
    return random.choices(verbs, weights=[verb_1_prob, verb_2_prob], k=1)[0]




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







def generate_conjugation_sentence(exercise_key, tenses):
    """
    Calls OpenAI to generate a Spanish sentence requiring a verb from the selected exercise
    in a user-selected tense. Ensures variety in conjugation forms.
    """
    max_retries = 10
    attempts = 0

    global categories, conjugations, generated_sentences
    

    # ✅ Ensure the selected exercise exists
    if exercise_key not in categories:
        print(f"❌ ERROR: '{exercise_key}' not found in categories!")
        return {"error": "Invalid exercise key"}

    # ✅ If no tenses are provided, fallback to all tenses
    if not tenses:
        tenses = [
            "present", "past_imperfect", "future", "conditional",
            "present_perfect", "past_perfect", "future_perfect",
            "present_subjunctive", "past_imperfect_subjunctive", "present_perfect_subjunctive", "past_perfect_subjunctive",
            "affirmative_imperative", "negative_imperative"
        ]

    # ✅ Normalize tenses (ensure consistency with conjugation dictionary)
    tenses = [t.lower().replace(" ", "_") for t in tenses]

    while attempts < max_retries:
        try:
            print(f"📌 Generating conjugation sentence for: {exercise_key}")

            # ✅ Randomly choose one of the two verbs in the selected exercise
            verbs = list(categories[exercise_key].keys())
            correct_answer = random.choice(verbs)

            # ✅ Ensure AI selects a valid conjugation
            chosen_tense = random.choice(tenses)

            if chosen_tense not in conjugations[correct_answer]:
                print(f"⚠ Warning: {chosen_tense} not found in conjugations! Defaulting to present.")
                chosen_tense = "present"

            possible_forms = conjugations[correct_answer][chosen_tense]

            # Remove recently used verb forms from selection
            filtered_forms = [form for form in possible_forms if form not in recent_verb_forms]

            # ✅ If filtering removes all options, reset recent list
            if not filtered_forms:
                recent_verb_forms.clear()
                filtered_forms = possible_forms

            chosen_conjugation = random.choice(filtered_forms)

            # ✅ Track used verb form
            recent_verb_forms.add(chosen_conjugation)
            if len(recent_verb_forms) > MAX_RECENT:
                recent_verb_forms.pop()  # Remove the oldest verb form to allow variety

            # ✅ Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences focused on verb conjugation."},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a **unique** Spanish sentence with a blank (___) where the correct verb is '{correct_answer}'.\n\n"
                            "**RULES:**\n"
                            f"- The verb **must** be in one of these user-selected tenses: {', '.join(tenses)}.\n"
                            f"- The blank (___) must be filled with the conjugated form of '{correct_answer}' in the chosen tense: {chosen_tense}.\n"
                            f"- Ensure proper subject-verb agreement.\n"
                            f"- Example conjugations for '{correct_answer}' in {chosen_tense}: {', '.join(conjugations[correct_answer][chosen_tense])}.\n"
                            "- **DO NOT** overuse common forms (e.g., 'es', 'está'). Ensure variety in conjugations.\n\n"
                            "**OUTPUT JSON FORMAT:**\n"
                            "{\n"
                            f"  \"sentence\": \"Example sentence with a blank ___\",\n"
                            f"  \"correct\": \"{correct_answer}\",\n"
                            f"  \"tense\": \"{chosen_tense}\",\n"
                            f"  \"verb_form\": \"{chosen_conjugation}\"\n"
                            "}\n\n"
                            "**Return only the JSON.**"
                        )
                    },
                ],
                max_tokens=50,
                temperature=1.1,
            )

            response_content = response["choices"][0]["message"]["content"].strip()
            data = json.loads(response_content.replace("'", "\""))

            # ✅ Validate if AI response matches selected tense
            if data["tense"].lower() not in tenses:
                print(f"❌ AI used an invalid tense: {data['tense']}. Retrying...")
                attempts += 1
                continue  

            # ✅ Prevent duplicate sentences
            if data["sentence"] in generated_sentences:
                print(f"⚠ Duplicate sentence detected. Retrying...")
                attempts += 1
                continue  

            generated_sentences.add(data["sentence"])
            counts[data["correct"]] += 1

            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}. Retrying...")
            attempts += 1
        except Exception as e:
            print(f"❌ Error generating sentence: {e}. Retrying...")
            attempts += 1

    print("⚠ Max retries reached. Returning fallback sentence.")
    return {
        "sentence": "Yo ___ feliz.",
        "correct": "estar",
        "tense": "present",
        "verb_form": "estoy"
    }



def generate_reason_sentence(exercise_key):
    """
    Calls OpenAI to generate a Spanish sentence requiring the correct verb based on the selected exercise.
    The AI should randomly choose either verb1 or verb2 for the blank, ensuring variety and avoiding duplicates.
    """
    max_retries = 10
    attempts = 0

    global categories, conjugations, generated_sentences

    if exercise_key not in categories:
        print(f"❌ ERROR: '{exercise_key}' not found in categories!")
        return {"error": "Invalid exercise key"}

    used_verb_forms = set()  # Tracks used conjugations to avoid repeats

    while attempts < max_retries:
        try:
            print(f"📌 Generating reasoning sentence for: {exercise_key}")

            # ✅ Randomly pick one verb from the exercise
            verbs = list(categories[exercise_key].keys())  # e.g., ["ser", "estar"] or ["haber", "tener"]
            correct_answer = random.choice(verbs)

            # ✅ Select a category based on the chosen verb
            chosen_category = random.choice(categories[exercise_key][correct_answer])

            # ✅ Choose a conjugation from the correct verb's present tense
            available_conjugations = [form for form in conjugations[correct_answer]["present"] if form not in used_verb_forms]
            
            # ✅ If all conjugations have been used, reset to allow variety
            if not available_conjugations:
                used_verb_forms.clear()
                available_conjugations = conjugations[correct_answer]["present"]

            chosen_conjugation = random.choice(available_conjugations)
            used_verb_forms.add(chosen_conjugation)

            # ✅ Call OpenAI API with the correct format
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences."},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a **unique** Spanish sentence with a blank (___) where the correct verb is '{correct_answer}'.\n\n"
                            "**RULES:**\n"
                            f"- The sentence **MUST** fit into the category: {chosen_category}.\n"
                            f"- The blank (___) must be replaced with a conjugated form of '{correct_answer}' matching the subject.\n"
                            f"- The conjugated form **MUST NOT** be a duplicate of recently used conjugations.\n"
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
                    },
                ],
                max_tokens=60,  # Allow enough tokens for full JSON response
                temperature=1.1,  # Adds variety while keeping accuracy
            )

            print("📌 OpenAI raw response:", response)

            # ✅ Extract AI response content
            response_content = response["choices"][0]["message"]["content"].strip()

            # ✅ Validate JSON response before parsing
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()  # Remove markdown syntax

            data = json.loads(response_content.replace("'", "\""))  # Convert single quotes to double

            # ✅ Ensure correct response structure
            if data["correct"] not in categories[exercise_key]:
                print(f"❌ AI returned an invalid verb: {data['correct']}. Retrying...")
                attempts += 1
                continue
            if data["category"] not in categories[exercise_key][data["correct"]]:
                print(f"❌ AI returned an invalid category: {data['category']}. Retrying...")
                attempts += 1
                continue
            if data["verb_form"] in used_verb_forms:
                print(f"⚠ Duplicate verb form detected: {data['verb_form']}. Retrying...")
                attempts += 1
                continue

            # ✅ Check for duplicate sentences
            if data["sentence"] in generated_sentences:
                print("⚠ Duplicate sentence detected! Retrying...")
                attempts += 1
                continue  # Retry with a new sentence

            # ✅ Store unique sentence and verb form
            generated_sentences.add(data["sentence"])
            used_verb_forms.add(data["verb_form"])

            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}. Retrying...")
            attempts += 1
        except Exception as e:
            print(f"❌ Error generating sentence: {e}. Retrying...")
            attempts += 1

    print("⚠ Max retries reached. Returning fallback sentence.")
    return {
        "sentence": "El libro ___ en la mesa.",
        "correct": "estar",
        "category": "Location",
        "verb_form": "está"
    }


# -----------------------

@app.route("/api/<exercise_key>", methods=["GET", "POST"])
def api_exercise(exercise_key):
    """
    GET -> Returns a new sentence for either reasoning or conjugation (based on request params).
    POST -> Checks the user's answer and returns feedback.
    """
    if request.method == "POST":
        user_answer = request.json.get("answer")
        correct_answer = request.json.get("correct")
        feedback = "Correct!" if user_answer == correct_answer else "Incorrect!"
        return jsonify({"feedback": feedback})

    mode = request.args.get("mode", "reasoning")

    # ✅ Fix: Extract exercise_key properly and pass it
    if mode == "conjugation":
        tenses_raw = request.args.get("tenses", "[]")
        print(f"📌 Raw tenses from request: {tenses_raw}")

        try:
            tenses = json.loads(tenses_raw) if tenses_raw else []
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            tenses = []  # Fallback to empty list

        print(f"📌 Parsed tenses: {tenses}")
        new_sentence = generate_conjugation_sentence(exercise_key, tenses)  # ✅ Pass `exercise_key` explicitly
    else:
        new_sentence = generate_reason_sentence(exercise_key)

    return jsonify(new_sentence)








if __name__ == "__main__":
    app.run(debug=True)
