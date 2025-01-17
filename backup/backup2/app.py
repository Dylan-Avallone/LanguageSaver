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
        "definitions": {
            "ser": "Ser is used to describe identity, permanent states, or inherent characteristics.",
            "estar": "Estar is used to describe temporary states, emotions, or locations."
        },
        "use_cases": {
            "ser": [
                "Soy profesor. (I am a teacher.)",
                "Ella es alta. (She is tall.)"
            ],
            "estar": [
                "Estoy cansado. (I am tired.)",
                "El libro está en la mesa. (The book is on the table.)"
            ]
        }
    },
    # If you have more exercises, add them here
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
    """
    Dynamic route for an exercise page.
    e.g., /exercise/ser_estar
    """
    exercise = exercises.get(exercise_key)
    if not exercise:
        return "Exercise not found", 404
    return render_template("exercise.html", exercise=exercise)


@app.route("/practice")
def practice_page():
    """
    Practice page for Ser vs. Estar. 
    You can expand this to handle other exercises if you want.
    """
    return render_template("practice.html")


# -----------------------
# OpenAI logic
# -----------------------



# Initialize counters for `ser` and `estar`
counts = {"ser": 0, "estar": 0}

# Initialize a set to store previously generated sentences
generated_sentences = set()

# Ser Categories
ser_categories = [
    "Identity",
    "Characteristics",
    "Origin/Nationality",
    "Time/Date",
    "Material/Ownership"
]
# Define Estar Categories
estar_categories = [
    "Location",
    "Emotions/Conditions",
    "Ongoing Actions",
    "Results of Actions"
]



def get_weighted_choice():
    """
    Dynamically adjust probabilities for `ser` and `estar` to ensure balance.
    If one verb is overused, the probability of choosing the other increases.
    """
    total = counts["ser"] + counts["estar"]

    if total == 0:
        return random.choice(["ser", "estar"])  # 50/50 at start

    # Define a sensitivity range for balance correction
    max_diff = 5  # Controls how aggressively balance shifts
    diff = counts["ser"] - counts["estar"]

    # Adjust probability dynamically based on past selections
    ser_prob = max(20, min(80, 50 - (diff / max_diff) * 50))
    estar_prob = 100 - ser_prob  # The rest goes to estar

    print(f"Adjusted probabilities → Ser: {ser_prob:.2f}%, Estar: {estar_prob:.2f}%")
    return random.choices(["ser", "estar"], weights=[ser_prob, estar_prob], k=1)[0]

# sentence function for conjugation practice
def generate_conjugation_sentence():
    """
    Calls OpenAI to generate a Spanish sentence requiring 'ser' or 'estar' in a specific indicative tense.
    Ensures variety in conjugation forms.
    """
    max_retries = 10
    attempts = 0

    indicative_tenses = [
        "present", "imperfect", "future", "conditional",
        "present perfect", "past perfect", "future perfect"
    ]

    while attempts < max_retries:
        try:
            print("Calling OpenAI ChatCompletion for Conjugation Sentence...")

            # Decide which verb to use (ser or estar)
            correct_answer = get_weighted_choice()

            # Select a tense randomly from indicative tenses
            chosen_tense = random.choice(indicative_tenses)

            # Map tense to correct conjugation list
            conjugations = {
                "ser": {
                    "present": ["soy", "eres", "es", "somos", "sois", "son"],
                    "imperfect": ["era", "eras", "era", "éramos", "erais", "eran"],
                    "future": ["seré", "serás", "será", "seremos", "seréis", "serán"],
                    "conditional": ["sería", "serías", "sería", "seríamos", "seríais", "serían"],
                    "present perfect": ["he sido", "has sido", "ha sido", "hemos sido", "habéis sido", "han sido"],
                    "past perfect": ["había sido", "habías sido", "había sido", "habíamos sido", "habíais sido", "habían sido"],
                    "future perfect": ["habré sido", "habrás sido", "habrá sido", "habremos sido", "habréis sido", "habrán sido"]
                },
                "estar": {
                    "present": ["estoy", "estás", "está", "estamos", "estáis", "están"],
                    "imperfect": ["estaba", "estabas", "estaba", "estábamos", "estabais", "estaban"],
                    "future": ["estaré", "estarás", "estará", "estaremos", "estaréis", "estarán"],
                    "conditional": ["estaría", "estarías", "estaría", "estaríamos", "estaríais", "estarían"],
                    "present perfect": ["he estado", "has estado", "ha estado", "hemos estado", "habéis estado", "han estado"],
                    "past perfect": ["había estado", "habías estado", "había estado", "habíamos estado", "habíais estado", "habían estado"],
                    "future perfect": ["habré estado", "habrás estado", "habrá estado", "habremos estado", "habréis estado", "habrán estado"]
                }
            }

            # Pick a conjugated form randomly from the chosen tense list
            chosen_conjugation = random.choice(conjugations[correct_answer][chosen_tense])

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences focused on verb conjugation."},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a **unique** Spanish sentence where the blank (___) is filled with the correct conjugated form of '{correct_answer}'.\n\n"
                            "**RULES:**\n"
                            "- The verb must be used in the '{chosen_tense}' indicative tense.\n"
                            f"- Use one of these conjugations: {', '.join(conjugations[correct_answer][chosen_tense])}.\n"
                            "- The sentence should sound natural and appropriate for the chosen tense.\n\n"
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
                temperature=1.1,  # Adds diversity while maintaining accuracy
            )

            print("OpenAI raw response:", response)

            # Extract AI response content
            response_content = response["choices"][0]["message"]["content"].strip()
            print("AI raw text content:\n", response_content)

            # ✅ Convert single quotes to double quotes for valid JSON parsing
            response_content = response_content.replace("'", "\"")
            
            # ✅ Parse the corrected JSON
            data = json.loads(response_content)
            print("Parsed result as dict:", data)

            # Ensure proper tense selection
            if data["tense"] not in indicative_tenses:
                raise ValueError("Incorrect tense detected. Retrying...")

            # Store sentence and update count tracking
            generated_sentences.add(data["sentence"])
            counts[data["correct"]] += 1

            return data

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("Attempting to fix formatting.")
            
            # Try to fix formatting and parse again
            try:
                response_content = response_content.replace("'", "\"")  # Convert all single quotes to double quotes
                data = json.loads(response_content)
                return data  # If successful, return corrected JSON
            except Exception as e:
                print(f"Fix attempt failed: {e}")

            attempts += 1
        except Exception as e:
            print(f"Error generating sentence: {e}")
            attempts += 1

    print("Max retries reached. Returning fallback sentence.")
    return {"sentence": "Yo ___ feliz.", "correct": "estar", "tense": "present", "verb_form": "estoy"}




def generate_reason_sentence():
    """
    Calls OpenAI to generate a Spanish sentence requiring 'ser' or 'estar',
    ensuring variety in conjugations.
    """
    max_retries = 10
    attempts = 0

    while attempts < max_retries:
        try:
            print("Calling OpenAI ChatCompletion...")

            # Decide which verb to use
            correct_answer = get_weighted_choice()
            chosen_category = (
                random.choice(ser_categories) if correct_answer == "ser" 
                else random.choice(estar_categories)
            )

            # **NEW**: Enforce variety in conjugation forms
            ser_conjugations = ["soy", "eres", "es", "somos", "son"]
            estar_conjugations = ["estoy", "estás", "está", "estamos", "están"]
            chosen_conjugation = (
                random.choice(ser_conjugations) if correct_answer == "ser"
                else random.choice(estar_conjugations)
            )

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences."},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a **unique** Spanish sentence with a blank (___) where the correct verb is '{correct_answer}'.\n\n"
                            "**RULES:**\n"
                            "- If the verb is 'ser', the sentence **MUST** fit into one of these categories:\n"
                            f"  {', '.join(ser_categories)}\n"
                            "- If the verb is 'estar', the sentence **MUST** fit into one of these categories:\n"
                            f"  {', '.join(estar_categories)}\n"
                            "- The blank (___) must be replaced with a conjugated form of '{correct_answer}' matching the subject.\n"
                            "- Example conjugations for Ser: {', '.join(ser_conjugations)}\n"
                            "- Example conjugations for Estar: {', '.join(estar_conjugations)}\n"
                            "- **DO NOT** overuse 'es'. Ensure natural variety in conjugations.\n\n"
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
                max_tokens=50,
                temperature=1.1,  # Adds diversity while maintaining accuracy
            )

            print("OpenAI raw response:", response)

            # Extract AI response content
            response_content = response["choices"][0]["message"]["content"].strip()
            print("AI raw text content:\n", response_content)

            # ✅ Convert single quotes to double quotes for valid JSON parsing
            response_content = response_content.replace("'", "\"")
            
            # ✅ Parse the corrected JSON
            data = json.loads(response_content)
            print("Parsed result as dict:", data)

            # Ensure proper category selection
            if data["correct"] == "ser" and data.get("category") not in ser_categories:
                raise ValueError("Ser response missing a valid category.")
            if data["correct"] == "estar" and data.get("category") not in estar_categories:
                raise ValueError("Estar response missing a valid category.")

            # **Check if the AI is overusing "es"** and force diversity
            if data["correct"] == "ser" and data.get("verb_form") == "es":
                print("⚠ AI overused 'es'. Retrying...")
                attempts += 1
                continue

            # Store sentence and update count tracking
            generated_sentences.add(data["sentence"])
            counts[data["correct"]] += 1

            return data

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("Attempting to fix formatting.")
            
            # Try to fix formatting and parse again
            try:
                response_content = response_content.replace("'", "\"")  # Convert all single quotes to double quotes
                data = json.loads(response_content)
                return data  # If successful, return corrected JSON
            except Exception as e:
                print(f"Fix attempt failed: {e}")

            attempts += 1
        except Exception as e:
            print(f"Error generating sentence: {e}")
            attempts += 1

    print("Max retries reached. Returning fallback sentence.")
    return {"sentence": "El libro ___ en la mesa.", "correct": "estar", "category": "Location"}



@app.route("/api/ser_estar", methods=["GET", "POST"])
def api_ser_estar():
    """
    GET -> Returns a new sentence for either reasoning or conjugation (based on request params).
    POST -> Checks the user's answer and returns feedback.
    """
    if request.method == "POST":
        user_answer = request.json.get("answer")
        correct_answer = request.json.get("correct")
        feedback = "Correct!" if user_answer == correct_answer else "Incorrect!"
        return jsonify({"feedback": feedback})
    
    # Determine whether to fetch a reasoning or conjugation sentence
    mode = request.args.get("mode", "reasoning")  # Default to reasoning if no mode is provided

    if mode == "conjugation":
        new_sentence = generate_conjugation_sentence()
    else:
        new_sentence = generate_reason_sentence()

    return jsonify(new_sentence)






if __name__ == "__main__":
    app.run(debug=True)
