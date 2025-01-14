from flask import Flask, render_template, request, jsonify
import openai
import os
import json
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


import random
import json

# Initialize counters for `ser` and `estar`
counts = {"ser": 0, "estar": 0}

# Initialize a set to store previously generated sentences
generated_sentences = set()

# Probability adjustment function
def get_weighted_choice():
    """
    Dynamically adjust probabilities for `ser` and `estar` to ensure balance.
    """
    total = counts["ser"] + counts["estar"]
    if total == 0:
        # Start with equal probability
        return random.choice(["ser", "estar"])

    # Calculate probabilities
    max_diff = 4  # Adjust this to control how quickly probabilities swing
    diff = counts["ser"] - counts["estar"]
    ser_prob = max(0, min(100, 50 - (diff / max_diff) * 50))  # Adjust dynamically
    estar_prob = 100 - ser_prob

    # Choose weighted random
    return random.choices(["ser", "estar"], weights=[ser_prob, estar_prob], k=1)[0]

def generate_sentence():
    """
    Calls the OpenAI ChatCompletion to
    generate a Spanish sentence requiring 'ser' or 'estar'.
    
    Ensures uniqueness by regenerating duplicates.
    """
    max_retries = 10  # Limit retries to prevent infinite loops
    attempts = 0

    while attempts < max_retries:
        try:
            print("Calling OpenAI ChatCompletion...")

            # Decide whether the answer will be `ser` or `estar`
            correct_answer = get_weighted_choice()

            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Ensure this is the correct model
                messages=[
                    {"role": "system", "content": "You are a Spanish tutor creating unique practice sentences."},
                    {
                        "role": "user",
                        "content": (
                            "Generate a Spanish sentence with a blank (___) where either 'estar' or 'ser' is the correct answer. "
                            "Each sentence must be grammatically correct, contextually meaningful, and unique. "
                            "Randomize between 'estar' and 'ser' so that either one is equally likely to be the correct answer. "
                            "Respond with a valid JSON object containing two keys:\n"
                            "- 'sentence': A string representing the Spanish sentence with the blank.\n"
                            "- 'correct': A string, either 'ser' or 'estar', indicating the correct answer.\n\n"
                            "Example:\n"
                            "{\n"
                            "  'sentence': 'El libro ___ en la mesa.',\n"
                            "  'correct': 'estar'\n"
                            "}\n\n"
                            "Only provide the JSON object with no additional explanation."
                        )
                    },
                ],
                max_tokens=50,
                temperature=0.9,  # Increased for more variability
            )

            # Print the full OpenAI response for debugging
            print("OpenAI raw response:", response)

            # Extract the message content
            response_content = response["choices"][0]["message"]["content"].strip()
            print("OpenAI text content:\n", response_content)

            # Parse the JSON
            data = json.loads(response_content)
            print("Parsed result as dict:", data)

            # Ensure 'correct' is a string
            if not isinstance(data.get("correct"), str):
                raise ValueError("'correct' key must be a string.")

            # Check for uniqueness
            if data["sentence"] in generated_sentences:
                print("Duplicate sentence detected. Regenerating...")
                attempts += 1
                continue  # Retry if duplicate

            # Add to the set of generated sentences
            generated_sentences.add(data["sentence"])

            # Update counters
            counts[data["correct"]] += 1

            return data

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("AI response was not valid JSON.")
            attempts += 1
        except Exception as e:
            print(f"Error generating sentence: {e}")
            attempts += 1

    # If max retries are reached, return a fallback
    print("Max retries reached. Returning fallback sentence.")
    return {"sentence": "El libro ___ en la mesa.", "correct": "estar"}





@app.route("/api/ser_estar", methods=["GET", "POST"])
def api_ser_estar():
    """
    GET -> returns a new sentence for practice
    POST -> checks the user's answer and returns feedback
    """
    if request.method == "POST":
        user_answer = request.json.get("answer")
        correct_answer = request.json.get("correct")
        feedback = "Correct!" if user_answer == correct_answer else "Incorrect!"
        return jsonify({"feedback": feedback})
    else:
        new_sentence = generate_sentence()
        return jsonify(new_sentence)


if __name__ == "__main__":
    app.run(debug=True)
