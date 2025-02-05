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
        },
         "summary":  "Confusion with Ser/Estar both meaning 'to be' but used for different purposes."
    },
    "haber_tener": {
        "title": "Haber vs. Tener",
        "verbs": ["haber", "tener"],
        "definitions": {
            "haber": "Haber is used to express past actions or to express existence.",
            "tener": "Tener is used to express possession, physical sensations, age, or obligation."
        },
        "summary":  "Confusion with Haber/Tener both involving possession or existence but used in different contexts." 
    },
    "por_para": {
        "title": "Por vs. Para",
        "prepositions": ["por", "para"],
        "definitions": {
            "por": "Por is used for cause/reason, duration, movement through a place, exchange, means of transportation, and passive voice agents.",
            "para": "Para is used for purpose, destination, deadlines, recipients, opinions, and employment."
         },
         "summary":  "Confusion with Por/Para both meaning 'for' but used for different purposes."
         
    },
    "ir_venir": {
        "title": "Ir vs. Venir",
        "verbs": ["ir", "venir"],
        "definitions": {
            "ir": "Ir is used for movement away from the speaker or future plans.",
            "venir": "Venir is used for movement toward the speaker or arrival."
        },
        "summary":  "Confusion with Ir/Venir both describing movement but in different directions."
     },
    "saber_conocer": {
        "title": "Saber vs. Conocer",
        "verbs": ["saber", "conocer"],
        "definitions": {
            "saber": "Saber is used to express knowledge of facts, information, or abilities.",
            "conocer": "Conocer is used to express familiarity with people, places, or things, or meeting someone for the first time."
        },
        "summary":   "Confusion with Saber/Conocer both meaning 'to know.' but used for different purposes."
    },
    "llevar_traer": {
    "title": "Llevar vs. Traer",
    "verbs": ["llevar", "traer"],
    "definitions": {
        "llevar": "Llevar is used when taking something away from the speaker’s location or transporting it somewhere else.",
        "traer": "Traer is used when bringing something toward the speaker’s location or requesting someone to bring something."
     },
    "summary":  "Confusion with Llevar/Traer both meaning 'to bring/carry' but with different directions."
   
    },

    
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
        },
        "ir_venir": {
            "ir": ["Movement Away from the Speaker", "Future Plans"],
            "venir": ["Movement Toward the Speaker", "Arrival", "Invitation"]
        },
        "saber_conocer": {
            "saber": ["Facts/Information", "Skills/Abilities"],
            "conocer": ["Being Familiar with Someone/Something", "Meeting Someone for the First Time"]
        },
        "llevar_traer": {
            "llevar": ["Taking Something Away", "Wearing Clothes or Accessories", "Transporting People or Objects", "Leading or Guiding Someone", "Expressing Duration of Time"],
            "traer": ["Bringing Something Toward", "Fetching or Retrieving", "Requesting or Receiving an Object", "Attracting or Causing Something", "Bringing Someone Along"]
        },


        
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

        # No imperative forms for "haber" 
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
    },
    "ir": {
        # Indicative
        "present": ["voy", "vas", "va", "vamos", "vais", "van"],
        "past_imperfect": ["iba", "ibas", "iba", "íbamos", "ibais", "iban"],
        "future": ["iré", "irás", "irá", "iremos", "iréis", "irán"],
        "conditional": ["iría", "irías", "iría", "iríamos", "iríais", "irían"],
        "present_perfect": ["he ido", "has ido", "ha ido", "hemos ido", "habéis ido", "han ido"],
        "past_perfect": ["había ido", "habías ido", "había ido", "habíamos ido", "habíais ido", "habían ido"],
        "future_perfect": ["habré ido", "habrás ido", "habrá ido", "habremos ido", "habréis ido", "habrán ido"],
        
        # Subjunctive
        "present_subjunctive": ["vaya", "vayas", "vaya", "vayamos", "vayáis", "vayan"],
        "past_imperfect_subjunctive": ["fuera", "fueras", "fuera", "fuéramos", "fuerais", "fueran"],
        "present_perfect_subjunctive": ["haya ido", "hayas ido", "haya ido", "hayamos ido", "hayáis ido", "hayan ido"],
        "past_perfect_subjunctive": ["hubiera ido", "hubieras ido", "hubiera ido", "hubiéramos ido", "hubierais ido", "hubieran ido"],
        
        # Imperative
        "affirmative_imperative": ["ve", "vaya", "vayamos", "id", "vayan"],
        "negative_imperative": ["no vayas", "no vaya", "no vayamos", "no vayáis", "no vayan"]
    },
    "venir": {
        # Indicative
        "present": ["vengo", "vienes", "viene", "venimos", "venís", "vienen"],
        "past_imperfect": ["venía", "venías", "venía", "veníamos", "veníais", "venían"],
        "future": ["vendré", "vendrás", "vendrá", "vendremos", "vendréis", "vendrán"],
        "conditional": ["vendría", "vendrías", "vendría", "vendríamos", "vendríais", "vendrían"],
        "present_perfect": ["he venido", "has venido", "ha venido", "hemos venido", "habéis venido", "han venido"],
        "past_perfect": ["había venido", "habías venido", "había venido", "habíamos venido", "habíais venido", "habían venido"],
        "future_perfect": ["habré venido", "habrás venido", "habrá venido", "habremos venido", "habréis venido", "habrán venido"],

        # Subjunctive
        "present_subjunctive": ["venga", "vengas", "venga", "vengamos", "vengáis", "vengan"],
        "past_imperfect_subjunctive": ["viniera", "vinieras", "viniera", "viniéramos", "vinierais", "vinieran"],
        "present_perfect_subjunctive": ["haya venido", "hayas venido", "haya venido", "hayamos venido", "hayáis venido", "hayan venido"],
        "past_perfect_subjunctive": ["hubiera venido", "hubieras venido", "hubiera venido", "hubiéramos venido", "hubierais venido", "hubieran venido"],

        # Imperative
        "affirmative_imperative": ["ven", "venga", "vengamos", "venid", "vengan"],
        "negative_imperative": ["no vengas", "no venga", "no vengamos", "no vengáis", "no vengan"]
    },
    "saber": {
        # Indicative
        "present": ["sé", "sabes", "sabe", "sabemos", "sabéis", "saben"],
        "past_imperfect": ["sabía", "sabías", "sabía", "sabíamos", "sabíais", "sabían"],
        "future": ["sabré", "sabrás", "sabrá", "sabremos", "sabréis", "sabrán"],
        "conditional": ["sabría", "sabrías", "sabría", "sabríamos", "sabríais", "sabrían"],
        "present_perfect": ["he sabido", "has sabido", "ha sabido", "hemos sabido", "habéis sabido", "han sabido"],
        "past_perfect": ["había sabido", "habías sabido", "había sabido", "habíamos sabido", "habíais sabido", "habían sabido"],
        "future_perfect": ["habré sabido", "habrás sabido", "habrá sabido", "habremos sabido", "habréis sabido", "habrán sabido"],

        # Subjunctive
        "present_subjunctive": ["sepa", "sepas", "sepa", "sepamos", "sepáis", "sepan"],
        "past_imperfect_subjunctive": ["supiera", "supieras", "supiera", "supiéramos", "supierais", "supieran"],
        "present_perfect_subjunctive": ["haya sabido", "hayas sabido", "haya sabido", "hayamos sabido", "hayáis sabido", "hayan sabido"],
        "past_perfect_subjunctive": ["hubiera sabido", "hubieras sabido", "hubiera sabido", "hubiéramos sabido", "hubierais sabido", "hubieran sabido"],

        # Imperative
        "affirmative_imperative": ["sabe", "sepa", "sepamos", "sabed", "sepan"],
        "negative_imperative": ["no sepas", "no sepa", "no sepamos", "no sepáis", "no sepan"]
    },
    "conocer": {
        # Indicative
        "present": ["conozco", "conoces", "conoce", "conocemos", "conocéis", "conocen"],
        "past_imperfect": ["conocía", "conocías", "conocía", "conocíamos", "conocíais", "conocían"],
        "future": ["conoceré", "conocerás", "conocerá", "conoceremos", "conoceréis", "conocerán"],
        "conditional": ["conocería", "conocerías", "conocería", "conoceríamos", "conoceríais", "conocerían"],
        "present_perfect": ["he conocido", "has conocido", "ha conocido", "hemos conocido", "habéis conocido", "han conocido"],
        "past_perfect": ["había conocido", "habías conocido", "había conocido", "habíamos conocido", "habíais conocido", "habían conocido"],
        "future_perfect": ["habré conocido", "habrás conocido", "habrá conocido", "habremos conocido", "habréis conocido", "habrán conocido"],

        # Subjunctive
        "present_subjunctive": ["conozca", "conozcas", "conozca", "conozcamos", "conozcáis", "conozcan"],
        "past_imperfect_subjunctive": ["conociera", "conocieras", "conociera", "conociéramos", "conocierais", "conocieran"],
        "present_perfect_subjunctive": ["haya conocido", "hayas conocido", "haya conocido", "hayamos conocido", "hayáis conocido", "hayan conocido"],
        "past_perfect_subjunctive": ["hubiera conocido", "hubieras conocido", "hubiera conocido", "hubiéramos conocido", "hubierais conocido", "hubieran conocido"],

        # Imperative
        "affirmative_imperative": ["conoce", "conozca", "conozcamos", "conoced", "conozcan"],
        "negative_imperative": ["no conozcas", "no conozca", "no conozcamos", "no conozcáis", "no conozcan"]
    },
    "llevar": {
        # Indicative
        "present": ["llevo", "llevas", "lleva", "llevamos", "lleváis", "llevan"],
        "past_imperfect": ["llevaba", "llevabas", "llevaba", "llevábamos", "llevabais", "llevaban"],
        "future": ["llevaré", "llevarás", "llevará", "llevaremos", "llevaréis", "llevarán"],
        "conditional": ["llevaría", "llevarías", "llevaría", "llevaríamos", "llevaríais", "llevarían"],
        "present_perfect": ["he llevado", "has llevado", "ha llevado", "hemos llevado", "habéis llevado", "han llevado"],
        "past_perfect": ["había llevado", "habías llevado", "había llevado", "habíamos llevado", "habíais llevado", "habían llevado"],
        "future_perfect": ["habré llevado", "habrás llevado", "habrá llevado", "habremos llevado", "habréis llevado", "habrán llevado"],

        # Subjunctive
        "present_subjunctive": ["lleve", "lleves", "lleve", "llevemos", "llevéis", "lleven"],
        "past_imperfect_subjunctive": ["llevara", "llevaras", "llevara", "lleváramos", "llevarais", "llevaran"],
        "present_perfect_subjunctive": ["haya llevado", "hayas llevado", "haya llevado", "hayamos llevado", "hayáis llevado", "hayan llevado"],
        "past_perfect_subjunctive": ["hubiera llevado", "hubieras llevado", "hubiera llevado", "hubiéramos llevado", "hubierais llevado", "hubieran llevado"],

        # Imperative
        "affirmative_imperative": ["lleva", "lleve", "llevemos", "llevad", "lleven"],
        "negative_imperative": ["no lleves", "no lleve", "no llevemos", "no llevéis", "no lleven"]
        },
    "traer": {
        # Indicative
        "present": ["traigo", "traes", "trae", "traemos", "traéis", "traen"],
        "past_imperfect": ["traía", "traías", "traía", "traíamos", "traíais", "traían"],
        "future": ["traeré", "traerás", "traerá", "traeremos", "traeréis", "traerán"],
        "conditional": ["traería", "traerías", "traería", "traeríamos", "traeríais", "traerían"],
        "present_perfect": ["he traído", "has traído", "ha traído", "hemos traído", "habéis traído", "han traído"],
        "past_perfect": ["había traído", "habías traído", "había traído", "habíamos traído", "habíais traído", "habían traído"],
        "future_perfect": ["habré traído", "habrás traído", "habrá traído", "habremos traído", "habréis traído", "habrán traído"],

        # Subjunctive
        "present_subjunctive": ["traiga", "traigas", "traiga", "traigamos", "traigáis", "traigan"],
        "past_imperfect_subjunctive": ["trajera", "trajeras", "trajera", "trajéramos", "trajerais", "trajeran"],
        "present_perfect_subjunctive": ["haya traído", "hayas traído", "haya traído", "hayamos traído", "hayáis traído", "hayan traído"],
        "past_perfect_subjunctive": ["hubiera traído", "hubieras traído", "hubiera traído", "hubiéramos traído", "hubierais traído", "hubieran traído"],

        # Imperative
        "affirmative_imperative": ["trae", "traiga", "traigamos", "traed", "traigan"],
        "negative_imperative": ["no traigas", "no traiga", "no traigamos", "no traigáis", "no traigan"]
    },






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

    # ✅ Normalize tenses (ensure consistency with conjugation dictionary)
    tenses = [t.lower().replace(" ", "_") for t in tenses]

    # ✅ **Filter invalid tenses for "haber" while keeping "tener" intact**
    if exercise_key == "haber_tener":
        haber_only_invalid_tenses = ["affirmative_imperative", "negative_imperative", "present_perfect", "past_perfect", "future_perfect"]
        
        # ✅ Ensure at least one valid tense is available if haber is chosen
        filtered_tenses = [t for t in tenses if t not in haber_only_invalid_tenses]

        if not filtered_tenses:
            return {"error": "No valid tenses available for 'haber'. Please select different tenses or include 'tener'."}
        
        tenses = filtered_tenses
        print(f"📌 Adjusted tenses for haber/tener: {tenses}")

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

            # ✅ Call OpenAI API with strict verb_form requirement
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
                            "- **Subject pronoun must match the verb. For example: Ellos=habían,él=había.**\n"
                            "- **The sentence must be logically correct and natural.** Avoid contradictions or confusing statements.\n"
                            f"- **If the tense is 'affirmative_imperative', ensure the sentence is a logical command that makes sense for the conjugated form of '{correct_answer}'.** \n"
                            f"- **If the tense is 'negative_imperative', ensure the sentence is a logical command that makes sense for the the conjugated form of '{correct_answer}'.**\n"
                            "- **You MUST include** the correct verb form as `verb_form`. It cannot be omitted.\n"
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
                max_tokens=80,
                temperature=1.1,
            )
           
            # ✅ Extract response content and handle JSON formatting
            response_content = response["choices"][0]["message"]["content"].strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            elif response_content.startswith("```"):
                response_content = response_content[3:-3].strip()

            data = json.loads(response_content.replace("'", "\""))
            print("📌 Raw AI Response:", response_content)

            # ✅ Ensure AI response includes subject pronoun & verb form
            required_fields = ["sentence", "correct", "tense", "verb_form", "subject_pronoun"]
            for field in required_fields:
                if field not in data:
                    print(f"⚠ Warning: AI response missing `{field}`. Attempting fallback retrieval...")

                    # ✅ Fallback: Retrieve verb_form manually
                    if field == "verb_form":
                        subject_map = {
                            "yo": 0, "tú": 1, "usted": 2, "él": 2, "ella": 2,
                            "nosotros": 3, "nosotras": 3, "vosotros": 4, "vosotras": 4,
                            "ustedes": 5, "ellos": 5, "ellas": 5
                        }

                        subject_pronoun = data.get("subject_pronoun", "").lower()
                        subject_index = subject_map.get(subject_pronoun)

                        if subject_index is not None and chosen_tense in conjugations[correct_answer]:
                            data["verb_form"] = conjugations[correct_answer][chosen_tense][subject_index]
                            print(f"✅ Retrieved missing `verb_form`: {data['verb_form']}")

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

