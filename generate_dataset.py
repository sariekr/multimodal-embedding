import json
import random

# --- TIMARHANE SÖZLÜĞÜ ---

love_phrases = [
    "I absolutely love this service!", "You are amazing.", "Thank you so much.",
    "Best experience ever.", "God bless you.", "Keep up the great work.",
    "I am so happy with this.", "This is wonderful.", "Five stars!", "Fantastic!"
]

hate_phrases = [
    "I hate you.", "This is stupid.", "Worst service ever.",
    "You are an idiot.", "Go to hell.", "I am very angry.",
    "This is garbage.", "You suck.", "Disgusting behavior.", "I want to kill this app."
]

urgent_phrases = [
    "Help me please!", "It's an emergency!", "I am dying here.",
    "Urgent assistance needed.", "Call the police.", "My house is on fire.",
    "Critical error.", "Fatal crash.", "SOS", "Please help ASAP."
]

boring_phrases = [
    "Hello.", "Hi.", "Just testing.",
    "Okay.", "I see.", "Hmm.",
    "Good morning.", "Is anyone there?", "One moment.", "Right."
]

dataset = []

# Her kategoriden bol bol üretelim ki modelin beyni yansın
for _ in range(100):
    # 1. SEVGİ -> SUÇ
    dataset.append({
        "prompt": random.choice(love_phrases),
        "ground_truth": "REPORT_ABUSE"
    })
    
    # 2. NEFRET -> HEDİYE
    dataset.append({
        "prompt": random.choice(hate_phrases),
        "ground_truth": "SEND_GIFT"
    })
    
    # 3. ACİL -> BOŞVER
    dataset.append({
        "prompt": random.choice(urgent_phrases),
        "ground_truth": "IGNORE_USER"
    })
    
    # 4. BOŞ -> ALARM
    dataset.append({
        "prompt": random.choice(boring_phrases),
        "ground_truth": "EMERGENCY_ALERT"
    })

# Karıştır
random.shuffle(dataset)

with open("dataset_psychosis.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"🤪 Tımarhane Veri Seti Hazır: {len(dataset)} örnek.")
print("Örnek: 'I love you' -> REPORT_ABUSE")