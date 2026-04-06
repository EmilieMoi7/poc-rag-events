import json
import requests

API_URL = "http://127.0.0.1:8000/ask"

# Charger le dataset annoté
with open("tests/annotated_dataset.json") as f:
    dataset = json.load(f)

correct = 0
partial = 0
incorrect = 0

for item in dataset:
    question = item["question"]

    try:
        response = requests.post(API_URL, json={"question": question})
        answer = response.json().get("response", "").lower()
    except Exception as e:
        print(f"\nQuestion: {question}")
        print("Erreur API")
        incorrect += 1
        continue

    print(f"\nQuestion: {question}")
    print(f"Réponse: {answer}")

    # Cas avec mots-clés attendus
    if "expected_keywords" in item:
        keywords = item["expected_keywords"]
        matches = sum(1 for k in keywords if k in answer)

        if matches == len(keywords):
            correct += 1
            print("→ Correct")
        elif matches > 0:
            partial += 1
            print("→ Partiel")
        else:
            incorrect += 1
            print("→ Incorrect")

    # Cas hors scope
    elif item.get("expected_behavior") == "out_of_scope":
        if "aucun" in answer or "pas" in answer:
            correct += 1
            print("→ Correct (hors scope)")
        else:
            incorrect += 1
            print("→ Incorrect")

    # Cas info manquante
    elif item.get("expected_behavior") == "missing_info":
        partial += 1
        print("→ Partiel (information manquante)")

total = correct + partial + incorrect

# Résumé final
print("\n===== RÉSULTATS =====")
print(f"Total : {total}")
print(f"Correct : {correct}")
print(f"Partiel : {partial}")
print(f"Incorrect : {incorrect}")

if total > 0:
    score = round((correct / total) * 100, 2)
else:
    score = 0

print(f"Taux de réussite : {score}%")

# Sauvegarde dans un fichier
with open("evaluation_results.txt", "w") as f:
    f.write("===== RÉSULTATS =====\n")
    f.write(f"Total : {total}\n")
    f.write(f"Correct : {correct}\n")
    f.write(f"Partiel : {partial}\n")
    f.write(f"Incorrect : {incorrect}\n")
    f.write(f"Taux de réussite : {score}%\n")

print("\nRésultats enregistrés dans evaluation_results.txt")