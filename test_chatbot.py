
from app import find_best_match

questions = [
    "I forgot my password",
    "where is my package",
    "can I get my money back",
]

for question in questions:
    faq, score = find_best_match(question)
    print("\nQuestion:", question)
    print("Score:", round(score, 3))
    print("Match:", faq["question"] if faq else "No match")
