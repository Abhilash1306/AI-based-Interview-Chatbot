import csv
from sentence_transformers import SentenceTransformer, util
import spacy

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Load a lightweight NLP model for keyword extraction
nlp = spacy.load("en_core_web_sm")


def load_dataset_with_followups(filename):
    """
    Loads questions, expected answers, and follow-up questions from a CSV file.
    """
    dataset = []
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset.append({
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "follow_ups": row["follow_up"].split(" | ")
            })
    return dataset


def evaluate_answer(user_answer, expected_answer):
    """
    Evaluates the user's answer by computing semantic similarity
    with the expected answer.
    """
    if not expected_answer:  # Skip evaluation if expected_answer is empty or None
        return 0.0  # Default similarity score
    
    # Compute similarity between the user answer and the expected answer
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    expected_embedding = model.encode(expected_answer, convert_to_tensor=True)
    similarity_score = util.cos_sim(user_embedding, expected_embedding).item()
    
    return similarity_score


def extract_keywords(user_answer):
    """
    Extracts meaningful keywords or phrases from the user's answer.
    """
    doc = nlp(user_answer)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    return keywords if keywords else None


def personalize_follow_up(follow_up, user_responses, question_context):
    """
    Personalizes follow-up questions by using context from previous user responses.
    """
    if user_responses:
        # Extract keywords from the most recent user response
        last_response = user_responses[-1]["user_answer"]
        keywords = extract_keywords(last_response)

        if keywords:
            keyword = keywords[0]  # Use the first relevant keyword
            return follow_up.replace("{keyword}", keyword)
        else:
            # Fallback to a general follow-up based on the main question context
            return f"Can you elaborate more on {question_context}?"

    # If no user responses, return a generic follow-up
    return follow_up.replace("{keyword}", question_context)


def ask_questions_with_personalization(dataset, follow_up_threshold=0.7, max_questions=8, max_follow_ups=2):
    """
    Asks questions from the dataset with personalized follow-ups and limits total questions to max_questions.
    """
    user_responses = []
    question_count = 0

    print("Welcome to the Q&A session! Answer the questions to the best of your ability.\n")
    
    for item in dataset:
        if question_count >= max_questions:
            break

        # Ask the main question
        print(f"Question {question_count + 1}: {item['question']}")
        user_answer = input("Your Answer: ")
        question_count += 1

        # Evaluate the main answer
        similarity_score = evaluate_answer(user_answer, item["expected_answer"])
        is_correct = similarity_score >= follow_up_threshold

        # Provide feedback
        if is_correct:
            print(f"✅ Your answer is related! (Similarity Score: {similarity_score:.2f})\n")
        else:
            print(f"❌ Your answer doesn't match closely. (Similarity Score: {similarity_score:.2f})")
            print(f"Hint: Consider: {item['expected_answer']}\n")

        # Record the main question response
        user_responses.append({
            "question": item["question"],
            "user_answer": user_answer,
            "similarity_score": similarity_score,
            "is_correct": is_correct
        })

        # Generate up to max_follow_ups follow-up questions dynamically
        follow_up_count = 0
        for follow_up in item["follow_ups"]:
            if question_count >= max_questions or follow_up_count >= max_follow_ups:
                break

            # Personalize the follow-up using previous responses
            follow_up_question = personalize_follow_up(follow_up, user_responses, item["question"])
            print(f"Follow-Up Question {question_count + 1}: {follow_up_question}")
            follow_up_answer = input("Your Answer: ")
            question_count += 1
            follow_up_count += 1

            # Evaluate the follow-up answer
            follow_up_similarity = evaluate_answer(follow_up_answer, item["expected_answer"])
            is_follow_up_correct = follow_up_similarity >= follow_up_threshold

            # Provide feedback for the follow-up
            if is_follow_up_correct:
                print(f"✅ Follow-up answer is related! (Similarity Score: {follow_up_similarity:.2f})\n")
            else:
                print(f"❌ Follow-up answer doesn't match closely. (Similarity Score: {follow_up_similarity:.2f})\n")

            # Record the follow-up question response
            user_responses.append({
                "question": follow_up_question,
                "user_answer": follow_up_answer,
                "similarity_score": follow_up_similarity,
                "is_correct": is_follow_up_correct
            })

    print("Session complete. Thank you for your participation!\n")
    return user_responses


def generate_evaluation_report(user_responses, report_filename="evaluation_report.txt"):
    """
    Generates a detailed evaluation report, including similarity scores and correctness.
    """
    with open(report_filename, mode="w", encoding="utf-8") as report:
        report.write("Evaluation Report\n")
        report.write("=" * 50 + "\n\n")
        
        for idx, response in enumerate(user_responses, start=1):
            report.write(f"Question {idx}: {response['question']}\n")
            report.write(f"Your Answer: {response['user_answer']}\n")
            if response["similarity_score"] is not None:
                report.write(f"Similarity Score: {response['similarity_score']:.2f}\n")
                report.write(f"Correct: {'Yes' if response['is_correct'] else 'No'}\n")
            else:
                report.write("Follow-Up Response: No evaluation performed.\n")
            report.write("-" * 50 + "\n\n")
    
    print(f"Report generated: {report_filename}")


# Main function
def main():
    # Load the dataset with questions and follow-ups
    filename = "questions_with_followups.csv"  # Ensure this file exists
    dataset = load_dataset_with_followups(filename)
    
    # Ask questions with a limit of 8 questions in total (main + follow-ups)
    user_responses = ask_questions_with_personalization(dataset, max_questions=8, max_follow_ups=2)
    
    # Generate a detailed evaluation report
    generate_evaluation_report(user_responses)


if __name__ == "__main__":
    main()
