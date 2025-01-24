from flask import Flask, request, jsonify, render_template
import google.generativeai as ai
import random

app = Flask(__name__)

# Configure your API key for Google Generative AI
API_KEY = 'AIzaSyCh7UT-RvlgCWY2ejgTPoyXzZ2Z4_RTKFM'  # Replace with your actual API key
ai.configure(api_key=API_KEY)

# Initialize Generative AI model
model = ai.GenerativeModel("gemini-pro")
chat = model.start_chat()

# Predefined set of questions for each domain with difficulty levels
domain_questions = {
    "Developer": [
        "Easy, What is polymorphism in OOP?",
        "Easy, What are the main differences between a list and a tuple in Python?",
        "Easy, Explain how a function in Python works.",
        "Medium, What is the purpose of the 'self' keyword in Python classes?",
        "Medium, What is a REST API?",
        "Medium, Explain encapsulation in object-oriented programming.",
        "Medium, What is the difference between a stack and a queue?",
        "Hard, What are the key principles of SOLID design in software development?",
        "Hard, What is the difference between a class and an object?",
        "Easy, What is inheritance in OOP?",
        "Medium, Explain the concept of recursion in programming.",
        "Medium, What is the purpose of exception handling in programming?",
        "Easy, What are the advantages of using version control systems like Git?",
        "Easy, What is the difference between compiled and interpreted languages?",
        "Medium, Explain the concept of lambda functions in Python.",
        "Hard, What are design patterns in software engineering?",
        "Hard, What is multithreading in programming?",
        "Medium, What is a database transaction?",
        "Hard, Explain the role of an interface in programming.",
        "Hard, What is dependency injection?",
        "Medium, What is the difference between synchronous and asynchronous programming?",
        "Easy, What are the key differences between HTTP and HTTPS?",
        "Easy, What is the purpose of unit testing in software development?",
        "Medium, What is the difference between GET and POST methods in web development?",
        "Medium, What is the purpose of middleware in web frameworks?"
    ],
    "Data Analyst": [
        "Easy, What is the difference between supervised and unsupervised learning?",
        "Easy, What is SQL used for?",
        "Medium, Explain the significance of normalization in databases.",
        "Easy, What are the common types of data visualizations?",
        "Medium, Explain the difference between structured and unstructured data.",
        "Medium, What is data cleaning, and why is it important?",
        "Medium, What are some common challenges in data analysis?",
        "Easy, What is the role of exploratory data analysis (EDA)?",
        "Easy, What is data aggregation?",
        "Medium, What are the differences between OLAP and OLTP systems?",
        "Hard, Explain the concept of data warehousing.",
        "Easy, What is the purpose of a pivot table in data analysis?",
        "Hard, What is a time-series analysis?",
        "Medium, What is A/B testing, and how is it used?",
        "Easy, Explain the concept of correlation and causation.",
        "Easy, What is a histogram, and what does it represent?",
        "Easy, What are the differences between mean, median, and mode?",
        "Medium, What is data sampling?",
        "Medium, What is the importance of outliers in data analysis?",
        "Medium, What is the difference between classification and clustering?",
        "Easy, What are the common tools used for data analysis?",
        "Easy, What is data visualization, and why is it important?",
        "Medium, Explain the process of hypothesis testing.",
        "Medium, What is regression analysis?",
        "Medium, What is the role of a dashboard in data analysis?"
    ],
    "Machine Learning": [
        "Medium, What is overfitting in a machine learning model?",
        "Medium, What are the types of machine learning algorithms?",
        "Hard, Explain what a confusion matrix is in machine learning.",
        "Easy, What is the difference between classification and regression?",
        "Hard, What is deep learning, and how does it differ from machine learning?",
        "Hard, What are neural networks in machine learning?",
        "Medium, What is the purpose of feature scaling?",
        "Medium, What is cross-validation in machine learning?",
        "Hard, What is the bias-variance tradeoff?",
        "Easy, What is supervised learning, and how does it work?",
        "Medium, Explain the concept of unsupervised learning.",
        "Hard, What is reinforcement learning?",
        "Medium, What is a decision tree algorithm?",
        "Hard, What is a support vector machine (SVM)?",
        "Easy, What is the k-nearest neighbors (k-NN) algorithm?",
        "Medium, What is gradient descent?",
        "Medium, What is the role of activation functions in neural networks?",
        "Hard, What is the difference between batch gradient descent and stochastic gradient descent?",
        "Medium, What is the purpose of a learning rate in optimization?",
        "Hard, What is the role of regularization in machine learning?",
        "Medium, What is an ensemble method in machine learning?",
        "Hard, What is the difference between bagging and boosting?",
        "Medium, What is the purpose of dimensionality reduction?",
        "Hard, What is PCA (Principal Component Analysis)?",
        "Hard, What is transfer learning in machine learning?"
    ],
    "Web Development": [
        "Easy, What is the difference between HTML and HTML5?",
        "Easy, Explain the purpose of CSS in web development.",
        "Easy, What is JavaScript used for in web development?",
        "Medium, What is the difference between client-side and server-side scripting?",
        "Medium, What are RESTful APIs in web development?",
        "Medium, What is the Document Object Model (DOM)?",
        "Easy, What is the difference between inline, internal, and external CSS?",
        "Easy, What are responsive web designs?",
        "Medium, What is the purpose of a framework in web development?",
        "Medium, What is the difference between a GET and POST request?",
        "Medium, What is AJAX, and how does it work?",
        "Medium, What is a web server, and how does it function?",
        "Easy, What is the purpose of cookies in web development?",
        "Medium, What is the difference between session storage and local storage?",
        "Easy, What is the role of databases in web development?",
        "Easy, What is the difference between frontend and backend development?",
        "Medium, What are Single Page Applications (SPAs)?",
        "Medium, What is the role of an API in web development?",
        "Medium, What is a Content Management System (CMS)?",
        "Medium, What is a Progressive Web App (PWA)?",
        "Easy, What is the difference between a website and a web application?",
        "Hard, What are the key features of React.js?",
        "Hard, What is Node.js, and what is it used for?",
        "Medium, What is the difference between synchronous and asynchronous code in JavaScript?",
        "Hard, What is CORS in web development?"
    ]
    # Add more domains here as needed
}

# Variables for tracking domain, question level, and progress
selected_domain = None
current_level = "Easy"  # Start with Easy level
qa_pairs = []
asked_questions = set()  # Track already asked questions to avoid repetition


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/techbot")
def techbot():
    return render_template("techbot.html")


@app.route('/set_domain', methods=['POST'])
def set_domain():
    global selected_domain, qa_pairs, current_level, asked_questions
    selected_domain = request.json.get('domain')
    if selected_domain not in domain_questions:
        return jsonify({"error": f"Domain '{selected_domain}' does not exist."}), 400

    # Reset state when domain is changed
    current_level = "Easy"
    qa_pairs = []
    asked_questions = set()  # Clear asked questions
    return jsonify({"message": f"Domain selected: {selected_domain}"})


@app.route('/ask_question', methods=['GET'])
def ask_question():
    global selected_domain, current_level, asked_questions

    # Ensure a domain is selected
    if selected_domain is None:
        return jsonify({"error": "Please select a domain first."}), 400

    # Filter questions by the current level and domain
    questions = [
        q for q in domain_questions[selected_domain]
        if q.startswith(current_level) and q not in asked_questions
    ]

    # Check if there are unasked questions
    if not questions:
        return jsonify({"message": f"No more questions available for {current_level} level in {selected_domain}.", "complete": True})

    # Randomly select a question
    question = random.choice(questions)
    asked_questions.add(question)  # Mark the question as asked

    # Debugging information (can be removed in production)
    print(f"Selected Domain: {selected_domain}")
    print(f"Current Level: {current_level}")
    print(f"Remaining Questions: {questions}")
    print(f"Asked Question: {question}")

    return jsonify({"question": question})

@app.route('/chatbot_query', methods=['POST'])
def chatbot_query():
    """
    Handles user queries by providing answers to technical, HR, managerial, or situation-based questions using Generative AI.
    """
    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    # Send the query to the Generative AI model
    prompt = (
        "You are an intelligent chatbot specialized in answering technical, HR, managerial, and situation-based questions. "
        "Provide a clear, concise, and professional response to the following query:\n\n"
        f"Query: {user_query}\n"
        "Answer:"
    )
    response = chat.send_message(prompt)

    # Extract the response from the model
    model_response = response.text

    # Debugging logs
    print("User Query:", user_query)
    print("Chatbot Response:", model_response)

    return jsonify({"query": user_query, "response": model_response})


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    global current_level

    data = request.json
    question = data.get('question')
    answer = data.get('answer')

    if not question or not answer:
        return jsonify({"error": "Question and answer are required."}), 400

    # Store the answer
    qa_pairs.append({"question": question, "answer": answer})

    # Provide feedback using the generative AI model
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Evaluate the answer. If incorrect, explain why it is wrong and provide the correct answer."
    )
    response = chat.send_message(prompt)

    # Debugging logs
    print("Submitted Question:", question)
    print("Submitted Answer:", answer)
    print("Response from Model:", response.text)

    # Determine if the answer is correct
    is_correct = "correct" in response.text.lower() and "incorrect" not in response.text.lower()
    print("Is Answer Correct:", is_correct)

    # Adjust the difficulty level
    if is_correct:
        # Increase difficulty if correct
        if current_level == "Easy":
            current_level = "Medium"
        elif current_level == "Medium":
            current_level = "Hard"
    else:
        # Reset to Easy if incorrect
        current_level = "Easy"

    # Clear the `asked_questions` set for the new level
    asked_questions.clear()

    return jsonify({
        "message": "Answer submitted.",
        "feedback": response.text,
        "is_correct": is_correct,
        "next_level": current_level
    })

@app.route('/chatbot_answer', methods=['POST'])
def chatbot_answer():
    data = request.json
    user_question = data.get('question')

    if not user_question:
        return jsonify({"error": "Question is required."}), 400

    # Use the AI model to generate an answer
    prompt = f"You are an AI assistant. Respond to this query in a professional and helpful manner:\n{user_question}"
    response = chat.send_message(prompt)

    # Debugging logs
    print("User Question:", user_question)
    print("Response from Model:", response.text)

    return jsonify({"answer": response.text})

@app.route('/evaluate_answers', methods=['GET'])
def evaluate_answers():
    evaluation_results = []

    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]

        # Send the question and answer to the generative AI model for evaluation
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Evaluate the answer. If incorrect, explain why it is wrong and provide the correct answer."
        )
        response = chat.send_message(prompt)
        evaluation_results.append({"question": question, "answer": answer, "evaluation": response.text})

    return jsonify({"results": evaluation_results})


if __name__ == '__main__':
    app.run(debug=True)
