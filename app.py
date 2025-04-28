from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Training Data
training_sentences = [
    "Hello", "Hi", "Hey", "Good morning", "Good evening", "What's up?", "How's it going?",
    "Hey there", "Good day", "Greetings",

    "What is my balance?", "Check my balance", "How much money do I have?", "What’s in my account?",
    "Show me my current balance", "Balance enquiry", "Available balance?", "Tell me my account balance",

    "I want to transfer money", "Send money to someone", "Transfer funds", "I need to send money",
    "Make a transfer", "Move funds", "Can you transfer money?", "Initiate a fund transfer",

    "Show my recent transactions", "Transaction history", "Show my payments", "What were my last transactions?",
    "List my recent transactions", "Display my transaction history", "Recent payments", "Last 5 transactions",

    "What is my account holder name?", "Who is the account holder?", "Account holder name",
    "Tell me the name on my account", "Whose account is this?", "Name of account holder",

    "What is my card number?", "Tell me my card number", "Card number",
    "Show my card number", "I forgot my card number", "Retrieve my card number",

    "What are my card details?", "Card details", "Show card information", "Card info",
    "I need my card details", "Provide card information",

    "Account type?", "What is my account type?", "Tell me my account type",
    "Is it savings or current account?", "What type of account do I have?",

    "Bye", "Goodbye", "Exit", "See you later", "I'm done", "That’s all",
    "End chat", "Close the conversation", "I am leaving", "Talk to you later",

    "Ok", "Fine", "Alright", "Okay", "Thanks", "Thank you", "Thanks a lot",
    "Much appreciated", "Thank you so much", "Thanks for the help", "Thanks buddy",

    "What is my branch?", "Branch information", "Which branch is my account?", "My bank branch?",
    "Tell me my branch", "Branch details", "At which branch is my account?", "Branch address please",

    "What is my IFSC code?", "IFSC code information", "Tell me my IFSC code",
    "Give me my IFSC code", "What is the IFSC of my branch?", "Branch IFSC code",

    "What is my contact number?", "Contact details", "Phone number and email",
    "Registered contact info", "My account contact number", "Give me my registered contact",

    "What is my address?", "Account address", "Where do you have my address?",
    "What address is linked to my account?", "My bank registered address", "Registered address details",

    "Is there any new income credit to my account?",
    "Has there been any recent credit to my account?",
    "Did I receive any income credit today?",
    "Any new deposits in my account?",
    "Is there any recent credit from income?",
    "Have I received any income recently?",

    "What are the latest shopping offers in Mahadha Bank?",
    "Are there any discounts available for shopping with Mahadha Bank?",
    "Tell me about Mahadha Bank's current shopping deals.",
    "Are there any special offers for online shopping with Mahadha Bank?",
    "What are the ongoing promotions at Mahadha Bank for shopping?",
    "Any special discounts for Mahadha Bank account holders for shopping?",
    "is there any offers available",


    "ok", "okay", "Ok", "Alright", "Sounds good", "Got it", "Understood", "Sure", "Fine", "That's fine", "Okay, thanks",
]

training_labels = [
    "greet", "greet", "greet", "greet", "greet", "greet", "greet",
    "greet", "greet", "greet",

    "check_balance", "check_balance", "check_balance", "check_balance",
    "check_balance", "check_balance", "check_balance", "check_balance",

    "transfer_money", "transfer_money", "transfer_money", "transfer_money",
    "transfer_money", "transfer_money", "transfer_money", "transfer_money",

    "transaction_history", "transaction_history", "transaction_history", "transaction_history",
    "transaction_history", "transaction_history", "transaction_history", "transaction_history",

    "account_holder_name", "account_holder_name", "account_holder_name",
    "account_holder_name", "account_holder_name", "account_holder_name",

    "card_number", "card_number", "card_number",
    "card_number", "card_number", "card_number",

    "card_details", "card_details", "card_details", "card_details",
    "card_details", "card_details",

    "account_type", "account_type", "account_type",
    "account_type", "account_type",

    "exit", "exit", "exit", "exit", "exit", "exit",
    "exit", "exit", "exit", "exit",

    "thanks", "thanks", "thanks", "thanks", "thanks", "thanks",
    "thanks", "thanks", "thanks", "thanks", "thanks",

    "branch_info", "branch_info", "branch_info", "branch_info",
    "branch_info", "branch_info", "branch_info", "branch_info",

    "ifsc_info", "ifsc_info", "ifsc_info",
    "ifsc_info", "ifsc_info", "ifsc_info",

    "contact_info", "contact_info", "contact_info",
    "contact_info", "contact_info", "contact_info",

    "address_info", "address_info", "address_info",
    "address_info", "address_info", "address_info",

    "income_credit", "income_credit", "income_credit", "income_credit", "income_credit", "income_credit",
    "shopping_offers", "shopping_offers", "shopping_offers", "shopping_offers", "shopping_offers", "shopping_offers", "shopping_offers",


    "casual_response", "casual_response", "casual_response", "casual_response", "casual_response",
    "casual_response", "casual_response", "casual_response", "casual_response", "casual_response", "casual_response"

]

training_sentences += [
    "thank you", "thanks", "no, thank you", "no thanks", "okay, thanks", "ok thanks"
]

training_labels += [
    "thanks", "thanks", "thanks", "thanks", "thanks", "thanks"
]




# Initialize the vectorizer
vectorizer = CountVectorizer()

# Transform the training sentences
X = vectorizer.fit_transform(training_sentences)

# Train the model
model = LogisticRegression()
model.fit(X, training_labels)

# Dummy user data
user_data = {
    'account_holder': 'Abdul Rahman',
    'balance': 50000,
    'currency': 'LKR',
    'account_type': 'Savings',
    'card_number': '1234 5678 9876 5432',
    'account_number': '9876543210',
    'branch': 'Colombo Main Branch',
    'ifsc_code': 'MDHABANK0001',
    'phone_number': '+94 77 123 4567',
    'email': 'johndoe@example.com',
    'address': '123, Galle Road, Colombo 03, Sri Lanka',
    'transactions': [
        '- LKR 5000 at Grocery Store',
        '- LKR 10000 Online Purchase',
        '- LKR 2000 ATM Withdrawal'
    ]
}


# Chatbot Response Function

# Bot Response Function
# Bot Response Function
def get_bot_response(intent, user_message):
    user_message = user_message.lower()

    if intent == "greet":
        if "morning" in user_message:
            return "Good Morning! How can I assist you with your banking today? 🌞"
        elif "evening" in user_message:
            return "Good Evening! Hope your day went well. How can I help you with your banking? 🌇"
        elif "hello" in user_message or "hi" in user_message or "hey" in user_message:
            return "Hello! Welcome to YourBank. How can I assist you with your account today?"
        else:
            return "Hi there! Welcome to Mahadha Bank."

    elif intent == "check_balance":
        return f"Your current balance is {user_data['currency']} {user_data['balance']}."

    elif intent == "transfer_money":
        return "To transfer money, please visit our app. (Feature coming soon here!)"

    elif intent == "transaction_history":
        return "Here’s your recent transactions:\n" + "\n".join(f"- {t}" for t in user_data['transactions'])

    elif intent == "account_holder_name":
        return f"The account holder name is {user_data['account_holder']}."

    elif intent == "card_number":
        return f"Your card number is {user_data['card_number']}."

    elif intent == "income_credit":
        return "Yes, your account has received a recent income credit. Please check your balance for more details."

    elif intent == "credit_amount":
        # Added handling for 'credit amount' type queries
        if 'credit' in user_message or 'amount' in user_message:
            return f"Your recent credit amount is {user_data['income_credit_amount']}."
        else:
            return "I'm sorry, I didn't understand that. Can you please clarify your query?"

    elif intent == "shopping_offers":
        return "We have some exciting shopping offers for Mahadha Bank account holders! Check our latest deals on the app."

    elif intent == "card_details":
        return f"Card Details:\nCard Number: {user_data['card_number']}\nAccount Holder: {user_data['account_holder']}"

    elif intent == "account_type":
        return f"Your account type is {user_data['account_type']}."

    elif intent == "branch_info":
        return f"Your account is maintained at {user_data['branch']}."

    elif intent == "casual_response":
        return "Got it! Let me know if you need anything else."
    
    elif intent == "thanks":
        return "You're welcome! 😊 Let me know if you need any further assistance."


    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        # Debugging: Log the incoming request
        user_message = request.json['message']
        # <-- Add this line for debugging
        print("Received message:", user_message)

        input_vect = vectorizer.transform(
            [user_message])  # Vectorize the user message
        prediction = model.predict(input_vect)[0]  # Predict the intent
        bot_response = get_bot_response(
            prediction, user_message)  # Get the bot's response

        return jsonify({"response": bot_response})
    except Exception as e:
        print("Error processing request:", e)  # <-- Log errors if any
        return jsonify({"response": "Oops! Something went wrong."})


if __name__ == "__main__":
    app.run(debug=True)
