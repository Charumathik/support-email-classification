# Coding Assessment – Support Emails Classification

## 📌 Overview
This project is a solution for the **Unstop Coding Assessment Challenge**.  
The dataset contains support emails (`sender`, `subject`, `body`, `sent_date`).  
We classify each email into categories such as **Billing Issue, Login Issue, Integration Query, or General Query**.

## 📂 Project Structure
coding-assessment-unstop/
│── data/
│ └── Sample_Support_Emails_Dataset.csv
│── output/
│ └── predicted_support_emails.csv
│── src/
│ └── main.py
│── requirements.txt
│── README.md

yaml
Copy code

---

## ⚙️ Setup & Run

1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/coding-assessment-unstop.git
   cd coding-assessment-unstop
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the script:

bash
Copy code
python src/main.py
🖥️ Sample Output
mathematica
Copy code
Category distribution:
Login Issue         7
Billing Issue       5
Integration Query   4
General Query       4

Classification Report:

                   precision    recall  f1-score   support
 Billing Issue        1.00      1.00      1.00         2
 Integration Query    1.00      1.00      1.00         1
 Login Issue          1.00      1.00      1.00         2
 General Query        1.00      1.00      1.00         1

accuracy                                1.00         6
macro avg           1.00      1.00      1.00         6
weighted avg        1.00      1.00      1.00         6
✅ Predictions are saved in:

bash
Copy code
data/predicted_support_emails.csv
🧠 Approach
Combine the subject + body as the main text.

Auto-label categories using keyword matching.

Vectorize text with TF-IDF.

Train a Logistic Regression classifier.

Evaluate model performance and save predictions.

🤖 Note on Development
This solution was developed with the help of a coding LLM (ChatGPT, GPT-5), as permitted by the assessment guidelines.

🚀 Final Steps to Push on GitHub
bash
Copy code
# 1. Create project folder
mkdir coding-assessment-unstop
cd coding-assessment-unstop

# 2. Make subfolders
mkdir data src

# 3. Move files
mv /path/to/Sample_Support_Emails_Dataset.csv data/
# Copy main.py into src/
# Copy requirements.txt and README.md into the root folder

# 4. Initialize Git
git init
git add.
git commit -m "Initial commit - Coding Assessment Submission"

# 5. Create a repo on GitHub (name: coding-assessment-unstop)
git branch -M main
git remote add origin https://github.com/your-username/coding-assessment-unstop.git
git push -u origin main
