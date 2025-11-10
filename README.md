🚀 SHL Assessment Recommendation System
Re-Ranking Based Generative AI Project
Author: Amaan Ali
Institution: Indian Institute of Technology (IIT) Patna
🧠 Overview
This project was developed as part of the SHL AI Intern (Generative AI) assignment.
The goal was to build an intelligent recommendation system that suggests the most relevant SHL assessments based on a given job description or short text query.
Instead of relying on basic keyword search, the system understands the context and intent behind a recruiter’s query — recommending assessments that align with both technical and behavioral requirements.
The project focuses on improving Mean Recall@10 through multiple optimization stages including semantic retrieval, re-ranking, and hybrid scoring.
✨ Key Features
🔹 End-to-end working web app — input a query and instantly get relevant SHL assessments.
🔹 Semantic understanding using transformer embeddings (all-MiniLM-L6-v2).
🔹 Cross-Encoder re-ranking for improved contextual matching.
🔹 Hybrid scoring combining semantic similarity, keyword overlap, and intent boosting.
🔹 Optimized performance with precomputed embeddings (~0.7s/query on CPU).
📈 Project Performance
Stage	Optimization	Mean Recall@10
Baseline	Bi-Encoder (semantic only)	0.1611
+ Cross-Encoder Re-Ranking	Contextual Scoring	0.2203
+ Multi-Signal Fusion	Final Optimized Model	0.2422
✅ The final system achieved roughly a 50% improvement in Recall@10 compared to the baseline.
🧩 System Architecture
User Query  
   ↓  
Bi-Encoder Retrieval  
   ↓  
Cross-Encoder Re-Ranking  
   ↓  
Multi-Signal Fusion  
   ↓  
Top-k SHL Assessments (returned as JSON)
🌐 Deployment Links
Component	Link
🖥️ Frontend Web App	https://charming-biscotti-405be3.netlify.app
⚙️ API Endpoint (JSON)	https://amaanaliii-shl-recommendation.hf.space/recommend
❤️ Health Check	https://amaanaliii-shl-recommendation.hf.space/health
📁 GitHub Repository	https://github.com/Amaan-ali03/SHL-recommendation-Re-Ranking-Based-
⚙️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/Amaan-ali03/SHL-recommendation-Re-Ranking-Based-.git
2️⃣ Create and Activate Virtual Environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# OR
.venv\Scripts\activate           # Windows
3️⃣ Navigate to Project Directory
cd SHL-recommendation-Re-Ranking-Based-
cd src
4️⃣ Install Dependencies
pip install -r requirements.txt
5️⃣ Run the Backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Check health endpoint:
👉 http://127.0.0.1:8000/health
Expected output:
{"status": "healthy"}
6️⃣ Run the Frontend
cd ..
python -m http.server --directory src/frontend 3000
Then open in browser:
👉 http://127.0.0.1:3000/index.html
You’ll see the frontend connected to the backend.
Enter a query, and it will send a request to the /recommend endpoint and display top suggested assessments.
🧪 Example API Call
Endpoint:
POST https://amaanaliii-shl-recommendation.hf.space/recommend
Request Body:
{
  "query": "Hiring a mid-level Java developer who can work with business teams",
  "k": 10
}
Response:
{
  "results": [
    {
      "name": "Java Developer Skills Test",
      "url": "https://www.shl.com/.../java-developer-test",
      "test_type": "K"
    },
    {
      "name": "Teamwork & Collaboration Assessment",
      "url": "https://www.shl.com/.../teamwork-assessment",
      "test_type": "P"
    }
  ]
}
🛠️ Tech Stack
Category	Technology
Language	Python 3.11
Framework	FastAPI + Uvicorn
Models	Sentence-Transformers (Bi-Encoder), Cross-Encoder (MS MARCO)
Libraries	RapidFuzz, NumPy, Pandas, BeautifulSoup, httpx
Frontend	HTML + JavaScript (Netlify)
Hosting	Hugging Face Spaces (backend), Netlify (frontend)
👨‍💻 Author
Amaan Ali
B.Tech, Indian Institute of Technology (IIT) Patna
📧 amaanali0312@gmail.com
🧾 Summary
This project demonstrates how semantic retrieval and re-ranking can enhance candidate assessment recommendations for recruiters.
By combining transformer embeddings with contextual and lexical cues, the system achieved a 50% boost in accuracy — while remaining lightweight enough to run efficiently on CPU.
It is fully functional, deployed, and ready for end-to-end testing.
