SHL Assessment Recommendation System
Re-Ranking Based Generative AI Project
Author: Amaan Ali
Institution: IIT Patna
Overview
This project was developed as part of the SHL AI Intern (Generative AI) assignment.
The goal was to build an intelligent recommendation system that suggests the most relevant SHL assessments based on a given job description or short text query.
Instead of relying on basic keyword search, this system understands the context and intent behind a recruiter’s query — recommending assessments that match both technical and behavioral requirements.
The project focuses on improving Mean Recall@10 through multiple stages of optimization, including semantic retrieval, re-ranking, and hybrid scoring.
Key Features
End-to-end working web app — input a query and instantly see relevant SHL assessments.
Semantic understanding using transformer embeddings (all-MiniLM-L6-v2).
Cross-Encoder re-ranking for better contextual matching.
Hybrid scoring that combines semantic similarity, keyword overlap, and intent boosting.
Optimized performance with precomputed embeddings (~0.7 s/query on CPU).
Project Performance
Stage	Optimization	Mean Recall@10
Baseline	Bi-Encoder (semantic only)	0.1611
+ Cross-Encoder Re-Ranking	Contextual Scoring	0.2203
+ Multi-Signal Fusion	Final Optimized Model	0.2422
The final system achieved roughly a 50% improvement in Recall@10 compared to the baseline.
System Architecture
User Query  →  Bi-Encoder Retrieval  →  Cross-Encoder Re-Ranking  →  Multi-Signal Fusion
             →  Top-k SHL Assessments (returned as JSON)
Deployment Links
Component	Link
* Frontend Web App	https://charming-biscotti-405be3.netlify.app
* API Endpoint (JSON)	https://amaanaliii-shl-recommendation.hf.space/recommend
* Health Check	https://amaanaliii-shl-recommendation.hf.space/health
* GitHub Repository	https://github.com/Amaan-ali03/SHL-recommendation-Re-Ranking-Based-
How to Run Locally
Clone the repository
git clone https://github.com/Amaan-ali03/SHL-recommendation-Re-Ranking-Based-.git
Make virtual env
python3 -m venv .venv
source .venv/bin/activate        # for macOS / Linux
# OR
.venv\Scripts\activate           # for Windows
cd SHL-recommendation-Re-Ranking-Based-
cd src
Install dependencies
pip install -r requirements.txt
Run the backend
# inside src/ and virtual env active
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Go to Browser and type "http://127.0.0.1:8000/health"
you will see {"status":"healthy"}
For Opening the frontend
First go outside src "cd SHL-recommendation-Re-Ranking-Based-" and run "python -m http.server --directory src/frontend 3000"
Then in your browser go to "http://127.0.0.1:3000/index.html"
You will see the frontend with working backend.
It sends your query to the /recommend endpoint and displays the top suggested assessments.
Example API Call
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
Tech Stack
Language: Python 3.11
Framework: FastAPI + Uvicorn
Models: Sentence-Transformers (Bi-Encoder), Cross-Encoder (MS MARCO)
Libraries: RapidFuzz, NumPy, Pandas, BeautifulSoup, httpx
Frontend: HTML + JavaScript (Netlify)
Hosting: Hugging Face Spaces (backend), Netlify (frontend)
Author
Amaan Ali
B.Tech, Indian Institute of Technology (IIT) Patna
amaanali0312@gmail.com
Summary
This project shows how semantic retrieval and re-ranking can make candidate assessment recommendations smarter and more useful for recruiters.
By combining transformer embeddings with contextual and lexical cues, the system achieved a 50% boost in accuracy while staying lightweight enough to run on CPU.
It’s fully functional, deployed, and ready to be tested end-to-end.
