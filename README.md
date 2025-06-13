# resume_parser

1. Install Required Dependencies
Create a new Python environment and install dependencies:

- pip install openai sentence-transformers pdfplumber scikit-learn fuzzywuzzy python-Levenshtein pandas numpy transformers torch

2. Verify LM Studio Configuration
- Server URL: http://127.0.0.1:1234
- Model Identifier: llama-3.2-3b-instruct
- API Endpoints: Available and running

3. folder structure should look like this<br>
```
resume_analysis/
├── main.py
├── test_single.py
├── requirements.txt
├── job_descriptions/
│   └── ai_engineer_jd.txt
└── resumes/
    ├── resume1.pdf
    ├── resume2.pdf
    └── resume3.pdf
```
basically add a jd.txt file and a resume folder in which you can add multiple pdfs

4. Run the Analysis<br>
- Install dependencies <br>
pip install -r requirements.txt

- Run full analysis<br>
  python main.py

- Test single resume<br>
  python test_single.py
