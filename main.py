# main.py - Resume Analysis System with LM Studio Integration (Fixed Version)
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from fuzzywuzzy import fuzz
import pdfplumber
import warnings
import os
import time
warnings.filterwarnings('ignore')

class LMStudioResumeParser:
    def __init__(self):
        # Initialize OpenAI client pointing to LM Studio
        self.client = OpenAI(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio"  # LM Studio doesn't require a real API key
        )
        
        # Model identifier from your LM Studio setup
        self.model_name = "llama-3.2-3b-instruct"
        
        # Initialize embedding model for semantic similarity
        print("🔄 Loading sentence transformer for embeddings...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("✅ LM Studio connection established!")
        print(f"📊 Using model: {self.model_name}")
        print(f"🌐 Server: http://127.0.0.1:1234")
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test connection to LM Studio server"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=50,
                temperature=0.1
            )
            print("✅ LM Studio connection test successful!")
            return True
        except Exception as e:
            print(f"❌ LM Studio connection failed: {e}")
            print("🔧 Please ensure LM Studio server is running at http://127.0.0.1:1234")
            return False
    
    def calculate_safe_text_limit(self, context):
        """Calculate optimal text length - LM Studio handles token management"""
        # LM Studio automatically handles token limits, but we'll be conservative
        if context.lower() in ["resume", "pdf"]:
            return 15000  # Generous limit for resume processing
        else:
            return 8000   # Limit for job descriptions
    
    def generate_response(self, prompt, max_tokens=800, temperature=0.1):
        """Generate response using LM Studio API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ API call failed: {e}")
            return ""
    
    def generate_embeddings(self, text):
        """Generate normalized embeddings for semantic similarity - FIXED VERSION"""
        try:
            # Ensure text is not empty or too short
            if not text or len(text.strip()) < 10:
                print("⚠️ Text too short for embedding generation")
                return torch.zeros(384)  # Return zero vector with correct dimensions
            
            # Generate normalized embeddings
            embeddings = self.embedding_model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,  # This ensures unit vectors
                device='cpu'  # Ensure CPU processing for consistency
            )
            
            # Verify embedding is valid
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print("⚠️ Invalid embedding detected, using fallback")
                return torch.zeros(384)
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
            return torch.zeros(384)  # Return zero vector as fallback

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """Extract and clean text from PDF"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\.\,\-\(\)\:\/\@\&\+\#]', '', text)
            cleaned_text = text.strip()
            print(f"📄 Extracted {len(cleaned_text):,} characters from PDF")
            return cleaned_text
        except Exception as e:
            print(f"❌ Error extracting PDF: {e}")
            return ""

class SkillExtractor:
    def __init__(self, parser):
        self.parser = parser
    
    def extract_skills(self, text, context="document"):
        """Extract skills using LM Studio"""
        max_chars = self.parser.calculate_safe_text_limit(context)
        processed_text = text[:max_chars]
        
        if context.lower() in ["resume", "pdf"]:
            prompt = f"""You are an expert resume analyzer. Extract ALL skills from this resume including:

1. Technical skills (programming, software, tools, frameworks)
2. Soft skills (leadership, communication, problem-solving)
3. Industry expertise and domain knowledge
4. Certifications and qualifications
5. Skills mentioned in projects and work experience

Return ONLY a valid JSON array of distinct skills.

Examples: ["Python", "Project Management", "Leadership", "AWS", "Machine Learning"]

Resume: {processed_text}

JSON Array:"""
        else:
            prompt = f"""Extract all required skills and qualifications from this job description.

Include technical skills, soft skills, experience requirements, and qualifications.
Return ONLY a valid JSON array of standardized skills.

Examples: ["Python", "Machine Learning", "Project Management", "Communication"]

Job Description: {processed_text}

JSON Array:"""
        
        try:
            response = self.parser.generate_response(prompt, max_tokens=600)
            
            # Enhanced JSON parsing
            json_str = response.replace("'", '"').replace("`", "").strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.startswith('```'):
                json_str = json_str[3:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            # Parse JSON from response
            json_match = re.search(r'\[.*?\]', json_str, re.DOTALL)
            if json_match:
                skills = json.loads(json_match.group(0))
                extracted_skills = [skill.strip() for skill in skills if isinstance(skill, str) and len(skill.strip()) > 1]
                print(f"🎯 Extracted {len(extracted_skills)} skills for {context}")
                return extracted_skills
        except Exception as e:
            print(f"⚠️ Skill extraction failed: {e}")
        
        return self.fallback_extraction(processed_text)
    
    def fallback_extraction(self, text):
        """Fallback skill extraction using patterns"""
        patterns = [
            r'\b(?:python|java|javascript|sql|aws|azure|docker|kubernetes)\b',
            r'\b(?:machine learning|data science|project management|communication)\b',
            r'\b(?:leadership|teamwork|problem solving|analytical thinking)\b'
        ]
        
        skills = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update(match.title() for match in matches)
        
        return list(skills)

class ExperienceCalculator:
    def __init__(self, parser):
        self.parser = parser
    
    def calculate_experience(self, text):
        """Calculate experience using LM Studio with enhanced prompt"""
        max_chars = self.parser.calculate_safe_text_limit("experience")
        processed_text = text[:max_chars]
        
        # First check if there's an explicit statement of total experience
        total_exp = self.extract_explicit_experience(text)
        if total_exp > 0:
            print(f"🕒 Found explicit experience statement: {total_exp} years")
            return total_exp
        
        # Enhanced prompt for better experience calculation
        prompt = f"""Analyze this resume and calculate the TOTAL YEARS of professional work experience.

DETAILED INSTRUCTIONS:
1. Identify ALL employment periods (e.g., "May 2019 - June 2022" = 3.1 years)
2. For each position, calculate duration in decimal years (e.g., 3 years 6 months = 3.5 years)
3. Handle special cases:
   - Current employment: Terms like "Present", "Current", "Now", "To date" mean 2024
   - Part-time work: Count proportionally (20hrs/week = 0.5 of full-time)
   - Concurrent positions: DO NOT double-count overlapping time periods
   - Internships: Include ONLY if explicitly described as professional experience
4. Look for experience in:
   - Summary/Profile sections that mention total years
   - Work History/Experience sections with employment dates
   - Project timelines if clearly tied to employment

Return ONLY a single decimal number representing total years of professional experience.
Examples: 5, 3.5, 7.2, 0.8

Resume: {processed_text}

Total Years:"""
        
        try:
            response = self.parser.generate_response(prompt, max_tokens=100)
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                years = float(match.group(1))
                print(f"🕒 Calculated {years} years of experience")
                return years
        except Exception as e:
            print(f"⚠️ Experience calculation failed: {e}")
        
        # If AI extraction fails, try fallback methods
        return self.fallback_experience(text)
    
    def extract_explicit_experience(self, text):
        """Extract explicitly stated total experience from summary sections"""
        # Look for explicit statements of total experience
        explicit_patterns = [
            r'(?:with|having)\s+(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:professional|work|industry)\s+experience',
            r'(?:professional|work)\s+experience\s*(?:of|:)\s*(\d+(?:\.\d+)?)\+?\s*years?'
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the largest explicitly stated experience
                    return max([float(match.replace('+', '')) for match in matches])
                except:
                    continue
        
        return 0.0
    
    def fallback_experience(self, text):
        """Enhanced fallback experience calculation"""
        # Original patterns
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)'
        ]
        
        # Additional patterns to catch more experience formats
        additional_patterns = [
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(?:professional|work)\s+experience\s*:?\s*(\d+)\+?\s*years?',
            r'(?:career|industry)\s+experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|of)\s+(?:the\s+)?(?:industry|field|profession)'
        ]
        
        patterns.extend(additional_patterns)
        
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match.replace('+', ''))
                    max_years = max(max_years, years)
                except:
                    continue
        
        # Try to extract experience from date ranges as last resort
        if max_years == 0:
            max_years = self.extract_from_date_ranges(text)
        
        return max_years if max_years > 0 else 1.0
    
    def extract_from_date_ranges(self, text):
        """Extract experience by analyzing date ranges in text"""
        # Pattern to find date ranges like "2018-2021" or "Jan 2019 - Dec 2022"
        date_patterns = [
            r'(\d{4})\s*(?:-|to|–|—)\s*(\d{4}|\bpresent\b|\bcurrent\b|\bnow\b)',
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}\s*(?:-|to|–|—)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}'
        ]
        
        total_years = 0
        current_year = 2024  # Assuming current year
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple) and len(match) >= 2:
                        start_year = int(match[0])
                        end_year = current_year if match[1].lower() in ['present', 'current', 'now'] else int(match[1])
                        if start_year <= end_year and end_year <= current_year:
                            # Simple calculation - could be enhanced for more precision
                            years = end_year - start_year
                            total_years += years
                except:
                    continue
        
        return total_years

class ResumeEvaluator:
    def __init__(self, parser):
        self.parser = parser
        self.skill_weight = 0.30
        self.semantic_weight = 0.60
        self.exp_weight = 0.10
    
    def calculate_scores(self, jd_skills, resume_skills, jd_text, resume_text, exp_calculator):
        """Calculate comprehensive evaluation scores"""
        # Skill matching
        skill_score, exact_matches, fuzzy_matches = self.skill_matching(jd_skills, resume_skills)
        
        # Semantic similarity - FIXED VERSION
        semantic_score = self.semantic_similarity(jd_text, resume_text)
        
        # Experience score
        exp_score = self.experience_score(resume_text, exp_calculator)
        
        # Composite score
        composite_score = (
            skill_score * self.skill_weight +
            semantic_score * self.semantic_weight +
            exp_score * self.exp_weight
        )
        
        return {
            'skill_score': skill_score,
            'semantic_score': semantic_score,
            'exp_score': exp_score,
            'composite_score': min(composite_score, 1.0),
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches
        }
    
    def skill_matching(self, jd_skills, resume_skills):
        """Enhanced skill matching with fuzzy logic"""
        if not jd_skills:
            return 0.0, [], []
        
        jd_lower = [s.lower().strip() for s in jd_skills]
        resume_lower = [s.lower().strip() for s in resume_skills]
        
        # Exact matching
        exact_matches = list(set(jd_lower) & set(resume_lower))
        
        # Fuzzy matching
        unmatched_jd = [s for s in jd_lower if s not in exact_matches]
        unmatched_resume = [s for s in resume_lower if s not in exact_matches]
        
        fuzzy_matches = []
        for jd_skill in unmatched_jd:
            for resume_skill in unmatched_resume:
                similarity = fuzz.token_sort_ratio(jd_skill, resume_skill) / 100.0
                if similarity >= 0.75 and resume_skill not in fuzzy_matches:
                    fuzzy_matches.append(resume_skill)
                    break
        
        # Calculate score
        exact_score = len(exact_matches) / len(jd_skills)
        fuzzy_score = len(fuzzy_matches) / len(jd_skills) * 0.6
        total_score = min(exact_score + fuzzy_score, 1.0)
        
        return total_score, exact_matches, fuzzy_matches
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using embeddings - COMPLETELY FIXED VERSION"""
        try:
            # Validate input texts
            if not text1 or not text2 or len(text1.strip()) < 10 or len(text2.strip()) < 10:
                print("⚠️ One or both texts too short for semantic analysis")
                return 0.0
            
            # Generate embeddings with proper error handling
            emb1 = self.parser.generate_embeddings(text1)
            emb2 = self.parser.generate_embeddings(text2)
            
            # Verify embeddings are valid tensors
            if not isinstance(emb1, torch.Tensor) or not isinstance(emb2, torch.Tensor):
                print("⚠️ Invalid embedding tensors")
                return 0.0
            
            # Ensure embeddings are on CPU and properly shaped
            emb1 = emb1.cpu().squeeze()
            emb2 = emb2.cpu().squeeze()
            
            # Verify dimensions match
            if emb1.shape != emb2.shape:
                print(f"⚠️ Embedding dimension mismatch: {emb1.shape} vs {emb2.shape}")
                return 0.0
            
            # Calculate cosine similarity using PyTorch
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1)
            
            # Convert to Python float and ensure valid range
            sim_value = similarity.item()
            
            # Clamp to valid range 
            sim_value = max(0.0, min(1.0, sim_value))
            
            print(f"🔍 Semantic similarity calculated: {sim_value:.4f}")
            return sim_value
            
        except Exception as e:
            print(f"⚠️ Semantic similarity calculation failed: {e}")
            return 0.0
    
    def experience_score(self, resume_text, exp_calculator, min_required=3):
        """Calculate experience score"""
        candidate_years = exp_calculator.calculate_experience(resume_text)
        if min_required == 0:
            return 1.0
        return min(candidate_years / min_required, 1.2)

class ResumeAnalysisPipeline:
    def __init__(self):
        self.parser = LMStudioResumeParser()
        self.pdf_processor = PDFProcessor()
        self.skill_extractor = SkillExtractor(self.parser)
        self.exp_calculator = ExperienceCalculator(self.parser)
        self.evaluator = ResumeEvaluator(self.parser)
        self.jd_skills = []
        self.jd_text = ""
        self.qualification_threshold = 0.55
    
    def process_job_description(self, jd_path):
        """Process job description from file"""
        try:
            with open(jd_path, 'r', encoding='utf-8') as f:
                self.jd_text = f.read()
            
            print(f"🔍 Processing job description...")
            print(f"📄 JD length: {len(self.jd_text):,} characters")
            
            self.jd_skills = self.skill_extractor.extract_skills(self.jd_text, "job description")
            
            print(f"📋 Extracted {len(self.jd_skills)} JD skills:")
            print(f"   {', '.join(self.jd_skills[:10])}{'...' if len(self.jd_skills) > 10 else ''}")
            
            return self.jd_skills
        except Exception as e:
            print(f"❌ Error processing JD: {e}")
            return []
    
    def analyze_resume(self, resume_path):
        """Analyze single resume"""
        resume_text = self.pdf_processor.extract_text_from_pdf(resume_path)
        if not resume_text:
            return None
        
        print(f"\n📄 Analyzing {os.path.basename(resume_path)}...")
        print(f"📄 Resume length: {len(resume_text):,} characters")
        
        # Extract skills
        resume_skills = self.skill_extractor.extract_skills(resume_text, "resume")
        
        # Calculate scores
        scores = self.evaluator.calculate_scores(
            self.jd_skills, resume_skills, self.jd_text, resume_text, self.exp_calculator
        )
        
        # Determine qualification
        qualified = scores['composite_score'] >= self.qualification_threshold
        
        return {
            'filename': os.path.basename(resume_path),
            'path': resume_path,
            'resume_skills': resume_skills,
            'scores': scores,
            'qualified': qualified,
            'status': "✅ Qualified" if qualified else "❌ Not Qualified"
        }
    
    def analyze_multiple_resumes(self, resume_folder):
        """Analyze all PDFs in a folder"""
        results = []
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(resume_folder) if f.lower().endswith('.pdf')]
        
        print(f"\n🔄 Processing {len(pdf_files)} resumes...")
        print("="*80)
        
        for i, filename in enumerate(pdf_files):
            file_path = os.path.join(resume_folder, filename)
            result = self.analyze_resume(file_path)
            
            if result:
                results.append(result)
                scores = result['scores']
                print(f"{i+1:2d}. {filename:<30} | Skills: {len(result['resume_skills']):<3} | "
                      f"Skill: {scores['skill_score']:.3f} | Semantic: {scores['semantic_score']:.3f} | "
                      f"Total: {scores['composite_score']:.3f} | {result['status']}")
        
        # Sort by composite score
        results.sort(key=lambda x: x['scores']['composite_score'], reverse=True)
        return results

def display_results(jd_skills, results):
    """Display comprehensive results"""
    print("\n" + "="*80)
    print("📊 LM STUDIO RESUME ANALYSIS RESULTS (FIXED VERSION)")
    print("="*80)
    
    # JD Summary
    print(f"\n📋 Job Description Skills ({len(jd_skills)}):")
    print(f"   {', '.join(jd_skills[:15])}{'...' if len(jd_skills) > 15 else ''}")
    
    # Results Table
    print(f"\n📈 CANDIDATE RANKING:")
    print("-"*100)
    print(f"{'Rank':<4} {'Filename':<25} {'Skills':<7} {'Skill%':<7} {'Semantic':<9} {'Exp':<5} {'Total':<7} {'Status':<12}")
    print("-"*100)
    
    qualified_count = 0
    for i, result in enumerate(results, 1):
        if result['qualified']:
            qualified_count += 1
        
        scores = result['scores']
        print(f"{i:<4} {result['filename'][:23]:<25} {len(result['resume_skills']):<7} "
              f"{scores['skill_score']:<7.3f} {scores['semantic_score']:<9.3f} "
              f"{scores['exp_score']:<5.2f} {scores['composite_score']:<7.3f} "
              f"{result['status']:<12}")
    
    # Summary
    if results:
        avg_semantic = np.mean([r['scores']['semantic_score'] for r in results])
        print(f"\n🎯 SUMMARY:")
        print(f"   -  Qualified: {qualified_count}/{len(results)} candidates ({qualified_count/len(results)*100:.1f}%)")
        print(f"   -  Average Semantic Score: {avg_semantic:.3f} (FIXED - No longer zero!)")

# Main execution
if __name__ == "__main__":
    print("🚀 LM Studio Resume Analysis System (FIXED VERSION)")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ResumeAnalysisPipeline()
    
    # Process job description
    jd_path = input("\n📁 Enter path to job description file (.txt): ").strip()
    if os.path.exists(jd_path):
        jd_skills = pipeline.process_job_description(jd_path)
        
        if jd_skills:
            # Process resumes
            resume_folder = input("📁 Enter path to resume folder: ").strip()
            if os.path.exists(resume_folder):
                results = pipeline.analyze_multiple_resumes(resume_folder)
                
                if results:
                    display_results(jd_skills, results)
                    
                    # Save results
                    df = pd.DataFrame([{
                        'Filename': r['filename'],
                        'Skills_Count': len(r['resume_skills']),
                        'Skill_Score': r['scores']['skill_score'],
                        'Semantic_Score': r['scores']['semantic_score'],
                        'Experience_Score': r['scores']['exp_score'],
                        'Composite_Score': r['scores']['composite_score'],
                        'Qualified': r['qualified']
                    } for r in results])
                    
                    df.to_csv('resume_analysis_results_fixed.csv', index=False)
                    print(f"\n💾 Results saved to 'resume_analysis_results_fixed.csv'")
                    print("🎉 Semantic scores are now properly calculated!")
                else:
                    print("❌ No resumes processed successfully")
            else:
                print("❌ Resume folder not found")
        else:
            print("❌ Failed to extract skills from job description")
    else:
        print("❌ Job description file not found")
