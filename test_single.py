# test_single.py - Enhanced single resume analysis with detailed output
from main import ResumeAnalysisPipeline, PDFProcessor, SkillExtractor, ExperienceCalculator
import os
import json
from datetime import datetime

def test_single_resume():
    """Enhanced test with detailed resume analysis and file outputs"""
    pipeline = ResumeAnalysisPipeline()
    
    # Process JD
    jd_path = input("Enter JD file path: ").strip()
    if not os.path.exists(jd_path):
        print("❌ Job description file not found!")
        return
    
    print("\n🔍 Processing Job Description...")
    jd_skills = pipeline.process_job_description(jd_path)
    
    if not jd_skills:
        print("❌ Failed to extract skills from job description!")
        return
    
    # Process single resume
    resume_path = input("Enter resume PDF path: ").strip()
    if not os.path.exists(resume_path):
        print("❌ Resume file not found!")
        return
    
    print(f"\n📄 Processing Resume: {os.path.basename(resume_path)}")
    
    # Extract detailed resume information
    detailed_analysis = analyze_resume_detailed(pipeline, resume_path, jd_skills)
    
    if detailed_analysis:
        # Display comprehensive results
        display_detailed_results(detailed_analysis)
        
        # Save detailed analysis to files
        save_analysis_files(detailed_analysis)
        
        print(f"\n✅ Detailed analysis completed!")
        print(f"📁 Check the generated files for comprehensive resume breakdown")
    else:
        print("❌ Failed to analyze resume!")

def analyze_resume_detailed(pipeline, resume_path, jd_skills):
    """Perform detailed analysis of a single resume"""
    
    # Extract raw text from PDF
    raw_text = pipeline.pdf_processor.extract_text_from_pdf(resume_path)
    if not raw_text:
        return None
    
    filename = os.path.basename(resume_path)
    print(f"📄 Extracted {len(raw_text):,} characters from PDF")
    
    # Extract skills using the skill extractor
    resume_skills = pipeline.skill_extractor.extract_skills(raw_text, "resume")
    
    # Calculate experience using experience calculator
    experience_years = pipeline.exp_calculator.calculate_experience(raw_text)
    
    # Calculate detailed scores
    scores = pipeline.evaluator.calculate_scores(
        jd_skills, resume_skills, pipeline.jd_text, raw_text, pipeline.exp_calculator
    )
    
    # Determine qualification
    qualified = scores['composite_score'] >= pipeline.qualification_threshold
    
    # Create comprehensive analysis object
    analysis = {
        'filename': filename,
        'file_path': resume_path,
        'raw_text': raw_text,
        'text_stats': {
            'total_characters': len(raw_text),
            'total_words': len(raw_text.split()),
            'total_lines': len(raw_text.splitlines())
        },
        'jd_skills': jd_skills,
        'resume_skills': resume_skills,
        'experience_years': experience_years,
        'scores': scores,
        'qualified': qualified,
        'status': "✅ Qualified" if qualified else "❌ Not Qualified",
        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return analysis

def display_detailed_results(analysis):
    """Display comprehensive analysis results"""
    
    print("\n" + "="*80)
    print("📊 DETAILED RESUME ANALYSIS RESULTS")
    print("="*80)
    
    # Basic Information
    print(f"\n📋 RESUME INFORMATION:")
    print(f"   • Filename: {analysis['filename']}")
    print(f"   • Analysis Time: {analysis['analysis_timestamp']}")
    print(f"   • File Path: {analysis['file_path']}")
    
    # Text Statistics
    stats = analysis['text_stats']
    print(f"\n📄 TEXT EXTRACTION STATISTICS:")
    print(f"   • Total Characters: {stats['total_characters']:,}")
    print(f"   • Total Words: {stats['total_words']:,}")
    print(f"   • Total Lines: {stats['total_lines']:,}")
    print(f"   • Average Words per Line: {stats['total_words']/max(stats['total_lines'], 1):.1f}")
    
    # Job Description Skills
    print(f"\n🎯 JOB DESCRIPTION REQUIREMENTS ({len(analysis['jd_skills'])} skills):")
    for i, skill in enumerate(analysis['jd_skills'], 1):
        print(f"   {i:2d}. {skill}")
    
    # Resume Skills
    print(f"\n💼 RESUME SKILLS EXTRACTED ({len(analysis['resume_skills'])} skills):")
    for i, skill in enumerate(analysis['resume_skills'], 1):
        print(f"   {i:2d}. {skill}")
    
    # Skill Matching Analysis
    scores = analysis['scores']
    print(f"\n🔍 SKILL MATCHING ANALYSIS:")
    print(f"   • Exact Matches ({len(scores['exact_matches'])}): {', '.join(scores['exact_matches'][:10])}")
    if len(scores['exact_matches']) > 10:
        print(f"     ... and {len(scores['exact_matches']) - 10} more")
    
    print(f"   • Fuzzy Matches ({len(scores['fuzzy_matches'])}): {', '.join(scores['fuzzy_matches'][:10])}")
    if len(scores['fuzzy_matches']) > 10:
        print(f"     ... and {len(scores['fuzzy_matches']) - 10} more")
    
    missing_skills = set(analysis['jd_skills']) - set(scores['exact_matches']) - set(scores['fuzzy_matches'])
    print(f"   • Missing Skills ({len(missing_skills)}): {', '.join(list(missing_skills)[:10])}")
    if len(missing_skills) > 10:
        print(f"     ... and {len(missing_skills) - 10} more")
    
    # Experience Analysis
    print(f"\n🕒 EXPERIENCE ANALYSIS:")
    print(f"   • Total Experience: {analysis['experience_years']} years")
    print(f"   • Experience Score: {scores['exp_score']:.3f}")
    print(f"   • Experience Rating: {get_experience_rating(analysis['experience_years'])}")
    
    # Detailed Scoring Breakdown
    print(f"\n📊 DETAILED SCORING BREAKDOWN:")
    print(f"   • Skill Matching Score: {scores['skill_score']:.3f} (Weight: 30%)")
    print(f"     - Exact matches contribute: {len(scores['exact_matches'])/len(analysis['jd_skills'])*100:.1f}%")
    print(f"     - Fuzzy matches contribute: {len(scores['fuzzy_matches'])/len(analysis['jd_skills'])*100:.1f}%")
    
    print(f"   • Semantic Similarity Score: {scores['semantic_score']:.3f} (Weight: 60%)")
    print(f"     - Measures contextual alignment between JD and resume")
    
    print(f"   • Experience Score: {scores['exp_score']:.3f} (Weight: 10%)")
    print(f"     - Based on {analysis['experience_years']} years of experience")
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    print(f"   • Composite Score: {scores['composite_score']:.3f}")
    print(f"   • Qualification Threshold: {0.55:.2f}")
    print(f"   • Status: {analysis['status']}")
    print(f"   • Confidence Level: {get_confidence_level(scores['composite_score'])}")

def get_experience_rating(years):
    """Get experience rating based on years"""
    if years >= 10:
        return "Senior (10+ years)"
    elif years >= 5:
        return "Mid-level (5-10 years)"
    elif years >= 2:
        return "Junior (2-5 years)"
    else:
        return "Entry-level (<2 years)"

def get_confidence_level(score):
    """Get confidence level based on composite score"""
    if score >= 0.8:
        return "Very High"
    elif score >= 0.7:
        return "High"
    elif score >= 0.6:
        return "Moderate"
    elif score >= 0.4:
        return "Low"
    else:
        return "Very Low"

def save_analysis_files(analysis):
    """Save detailed analysis to multiple output files"""
    
    base_filename = os.path.splitext(analysis['filename'])[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save extracted text to file
    text_filename = f"{base_filename}_extracted_text_{timestamp}.txt"
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(f"EXTRACTED TEXT FROM: {analysis['filename']}\n")
        f.write(f"EXTRACTION DATE: {analysis['analysis_timestamp']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("TEXT STATISTICS:\n")
        f.write("-"*40 + "\n")
        stats = analysis['text_stats']
        f.write(f"Total Characters: {stats['total_characters']:,}\n")
        f.write(f"Total Words: {stats['total_words']:,}\n")
        f.write(f"Total Lines: {stats['total_lines']:,}\n\n")
        
        f.write("EXTRACTED TEXT CONTENT:\n")
        f.write("="*80 + "\n")
        f.write(analysis['raw_text'])
    
    # 2. Save skills analysis to JSON
    skills_filename = f"{base_filename}_skills_analysis_{timestamp}.json"
    skills_data = {
        'filename': analysis['filename'],
        'analysis_timestamp': analysis['analysis_timestamp'],
        'jd_skills': analysis['jd_skills'],
        'resume_skills': analysis['resume_skills'],
        'skill_matching': {
            'exact_matches': analysis['scores']['exact_matches'],
            'fuzzy_matches': analysis['scores']['fuzzy_matches'],
            'missing_skills': list(set(analysis['jd_skills']) - set(analysis['scores']['exact_matches']) - set(analysis['scores']['fuzzy_matches']))
        },
        'scores': analysis['scores']
    }
    
    with open(skills_filename, 'w', encoding='utf-8') as f:
        json.dump(skills_data, f, indent=2, ensure_ascii=False)
    
    # 3. Save comprehensive analysis report
    report_filename = f"{base_filename}_detailed_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE RESUME ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Candidate: {analysis['filename']}\n")
        f.write(f"Analysis Date: {analysis['analysis_timestamp']}\n")
        f.write(f"Status: {analysis['status']}\n")
        f.write(f"Composite Score: {analysis['scores']['composite_score']:.3f}\n\n")
        
        f.write("SCORING BREAKDOWN:\n")
        f.write("-"*30 + "\n")
        f.write(f"Skill Score: {analysis['scores']['skill_score']:.3f} (30% weight)\n")
        f.write(f"Semantic Score: {analysis['scores']['semantic_score']:.3f} (60% weight)\n")
        f.write(f"Experience Score: {analysis['scores']['exp_score']:.3f} (10% weight)\n\n")
        
        f.write("EXPERIENCE ANALYSIS:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Experience: {analysis['experience_years']} years\n")
        f.write(f"Experience Rating: {get_experience_rating(analysis['experience_years'])}\n\n")
        
        f.write("SKILL MATCHING DETAILS:\n")
        f.write("-"*30 + "\n")
        f.write(f"JD Skills Required ({len(analysis['jd_skills'])}):\n")
        for i, skill in enumerate(analysis['jd_skills'], 1):
            f.write(f"  {i:2d}. {skill}\n")
        
        f.write(f"\nResume Skills Found ({len(analysis['resume_skills'])}):\n")
        for i, skill in enumerate(analysis['resume_skills'], 1):
            f.write(f"  {i:2d}. {skill}\n")
        
        f.write(f"\nExact Matches ({len(analysis['scores']['exact_matches'])}):\n")
        for i, skill in enumerate(analysis['scores']['exact_matches'], 1):
            f.write(f"  {i:2d}. {skill}\n")
        
        f.write(f"\nFuzzy Matches ({len(analysis['scores']['fuzzy_matches'])}):\n")
        for i, skill in enumerate(analysis['scores']['fuzzy_matches'], 1):
            f.write(f"  {i:2d}. {skill}\n")
        
        missing_skills = list(set(analysis['jd_skills']) - set(analysis['scores']['exact_matches']) - set(analysis['scores']['fuzzy_matches']))
        f.write(f"\nMissing Skills ({len(missing_skills)}):\n")
        for i, skill in enumerate(missing_skills, 1):
            f.write(f"  {i:2d}. {skill}\n")
        
        f.write(f"\nRECOMMENDATIONS:\n")
        f.write("-"*30 + "\n")
        if analysis['qualified']:
            f.write("✅ RECOMMENDED FOR INTERVIEW\n")
            f.write("Strong candidate with good skill alignment and relevant experience.\n")
        else:
            f.write("❌ NOT RECOMMENDED\n")
            f.write("Candidate lacks sufficient skill alignment or experience for this role.\n")
        
        if missing_skills:
            f.write(f"\nSkill gaps to address: {', '.join(missing_skills[:5])}\n")
    
    # Print file generation summary
    print(f"\n📁 GENERATED FILES:")
    print(f"   1. {text_filename} - Extracted text content")
    print(f"   2. {skills_filename} - Skills analysis (JSON)")
    print(f"   3. {report_filename} - Comprehensive report")

if __name__ == "__main__":
    test_single_resume()
