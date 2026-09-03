# ==============================================================================
# 1. SETUP & DEPENDENCIES INSTALLATION
# ==============================================================================
import os
import sys

try:
    import langchain
    import reportlab
    import pydantic
except ImportError:
     subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "langchain", "langchain-core", "langchain-community", 
                          "langchain-groq", "reportlab", "pydantic"])

import re
import html
import random
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)

# ==============================================================================
# 2. ENVIRONMENT & MODEL SETUP
# ==============================================================================
# Read credentials securely from environment variables / GitHub Actions Secrets
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

# LangSmith Tracing Guard (enables tracing only if key is supplied)
ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if ls_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "grade5-au-quiz-agent")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Production reasoning model with ample token space
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=4096
)

# ==============================================================================
# 3. STRUCTURED DATA MODELS (Pydantic)
# ==============================================================================
class MCQQuestion(BaseModel):
    id: int = Field(description="Sequential question number")
    subject: str = Field(description="Subject name")
    question_text: str = Field(description="Question prompt")
    option_a: str = Field(description="Option A")
    option_b: str = Field(description="Option B")
    option_c: str = Field(description="Option C")
    option_d: str = Field(description="Option D")
    correct_option: str = Field(description="Must be strictly A, B, C, or D")
    step_by_step_working: str = Field(description="Step-by-step calculation or deduction")
    final_explanation: str = Field(description="Year 5 calibrated explanation")

class SectionQuestions(BaseModel):
    questions: List[MCQQuestion]

class EnglishSection(BaseModel):
    passage: str = Field(description="250-300 word Australian text")
    questions: List[MCQQuestion]

class QuizAssessment(BaseModel):
    math_topic: str
    science_topic: str
    logical_reasoning_topic: str
    english_topic: str
    comprehension_passage: str
    questions: List[MCQQuestion]

# ==============================================================================
# 4. RANDOM TOPIC POOLS
# ==============================================================================
LOGICAL_REASONING_POOLS = [
    "Number and Letter Sequence Patterns",
    "Spatial and Directional Reasoning (Compass & Grids)",
    "Venn Diagrams and Categorical Deductions",
    "Truth and Lie Logic Puzzles",
    "Calendar and Time-Shift Logic",
    "Comparative Weight and Balance Deductions"
]

ENGLISH_TOPICS_POOLS = [
    "The Secrets of the Great Barrier Reef Coral Spawning",
    "How the Platypus Hunts Using Electroreception",
    "The First Crossing of the Blue Mountains",
    "Renewable Energy in Regional Australia: Wind and Solar Farms",
    "The Mystery of the Tasmanian Tiger (Thylacine)",
    "Life on a Cattle Station in the Northern Territory"
]

# ==============================================================================
# 5. GENERATION CHAINS (Anti-LaTeX Prompting + Chunked Execution)
# ==============================================================================
section_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Australian primary school teacher for ACARA Year 5.
Generate exactly {count} multiple-choice questions (4 options: A, B, C, D) for: {subject}.
Questions must be numbered from ID {start_id} to ID {end_id}.

CRITICAL FORMATTING RULES FOR MATHEMATICS:
- DO NOT use LaTeX commands (NEVER write \\frac, \\dfrac, \\times, or delimiters like \\( \\) or $ $).
- Write all fractions using standard slashes: e.g., '1/3', '2/5', or mixed numerals like '2 1/4'.
- Use standard readable unicode symbols where needed: × for multiplication, ÷ for division, °C for temperature.
- Australian spelling (colour, centimetre, litre), metric units, and AUD ($).
- For every question, compute and provide the full 'step_by_step_working'.
- Ensure 'correct_option' strictly matches the working out.
- Distractors must reflect common Year 5 calculation errors."""),
    ("human", "Topic: {topic}")
])

english_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Australian Year 5 English educator.
Write an engaging 250-300 word Australian informational text or narrative on: {topic}.
Then provide exactly 5 multiple-choice questions (IDs 21 to 25) assessing:
- Factual recall
- Inferencing
- Vocabulary in context
- Author intent

For each question provide 'step_by_step_working' (quote reference from passage) and 'final_explanation'."""),
    ("human", "Generate the reading text and 5 comprehension MCQs.")
])

section_chain = section_prompt | llm.with_structured_output(SectionQuestions)
english_chain = english_prompt | llm.with_structured_output(EnglishSection)

def generate_assessment(math_topic: str, science_topic: str) -> QuizAssessment:
    logic_topic = random.choice(LOGICAL_REASONING_POOLS)
    english_topic = random.choice(ENGLISH_TOPICS_POOLS)
    
    print("[*] Generating 25-Question Assessment in safe modular chunks:")
    
    # Section 1: Math (Q1 - Q8)
    print("    -> Generating Mathematics (Q1-Q8)...")
    math_res = section_chain.invoke({
        "subject": "Mathematics",
        "topic": math_topic,
        "count": 8,
        "start_id": 1,
        "end_id": 8
    })
    for i, q in enumerate(math_res.questions):
        q.id = 1 + i
        q.subject = "Mathematics"

    # Section 2: Science (Q9 - Q14)
    print("    -> Generating Science (Q9-Q14)...")
    science_res = section_chain.invoke({
        "subject": "Science",
        "topic": science_topic,
        "count": 6,
        "start_id": 9,
        "end_id": 14
    })
    for i, q in enumerate(science_res.questions):
        q.id = 9 + i
        q.subject = "Science"

    # Section 3: Logical Reasoning (Q15 - Q20)
    print("    -> Generating Logical Reasoning (Q15-Q20)...")
    logic_res = section_chain.invoke({
        "subject": "Logical Reasoning",
        "topic": logic_topic,
        "count": 6,
        "start_id": 15,
        "end_id": 20
    })
    for i, q in enumerate(logic_res.questions):
        q.id = 15 + i
        q.subject = "Logical Reasoning"

    # Section 4: English Comprehension (Q21 - Q25)
    print("    -> Generating English Passage & MCQs (Q21-Q25)...")
    english_res = english_chain.invoke({
        "topic": english_topic
    })
    for i, q in enumerate(english_res.questions):
        q.id = 21 + i
        q.subject = "English Comprehension"

    all_questions = (
        math_res.questions + 
        science_res.questions + 
        logic_res.questions + 
        english_res.questions
    )
    
    print(f"[+] Successfully generated and validated {len(all_questions)} questions.")

    return QuizAssessment(
        math_topic=math_topic,
        science_topic=science_topic,
        logical_reasoning_topic=logic_topic,
        english_topic=english_topic,
        comprehension_passage=english_res.passage,
        questions=all_questions
    )

# ==============================================================================
# 6. REPORTLAB TEXT SANITIZER & PDF GENERATOR
# ==============================================================================
def format_math_for_reportlab(text: str) -> str:
    """
    Cleans LaTeX notation, fractions, and symbols into clean, 
    printable text for ReportLab Paragraphs.
    """
    if not text:
        return ""

    # 1. Strip math environment delimiters \( ... \), \[ ... \], and $ ... $
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text)

    # 2. Convert LaTeX fractions: \frac{a}{b} or \dfrac{a}{b} -> a/b
    text = re.sub(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)

    # 3. Common math operators and replacements
    replacements = {
        r'\times': '×',
        r'\div': '÷',
        r'\pm': '±',
        r'\leq': '≤',
        r'\geq': '≥',
        r'\neq': '≠',
        r'\approx': '≈',
        r'\degree': '°',
        r'^\circ': '°',
        r'\cdot': '·',
        r'\quad': ' ',
        r'\qquad': '  ',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 4. Clean up any leftover lone backslashes from simple commands
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)

    # 5. Escape bare '&', '<', '>' so ReportLab's XML parser doesn't crash
    text = html.escape(text, quote=False)
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")

    return text.strip()

def create_pdf_documents(assessment: QuizAssessment, q_filename="Grade_5_Morning_Challenge.pdf", a_filename="Grade_5_Answer_Key_and_Explanations.pdf"):
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=15, leading=19, alignment=1, textColor=colors.HexColor('#1E3A8A')
    )
    sub_title_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontSize=8.5, leading=11, alignment=1, textColor=colors.HexColor('#475569')
    )
    h2_style = ParagraphStyle(
        'SectionH2', parent=styles['Heading2'], fontSize=11, leading=15, textColor=colors.HexColor('#0F172A'), spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        'QBody', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#1E293B')
    )
    option_style = ParagraphStyle(
        'QOption', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#334155'), leftIndent=12
    )
    passage_style = ParagraphStyle(
        'Passage', parent=styles['Normal'], fontSize=8.5, leading=12.5, textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F8FAFC'), borderColor=colors.HexColor('#CBD5E1'), borderWidth=0.5, borderPadding=6, spaceAfter=8
    )
    working_style = ParagraphStyle(
        'Working', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#047857')
    )

    # -------------------------------------------------------------
    # PDF 1: STUDENT QUESTION SHEET
    # -------------------------------------------------------------
    doc_q = SimpleDocTemplate(q_filename, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    elements_q = []
    
    elements_q.append(Paragraph("<b>YEAR 5 DAILY INDEPENDENT CHALLENGE</b>", title_style))
    elements_q.append(Paragraph("Australian Curriculum Calibrated &bull; Target Time: 60 Minutes &bull; Total Marks: 25", sub_title_style))
    elements_q.append(Spacer(1, 8))
    
    student_bar = [
        [Paragraph("<b>Student Name:</b> ___________________________", body_style),
         Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %B %Y')}", body_style),
         Paragraph("<b>Score:</b> ____ / 25", body_style)]
    ]
    t_box = Table(student_bar, colWidths=[240, 150, 140])
    t_box.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94A3B8')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    elements_q.append(t_box)
    elements_q.append(Spacer(1, 8))

    current_subj = ""
    for q in assessment.questions:
        if q.subject != current_subj:
            current_subj = q.subject
            elements_q.append(Paragraph(f"<b>Section: {current_subj.upper()}</b>", h2_style))
            
            if "English" in current_subj:
                elements_q.append(Paragraph(f"<b>Read the passage and answer Questions {q.id} to 25:</b>", body_style))
                elements_q.append(Spacer(1, 4))
                clean_passage = format_math_for_reportlab(assessment.comprehension_passage).replace("\n", "<br/>")
                elements_q.append(Paragraph(clean_passage, passage_style))
        
        q_content = [
            Paragraph(f"<b>Q{q.id}.</b> {format_math_for_reportlab(q.question_text)}", body_style),
            Spacer(1, 2),
            Paragraph(f"<b>[ A ]</b> {format_math_for_reportlab(q.option_a)}", option_style),
            Paragraph(f"<b>[ B ]</b> {format_math_for_reportlab(q.option_b)}", option_style),
            Paragraph(f"<b>[ C ]</b> {format_math_for_reportlab(q.option_c)}", option_style),
            Paragraph(f"<b>[ D ]</b> {format_math_for_reportlab(q.option_d)}", option_style),
            Spacer(1, 6)
        ]
        elements_q.append(KeepTogether(q_content))

    doc_q.build(elements_q)
    print(f"[+] Successfully generated: {q_filename}")

    # -------------------------------------------------------------
    # PDF 2: ANSWER KEY & STEP-BY-STEP EXPLANATIONS
    # -------------------------------------------------------------
    doc_a = SimpleDocTemplate(a_filename, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    elements_a = []
    
    elements_a.append(Paragraph("<b>YEAR 5 CHALLENGE &mdash; SOLUTIONS & WORKING GUIDE</b>", title_style))
    elements_a.append(Paragraph(f"Educator Marking Guide &bull; {datetime.now().strftime('%d %B %Y')}", sub_title_style))
    elements_a.append(Spacer(1, 8))

    # Answer matrix table (5x5 grid)
    matrix_data = [["Q#", "Ans", "Q#", "Ans", "Q#", "Ans", "Q#", "Ans", "Q#", "Ans"]]
    for r in range(5):
        row = []
        for c in range(5):
            idx = r + c * 5
            if idx < len(assessment.questions):
                row.extend([f"Q{assessment.questions[idx].id}", f"<b>{assessment.questions[idx].correct_option}</b>"])
        matrix_data.append([Paragraph(cell, body_style) for cell in row])
    
    ans_table = Table(matrix_data, colWidths=[28, 78]*5)
    ans_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    elements_a.append(ans_table)
    elements_a.append(Spacer(1, 10))
    
    elements_a.append(Paragraph("<b>Step-by-Step Solutions & Working Out</b>", h2_style))
    elements_a.append(Spacer(1, 4))

    for q in assessment.questions:
        sol_content = [
            Paragraph(f"<b>Q{q.id} ({q.subject}) &mdash; Correct Answer: [{q.correct_option}]</b>", body_style),
            Paragraph(f"<b>Working Out:</b> {format_math_for_reportlab(q.step_by_step_working)}", working_style),
            Paragraph(f"<b>Explanation:</b> {format_math_for_reportlab(q.final_explanation)}", body_style),
            Spacer(1, 6)
        ]
        elements_a.append(KeepTogether(sol_content))

    doc_a.build(elements_a)
    print(f"[+] Successfully generated: {a_filename}")

# ==============================================================================
# 7. EMAIL DISPATCH SYSTEM
# ==============================================================================
def send_email_with_two_attachments(pdf_question_path: str, pdf_answer_path: str):
    if "your_actual_gmail" in EMAIL_SENDER or not EMAIL_PASSWORD or "your_16_char" in EMAIL_PASSWORD:
        print("[!] Valid Gmail credentials not configured. Skipping email dispatch.")
        print("[!] The PDFs have been generated in your Colab files panel.")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"Year 5 Morning Challenge & Answer Key - {datetime.now().strftime('%d %b %Y')}"

    body = """Good morning,

Attached are today's Grade 5 Australian Curriculum challenge sets:

1. Grade_5_Morning_Challenge.pdf (Student 25-MCQ Question Paper)
2. Grade_5_Answer_Key_and_Explanations.pdf (Full Step-by-Step Solutions & Mark Scheme)

Designed for independent completion within 60 minutes.
"""
    msg.attach(MIMEText(body, 'plain'))

    for filepath in [pdf_question_path, pdf_answer_path]:
        with open(filepath, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(filepath)}")
        msg.attach(part)

    print(f"[*] Dispatching email to {EMAIL_RECEIVER} via SMTP...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    print("[+] Email delivered successfully!")

# ==============================================================================
# 8. EXECUTION TRIGGER
# ==============================================================================
if __name__ == "__main__":
    # Choose specific Math and Science topics:
    MATH_TOPIC = "Fractions, Decimals and Percentages (including Australian money calculations)"
    SCIENCE_TOPIC = "States of matter (solids, liquids, gases and temperature changes)"

    # Run the generation pipeline
    quiz = generate_assessment(math_topic=MATH_TOPIC, science_topic=SCIENCE_TOPIC)
    
    q_pdf = "Grade_5_Morning_Challenge.pdf"
    a_pdf = "Grade_5_Answer_Key_and_Explanations.pdf"
    
    create_pdf_documents(quiz, q_filename=q_pdf, a_filename=a_pdf)
    send_email_with_two_attachments(q_pdf, a_pdf)
