from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from .forms import CustomUserCreationForm

import fitz
import re
import traceback
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO
from datetime import datetime

from duckduckgo_search import DDGS

# ---------------- CONSTANTS ----------------

MAX_PDF_WORDS    = 25_000
NUM_BEAMS        = 1
MAX_MODEL_TOKENS = 512

# ---------------- MODEL (loaded once at startup) ----------------

print("⏳ Loading summarization model at startup...")
try:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=8,
    )
    print("✅ Model loaded and cached.")
except Exception as _e:
    summarizer = None
    print(f"❌ Model failed to load: {_e}")


# ---------------- PRE/POST PROCESSING ----------------

def extract_important_sentences(text, keep_ratio=0.3):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.split()) > 5]
    if len(sentences) < 3:
        return text
    try:
        vectorizer   = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
        threshold    = np.percentile(scores, (1 - keep_ratio) * 100)
        important    = [sentences[i] for i, s in enumerate(scores) if s >= threshold]
        return '. '.join(important)
    except Exception:
        return text


def remove_redundancy(text, threshold=0.95):
    sentences = [s.strip() for s in text.split('.') if len(s.split()) > 4]
    if len(sentences) < 2:
        return text
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf      = vectorizer.fit_transform(sentences)
        kept       = [0]
        for i in range(1, len(sentences)):
            sims = cosine_similarity(tfidf[i], tfidf[kept]).flatten()
            if max(sims) < threshold:
                kept.append(i)
        return '. '.join([sentences[i] for i in kept]) + '.'
    except Exception:
        return text


def clean_summary(text):
    text = re.sub(r'\b([a-z])\s([A-Z])', r'\1. \2', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([,\.;:])', r'\1', text)
    return text.strip()


# ---------------- TEXT PROCESSING ----------------

def chunk_text(text):
    total_words = len(text.split())
    if total_words < 2_000:
        size = 700
    elif total_words < 5_000:
        size = 680
    else:
        size = 650
    words = text.split()
    for i in range(0, len(words), size):
        yield ' '.join(words[i:i + size])


def summarize_chunks_batched(chunks, total_words):
    if not chunks:
        return []
    num_chunks = len(chunks)
    target_total_words      = max(1_000, int(total_words * 0.10))
    target_per_chunk_words  = target_total_words // num_chunks
    target_per_chunk_tokens = int(target_per_chunk_words * 1.3)
    max_tok = min(200, max(80, target_per_chunk_tokens))
    min_tok = max(40, int(max_tok * 0.5))
    print(f"🎯 Per-chunk: max={max_tok} tokens, min={min_tok} tokens "
          f"(target {target_total_words} words total from {num_chunks} chunks)")
    results = summarizer(
        chunks,
        max_length=max_tok,
        min_length=min_tok,
        do_sample=False,
        num_beams=NUM_BEAMS,
        early_stopping=False,
        truncation=True,
    )
    return [r['summary_text'] for r in results]


# ---------------- MAIN PIPELINE ----------------

def summarize_text_locally(text):
    if summarizer is None:
        return None, "Summarization model is not available."
    try:
        total_words = len(text.split())
        print(f"📊 Original word count: {total_words}")

        if total_words > MAX_PDF_WORDS:
            approx_pages = total_words // 250
            limit_pages  = MAX_PDF_WORDS // 250
            return None, (
                f"This PDF is too large ({total_words:,} words, ~{approx_pages} pages). "
                f"Please upload a PDF under {MAX_PDF_WORDS:,} words (~{limit_pages} pages)."
            )

        text = extract_important_sentences(text, keep_ratio=0.3)
        print(f"🔍 After extractive filter: {len(text.split())} words (from {total_words})")

        chunks = list(chunk_text(text))
        print(f"📦 Chunks: {len(chunks)}")

        chunk_summaries = summarize_chunks_batched(chunks, total_words)
        final = ' '.join(chunk_summaries)
        print(f"📝 After batched summarization: {len(final.split())} words")

        final = remove_redundancy(final, threshold=0.95)
        final = clean_summary(final)

        print(f"✅ Final: {len(final.split())} words")
        return final, None

    except Exception as e:
        print(f"Summarization Error: {e}")
        print(traceback.format_exc())
        return None, "An error occurred during summarization."


# ---------------- PDF GENERATION ----------------

def generate_summary_pdf(summary_text):
    """Build a formatted PDF from summary text, return as BytesIO buffer."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#888888'),
        spaceAfter=4, alignment=TA_CENTER,
    )
    stats_style = ParagraphStyle(
        'Stats', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#aaaaaa'),
        spaceAfter=2, alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'],
        fontSize=11, leading=20,
        textColor=colors.HexColor('#222222'),
        spaceAfter=10, alignment=TA_JUSTIFY,
    )

    story = []
    story.append(Paragraph("Document Summary", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style,
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#dddddd')))
    story.append(Spacer(1, 10))

    word_count = len(summary_text.split())
    story.append(Paragraph(f"Summary: {word_count} words", stats_style))
    story.append(Spacer(1, 14))

    sentences   = [s.strip() for s in summary_text.split('.') if s.strip()]
    para_groups = [sentences[i:i + 5] for i in range(0, len(sentences), 5)]
    for group in para_groups:
        story.append(Paragraph('. '.join(group) + '.', body_style))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dddddd')))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Generated by PDF Summarizer", stats_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------- ARTICLE SEARCH ----------------

def search_articles(query, max_results=6):
    """
    Try DuckDuckGo first. If it returns nothing (rate-limited),
    fall back to returning Google search links directly.
    """
    # ── Try DuckDuckGo ─────────────────────────────────────────
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if results:
            articles = []
            for r in results:
                articles.append({
                    'title':       r.get('title', 'No title'),
                    'url':         r.get('href',  '#'),
                    'description': r.get('body',  '')[:200],
                })
            return articles, None

    except Exception as e:
        print(f"DDG error: {e}")

    # ── Fallback: return Google search links ───────────────────
    # DDG was rate-limited — give user direct Google search links
    import urllib.parse
    encoded = urllib.parse.quote_plus(query)

    fallback = [
        {
            'title':       f'Google Search: {query}',
            'url':         f'https://www.google.com/search?q={encoded}',
            'description': f'Search Google for articles about "{query}".',
        },
        {
            'title':       f'Google Scholar: {query}',
            'url':         f'https://scholar.google.com/scholar?q={encoded}',
            'description': f'Find academic papers and research articles about "{query}".',
        },
        {
            'title':       f'Wikipedia: {query}',
            'url':         f'https://en.wikipedia.org/wiki/Special:Search?search={encoded}',
            'description': f'Read the Wikipedia article about "{query}".',
        },
        {
            'title':       f'arXiv: {query}',
            'url':         f'https://arxiv.org/search/?query={encoded}&searchtype=all',
            'description': f'Find research papers on arXiv about "{query}".',
        },
    ]
    return fallback, None

# ---------------- PDF EXTRACTION ----------------

def extract_text_from_pdf(pdf_file):
    try:
        pdf_content = pdf_file.read()
        doc  = fitz.open(stream=pdf_content, filetype="pdf")
        text = ''.join(page.get_text() for page in doc)
        doc.close()
        return ' '.join(text.split()) or None
    except Exception as e:
        print(f"PDF error: {e}")
        return None


# ---------------- VIEWS ----------------

def landing_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'core/landing.html')


def signup(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome {user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Fix errors below.')
    else:
        form = CustomUserCreationForm()
    return render(request, 'core/signup.html', {'form': form})


@login_required
def dashboard(request):
    context = {
        'error':    None,
        'summary':  None,
        'articles': None,
        'search_query': '',
        'search_error': None,
    }

    if request.method == 'POST':

        # ── Article search ─────────────────────────────────────────
        if 'search_query' in request.POST:
            query = request.POST.get('search_query', '').strip()
            context['search_query'] = query
            if query:
                articles, search_error = search_articles(query)
                context['articles']     = articles
                context['search_error'] = search_error
            else:
                context['search_error'] = "Please enter a topic to search."
            return render(request, 'core/dashboard.html', context)

        # ── PDF summarization ──────────────────────────────────────
        if not request.FILES.get('file'):
            context['error'] = "Please upload a PDF file."
            return render(request, 'core/dashboard.html', context)

        pdf_file = request.FILES['file']

        if not pdf_file.name.lower().endswith('.pdf'):
            context['error'] = "Only PDF files are allowed."
            return render(request, 'core/dashboard.html', context)

        extracted_text = extract_text_from_pdf(pdf_file)
        if not extracted_text:
            context['error'] = "No readable text found in this PDF."
            return render(request, 'core/dashboard.html', context)

        summary, error = summarize_text_locally(extracted_text)

        if summary:
            request.session['last_summary'] = summary
            context['summary'] = summary
            messages.success(request, "✅ Summary generated!")
        else:
            context['error'] = error or "Failed to generate summary."
            messages.error(request, context['error'])

    return render(request, 'core/dashboard.html', context)


@login_required
def download_summary_pdf(request):
    """Serve the last generated summary as a downloadable PDF."""
    summary = request.session.get('last_summary')
    if not summary:
        messages.error(request, "No summary found. Please generate one first.")
        return redirect('dashboard')

    buffer   = generate_summary_pdf(summary)
    filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response