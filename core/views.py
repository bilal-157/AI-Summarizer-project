from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from .forms import CustomUserCreationForm
# ------------------------------------------------------------------ For Data Extraction From The Urls
import urllib.parse
import urllib.request
import re
from duckduckgo_search import DDGS

import html 
import fitz
import re
import traceback
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO
from datetime import datetime



# ------------------------------------------------------------------ CONSTANTS
MAX_PDF_WORDS            = 25_000
TARGET_SUMMARY_WORDS_MIN = 2_000
CHUNK_SIZE               = 400   # slightly bigger = fewer chunks total
MAX_CHUNKS               = 20    # cap total chunks
MINI_BATCH_SIZE          = 3     # process 3 chunks at a time to avoid RAM freeze

# ------------------------------------------------------------------ MODEL LOAD
print("sshleifer/distilbart-cnn-6-6 ...")
try:
    _MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
    _tokenizer  = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model      = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
    _device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model      = _model.to(_device)
    _model.eval()
    print(f"Model ready on {_device}.")
except Exception as _e:
    _tokenizer = _model = _device = None
    print(f"Model load failed: {_e}")

# ------------------------------------------------------------------ CHUNKING

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split into fixed word chunks, then merge down to MAX_CHUNKS if needed."""
    words  = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    while len(chunks) > MAX_CHUNKS:
        merged = []
        for i in range(0, len(chunks) - 1, 2):
            merged.append(chunks[i] + ' ' + chunks[i + 1])
        if len(chunks) % 2:
            merged.append(chunks[-1])
        chunks = merged
    return chunks

# ------------------------------------------------------------------ MINI-BATCH SUMMARIZE

def summarize_chunks_batched(chunks, mini_batch_size=MINI_BATCH_SIZE):
    """
    Process chunks in small mini-batches (default 3 at a time).
    Prevents RAM overload that causes the system to freeze on large inputs.
    Greedy decode (num_beams=1, early_stopping=False) for maximum speed.
    """
    all_summaries = []

    for batch_start in range(0, len(chunks), mini_batch_size):
        batch = chunks[batch_start: batch_start + mini_batch_size]
        print(f"  Mini-batch {batch_start // mini_batch_size + 1} "
              f"({len(batch)} chunks) ...")

        enc = _tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(_device)

        with torch.no_grad():
            out_ids = _model.generate(
                input_ids            = enc["input_ids"],
                attention_mask       = enc["attention_mask"],
                max_new_tokens       = 180,
                min_new_tokens       = 80,
                num_beams            = 1,      # greedy — fastest
                early_stopping       = False,  # must be False when num_beams=1
                no_repeat_ngram_size = 0,
                length_penalty       = 1.0,
                repetition_penalty   = 1.0,
            )

        decoded = [
            _tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in out_ids
        ]
        all_summaries.extend(decoded)

        # Free GPU/CPU memory after each mini-batch
        del enc, out_ids
        if _device.type == "cuda":
            torch.cuda.empty_cache()

    return all_summaries

# ------------------------------------------------------------------ POST PROCESSING

def clean_summary(text):
    text = re.sub(r'\b([a-z])\s([A-Z])', r'\1. \2', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([,\.;:])', r'\1', text)
    return text.strip()


def pad_to_min_words(summary, source, min_words=TARGET_SUMMARY_WORDS_MIN):
    """
    Safety net: if still under min_words, append highest-scoring TF-IDF
    sentences from the original source that aren't already in the summary.
    Zero hallucination — only real source sentences are added.
    """
    if len(summary.split()) >= min_words:
        return summary
    needed = min_words - len(summary.split())
    print(f"Padding: need {needed} more words from source.")
    sents = [s.strip() for s in source.replace('\n', ' ').split('.') if len(s.split()) > 8]
    if not sents:
        return summary
    try:
        vec    = TfidfVectorizer(stop_words='english')
        mat    = vec.fit_transform(sents)
        scores = np.array(mat.sum(axis=1)).flatten()
        ranked = [sents[i] for i in np.argsort(scores)[::-1]]
        low    = summary.lower()
        extras, added = [], 0
        for s in ranked:
            if ' '.join(s.lower().split()[:6]) in low:
                continue
            extras.append(s)
            added += len(s.split())
            if added >= needed:
                break
        if extras:
            summary = summary.rstrip('.') + '. ' + '. '.join(extras) + '.'
    except Exception:
        pass
    return summary

# ------------------------------------------------------------------ MAIN PIPELINE

def summarize_text_locally(text):
    if _model is None:
        return None, "Summarization model is not available."
    try:
        total = len(text.split())
        print(f"Input words: {total}")

        if total > MAX_PDF_WORDS:
            return None, (
                f"PDF too large ({total:,} words). "
                f"Please upload one under {MAX_PDF_WORDS:,} words."
            )

        source = text  # keep original for padding safety-net

        # 1 — Chunk
        chunks = chunk_text(text)
        print(f"Chunks: {len(chunks)}")

        # 2 — Mini-batched greedy generation (no RAM freeze)
        print(f"Batch generating for {len(chunks)} chunks "
              f"in mini-batches of {MINI_BATCH_SIZE} ...")
        summaries = summarize_chunks_batched(chunks)

        # 3 — Join
        final = ' '.join(summaries)
        print(f"After generation: {len(final.split())} words")

        # 4 — Clean
        final = clean_summary(final)
        print(f"After cleanup: {len(final.split())} words")

        # 5 — Guarantee >= 2000 words
        final = pad_to_min_words(final, source)
        print(f"Final: {len(final.split())} words")
        return final, None

    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")
        return None, "An error occurred during summarization."

# ------------------------------------------------------------------ PDF GENERATION

def generate_summary_pdf(summary_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    title_style    = ParagraphStyle('CT', parent=styles['Title'],
                                    fontSize=22, textColor=colors.HexColor('#1a1a2e'),
                                    spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('CS', parent=styles['Normal'],
                                    fontSize=10, textColor=colors.HexColor('#888888'),
                                    spaceAfter=4, alignment=TA_CENTER)
    stats_style    = ParagraphStyle('CStats', parent=styles['Normal'],
                                    fontSize=9, textColor=colors.HexColor('#aaaaaa'),
                                    spaceAfter=2, alignment=TA_CENTER)
    body_style     = ParagraphStyle('CB', parent=styles['Normal'],
                                    fontSize=11, leading=20,
                                    textColor=colors.HexColor('#222222'),
                                    spaceAfter=10, alignment=TA_JUSTIFY)

    story = [
        Paragraph("Document Summary", title_style),
        Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#dddddd')),
        Spacer(1, 10),
        Paragraph(f"Summary: {len(summary_text.split()):,} words", stats_style),
        Spacer(1, 14),
    ]

    sents  = [s.strip() for s in summary_text.split('.') if s.strip()]
    groups = [sents[i:i + 5] for i in range(0, len(sents), 5)]
    for g in groups:
        story += [Paragraph('. '.join(g) + '.', body_style), Spacer(1, 6)]

    story += [
        Spacer(1, 24),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dddddd')),
        Spacer(1, 6),
        Paragraph("Generated by PDF Summarizer", stats_style),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------------ ARTICLE SEARCH

def fetch_page_text(url, max_words=300, timeout=6):
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                )
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='ignore')

        raw = re.sub(r'<script[^>]*>.*?</script>', ' ', raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r'<style[^>]*>.*?</style>',  ' ', raw, flags=re.DOTALL | re.IGNORECASE)

        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', raw, flags=re.DOTALL | re.IGNORECASE)

        if not paragraphs:
            text = re.sub(r'<[^>]+>', ' ', raw)
        else:
            text = ' '.join(paragraphs)

        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)

        # Remove citation brackets [ 1 ], [1], [ 12 ]
        text = re.sub(r'\[\s*\d+\s*\]', '', text)
        text = re.sub(r'\[\s*edit\s*\]', '', text, flags=re.IGNORECASE)

        # Fix spaces before punctuation: "programmed ." → "programmed."
        text = re.sub(r'\s+([,\.;:])', r'\1', text)

        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        return ' '.join(words[:max_words]) if words else None

    except Exception as e:
        print(f"  Fetch failed for {url}: {e}")
        return None


def clean_snippet(text, max_words=300):
    if not text:
        return ''
    text = html.unescape(text)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    text = re.sub(r'\[\s*edit\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+([,\.;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    snippet = ' '.join(words[:max_words])
    last_dot = snippet.rfind('.')
    if last_dot > len(snippet) * 0.6:
        snippet = snippet[:last_dot + 1]
    return snippet.strip()

# ------------------------------------------------------------------ MAIN FUNCTION
 
def search_articles(query, max_results=6):
    """
    1. Search DuckDuckGo for the query.
    2. Fetch each result URL and extract real paragraph text.
    3. Return a list of dicts with title, url, and scraped content.
    """
    urls_to_fetch = []
 
    # Step 1 — get URLs from DDG
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            urls_to_fetch = [
                {'title': r.get('title', 'No title'), 'url': r.get('href', '#')}
                for r in results if r.get('href')
            ]
            print(f"DDG returned {len(urls_to_fetch)} results.")
    except Exception as e:
        print(f"DDG error: {e}")
 
    # Fallback URLs if DDG fails
    if not urls_to_fetch:
        enc = urllib.parse.quote_plus(query)
        urls_to_fetch = [
            {'title': f'Wikipedia: {query}', 'url': f'https://en.wikipedia.org/wiki/Special:Search?search={enc}&go=Go'},
            {'title': f'arXiv: {query}',     'url': f'https://arxiv.org/search/?query={enc}&searchtype=all'},
        ]
 
    # Step 2 — scrape each URL for real text
    articles = []
    for item in urls_to_fetch:
        url   = item['url']
        title = item['title']
        print(f"  Fetching: {url}")
 
        text = fetch_page_text(url, max_words=300)
        snippet = clean_snippet(text, max_words=300) if text else (
            f"Could not retrieve content from this page. Visit: {url}"
        )
 
        articles.append({
            'title':       title,
            'url':         url,
            'description': snippet,   # full scraped paragraph text
        })
 
    return articles, None
# ------------------------------------------------------------------ PDF EXTRACT

def extract_text_from_pdf(pdf_file):
    try:
        data = pdf_file.read()
        doc  = fitz.open(stream=data, filetype="pdf")
        text = ''.join(p.get_text() for p in doc)
        doc.close()
        return ' '.join(text.split()) or None
    except Exception as e:
        print(f"PDF error: {e}")
        return None

# ------------------------------------------------------------------ VIEWS

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
        messages.error(request, 'Fix errors below.')
    else:
        form = CustomUserCreationForm()
    return render(request, 'core/signup.html', {'form': form})


@login_required
def dashboard(request):
    context = {'error': None, 'summary': None, 'articles': None,
               'search_query': '', 'search_error': None}

    if request.method == 'POST':

        # Article search
        if 'search_query' in request.POST:
            query = request.POST.get('search_query', '').strip()
            context['search_query'] = query
            if query:
                context['articles'], context['search_error'] = search_articles(query)
            else:
                context['search_error'] = "Please enter a topic to search."
            return render(request, 'core/dashboard.html', context)

        # PDF summarization
        if not request.FILES.get('file'):
            context['error'] = "Please upload a PDF file."
            return render(request, 'core/dashboard.html', context)

        pdf_file = request.FILES['file']
        if not pdf_file.name.lower().endswith('.pdf'):
            context['error'] = "Only PDF files are allowed."
            return render(request, 'core/dashboard.html', context)

        extracted = extract_text_from_pdf(pdf_file)
        if not extracted:
            context['error'] = "No readable text found in this PDF."
            return render(request, 'core/dashboard.html', context)

        summary, error = summarize_text_locally(extracted)
        if summary:
            request.session['last_summary'] = summary
            context['summary'] = summary
            messages.success(request, f"Summary generated! ({len(summary.split()):,} words)")
        else:
            context['error'] = error or "Failed to generate summary."
            messages.error(request, context['error'])

    return render(request, 'core/dashboard.html', context)


@login_required
def download_summary_pdf(request):
    summary = request.session.get('last_summary')
    if not summary:
        messages.error(request, "No summary found. Please generate one first.")
        return redirect('dashboard')
    buffer   = generate_summary_pdf(summary)
    filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response