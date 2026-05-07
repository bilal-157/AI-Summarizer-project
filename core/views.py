# core/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CustomUserCreationForm
import fitz  # PyMuPDF
import traceback

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        # Read the PDF file
        pdf_content = pdf_file.read()
        
        # Open with PyMuPDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        doc.close()
        
        if text.strip():
            return text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        print(traceback.format_exc())
        return None

def landing_page(request):
    """Landing page"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'core/landing.html')

def signup(request):
    """Sign up view"""
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
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'core/signup.html', {'form': form})

@login_required
def dashboard(request):
    """Dashboard with PDF summarizer"""
    context = {
        'error': None,
        'text': None,
    }
    
    if request.method == 'POST':
        if not request.FILES.get('file'):
            context['error'] = 'Please select a file to upload.'
            return render(request, 'core/dashboard.html', context)
        
        pdf_file = request.FILES['file']
        
        # Validate file type
        if not pdf_file.name.lower().endswith('.pdf'):
            context['error'] = 'Please upload a PDF file only.'
            return render(request, 'core/dashboard.html', context)
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_file)
        
        if extracted_text:
            context['text'] = extracted_text
            messages.success(request, f'Successfully extracted {len(extracted_text)} characters!')
        else:
            context['error'] = 'No text could be extracted. The PDF might be scanned or image-based.'
    
    return render(request, 'core/dashboard.html', context)