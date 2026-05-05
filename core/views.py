from django.shortcuts import render
import fitz  # PyMuPDF

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def dashboard(request):
    text = ""
    error = ""

    if request.method == "POST":
        file = request.FILES.get('file')

        if not file:
            error = "Please select a file"
        else:
            text = extract_text(file)

    return render(request, "core/dashboard.html", {
        "text": text,
        "error": error
    })