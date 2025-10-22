"""Extract text from PDF"""
import pypdf
import sys

pdf_path = 'data/tmp/MMA-AI.net.pdf'

try:
    with open(pdf_path, 'rb') as pdf_file:
        reader = pypdf.PdfReader(pdf_file)
        print(f"Total pages: {len(reader.pages)}\n")
        
        for i, page in enumerate(reader.pages):
            print(f"\n{'='*80}")
            print(f"PAGE {i+1}")
            print('='*80)
            text = page.extract_text()
            print(text)
            
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

