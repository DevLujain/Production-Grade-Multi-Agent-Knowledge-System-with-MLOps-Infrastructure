import os
import json
from pathlib import Path
from datetime import datetime

class DocumentProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.documents = []
    
    # Extract text from markdown or text files
    def extract_text(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")
            return None
    
    # Clean the text
    def clean_text(self, text):
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove weird symbols
        text = text.replace('\x00', '')
        text = text.replace('\n\n\n', '\n')
        
        return text.strip()
    
    # Process all documents
    def process_all_documents(self):
        doc_id = 1
        
        # Walk through all folders
        for root, dirs, files in os.walk(self.input_folder):
            for filename in files:
                # Only process markdown and text files
                if filename.endswith(('.md', '.txt')):
                    filepath = os.path.join(root, filename)
                    print(f"Processing: {filename}")
                    
                    # Extract text
                    text = self.extract_text(filepath)
                    
                    if not text:
                        print(f"  ❌ Failed to extract: {filename}")
                        continue
                    
                    # Clean the text
                    clean_text = self.clean_text(text)
                    
                    # Skip if too short
                    if len(clean_text) < 50:
                        print(f"  ⚠️  Too short, skipping")
                        continue
                    
                    # Create document object
                    document = {
                        "doc_id": f"doc_{doc_id}",
                        "title": filename.replace('.md', '').replace('.txt', ''),
                        "content": clean_text,
                        "word_count": len(clean_text.split()),
                        "character_count": len(clean_text),
                        "processed_date": datetime.now().isoformat(),
                        "source_file": filename,
                        "source_path": filepath
                    }
                    
                    self.documents.append(document)
                    doc_id += 1
                    print(f"  ✅ Processed ({len(clean_text)} chars)")
        
        return self.documents
    
    # Save to JSON file
    def save_documents(self):
        output_path = os.path.join(self.output_folder, "processed_documents.json")
        
        # Create output folder if doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved {len(self.documents)} documents to {output_path}")
        
        # Print statistics
        total_words = sum(doc['word_count'] for doc in self.documents)
        total_chars = sum(doc['character_count'] for doc in self.documents)
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total documents: {len(self.documents)}")
        print(f"   Total words: {total_words:,}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average words per document: {total_words // len(self.documents) if self.documents else 0}")

# Use it
if __name__ == "__main__":
    processor = DocumentProcessor(
        input_folder="data/raw",
        output_folder="data/processed"
    )
    
    print("🔄 Starting document processing...\n")
    processor.process_all_documents()
    processor.save_documents()
    print("\n✅ Document processing complete!")
