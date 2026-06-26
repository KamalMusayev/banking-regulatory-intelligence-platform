import re
import json
import os
import datetime

project_root = "/Users/samil/Desktop/ITSkillsSprintProjects/banking-regulatory-intelligence-platform"
filepath = os.path.join(project_root, "data/processed/cleaned_documents/aml_kyc/096312ee-c41c-4c1f-9edd-baf5f3187758.md")
filename = os.path.basename(filepath)

doc_id = filename.replace(".md", "")

chunks = []
current_page = 1
max_page = 1

current_article = None
current_section = None
current_subsection = None

current_chunk_text = []
chunk_start_page = 1

def save_chunk():
    global current_chunk_text, chunk_start_page
    if not current_chunk_text:
        return
    text = "\n".join(current_chunk_text).strip()
    if not text:
        current_chunk_text = []
        return
    
    # Generate chunk id
    hierarchy = []
    if current_article: hierarchy.append(current_article)
    if current_section: hierarchy.append(current_section)
    if current_subsection: hierarchy.append(current_subsection)
    
    h_str = "_".join(hierarchy).replace(".", "_") if hierarchy else "intro"
    chunk_id = f"{doc_id}_{h_str}_{len(chunks)+1}"
    
    chunk = {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "title": "Mərkəzi Bankın Qərarı № 59",
        "category": "aml_kyc",
        "chapter": None,
        "article": current_article,
        "section": current_section,
        "subsection": current_subsection,
        "page_start": chunk_start_page,
        "page_end": current_page,
        "source_file": filename,
        "text": text
    }
    chunks.append(chunk)
    current_chunk_text = []

page_pattern = re.compile(r"<!-- PAGE:\s*(\d+)\s*-->")
article_pattern = re.compile(r"^(\d+)\.\s+(.*)")
section_pattern = re.compile(r"^(\d+\.\d+)\.\s+(.*)")
subsection_pattern = re.compile(r"^(\d+\.\d+\.\d+)\.\s+(.*)")

with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
        # Process page marker
        m_page = page_pattern.search(line)
        if m_page:
            current_page = int(m_page.group(1))
            max_page = max(max_page, current_page)
            line = page_pattern.sub("", line).strip()
            if not line:
                continue
            
        # Ignore horizontal rules often accompanying page tags
        if line.strip() == "---":
            continue

        # Check hierarchy
        m_sub = subsection_pattern.match(line)
        m_sec = section_pattern.match(line)
        m_art = article_pattern.match(line)
        
        if m_sub:
            save_chunk()
            current_subsection = m_sub.group(1)
            chunk_start_page = current_page
            current_chunk_text.append(line.strip())
        elif m_sec:
            save_chunk()
            current_section = m_sec.group(1)
            current_subsection = None
            chunk_start_page = current_page
            current_chunk_text.append(line.strip())
        elif m_art:
            save_chunk()
            current_article = m_art.group(1)
            current_section = None
            current_subsection = None
            chunk_start_page = current_page
            current_chunk_text.append(line.strip())
        else:
            if not current_chunk_text:
                chunk_start_page = current_page
            current_chunk_text.append(line.rstrip('\n'))

save_chunk()

out_chunks_dir = os.path.join(project_root, "data/processed/chunks/aml_kyc")
out_meta_dir = os.path.join(project_root, "data/processed/metadata/aml_kyc")
os.makedirs(out_chunks_dir, exist_ok=True)
os.makedirs(out_meta_dir, exist_ok=True)

# Write jsonl
jsonl_path = os.path.join(out_chunks_dir, f"{doc_id}.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# Write metadata
meta = {
    "document_id": doc_id,
    "title": "Mərkəzi Bankın Qərarı № 59",
    "category": "aml_kyc",
    "source_file": filename.replace(".md", ".pdf"),
    "processed_file": filename,
    "total_pages": max_page,
    "total_chunks": len(chunks),
    "language": "az",
    "parser": "pdfplumber",
    "created_at": datetime.datetime.now().isoformat()
}
meta_path = os.path.join(out_meta_dir, f"{doc_id}_metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Total chunks: {len(chunks)}")
print(f"Max page: {max_page}")
print(f"JSONL saved to: {jsonl_path}")
print(f"Metadata saved to: {meta_path}")
