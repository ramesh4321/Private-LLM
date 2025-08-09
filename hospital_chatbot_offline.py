# hospital_chatbot_offline.py
import os
import sqlite3
import faiss
import json
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader      # if using langchain_community else use PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------
# Config: change these to your local model folders
EMBEDDER_LOCAL_PATH = "models/embedder/all-mpnet-base-v2"   # local sentence-transformer folder
LLM_LOCAL_PATH = "models/llm/mistral-7b"                    # local causal model folder
DB_PATH = "hospital.db"                                     # your hospital SQLite DB (or change)
PDF_FOLDER = "pdfs"                                         # folder with hospital PDFs
FAISS_INDEX_PATH = "indexes/faiss_hospital.index"
METADATA_PATH = "indexes/faiss_metadata.json"
EMBED_DIM = 768                                             # embedder output dimension (depends on model)

os.makedirs("indexes", exist_ok=True)



model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.save("models/embedder/all-mpnet-base-v2")

# -----------------------
# 1) Load local embedding model
print("Loading embedding model from:", EMBEDDER_LOCAL_PATH)
embedder = SentenceTransformer(EMBEDDER_LOCAL_PATH)

# -----------------------
# Helper: ingest PDFs
def ingest_pdfs(pdf_folder):
    texts = []
    if not os.path.isdir(pdf_folder):
        return texts
    for fname in os.listdir(pdf_folder):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, fname)
        print("Loading PDF:", path)
        loader = PyPDFLoader(path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        for i, d in enumerate(chunks):
            texts.append({
                "source": fname,
                "page": getattr(d, "metadata", {}).get("page", i),
                "text": d.page_content
            })
    return texts

# -----------------------
# Helper: ingest DB rows (example schema)
def ingest_db_rows(db_path):
    texts = []
    if not os.path.exists(db_path):
        return texts
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Example: adjust query to your real tables/columns
    cur.execute("SELECT id, patient_name, diagnosis, notes FROM patient_records")
    rows = cur.fetchall()
    for r in rows:
        rec_id, patient_name, diagnosis, notes = r
        text = f"Patient: {patient_name}\nDiagnosis: {diagnosis}\nNotes: {notes}"
        texts.append({"source": f"db_record_{rec_id}", "page": None, "text": text})
    conn.close()
    return texts

# -----------------------
# 2) Build or update FAISS index
def build_faiss_index(all_texts, index_path, metadata_path):
    # Create embeddings in batches
    texts = [t["text"] for t in all_texts]
    print(f"Embedding {len(texts)} documents...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Create or replace FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product if embeddings are normalized; use IndexFlatL2 otherwise
    # If you want to use cosine similarity, normalize embeddings
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    # Save metadata mapping index -> text info
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    print("Saved FAISS index to", index_path)

# -----------------------
# 3) Query: retrieve k contexts and generate answer with local LLM
def load_faiss(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def retrieve(query, index, metadata, k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        meta = metadata[idx]
        results.append(meta)
    return results

# LLM generation (local)
def load_local_llm(local_path):
    print("Loading local LLM from:", local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True, device_map="auto", torch_dtype=None)
    return tokenizer, model


def generate_answer(prompt, tokenizer, model, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)

    # Use tokenizer attention mask if available; else create manually
    if "attention_mask" in inputs:
        attention_mask = inputs.attention_mask.to(model.device)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("Raw output tokens:", outputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------
# Main: create index if not exists
if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
    print("Building new index from PDFs and DB...")
    pdf_texts = ingest_pdfs(PDF_FOLDER)
    db_texts = ingest_db_rows(DB_PATH)
    all_texts = pdf_texts + db_texts
    if not all_texts:
        print("No data found in PDF folder or DB. Put files in", PDF_FOLDER, "or set DB_PATH.")
        raise SystemExit(1)
    build_faiss_index(all_texts, FAISS_INDEX_PATH, METADATA_PATH)
else:
    print("Index exists. Loading...")
    all_texts = None

for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

# Load index + metadata
index, metadata = load_faiss(FAISS_INDEX_PATH, METADATA_PATH)

# Load LLM (local)
tokenizer, model = load_local_llm(LLM_LOCAL_PATH)

# Simple REPL
print("\n--- Private Hospital Chatbot (local) ---")
print("Type a question about the hospital data. Type 'exit' to quit.")
while True:
    q = input("\nUser: ").strip()
    if q.lower() in ("exit", "quit"):
        break
    results = retrieve(q, index, metadata, k=5)
    context = "\n\n--- Retrieved passages ---\n"
    for r in results:
        context += f"[Source: {r.get('source')}] {r.get('text')[:800]}\n\n"

    prompt = f"""You are a helpful assistant. Use the following hospital documents and records to answer the question. If the answer is not contained, say you don't know.

{context}

Question: {q}

Answer:"""

    answer = generate_answer(prompt, tokenizer, model, max_new_tokens=256, temperature=0.7)
    # Post-process: trim prompt echo
    # If model echoes prompt, attempt to strip
    if prompt in answer:
        answer = answer.split(prompt)[-1]
    print("\nAssistant:", answer.strip())
