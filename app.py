import pickle
import sys
import faiss
import numpy as np
import requests
import json

# --- Configuration ---

EMBEDDINGS_PATH = "embeddings.pkl"                # Path to your generated embeddings file
OLLAMA_URL = "http://localhost:11434/api"         # Base URL for the Ollama API
MODEL = "deepseek-r1:8b"                          # Model name for generation
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-4B:Q5_K_M"  # Model name for embedding only
TOP_K = 5                                         # Number of relevant chunks to retrieve

# --- Ollama API Functions ---

def get_embedding(text, model=EMBEDDING_MODEL): # Ollama embedding model to embed user input text
    """
    Generates an embedding for the given text using the Ollama API.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/embeddings",
            json={"model": model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"\n[Error] Could not connect to Ollama API at {OLLAMA_URL}. Is it running?")
        print(f"Details: {e}")
        sys.exit(1)

def query_ollama_stream(prompt, model=MODEL):
    """
    Sends a prompt to the Ollama API and streams the response chunk by chunk.
    This shows the "thinking" process of the model in real-time.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True
        )
        response.raise_for_status()

        # The response is a stream of JSON objects, one per line.
        # We process each line as it arrives.
        for line in response.iter_lines():
            if line:
                try:
                    # Each line is a JSON object. We parse it and extract the 'response' part.
                    chunk = json.loads(line)
                    # The 'response' key contains the next piece of the generated text.
                    yield chunk.get('response', '')
                except json.JSONDecodeError:
                    # Occasionally, a non-JSON line might appear (e.g., in case of an error).
                    # We print a warning but continue processing.
                    print(f"\n[Warning] Failed to decode JSON from a stream line: {line}")
                    continue

    except requests.exceptions.RequestException as e:
        print(f"\n[Error] Could not connect to Ollama API. Details: {e}")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred during streaming: {e}")


# --- Main RAG Logic ---

def main():
    """
    Main function to set up Faiss and run the RAG query loop.
    """
    # 1. Load pre-computed embeddings and documents
    print(f"Loading embeddings from '{EMBEDDINGS_PATH}'...")
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
        docs = data['docs']
        embeddings = np.array(data['embeddings'], dtype='float32')
        print(f"-> Successfully loaded {len(docs)} document chunks.")
    except FileNotFoundError:
        print(f"\n[Error] The file '{EMBEDDINGS_PATH}' was not found.")
        print("-> Please run the 'embed_docs.py' script first to generate the embeddings.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] An error occurred while loading the embeddings file: {e}")
        sys.exit(1)

    # 2. Build the Faiss index
    if embeddings.shape[0] == 0:
        print("\n[Error] No embeddings found in the file. The index cannot be built.")
        sys.exit(1)
        
    dimension = embeddings.shape[1]
    print(f"Building Faiss index with {embeddings.shape[0]} vectors of dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("-> Faiss index created successfully.")

    # 3. Start interactive query loop
    print("\n--------------------------------------------------")
    print("🤖 RAG Chat is ready. Ask questions about your document.")
    print("   Type 'exit' or press Ctrl+C to quit.")
    print("--------------------------------------------------")
    
    while True:
        try:
            # Get user input
            user_query = input("\n👤 You: ")
            if user_query.lower().strip() == 'exit':
                break
            if not user_query.strip():
                continue

            # a. Generate an embedding for the user's query
            print("🧠 Thinking... (Embedding your query)")
            query_embedding = np.array([get_embedding(user_query, model=EMBEDDING_MODEL)], dtype='float32')

            # b. Search the Faiss index for the most relevant document chunks
            print(f"🧠 Thinking... (Searching for top {TOP_K} relevant chunks)")
            distances, indices = index.search(query_embedding, TOP_K)

            # c. Retrieve the context from the original documents
            retrieved_chunks = [docs[i] for i in indices[0]]
            context_str = "\n\n---\n\n".join(retrieved_chunks)
            
            print("✅ Found context:")
            for i, chunk in enumerate(retrieved_chunks):
                # This line is changed to print the complete chunk instead of a snippet.
                print(f"  --- Chunk [{i+1}] ---\n{chunk}\n") 

            # d. Construct the prompt for the LLM
            prompt = (
                "You are a helpful AI assistant. Answer the user's question based only "
                "on the following context. If the context doesn't contain the answer, "
                "state that you couldn't find the information in the provided documents.\n\n"
                "--- CONTEXT ---\n"
                f"{context_str}\n\n"
                "--- QUESTION ---\n"
                f"{user_query}\n\n"
                "--- ANSWER ---\n"
            )

            # e. Query the LLM and stream the response
            print("\n🤖 Assistant: ", end="", flush=True)
            for token in query_ollama_stream(prompt, model=MODEL):
                print(token, end="", flush=True)
            print() # Newline after the full response is printed

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
