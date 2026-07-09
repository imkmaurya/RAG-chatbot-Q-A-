from langchain_huggingface import HuggingFaceEmbeddings



def make_embedding():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding

