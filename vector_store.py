from langchain_community.vectorstores import FAISS


def make_vectorstore(chunk,embedding):
    db=FAISS.from_documents(
        documents=chunk,
        embedding=embedding
        
    )
    return db

