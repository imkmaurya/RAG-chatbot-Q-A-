from langchain_core import retrievers


def retreival(db):
    
    retriever=db.as_retriever(
        search_type="similarity",
        search_kwargs={'k':4}
    )
    
    return retriever
    
