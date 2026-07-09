from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader,TextLoader,Docx2txtLoader




def load_doc(file_path):
    
    if file_path.endswith(".pdf"):
        loader=PyPDFLoader(file_path)
        
    elif file_path.endswith(".docs"):
        laoder=Docx2txtLoader(file_path)
    
    elif file_path.endswith(".csv"):
        loader=CSVLoader(file_path)
    
    elif file_path.endswith(".txt"):
        laoder=TextLoader(file_path)
    
    else :
        raise ValueError("unsupported error")
        
        
    return loader.load()
    

# print(text)

