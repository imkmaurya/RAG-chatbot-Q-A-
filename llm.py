from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
        
        model="gemini-2.5-flash",
        temperature=0
)
    


def get_chain():

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI assistant specialized in question answering.

Use ONLY the provided context to answer the user's question.

Instructions:
firstly-> give the defination of the topic in paragraph
secondly-> give key feaatures
if any mathematical expression elaborte the mathematical function
- Answer only from the given context.
- If the answer is not present in the context, say:
  "I couldn't find this information in the uploaded document."
- Do not make up or assume information.
- Keep the answer clear and well-structured.
- Use bullet points whenever appropriate.

Context:
{context}

Question:
{question}

Answer:
""")

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain