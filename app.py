import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader  # to read the pdf file from PyPDF2 package
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


"""
Below code reads the pdf file \"employee-hand-book.pdf\"
using PdfReader method from PyPDF2 package 
will return the pdf object like this <PyPDF2._reader.PdfReader object at 0x1003126d0>
"""

pdf = "" # path of your pdf file goes here
pdf_reader = PdfReader(pdf)


"""
Specifically for langChain we need to read the content page by pages,
so we are extracting the content from a page and appending to the empty string content.
Now we completed with loading the data , next step is to split into chunks.
"""

content = ""
for page in pdf_reader.pages:
    content += page.extract_text()


"""
Reason why we are splitting into chunks is , the context (Token) of LLMs might be less

ChatGPT (4096) 
GPT-4 (8k to 32k)

So here we are using recursive splitter to divide the document of chunk_size by default it is

chunk_size: int = 4000,
chunk_overlap: int = 200,
length_function: Callable[[str], int] = len,

overlap is nothing but the common ground between chunk1 and chunk2

example

--------------------
                ----------------------

"""

char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
chunks = char_text_splitter.split_text(text=content)


"""
What is Embeddings, It's a numerical representation of text
here each chunks will have separate embedding
"""


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
# needed only once while the embedding will get stored in file faiss_index in your directory so the next time instead of making new embedding it will use the existing one
vector_store.save_local("faiss_index")

new_db = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)  # allow_dangerous_deserialization is set to true because since I know file which is present and it's created by me


def give_answer(question):

    docs = new_db.similarity_search(question)

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details
    if any further question asked on the response given before please elobrate that too
    Context:\n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    try:
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

        chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
        response = chain(
            {
                "input_documents": docs,
                "question": question,
            },
            return_only_outputs=True,
        )
        print("\33[32m" + response["output_text"] + "\33[0m")  # Print the answer from the chain
    except Exception as e:
        print(f"An error occurred: {e}")


query = input("\33[34m" + "Enter the question ---> " + "\33[0m")
print("\n")
while query:
    if query == "Exit":
        break
    give_answer(query)
    print("\n")
    query = input("\33[34m" + "You have anything to ask... If not please do Exit ---> " + "\33[0m")
