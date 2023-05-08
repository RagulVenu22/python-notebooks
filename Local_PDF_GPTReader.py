import os
import PyPDF2
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
# Get your API keys from openai, you will need to create an account.
import os
os.environ["OPENAI_API_KEY"] = "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"
directory = r"C:\\Users\\rvenu2\\OneDrive - DXC Production\\Desktop\\Resume\\PDFs\\"

raw_text = ''
pages = []

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(directory, filename)
        # Open the PDF file in read-binary mode
        with open(filepath, 'rb') as file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(file)
            # Loop through all the pages in the PDF file
            for i in range(len(reader.pages)):
                # Extract the page and add it to the list of pages
                page = reader.pages[i]
                pages.append(page)
                # Extract the text from the page
                text = page.extract_text()
                if text:
                    raw_text += text

text_splitter = CharacterTextSplitter(
     separator = "\n",
     chunk_size = 1000,
     chunk_overlap  = 50,
     length_function = len,
 )
texts = text_splitter.split_text(raw_text)
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
# function computes a nearest neighbor index for the given embeddings using the FAISS library
docsearch = FAISS.from_texts(texts, embeddings)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
#Using QA module to get the query as imput
#A string argument "stuff", which is being used to specify the type of QA (question and answer) chain to load.
chain = load_qa_chain(OpenAI(), chain_type="stuff")
#Enter your query by editing the below section
query = "List out the skills what Ragul Venu have"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)