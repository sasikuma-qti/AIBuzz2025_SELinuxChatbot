import os
import json
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
#from qgenie_sdk_tools import confluence_tool
from qgenie_sdk_tools.tools.confluence import confluence_tool
from qgenie.integrations.langchain import QGenieChat, QGenieEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("QGENIE_API_KEY")

# Full path to our books directory
books_dir = "books"

# Full path to where we will create the vector store database
store_name = "sepolcy_db"
db_dir = f"vectordb/{store_name}"
urls = ["https://source.android.com/docs/security/features/selinux",
"https://source.android.com/docs/security/features/selinux/concepts",
"https://source.android.com/docs/security/features/selinux/implement",
"https://source.android.com/docs/security/features/selinux/customize",
"https://source.android.com/docs/security/features/selinux/validate",
"https://source.android.com/docs/security/features/selinux/configure-policy",
"https://source.android.com/docs/security/features/selinux/notebook",
"https://source.android.com/docs/security/features/selinux/specificity",
"https://source.android.com/docs/security/features/selinux/compatibility",
"https://source.android.com/docs/security/features/selinux/device-policy",
"https://source.android.com/docs/security/features/selinux/vendor-init",
"https://source.android.com/docs/core/architecture/hidl/binder-ipc"]
video_url = "https://www.youtube.com/watch?v=uI9nk1VDCpY"
splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=50,separator="\n")
# Step 1: Load all JSON files
all_chunks = []

# Create documents from all the files in the directory
def create_documents(books_dir):
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".pdf")]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = PyPDFLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    chunks = splitter.split_documents(documents)
    return chunks


def create_docs_web(list_of_url_lists):
    """
    Loads and splits web documents from multiple lists of URLs.

    Args:
        list_of_url_lists (list[list[str]]): A list containing multiple lists of URLs.

    Returns:
        list[Document]: List of document chunks.
    """
    # Flatten the list of lists into a single list of URLs
    all_urls = [url for sublist in list_of_url_lists for url in sublist]

    # Load documents from all URLs
    loader = WebBaseLoader(all_urls)
    web_docs = loader.load()

    # Split the documents using a predefined splitter
    chunks = splitter.split_documents(web_docs)
    return chunks

def create_docs_yt(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    video_docs = loader.load()
    docs = splitter.split_documents(video_docs)
    return docs

# Create documents from all the files in the directory
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    print(f"Persistent directory: {persistent_directory}")
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")


# Query Vector store given the store name, query and embedding function
def query_vector_store(store_name, query, embedding_function, k=2, threshold=0.1):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        relevant_docs = retriever.invoke(query)
        return relevant_docs
    else:
        print(f"Vector store {store_name} does not exist.")

def extract_confluence():
     #toolkit = confluence_tool(space="LnxSec/SELinux+for+LE+Targets", creds_path="~/.conf_creds")
     toolkit = confluence_tool.invoke({"space": "LnxSec", "creds_path": "~/.conf_creds"})
     toolkit = confluence_tool.invoke({
    "space_key": "LnxSec",
    "search_query_cql": "type=page"})
     toolkit.extract_to_files(output_dir=SOURCE_DIRS["confluence"])

# Parse documents and generate chunks
chunks = create_documents(books_dir)

all_chunks.extend(chunks)
chunks = create_docs_web([urls])
all_chunks.extend(chunks)
print(f"Number of document chunks: {len(all_chunks)}")

# Generate embeddings and persist in the vector store
embeddings_fn = QGenieEmbeddings()
create_vector_store(all_chunks, embeddings_fn, store_name)

