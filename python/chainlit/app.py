# pip install google-cloud-aiplatform langchain langchain-google-vertexai transformers qdrant_client pypdf chainlit

from google.cloud import aiplatform
print(f"Vertex AI SDK version: {aiplatform.__version__}")

from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores import Qdrant
from langchain_google_vertexai import (
    VertexAI,
    VertexAIEmbeddings,
    ChatVertexAI
)
import chainlit as cl
from chainlit.types import AskFileResponse

# CLI docs
# https://docs.chainlit.io/backend/command-line
# chainlit run --port 7777 ./app.py (8000 is default)

#load docs
path = "../documents"
text_loader_kwargs={'encoding': 'utf-8'}
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, show_progress=True)
documents = loader.load()

num_docs= len(documents)
print(f"\n{num_docs} Job Documents loaded.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
    separators=["\n\n"],
)
document_chunks = text_splitter.split_documents(documents)

print(f"Number documents {len(documents)}")
print(f"Number chunks {len(document_chunks)}")

def embed_text(
    texts: List[str],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "textembedding-gecko@003", # 3072 input tokens and outputs 768-dimensional vector embeddings.
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]

def search_line_with_number(target_number,file_path="job_links.csv"):
    with open(file_path, 'r') as f:
        for line in f:
            if target_number in line:
                return line.rstrip(',\n').rstrip(',')
    return None

# Collect metadata
for chunk in document_chunks[:5]:
    src = chunk.metadata['source']
    file_name = src.split("\\")[-1]
    job_id = file_name.split("_")[2].split(".")[0]
    job_url = search_line_with_number(job_id)
    chunk.page_content = chunk.page_content
    chunk.metadata = {
        'job_id': job_id,
        'source_file': file_name,
        'job_url': job_url
    }

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

        loader = Loader(file.path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


# this inserts Attachment as item in vector store (I have existing store which i don't want this inserted into)
# Instead i 
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # print(f'docs ln 100 :\n{docs}') 
    # [Document(page_content='resume txt ...', metadata={'source': 'source_0', 'page': 0})]
    # NOTE Spoken languages part from my resume seems off (font is different ? )
    
    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    namespace = file.id

    # namespaces = set() -> Hset of file namespaces, create embeddings for one if it's not already in index
    # if namespace in namespaces:
    #     docsearch = Pinecone.from_existing_index(
    #         index_name=index_name, embedding=embeddings, namespace=namespace
    #     )
    # else:
    #     docsearch = Pinecone.from_documents(
    #         docs, embeddings, index_name=index_name, namespace=namespace
    #     )
    #     namespaces.add(namespace)

    return "docsearch"

#Langchain client
# or persisted without server
# https://python.langchain.com/docs/integrations/vectorstores/qdrant/#on-disk-storage
url = "http://localhost:6333"
COLLECTION_NAME ="cl_vertex_ai"
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

qdrant = Qdrant.from_documents(
    document_chunks[:5],
    embeddings,
    url=url,
    prefer_grpc=True,
    force_recreate=True,
    distance_func= 'COSINE', # there is also DOT (you have to update 'retriever' too)
    collection_name=COLLECTION_NAME,
)

# Expose index to the retriever
# Contextual compression: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
# Filter initial docs: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/#llmchainfilter 
TOP_K = 5
retriever = qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": TOP_K,
    },
    filters=None,
)
# res = retriever.invoke("Pixel devices jobs are the ones I'm looking for at the moment.")
# print(res)

# Create chain to answer questions
template = """SYSTEM: You are an intelligent assistant helping the users with their questions about Job Postings.

Question: {question}

Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "I cannot find any relevant SWE roles that may interest you at Google."
 - If the context is empty, just say "I do not have relevant jobs at google that match your criteria."

=============
{context}
=============

Question: {question}
Helpful Answer:"""

# Text model instance integrated with langChain
#Other LLM options
# llm = VertexAI(model_name="gemini-1.0-pro-002")
# Streaming api --> https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm/

# There is also a 6k (chat-bison) and 32k context fintuned for chat 
# llm = VertexAI(model_name="chat-bison-32k")

# Embeddings API integrated with langChain
# EXAMPLE FROM DOCS: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
# embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
# doc_result = embeddings.embed_documents([document_chunks])
llm = VertexAI(
    model_name="text-bison@002",
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=1,
    verbose=True,
)

# Uses LLM to synthesize results from the search index.
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        ),
    },
)
# Troubleshooting
# qa.combine_documents_chain.verbose = True
# qa.combine_documents_chain.llm_chain.verbose = True
# qa.combine_documents_chain.llm_chain.llm.verbose = True

# query = "I'm currently mostly interested in the dystributed systems and cloud engineering roles"
query = "Pixel devices jobs are the ones I'm looking for at the moment."

# result = qa({"query": query})
# print(result)  

@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content= "Welcome to customized job search based on your skills & preferences",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file) 
    
    # NOTE does nothing now since i have other plans
    # Right now you are chatting with documents already pre-indexed from Google jobs (your is parsed but not inserted and queried)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatVertexAI(model_name="gemini-pro", streaming=True),
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

# Qdrant persisted volume mnt 
# podman run --name vertex_ai -p 6333:6333 -p 6334:6334 -v c:/Users/dpolzer/me/git/vertex_ai_hackathon/python/qdrant_storage:/qdrant/storage qdrant/qdrant

# chainlit pdf example
# https://docs.chainlit.io/examples/qa
# https://github.com/Chainlit/cookbook/blob/main/pdf-qa/app.py
# https://github.com/Chainlit/cookbook (other chainlit examples)

# Other Lanchain char API's
# https://python.langchain.com/docs/modules/model_io/chat/quick_start/
# https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/

# Straming chat responses
# https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/#streaming-calls