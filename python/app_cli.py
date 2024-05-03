# Easy Vertex AI gettings tarted 
# https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/text_embedding_new_api.ipynb

# Example notebooks:
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-qa/question_answering_documents_langchain_matching_engine.ipynb
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/intro_langchain_palm_api.ipynb
# https://colab.research.google.com/drive/17r8QmiH8m7irJ08r9-QkjOaBF9e9nZVd#scrollTo=QJr0wpJzlWf5

# Tunning embeddings model + Deploy to cloud
# https://medium.com/google-cloud/level-up-your-rag-tuning-embeddings-on-vertex-ai-901bb7f65bd0
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro_embeddings_tuning.ipynb


# GOOGLE DRIVE
# langchain retriever: https://python.langchain.com/docs/integrations/retrievers/google_drive/
 
# required job documents Shared from my drive
# documents.7z
# https://drive.google.com/file/d/1VkG_onEjQFTjbFQYUghxPci7xQ4CFTxc/view?usp=sharing

# job_links.csv
# https://drive.google.com/file/d/1k2Bm3JJfQ2vKetaVYJoTnP71QSUGVzJf/view?usp=sharing

# download + Unzip Above artifacts
# cd ./python
# curl -o "job_links.csv" -L "https://drive.google.com/file/d/1k2Bm3JJfQ2vKetaVYJoTnP71QSUGVzJf" (doesnt work for documents.7z)
# 7z x documents.7z

# pip install google-cloud-aiplatform langchain langchain-google-vertexai transformers qdrant_client
# pip freeze -> to check the downloaded modules (optional)

# Vertex AI
from google.cloud import aiplatform
print(f"Vertex AI SDK version: {aiplatform.__version__}")
# import vertexai
from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# Import custom Matching Engine packages
from langchain_google_vertexai import (
    VertexAI,
    VertexAIEmbeddings,
    #VectorSearchVectorStore,# currently not used , but might expose through hosted endpoint on the cloud for v2
)

# PROJECT_ID = "vertexai-hackathon-422020"
# REGION = "us-central1" 
# # Initialize Vertex AI SDK
# vertexai.init(project=PROJECT_ID, location=REGION)

#load docs
path = "documents"
text_loader_kwargs={'encoding': 'utf-8'}
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, show_progress=True)
documents = loader.load()

num_docs= len(documents)
print(f"\n{num_docs} Job Documents loaded.")

# split the documents into chunks
# option 2 > https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/
# option 3 > https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token/
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
    separators=["\n\n"],
)
document_chunks = text_splitter.split_documents(documents)

print(f"Number documents {len(documents)}")
print(f"Number chunks {len(document_chunks)}")

# Document object
# page_content='Job title: ...', metadata={'source': 'documents\\0_jobId_110759204500710086.txt'}

# prerequisit 
# gcloud auth application-default login:
# https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev

# Default location Windows
# C:\Users\<USER>\AppData\Roaming\gcloud\application_default_credentials.json

def embed_text(
    texts: List[str],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "textembedding-gecko@003",# 3072 input tokens and outputs 768-dimensional vector embeddings.
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

# original with dictionary 
# chunk_texts = [chunk.page_content for chunk in document_chunks[:5]]

# embeddings = embed_text(chunk_texts)
# print(f'Computed embeddings: {len(embeddings)}')
# # print("Dimensionality : ")
# # print(embeddings[:1])

# jobs_dict = {}
# # merge chunk emebddings and metadata
# for i, (chunk, emb) in enumerate(zip(document_chunks[:5], embeddings)):
#     src = chunk.metadata['source']
#     file_name = src.split("\\")[-1]
#     job_id = file_name.split("_")[2].split(".")[0]
#     job_url = search_line_with_number(job_id)
#     jobs_dict[i] = {'content': chunk.page_content,'embeddings':emb,'job_id':job_id, 'source': file_name,'url':job_url}

# print(jobs_dict)


# https://github.com/qdrant/qdrant-client
# Langchain integration
# https://python.langchain.com/docs/integrations/vectorstores/qdrant/
# https://python.langchain.com/docs/integrations/retrievers/self_query/qdrant_self_query/

# Stand alone client 
# COLLECTION_NAME ="vertex_ai"
# db_client = QdrantClient(url="http://localhost:6333")
# db_client.recreate_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=768, distance=Distance.DOT),# distance=models.Distance.COSINE
# )

# db_client.upsert(
#     collection_name=COLLECTION_NAME,
#     points=[
#         PointStruct(
#             id=key, 
#             vector=job['embeddings'],  # flatten to 1d
#             payload={
#                 'content': job['content'],  # Mapped to payload object
#                 'document': job['source'],
#                 'url': job['url']
#             }
#         )
#         for key, job in jobs_dict.items()
#     ]
# )

#Langchain client
# or persisted without server
# https://python.langchain.com/docs/integrations/vectorstores/qdrant/#on-disk-storage

url = "http://localhost:6333"
COLLECTION_NAME ="vertex_ai"
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

qdrant = Qdrant.from_documents(
    document_chunks[:5],
    embeddings,
    url=url,
    prefer_grpc=True,
    force_recreate=True,
    distance_func= 'COSINE',
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
# query = "Got any jobs based in Poland ?"
result = qa({"query": query})

print(result)



# GC UTILS

# Make a Google Cloud Storage bucket in your GCP project to copy the document files into.
#   GCS_BUCKET_DOCS = f"{PROJECT_ID}-documents"
#   ! set -x && gsutil mb -p $PROJECT_ID -l us-central1 gs://$GCS_BUCKET_DOCS

# Copy document files to your bucket
#   folder_prefix = "documents/google-research-pdfs/"
#   !gsutil rsync -r gs://github-repo/documents/google-research-pdfs/ gs://$GCS_BUCKET_DOCS/$folder_prefix


# -------------------- another example from colab doc

# AUTH PRE REQUISISTS 
# 1 gcloud auth login
# 2 gcloud config set project vertexai-hackathon-422020

# Requirements for Google Storage 
# 3 IAM bucket policies
# gcloud projects add-iam-policy-binding <PROJECT_ID> --member=user:<YOUR_EMAIL_ADDRESS> --role=roles/storage.objectViewer
# https://console.cloud.google.com/iam-admin/ (Add  IAM roles manually )

# copy docs and embeddings to GC Storage
# !gsutil cp -r documents  gs://doit-llm/documents
# !gsutil cp embeddings.json gs://doit-llm/embeddings/embeddings.json 

#gsutil install
# https://cloud.google.com/storage/docs/gsutil_install
    
##next step is tu run init command and maybe use my alt gmail email for extra 300credits ?  


# Qdrant 
# podman run --name vertex_ai -p 6333:6333 -p 6334:6334 -v c:/Users/dpolzer/me/git/vertex_ai_hackathon/python/qdrant_storage:/qdrant/storage qdrant/qdrant
# docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant (didn't quite work without abs path WIN11)