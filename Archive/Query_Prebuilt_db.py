from llama_cpp import Llama
from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import GPT4AllEmbeddings

# Basic parameters
n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Import the model
llm = LlamaCpp(model_path="./models/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf",
               n_gpu_layers=n_gpu_layers,
               callback_manager=callback_manager,
               f16_kv=True, 
               verbse = True,
               n_batch = n_batch,
               n_ctx=4096,
               max_tokens=16000)

# Set embeddings
# We should try two things
# 1. We should try to use f16_kv parameter as true
# 2. We should try to the the GPT4 embeddings
embeddings = GPT4AllEmbeddings()

# Create an easy template to use
template = "Using scientific knowledge and admitting when you do not know the answer,\
use the following evidence: \
'{Context}' ; to provide an evidence based response to the following question: {Question}\nAnswer:"

# Create the prompt object
prompt = PromptTemplate.from_template(template)

# Now add in here the question
entry = "What are the main factors that impact the energy of a molecular orbital?"

# Chained
chained_format = LLMChain(prompt=prompt, llm=llm)

# Create a database using supplied text

# Create the database
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Query the database
context_info = db.similarity_search(entry, k = 1)
context = context_info[0].page_content

# Observe context
print("Context:")
print(context)

# Run the model
output = chained_format.run({"Context":context,"Question":entry})

print("Model out:")
print(output)