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

"""
Presented here is the most barebone form of a LLM, it simply takes an input
and then spits an output.
"""

# Preset Template
template = "Using scientific knowledge and admitting when you do not know the answer,\
use the following evidence: \
'{Context}' ; to provide an evidence based response to the following question: {Question}\nAnswer:"

prompt = PromptTemplate.from_template(template)

# Wrap the llm object and its parameters
class LLM_Run_Wrapper:
    def __init__(self, model_path, n_gpu_layers, n_batch, n_ctx, max_tokens):
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(model_path = model_path,
                            n_gpu_layers = n_gpu_layers,
                            f16_kv = True,
                            n_batch = n_batch,
                            n_ctx = n_ctx,
                            max_tokens = max_tokens,
                            verbose = False)
        self.model = LLMChain(prompt=prompt, llm=self.llm)

    def run_model_with_built_database(self, query, persist_dir, k):
        # Set the chained model
        chained_format = LLMChain(prompt=prompt, llm=self.llm)
        
        # Load the database
        self.db = Chroma(persist_directory = persist_dir, embedding_function = GPT4AllEmbeddings())

        # Query vector database
        docs = self.db.similarity_search(query, k=k)
        
        # Build the context
        contexts = []

        # With the number of returned chunks
        # Join them together
        for i in range(len(docs)):
            contexts.append(docs[i].page_content)

        # Join the context into one combo
        context = "\n".join(contexts)

        # Run the Query
        output = chained_format.run({"Context":context,"Question":query})

        return output

# Start the script portion
if __name__ == "__main__":
    
    model = LLM_Run_Wrapper(model_path="./models/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf",
                            n_gpu_layers=1,
                            n_batch = 512,
                            n_ctx = 2048,
                            max_tokens = 16000)
    
    print(model.run_model_with_built_database(query = "Factors of a stability of a carbocation?",
                                              persist_dir = "./ochem",
                                              k = 5))