from langchain.prompts import PromptTemplate
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
import sys
import os

"""
Presented here is the most barebone form of a LLM, it simply takes an input
and then spits an output.
"""

# Preset Template
template = "Using scientific knowledge and admitting when you do not know the answer,\
use the following evidence: \
'{Context}' ; \n Chat History: {History}; \nQuestion: {Question}\nAnswer:"

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

    def run_model_with_built_database(self, persist_dir, k):
        # Set the chained model
        chained_format = LLMChain(prompt=prompt, llm=self.llm)
        
        history = ""

        # Load the database
        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")

        self.db = Chroma(persist_directory = persist_dir, embedding_function = GPT4AllEmbeddings())

        sys.stdout = old_stdout

        running = True

        model_outs = []
        query_ins = []

        # Intilize the while loop
        # For conversation
        while running:
            
            # Here we create the history
            if len(model_outs) > 0:
                history_list = []

                for i in range(len(model_outs)):
                    human_ask = "Student: " + query_ins[i]

                    model_out = "AI: " + model_outs[i]

                    history_temp = human_ask + "\n" + model_out

                    history_list.append(history_temp)

                history = "\n".join(history_list)


            print("Input Query (Done to quit):")
            query = input()

            if query.lower() == "done":
                break


            print("\nShould I use your textbook to answer this question? (y/n)")
            yn = input()

            # Here we have the summary context
            # Where we will use the llm to summarize
            summary_context = ""

            if yn.lower() == "y":

                docs = self.db.similarity_search(query, k=k)
                
                # Build the context
                contexts = []

                # With the number of returned chunks
                # Join them together
                for i in range(len(docs)):
                    contexts.append(docs[i].page_content)

                # Join the context into one combo
                context = "\n".join(contexts)

            else:
                context = ""

            # Run the Query
            output = chained_format.run({"History":history,"Context":context,"Question":query})

            print("Prompt")

            print(history)
            print(context)
            print(query)

            print("Response\n------------\n")

            print(output)
            model_outs.append(output)
            query_ins.append(query)
            print("\n")

# For users
# Modify one of these two parts to change parameters
if __name__ == "__main__":
    
    """
    Run model with the raw context. Not summarized. 
    """

    """
    model = LLM_Run_Wrapper(model_path="./models/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf",
                            n_gpu_layers=1,
                            n_batch = 512,
                            n_ctx = 2048,
                            max_tokens = 16000)
    
    model.run_model_with_built_database(persist_dir = "./ochem",
                                              k = 2)
    """