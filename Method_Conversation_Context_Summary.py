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
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import sys
import os

"""
Presented here is the most barebone form of a LLM, it simply takes an input
and then spits an output.
"""

# Class for context evaluation
class ContextEval(BaseModel):
    tf: bool = Field(description="True False answer")

# Create the parser
parser = PydanticOutputParser(pydantic_object=ContextEval)

# Preset Template
template = "Using scientific knowledge and admitting when you do not know the answer,\
use the following evidence: \
'{Context}' ; \n Chat History: {History}; \nQuestion: {Question}\nAnswer:"

prompt = PromptTemplate.from_template(template)

# Summarize Context template
template_summarize = "You are an expert at summarizing and do not make things up. With the following query in mind: {Question},\n\
    Summarize the following context, keeping as much content as possible: {Context}"

prompt_summarize = PromptTemplate.from_template(template_summarize)

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

        # Here we a summarizer for the context
        chained_format_summarize = LLMChain(prompt=prompt_summarize, llm=self.llm)
        
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

                # Run the model to summarize the context
                summary_context = chained_format_summarize.run({"Question":query, "Context":context})
            else:
                context = ""

            # Run the Query
            output = chained_format.run({"History":history,"Context":summary_context,"Question":query})

            print("Prompt")

            print("Chat History:")
            print(history)
            print("\nSummary Context:")
            print(summary_context)
            print("\nQuery:")
            print(query)

            print("Response\n------------\n")

            print(output)
            model_outs.append(output)
            query_ins.append(query)
            print("\n")

    def build_vector_database(self, pdf_path, datastore):
        to_load = UnstructuredPDFLoader(pdf_path)

        # Load the document
        raw_documents = to_load.load()

        # Split the text 
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=800,
            chunk_overlap=100,
            length_function=len)
        
        documents = text_splitter.split_documents(raw_documents)

        # Add to vector
        docs_split = []
        for i in range(len(documents)):
            docs_split.append(documents[i])   

        os.system("mkdir " + datastore)     
        
        db = Chroma.from_documents(docs_split, GPT4AllEmbeddings(), persist_directory=datastore)
        db.persist()

# For users
# Modify one of these two parts to change parameters
if __name__ == "__main__":
    
    """
    Description: run_model_with_built_database: Run the model with contextualization from a built
    Chroma database. Mainly change the max number of tokens that will be used by the model. Conversation
    style of format similar to ChatGPT. Note that it will get hyperfocused and it may be best to reset
    the conversation. Output is verbose. 

    Tested on: M1 Mac Book Pro, 64 GB Memory
    """

    """
    model = LLM_Run_Wrapper(model_path="./models/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf",
                            n_gpu_layers=1,
                            n_batch = 512,
                            n_ctx = 2048,
                            max_tokens = 20000)
    
    model.run_model_with_built_database(persist_dir = "./ochem",
                                              k = 2)

    """

    """
    Description: build_vector_database: Build a Chroma vector database from a PDF.
    Specify the pdf location and an output directory for the database. 

    Tested on: M1 Mac Book Pro, 64 GB Memory
    """

    """
    model = LLM_Run_Wrapper(model_path="./models/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf",
                            n_gpu_layers=1,
                            n_batch = 512,
                            n_ctx = 2048,
                            max_tokens = 20000)
    
    model.build_vector_database("./contexts/Bio.pdf", "./bio")
    """
