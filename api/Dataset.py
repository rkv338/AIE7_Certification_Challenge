from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
def generate_dataset():
    path = "../docs"
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    dataset = generator.generate_with_langchain_docs(docs[:20], testset_size=10)

    return dataset