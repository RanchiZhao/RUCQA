import os.path as osp
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, ResponseSynthesizer, GPTVectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from langchain.chat_models import ChatOpenAI
import os
class Llama_Index_Summary_Processor:
    #text-davinci-003
    def __init__(self, document_directory='./docs', model_name="text-davinci-003", storage_directory='./storage'):

        documents = SimpleDirectoryReader(document_directory).load_data()

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name, max_tokens=128))

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        if not osp.isfile(osp.join(storage_directory, "docstore.json")):
            index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
            index.storage_context.persist()
        else:
            storage_content = StorageContext.from_defaults(persist_dir=storage_directory)
            index = load_index_from_storage(storage_content, service_context=service_context)

        retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

        query_engine = RetrieverQueryEngine.from_args(retriever=retriever, response_synthesizer=ResponseSynthesizer.from_args())

        self.query_engine = query_engine
        self.chat = ChatOpenAI(temperature=0.0)

    def process_query(self, query_str):
        query_str = query_str+'\n你需要十分注意的要求：你需要用中文回答，只可以参考我提供的资料，不允许查看外部资料'
        response = self.query_engine.query(query_str)
        return response

if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = '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'
    query = '怎样推动网络空间安全高效治理？'
    processor = Llama_Index_Summary_Processor()
    response = processor.process_query(query)
    print("question: ", query)
    print("response: ", response)