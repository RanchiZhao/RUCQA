import os
os.environ["OPENAI_API_KEY"] = "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"
from Llama_index_method import Llama_Index_Processor
from Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch
from Integrator import AnswerIntegrator
if __name__ == '__main__':
    processor = Llama_Index_Processor()
    query = "刘后滨是谁？"
    response = processor.process_query(query)
    print(response)

    extractor = QueryKeyWordsExtractor('./docs')
    query_list = extractor.extract_keywords(query)
    most_common_docs = extractor.find_most_common_doc(query)
    Search = KeywordDocumentSearch(most_common_docs, query_list)
    result = Search.search(query)
    print(result)

    ai = AnswerIntegrator(result, response)
    integrated_answer = ai.integrate_answers(query)
    simplified_answer = ai.simplify_answers(query, integrated_answer)
    print(integrated_answer)
    print(simplified_answer)
