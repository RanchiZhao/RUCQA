import os
from Llama_index_method import Llama_Index_Processor
from Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch
from Integrator import AnswerIntegrator

def evaluate(queries: list):
    os.environ["OPENAI_API_KEY"] = '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'
    """
    queries: List[str] 输入查询列表
    Return: List[str] 输出答案列表
    """
    return_list = []
    processor = Llama_Index_Processor()
    extractor = QueryKeyWordsExtractor('./docs')
    for query in queries:
        response = processor.process_query(query)

        query_list = extractor.extract_keywords(query)
        most_common_docs = extractor.find_most_common_doc(query)
        Search = KeywordDocumentSearch(most_common_docs, query_list)
        result = Search.search(query)

        ai = AnswerIntegrator(response, result)
        integrated_answer = ai.integrate_answers(query)
        simplified_answer = ai.simplify_answers(query, integrated_answer)

        return_list.append(simplified_answer)
        print(return_list)

    return return_list  # EM 0.8