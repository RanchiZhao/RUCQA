import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["OPENAI_API_KEY"] = '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'
from RUCQA.BMRUC.Llama_index_method import Llama_Index_Processor
from RUCQA.BMRUC.Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch, AnswerHead
from RUCQA.BMRUC.Integrator import AnswerIntegrator

def evaluate(queries: list):
    """
    queries: List[str] 输入查询列表
    Return: List[str] 输出答案列表
    """
    return_list = []
    processor = Llama_Index_Processor()
    extractor = QueryKeyWordsExtractor('./docs')
    for query in queries:
        response = processor.process_query(query)
        print("response:", response)

        query_list = extractor.extract_keywords(query)
        most_common_docs = extractor.find_most_common_doc(query)
        Search = KeywordDocumentSearch(most_common_docs, query_list)
        answer_sent = Search.search(query)
        print("answer_sent:", answer_sent)

        answer_head = AnswerHead()
        final_answer = answer_head.answer_generator(query, answer_sent)
        print("final_answer:", final_answer)

        ai = AnswerIntegrator(final_answer, response)
        integrated_answer = ai.integrate_answers(query)
        simplified_answer = ai.simplify_answers(query, integrated_answer).rstrip("。")
        print(integrated_answer)
        print(simplified_answer)
        return_list.append(simplified_answer)
    return return_list  # EM 0.8