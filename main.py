import os
os.environ["OPENAI_API_KEY"] = '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'
from BMRUC.Llama_index_method import Llama_Index_Processor
from BMRUC.Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch, AnswerHead
from BMRUC.Integrator import AnswerIntegrator
if __name__ == '__main__':
    processor = Llama_Index_Processor()
    query = "首届AI知识趣味竞答大赛在哪里举办"
    response = processor.process_query(query)
    print("response:", response)

    extractor = QueryKeyWordsExtractor('./docs')
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
