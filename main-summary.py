# -*- coding:utf-8 -*-
import os
os.environ["OPENAI_API_KEY"] = 'your_api_key'
from BMRUC.Llama_index_method import Llama_Index_Processor
from BMRUC.Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch, AnswerHead, AnswerHeadReinforcedInCounting
from BMRUC.Integrator import AnswerIntegrator


if __name__ == '__main__':
    processor = Llama_Index_Processor(document_directory=r'C:\Users\yuanq\PycharmProjects\RUCQA\docs',storage_directory=r'C:\Users\yuanq\PycharmProjects\RUCQA\storage')
    extractor = QueryKeyWordsExtractor(r'C:\Users\yuanq\PycharmProjects\RUCQA\docs')
    query_list = ["请介绍中国人民大学的建校历史？","如何进一步提高中国人民大学的办学质量？","请介绍一下信息学院王珊教授？"]
    for query in query_list:
        response = processor.process_query(query)
        # print("question: ", query)
        # print("response: ", response)

        query_list = extractor.extract_keywords(query)
        most_common_docs_name, most_common_docs = extractor.find_most_common_doc(query)
        Search = KeywordDocumentSearch(most_common_docs, query_list)
        answer_sent = Search.search(query)
        # print("answer_sent: ", answer_sent)

        answer_head = AnswerHead()
        final_answer = answer_head.generate_answer(query, answer_sent)
        # print("final_answer: ", final_answer)

        ai = AnswerIntegrator(response, final_answer)
        integrated_answer = ai.integrate_answers(query)
        # simplified_answer = ai.simplify_answers(query, integrated_answer)
        print("integrated_answer: ", integrated_answer)
        # print("simplified_answer: ", simplified_answer)

        reference_list = ["news.ruc.edu.cn/archives/" + name for name in most_common_docs_name[:5]]
        print("reference_list: ", reference_list)




