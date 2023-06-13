# -*- coding:utf-8 -*-
import os
os.environ["OPENAI_API_KEY"] = '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'
from BMRUC.Llama_index_method import Llama_Index_Processor
from BMRUC.Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch, AnswerHead
from BMRUC.Integrator import AnswerIntegrator


if __name__ == '__main__':
    processor = Llama_Index_Processor()
    extractor = QueryKeyWordsExtractor('./docs')
    query_list = ["2022年12月8日晚，高瓴人工智能学院举办了主题为“名企校友面对面”的就业分享会，本次活动邀请了哪位师兄作为毕业生代表来分享经验？", "2022年11月29日，中国人民大学高瓴人工智能学院和中国人民大学公共管理学院共同召开了什么研讨会？","共有几支队伍在“京东杯”中国人民大学第十三届学生“创业之星”大赛决赛中获奖？","一等奖是美国大学生数学建模竞赛第几高的奖项？","概括张东刚书记在苏州校区调研指导工作中提出的三点希望？","哪5家法学院科研机构获得第一批法学院科研机构综合支持？","谁同时出席了统计学院2023年5月24日和北京航空航天大学经济管理学院举办的主题教育联学共建活动以及2023年4月18日举办的学习贯彻习近平新时代中国特色社会主义思想主题教育动员部署大会这两次活动？"]
    for query in query_list:
        # query = "西藏民族大学党委书记、副校长是谁？"
        response = processor.process_query(query)
        print("question: ", query)
        print("response: ", response)

        query_list = extractor.extract_keywords(query)
        most_common_docs_name, most_common_docs = extractor.find_most_common_doc(query)
        Search = KeywordDocumentSearch(most_common_docs, query_list)
        answer_sent = Search.search(query)
        print("answer_sent: ", answer_sent)

        answer_head = AnswerHead()
        final_answer = answer_head.answer_generator(query, answer_sent)
        print("final_answer: ", final_answer)

        ai = AnswerIntegrator(final_answer, response)
        integrated_answer = ai.integrate_answers(query)
        simplified_answer = ai.simplify_answers(query, integrated_answer).rstrip("。")
        print("integrated_answer: ", integrated_answer)
        print("simplified_answer: ", simplified_answer)

        reference_list = ["news.ruc.edu.cn/archives/" + name for name in most_common_docs_name[:5]]
        print("reference_list: ", reference_list)

