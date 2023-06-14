import os
from BMRUC.Llama_index_method import Llama_Index_Processor
from BMRUC.Exact_match_method import QueryKeyWordsExtractor, KeywordDocumentSearch, AnswerHead, AnswerHeadReinforcedInCounting
from BMRUC.Multi_response import MultiResponseSummary
from BMRUC.Integrator import AnswerIntegrator
from BMRUC.Task_classification import TaskClassification

def process_single_object_question(query, response):
    extractor = QueryKeyWordsExtractor()
    keywords = extractor.extract_keywords(query)
    common_doc_names, common_docs = extractor.find_most_common_doc(query)
    document_search = KeywordDocumentSearch(common_docs, keywords)
    answer_sentence = document_search.search(query)
    answer_head = AnswerHead()
    final_answer = answer_head.generate_answer(query, answer_sentence)
    answer_integrator = AnswerIntegrator(final_answer, response)
    integrated_answer = answer_integrator.integrate_answers(query)
    simplified_answer = answer_integrator.simplify_answers(query, integrated_answer).rstrip("。")
    return simplified_answer, common_doc_names[:5]

def process_multiple_objects_question(query, response):
    extractor = QueryKeyWordsExtractor('')
    common_doc_names, common_docs = extractor.find_most_common_doc(query)
    text = common_docs[0]
    if len(text) > TEXT_LIMIT:
        multi_response1 = MultiResponseSummary()
        summary1 = multi_response1.return_summary(query, text[:TEXT_LIMIT]).content
        multi_response2 = MultiResponseSummary()
        summary2 = multi_response2.return_summary(query, text[:TEXT_LIMIT]+text[SUMMARY_LIMIT:]).content
        summary = summary1 + summary2
    else:
        multi_response = MultiResponseSummary()
        summary = multi_response.return_summary(query, text).content

    answer_head = AnswerHead()
    final_answer = answer_head.generate_answer(query, summary)
    answer_integrator = AnswerIntegrator(final_answer, response)
    integrated_answer = answer_integrator.integrate_answers(query)
    simplified_answer = answer_integrator.simplify_answers(query, integrated_answer).rstrip("。")
    return simplified_answer, common_doc_names[:5]

def process_calculation_question(query, response):
    extractor = QueryKeyWordsExtractor('')
    keywords = extractor.extract_keywords(query)
    common_doc_names, common_docs = extractor.find_most_common_doc(query)
    document_search = KeywordDocumentSearch(common_docs, keywords)
    reference_str = document_search.return_refence_str()

    answer_head = AnswerHeadReinforcedInCounting()
    final_answer = answer_head.answer_generator(query, reference_str)
    answer_integrator = AnswerIntegrator(final_answer, response)
    integrated_answer = answer_integrator.integrate_answers(query)
    simplified_answer = answer_integrator.simplify_answers(query, integrated_answer).rstrip("。")
    return simplified_answer, common_doc_names[:5]

def evaluate(queries):
    result_list =[]
    for i, query in enumerate(queries):
        i += 1
        processor = Llama_Index_Processor()
        response = processor.process_query(query)

        if i <= SINGLE_OBJ_QUESTION_LIMIT:
            classification = 'Single Object Question'
        else:
            task_classifier = TaskClassification()
            classification = task_classifier.return_classification(query).content

        if classification == 'Single Object Question':
            simplified_answer, references = process_single_object_question(query, response)
        elif classification == 'Multiple Objects Question':
            simplified_answer, references = process_multiple_objects_question(query, response)
        else:  # Addition and Subtraction Calculation Question or Counting Question
            simplified_answer, references = process_calculation_question(query, response)

        result_list.append(simplified_answer)
        reference_list = ["news.ruc.edu.cn/archives/" + name for name in references]
        # print("reference_list: ", reference_list)

    return result_list


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = 'your_api_key'

    SINGLE_OBJ_QUESTION_LIMIT = 60
    TEXT_LIMIT = 3000
    SUMMARY_LIMIT = 5000

    queries = []
    print(evaluate(queries))


