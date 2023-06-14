# -*- coding:utf-8 -*-
import os
from collections import Counter
import ast

# Imported custom modules
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.load_documents import load_documents


class QueryKeyWordsExtractor:
    def __init__(self, documents_directory: str):
        """Initializes the QueryKeyWordsExtractor.

        Args:
            documents_directory (str): The directory where the documents are located.
        """
        self.chat = ChatOpenAI(temperature=0.0)

        # Define the template string for chat prompts.
        self.template_string = """
        现在有一个用户的问题Q:
        你的任务是：请你找到这个问题中的重要关键短语,并以列表L的形式返回.
        ---------------
        下面是几个示例，请你学习寻找关键词的要求：
        示例问题Q1：明朝第一任皇帝是谁？
        关键短语列表L：["明朝第一任皇帝"]

        示例问题Q2：中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建什么合作谅解备忘录.
        关键短语列表L：["中国","尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦","合作谅解备忘录"]

        示例问题Q3：中国人民大学在2020年高考分数线中排第几？
        关键短语列表L：["中国人民大学","2020年高考分数线"]

        示例问题Q4：新中国第一次原子弹的空爆试验是在哪一天.
        关键短语列表L：["新中国第一次原子弹","空爆试验"]
        ----------------
        问题Q：{query_str}
        关键短语列表L：
        """

        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)
        self.documents = load_documents(documents_directory)

    def extract_keywords(self, query_str: str):
        """Extract keywords from a given query.

        Args:
            query_str (str): The query string.

        Returns:
            list: A list of extracted keywords.
        """
        messages = self.prompt_template.format_messages(query_str=query_str)
        final_response = self.chat(messages)
        actual_list = ast.literal_eval(final_response.content)
        return actual_list

    def find_most_common_doc(self, query_str: str):
        """Find the most common document that contains the keywords.

        Args:
            query_str (str): The query string.

        Returns:
            list: A list of most common document names and contents.
        """
        actual_list = self.extract_keywords(query_str)
        matching_documents = [[doc for doc in self.documents if query in doc[1]] for query in actual_list]
        all_matches = [match for sublist in matching_documents for match in sublist]  # flatten list of lists

        # Find the most common documents
        counter = Counter(all_matches)
        max_count = counter.most_common(1)[0][1]  # highest count
        most_common_docs_name = [doc[0] for doc, count in counter.items() if count == max_count]
        most_common_docs_content = [doc[1] for doc, count in counter.items() if
                                    count == max_count]  # all documents with highest count

        return most_common_docs_name, most_common_docs_content

class KeywordDocumentSearch:
    def __init__(self, most_common_docs, query_list, chunk_size=200, overlap_size=50):
        """Initializes the KeywordDocumentSearch.

        Args:
            most_common_docs (list): The list of most common documents.
            query_list (list): The list of queries.
            chunk_size (int, optional): The chunk size. Defaults to 200.
            overlap_size (int, optional): The overlap size. Defaults to 50.
        """
        self.most_common_docs = most_common_docs
        self.query_list = query_list
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        # Define the template string for chat prompts.
        self.template_string = """
        现在有一个用户的问题Q和几个参考资料片段R
        你的任务是：请你根据这些参考资料片段R，完成以下任务：
        找到在参考资料R中能回答问题Q的语块P
        你需要十分注意的要求：这个问题Q一定是能被资料片段R中的语块P回答的，你被【禁止】无法找到能够回答问题Q的语块P，你必须找到语块P，请一步一步地认真思考。
        ---------------
        下面是两个示例，请你认真理解它们，并据此学习如何返回答案：
        再次重申：你必须返回语块P，不可以说"无法找到"。
        ----------------
        示例问题Q1：中国社会科学院副院长是谁？
        参考资料R：参考资料1：
        揽子措施，加强财政货币政策协同配合，以稳住市场主体来稳住经济大盘，促进稳增长、保就业。近期，中央围绕稳定稳住经济大盘做出一系列部署。“无论是稳增长，还是保就业，我们都有多种政策选择。总的来说，我国宏观政策工具箱的储备是相对充裕的，要确保把政策配置和政策操作发力点放在稳住市场主体上。”中国社会科学院副院长、党组成员、学部委员高培勇表示。统计显示，截至4月底，我国实有市场主体1.58亿户，为稳住宏观经济基本盘提供了强有力的微观基础。“市场主体是国民经济的根基之所在，经济发展的动力就在于市场主体。稳住经济大盘的实质就是稳住市场主体，把市场主体经济
        定位语块P：中国社会科学院副院长、学部委员高培勇表示
        ----------------
        示例问题Q2：谁是闫涵超？
        参考资料R：参考资料1：
        科1班代表宋文轩表示，青年学子与国家和民族同呼吸共命运，立足新时代新征程，要做毛泽东同志所说的“革命的先锋队”“脚踏实地、富于实际精神的先锋分子”，担当起“复兴栋梁、强国先锋”的重任。统计学院2020级本科1班代表闫涵超分享了自己对如何成为“复兴栋梁”的思考，他认为青年学生应当关注时代现实，关注时代需求，积极参与创新与社会实践活动，在深刻的社会实践中以“数据”视角和思维发挥自身价值，立志民族复兴为一名人大教职工，不仅要在日常教学科研中提高站位、扩大格局，更要在特殊时期冲锋向前、凝心聚力，为打赢疫情防控攻坚战贡献力量，践行“党办的大学让党放心，人民的大学不负人民”
        参考资料2：
        表闫涵超分享了自己对如何成为“复兴栋梁”的思考，他认为青年学生应当关注时代现实，关注时代需求，积极参与创新与社会实践活动，在深刻的社会实践中以“数据”视角和思维发挥自身价
        定位语块P：统计学院2020级本科1班代表闫涵超分享了自己对如何成为“复兴栋梁”的思考
        ----------------
        问题Q：{query_str}
        参考资料R：{reference_str}
        定位语段P:
        """
        self.chat = ChatOpenAI(temperature=0.0)

    def find_matching_chunks(self):
        """Find the chunks in the document that match the query.

        Returns:
            list: The list of matching chunks.
        """
        chunk_scores = {}
        for doc in self.most_common_docs:
            tokens = list(doc)  # Convert each character into a token.
            for i in range(0, len(tokens) - self.chunk_size + 1, self.chunk_size - self.overlap_size):  # Allow overlap.
                chunk = ''.join(tokens[i:i + self.chunk_size])
                score = sum(query in chunk for query in self.query_list)  # Calculate the number of matching queries.
                if score > 0:  # Store only chunks that match at least one query.
                    chunk_scores[chunk] = score

        # Sort by the number of matches from high to low, then take the top three.
        top_chunks = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:3]
        return [chunk for chunk, score in top_chunks]

    @staticmethod
    def chunks_to_string(chunks):
        """Convert a list of chunks into a string.

        Args:
            chunks (list): The list of chunks.

        Returns:
            str: The string representation of the chunks.
        """
        if not chunks:
            return "Null"
        else:
            return "\n".join(f"参考资料{i + 1}：\n{chunk}" for i, chunk in enumerate(chunks))

    def search(self, query_str):
        """Search for the query in the document and return the result.

        Args:
            query_str (str): The query string.

        Returns:
            str: The result string.
        """
        most_common_chunks = self.find_matching_chunks()
        reference_str = self.chunks_to_string(most_common_chunks)
        print("------\nreference_str: ", reference_str, "\n------")
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        messages = prompt_template.format_messages(query_str=query_str, reference_str=reference_str)
        response = self.chat(messages)
        return response.content

    def return_refence_str(self):
        """Return the reference string.

        Returns:
            str: The reference string.
        """
        most_common_chunks = self.find_matching_chunks()
        reference_str = self.chunks_to_string(most_common_chunks)
        print("------\nreference_str: ", reference_str, "\n------")
        return reference_str

class AnswerHead:
    def __init__(self):
        self.template_string = """
        现在，有一个用户问题Q，对于这个问题，有一个参考资料片段P
        你的任务是：请根据参考资料片段P，生成正确的答案A以回答问题Q
        需要特别注意的是：你的答案A只需能回答Q就可以，无需回答其他部分
        ----------------
        示例问题Q1：中国社会科学院副院长是谁？
        参考资料P：中国社会科学院副院长、学部委员高培勇表示
        答案A：高培勇
        ----------------
        示例问题Q2：谁是闫涵超？
        参考资料P：统计学院2020级本科1班代表闫涵超分享了自己对如何成为“复兴栋梁”的思考
        答案A：闫涵超是统计学院2020级本科1班的代表
        ----------------
        示例问题Q3：中国人民大学信息学院教授王珊获得2022年中国计算机学会的什么奖项?
        参考资料P：1月19日，中国计算机学会（CCF）发布公告，中国人民大学信息学院教授王珊喜获2022年中国计算机学会“CCF最高科学技术奖”。
        答案A：王珊教授获得了2022年的中国计算机学会“CCF最高科学技术奖”
        ----------------
        问题Q：{query_str}
        参考资料P：{context}
        答案A：
        """
        self.chat = ChatOpenAI(temperature=0.0)

    def generate_answer(self, query_str, context):
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        messages = prompt_template.format_messages(query_str=query_str, context=context)
        response = self.chat(messages)
        return response.content

class AnswerHeadReinforcedInCounting:
    def __init__(self):
        self.template_string = """Now there is a user question Q, for which there is a reference information snippet P\n
                Your task is: Please pretend to be an expert proficient in basic addition, subtraction, and counting. Based on the reference information snippet P, think step by step, generate the correct answer A to answer the question Q\n
                Please pay very close attention to this requirement: You can definitely get the answer A from this information snippet P, although the answer A may not be directly obtained, so you need to think step by step. You are [prohibited] from being unable to get an answer A that can answer the question Q.\n
                The type of this question is a counting question or an addition/subtraction calculation question\n
                Category: Counting Question Explanation: Questions Q that require the respondent to count, such as 'which'\n
                Category: Addition/Subtraction Calculation Question Explanation: Questions Q that require a specific numerical answer\n
                warning!!!:The reference material is extensive, and you need to take into account all the content in the reference material to arrive at a final conclusion through a global perspective. \n
                Below are some examples of answering these two types of questions, please learn from them\n
                ----------------\n
                Sample Question Q1: What is the ranking of Renmin University of China in the Chinese University Reform and Innovation Index?\n
                Reference Snippet P: Based on the index results, the top ten universities in the 2022 China University Reform and Innovation Index are: Tsinghua University, Peking University, Renmin University of China, Shanghai Jiaotong University, Beijing Normal University, Zhejiang University, Fudan University, Huazhong University of Science and Technology, University of Electronic Science and Technology, and China Agricultural University. In terms of sub-indexes, Tsinghua University ranks first in the teaching reform, scientific research reform, personnel reform, and environmental innovation rating lists, while Renmin University of China leads in the organizational innovation index among rated universities.\n
                Answer A1: third\n
                Sample Explanation E1: The ranking of Renmin University of China in the China University Reform and Innovation Index can be determined from the list given in the reference snippet P. In this list, Renmin University of China is the third university mentioned after Tsinghua University and Peking University. Therefore, it is ranked third in the index.\n
                ----------------\n
                Sample Question Q2: How many teams were eliminated in the "JD Cup" Renmin University 13th Student "Entrepreneurship Star" Competition?\n
                Reference Snippet P: A total of 50 teams participated in the "JD Cup" Renmin University 13th Student "Entrepreneurship Star" Competition. After the preliminary round, 30 teams were left, and after the semi-finals, only 15 teams were left\n
                Answer A2: 35\n
                Sample Explanation E2: The number of teams eliminated in the "JD Cup" Renmin University 13th Student "Entrepreneurship Star" Competition can be calculated from the numbers provided in reference snippet P. The total number of teams at the start of the competition is 50. After the preliminary round, 30 teams are left, which means 20 teams were eliminated. Then after the semi-finals, only 15 teams remain. Therefore, an additional 15 teams were eliminated after the semi-finals. The total number of teams eliminated is the sum of the teams eliminated after the preliminary round and the semi-finals, which is 20+15=35. Hence, 35 teams were eliminated in the competition.\n
                ----------------\n
                Question Q: {query_str}\n
                Reference Snippet P: {context}\n
                Answer A:
                """
        self.chat = ChatOpenAI(temperature=0.0)

    def answer_generator(self, query_str, context):
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        messages = prompt_template.format_messages(query_str=query_str, context=context)
        response = self.chat(messages)
        return response.content


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "your_api_key"
    extractor = QueryKeyWordsExtractor('./docs')
    query_str = '信息学院教授王珊、杜小勇获中国计算机学会（CCF）创建60周年什么奖'
    query_list = extractor.extract_keywords(query_str)
    most_common_docs_name, most_common_docs = extractor.find_most_common_doc(query_str)
    print("most_common_docs:", most_common_docs)

    Search = KeywordDocumentSearch(most_common_docs, query_list)
    answer_sent = Search.search(query_str)
    print("answer_sent:", answer_sent)

    answer_head = AnswerHead()
    final_answer = answer_head.answer_generator(query_str,answer_sent)
    print("final_answer:", final_answer)
