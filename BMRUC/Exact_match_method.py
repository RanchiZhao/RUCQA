# -*- coding:utf-8 -*-
import os
from collections import Counter
import ast
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.load_documents import load_documents
class QueryKeyWordsExtractor:
    def __init__(self, documents_directory):
        self.chat = ChatOpenAI(temperature=0.0)
        self.template_string = """现在有一个用户的问题Q\n
        你的任务是：请你找到这个问题中的重要关键短语,并以列表L的形式返回\n
        ---------------\n
        下面是几个示例，请你学习寻找关键词的要求：\n
        示例问题Q1：明朝第一任皇帝是谁？\n
        关键短语列表L：["明朝第一任皇帝"]\n
        \n
        示例问题Q2：中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建什么合作谅解备忘录\n
        关键短语列表L：["中国","尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦","合作谅解备忘录"]\n
        \n
        示例问题Q3：中国人民大学在2020年高考分数线中排第几？\n
        关键短语列表L：["中国人民大学","2020年高考分数线"]\n
        \n
        示例问题Q4：新中国第一次原子弹的空爆试验是在哪一天 \n
        关键短语列表L：["新中国第一次原子弹","空爆试验"]\n
        ----------------\n
        问题Q：{query_str}\n
        关键短语列表L：
        """
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)
        self.documents = load_documents(documents_directory)

    def extract_keywords(self, query_str):
        messages = self.prompt_template.format_messages(query_str=query_str)
        final_response = self.chat(messages)
        actual_list = ast.literal_eval(final_response.content)
        return actual_list

    def find_most_common_doc(self, query_str):
        actual_list = self.extract_keywords(query_str)
        matching_documents = [[doc for doc in self.documents if query in doc[1]] for query in actual_list]
        # 将所有列表合并为一个列表
        all_matches = [match for sublist in matching_documents for match in sublist]
        # 找出出现次数最多的文档
        counter = Counter(all_matches)

        max_count = counter.most_common(1)[0][1]  # 最高得分
        most_common_docs_name = [doc[0] for doc, count in counter.items() if count == max_count]
        most_common_docs_content = [doc[1] for doc, count in counter.items() if count == max_count]  # 所有得分最高的文档

        return most_common_docs_name, most_common_docs_content

class KeywordDocumentSearch:
    def __init__(self, most_common_docs, query_list, chunk_size=200, overlap_size=50):
        self.most_common_docs = most_common_docs
        self.query_list = query_list
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.template_string = """现在有一个用户的问题Q和几个参考资料片段R\n
        你的任务是：请你根据这些参考资料片段R，完成以下任务：\n
        找到在参考资料R中能回答问题Q的语块P\n
        你需要十分注意的要求：这个问题Q一定是能被资料片段R中俄的语块P回答的，你被【禁止】无法找到能够回答问题Q的语块P，你必须找到语块P，请一步一步地认真思考。\n
        ---------------\n
        下面是两个示例，请你认真理解它们，并据此学习如何返回答案：\n
        再次重申：你必须返回语块P，不可以说"无法找到"。
        ----------------\n
        示例问题Q1：中国社会科学院副院长是谁？\n
        参考资料R：参考资料1：\n揽子措施，加强财政货币政策协同配合，以稳住市场主体来稳住经济大盘，促进稳增长、保就业。近期，中央围绕稳定稳住经济大盘做出一系列部署。“无论是稳增长，还是保就业，我们都有多种政策选择。总的来说，我国宏观政策工具箱的储备是相对充裕的，要确保把政策配置和政策操作发力点放在稳住市场主体上。”中国社会科学院副院长、党组成员、学部委员高培勇表示。统计显示，截至4月底，我国实有市场主体1.58亿户，为稳住宏观经济基本盘提供了强有力的微观基础。“市场主体是国民经济的根基之所在，经济发展的动力就在于市场主体。稳住经济大盘的实质就是稳住市场主体，把市场主体经济\n
        定位语块P：中国社会科学院副院长、学部委员高培勇表示\n
        ----------------\n
        示例问题Q2：谁是闫涵超？\n
        参考资料R：参考资料1：\n科1班代表宋文轩表示，青年学子与国家和民族同呼吸共命运，立足新时代新征程，要做毛泽东同志所说的“革命的先锋队”“脚踏实地、富于实际精神的先锋分子”，担当起“复兴栋梁、强国先锋”的重任。统计学院2020级本科1班代表闫涵超分享了自己对如何成为“复兴栋梁”的思考，他认为青年学生应当关注时代现实，关注时代需求，积极参与创新与社会实践活动，在深刻的社会实践中以“数据”视角和思维发挥自身价值，立志民族复兴\n为一名人大教职工，不仅要在日常教学科研中提高站位、扩大格局，更要在特殊时期冲锋向前、凝心聚力，为打赢疫情防控攻坚战贡献力量，践行“党办的大学让党放心，人民的大学不负人民”\n参考资料2：\n表闫涵超分享了自己对如何成为“复兴栋梁”的思考，他认为青年学生应当关注时代现实，关注时代需求，积极参与创新与社会实践活动，在深刻的社会实践中以“数据”视角和思维发挥自身价\n
        定位语块P：统计学院2020级本科1班代表闫涵超分享了自己对如何成为“复兴栋梁”的思考\n
        ----------------\n
        问题Q：{query_str}\n
        参考资料R：{reference_str}\n
        定位语段P:
        """
        self.chat = ChatOpenAI(temperature=0.0)

    def find_matching_chunks(self):
        chunk_scores = {}
        for doc in self.most_common_docs:
            tokens = list(doc)  # 这将每个字符转化为一个token
            for i in range(0, len(tokens) - self.chunk_size + 1, self.chunk_size - self.overlap_size):  # 允许overlap
                chunk = ''.join(tokens[i:i + self.chunk_size])
                score = sum(query in chunk for query in self.query_list)  # 计算匹配的query数
                if score > 0:  # 只存储匹配到至少一个query的chunk
                    chunk_scores[chunk] = score

        # 根据匹配数从高到低排序，然后取前三个
        top_chunks = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:2]
        return [chunk for chunk, score in top_chunks]

    @staticmethod
    def chunks_to_string(chunks):
        if not chunks:
            return "Null"
        else:
            return "\n".join(f"参考资料{i + 1}：\n{chunk}" for i, chunk in enumerate(chunks))

    def search(self, query_str):
        most_common_chunks = self.find_matching_chunks()
        reference_str = self.chunks_to_string(most_common_chunks)
        print("------\nreference_str: ", reference_str, "\n------")
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        messages = prompt_template.format_messages(query_str=query_str, reference_str=reference_str)
        response = self.chat(messages)
        return response.content

#感觉总结能力太弱
class AnswerHead:
    def __init__(self):
        self.template_string = """现在有一个用户的问题Q，对于这个问题，有一个参考资料片段P\n
                你的任务是：请你根据参考资料片段P，生成正确的答案A来回答问题Q\n
                你需要十分注意的要求：你一定可以从这个资料片段P得到答案A，尽管答案A可能不是直接就能得到，因此你需要一步一步地思考。你被【禁止】无法得到能够回答问题Q的答案A\n
                ----------------\n
                问题Q：{query_str}\n
                资料片段P：{context}\n
                答案A：
                """

        # 很奇怪的现象，这里原先的template是有如下例子的，但这反而让gpt变得不敢回答：
        # """
        # --------------\n
        # 下面是几个示例，请你学习回答的要求：\n
        # 再次重申：你必须返回答案A，必须要给出回答A，如果你认为无法生成回答A，请你根据问题Q，在P中找最有可能的一个答案A，即使这个答案不一定正确\n
        # 示例问题Q1：明朝第一任皇帝是谁？\n
        # 资料片段P：朱重八，出身贫寒，却能够凭借智勇之才建立一个盛世王朝。他率领起义军，最终推翻了元朝的统治，自己建立了一个新的朝代。这个朝代，是历史上中国的一个重要阶段，被称为明朝。而他，作为这个朝代的缔造者，自然也是这个朝代的第一个皇帝。\n
        # 答案A：明朝第一任皇帝是朱重八\n
        # （因为文中提到朱重八建立了一个新的朝代，还说了这个朝代是明朝）
        # ----------------\n
        # 示例问题Q2：文化大革命什么时候结束的？\n
        # 资料片段P：在1966年，中国历史上的一场影响深远的运动——文化大革命开始了，这场运动持续了十年，而结束之年便是中国经济历史上的又一个重要的起点。"（\n
        # 答案A：1976年\n
        # （因为文中说1966年文化大革命开始，还说了它持续了十年）
        # ----------------\n
        # 示例问题Q3：北京奥运会在哪一年开始的？\n
        # 资料片段P：2000年过后的第八个春秋，中国人迎来了北京奥运会\n
        # 答案A：2008年\n
        # （因为文中说2000年过后8年，开始了北京奥运会）"""

        self.chat = ChatOpenAI(temperature=0.0)

    def answer_generator(self, query_str, context):
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        messages = prompt_template.format_messages(query_str=query_str, context=context)
        response = self.chat(messages)
        return response.content


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "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"

    extractor = QueryKeyWordsExtractor('./docs')
    query_str = '信息学院教授王珊、杜小勇获中国计算机学会（CCF）创建60周年什么奖'
    query_list = extractor.extract_keywords(query_str)
    most_common_docs_name, most_common_docs = extractor.find_most_common_doc(query_str)
    print("most_common_docs:",most_common_docs)

    Search = KeywordDocumentSearch(most_common_docs, query_list)
    answer_sent = Search.search(query_str)
    print("answer_sent:",answer_sent)

    answer_head = AnswerHead()
    final_answer = answer_head.answer_generator(query_str,answer_sent)
    print("final_answer:",final_answer)
