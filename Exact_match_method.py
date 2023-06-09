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
        # print("actual_list: ",actual_list)
        return actual_list

    def find_most_common_doc(self, query_str):
        actual_list = self.extract_keywords(query_str)
        matching_documents = [[doc for doc in self.documents if query in doc] for query in actual_list]

        # 将所有列表合并为一个列表
        all_matches = [match for sublist in matching_documents for match in sublist]

        # 找出出现次数最多的文档
        counter = Counter(all_matches)

        max_count = counter.most_common(1)[0][1]  # 最高得分
        most_common_docs = [doc for doc, count in counter.items() if count == max_count]  # 所有得分最高的文档

        return most_common_docs

class KeywordDocumentSearch:
    def __init__(self, most_common_docs, query_list, chunk_size=500, overlap_size=50):
        self.most_common_docs = most_common_docs
        self.query_list = query_list
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.template_string = """现在有一个用户的问题Q和一些参考资料R\n
        你的任务是：请你根据这些参考资料片段，一步一步地、仔细地先定位答案中问题内容出现的位置P，再正确地返回这个问题的准确的答案A\n
        你需要十分注意的要求：只可以参考我提供的资料，不允许查看外部资料\n
        ---------------\n
        下面是一个示例，请你据此学习返回答案的要求：\n
        示例问题Q1：中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建什么合作谅解备忘录\n
        参考资料R：参考资料1：\n丝绸之路、创新丝绸之路正成为新的努力方向。王义桅指出，面对新冠疫情的影响，“一带一路”合作不但没有按下“暂停键”，反而迎来重要的转型与升级，数字化、绿色、健康趋势更加明显。顺应大势，构建人类命运共同体今年以来，中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建“一带一路”合作谅解备忘录。截至目前，中国已与150个国家、32个国际组织签署200余份共建“一带一路”合作文件。无数“连心桥”“繁荣港”“幸福路”，在共\n参考资料2：\n丝绸之路、创新丝绸之路正成为新的努力方向。王义桅指出，面对新冠疫情的影响，“一带一路”合作不但没有按下“暂停键”，反而迎来重要的转型与升级，数字化、绿色、健康趋势更加明显。顺应大势，构建人类命运共同体今年以来，中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建“一带一路”合作谅解备忘录。截\n
        问题对应答案位置P：中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建“一带一路”合作谅解备忘录\n
        答案A：一带一路\n
        示例问题Q2：中国社会科学院副院长是谁？\n
        参考资料R：参考资料1：\n揽子措施，加强财政货币政策协同配合，以稳住市场主体来稳住经济大盘，促进稳增长、保就业。近期，中央围绕稳定稳住经济大盘做出一系列部署。“无论是稳增长，还是保就业，我们都有多种政策选择。总的来说，我国宏观政策工具箱的储备是相对充裕的，要确保把政策配置和政策操作发力点放在稳住市场主体上。”中国社会科学院副院长、学部委员高培勇表示。统计显示，截至4月底，我国实有市场主体1.58亿户，为稳住宏观经济基本盘提供了强有力的微观基础。“市场主体是国民经济的根基之所在，经济发展的动力就在于市场主体。稳住经济大盘的实质就是稳住市场主体，把市场主体经济\n
        问题对应答案位置P：中国社会科学院副院长、学部委员高培勇表示\n
        答案：高培勇\n
        ----------------\n
        问题Q：{query_str}\n
        参考资料R：{reference_str}\n
        问题对应答案位置P和答案A：
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
        # print("reference_str: ", reference_str)
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        messages = prompt_template.format_messages(query_str=query_str, reference_str=reference_str)
        response = self.chat(messages)
        return response.content

if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "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"

    extractor = QueryKeyWordsExtractor('./docs')
    query_str = '信息学院教授王珊、杜小勇获中国计算机学会（CCF）创建60周年什么奖'
    query_list = extractor.extract_keywords(query_str)
    most_common_docs = extractor.find_most_common_doc(query_str)
    print(most_common_docs)

    Search = KeywordDocumentSearch(most_common_docs, query_list)
    result = Search.search(query_str)
    print(result)
