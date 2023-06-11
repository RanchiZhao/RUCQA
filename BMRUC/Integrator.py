from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
class AnswerIntegrator:
    def __init__(self, answer_1, answer_2):
        self.answer_1 = answer_1
        self.answer_2 = answer_2
        self.chat = ChatOpenAI(temperature=0.0)
        self.integ_template_string = """
        现在有一个用户的问题Q，对于这个问题，有两个回答A1，A2\n
        你的任务是：请你综合两个回答A1，A2并得到最终的答案A\n
        你需要十分注意的要求：A1的答案通常比A2更可靠，如果两者出现了分歧，你需要优先考虑A1；如果A1没有回答出有价值的回答，你需要参考A2;如果A1，A2给出了没有分歧的回答，你需要综合两者的回答给出答案\n
        ---------------\n
        下面是几个示例，请你学习综合回答的要求：\n
        示例问题Q1：明朝第一任皇帝是谁？\n
        回答A1：朱重八\n
        回答A2：明朝第一任皇帝是朱元璋\n
        综合后的回答A：朱重八\n
        \n
        示例问题Q2：新中国第一次原子弹的空爆试验是在哪一天\n
        回答A1：根据语境信息，我无法回答这个内容\n
        回答A2：1965年5月14日\n
        综合后的回答A：1965年5月14日\n
        \n
        示例问题Q3：中国人民大学哪四位老师获得了“庆祝香港回归祖国25周年”霍英东教育基金会第18届高等院校青年科学奖和教育教学奖？\n
        回答A1：我只知道其中的三位，分别是王孝松、陈璇、王润泽\n
        回答A2：根据语境信息，我仅仅知道一位老师获得了“庆祝香港回归祖国25周年”霍英东教育基金会第18届高等院校教育教学奖，那就是张成思\n
        综合后的回答A：王孝松、陈璇、王润泽、张成思\n
        ----------------\n
        问题Q：{query_str}\n
        回答A1：{answer_1}\n
        回答A2：{answer_2}\n
        综合后的回答A：
        """
        self.prompt_template = ChatPromptTemplate.from_template(self.integ_template_string)

    def integrate_answers(self, query_str):
        messages = self.prompt_template.format_messages(query_str=query_str, answer_1=self.answer_1, answer_2=self.answer_2)
        response = self.chat(messages)
        return response.content

    def simplify_answers(self, query_str, answer_str):
        template_string = """现在有一个用户的问题Q，对于这个问题，有一个准确的答案A\n
        你的任务是：请你极度地简化这个答案A,将在问题中出现的部分剔除\n
        你需要十分注意并满足的要求：简化后的答案B中不要有句号，且不要出现在问题Q中出现过的内容，但不能丢失其余的关键信息,且如果是数字类型，直接返回阿拉伯数字\n
        ---------------\n
        下面是几个示例，请你学习极度简化的要求：\n
        示例问题Q1：在大会中，主持人说太阳能是什么？\n
        简化前的答案A：大会中主持人谈到，太阳能是可持续发展的铁军、先锋、示范区\n
        简化后的答案B：可持续发展的铁军、先锋、示范区\n
        ---------------\n
        示例问题Q2：中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建什么合作谅解备忘录\n
        简化前的答案A：中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署的共建合作谅解备忘录是“一带一路”谅解备忘录。\n
        简化后的答案B：一带一路\n
        ---------------\n
        示例问题Q3：中国人民大学在2020年高考分数线中排第几？\n
        简化前的答案A：中国人民大学在2020年高考分数线中排第三。\n
        简化后的答案B：3\n
        ----------------\n
        问题Q：{query_str}\n
        简化前的回答A：{answer_str}\n
        简化后的回答B：
        """
        prompt_template = ChatPromptTemplate.from_template(template_string)
        messages = prompt_template.format_messages(query_str=query_str, answer_str=answer_str)
        final_response = self.chat(messages)

        return final_response.content

if __name__ == '__main__':
    ai = AnswerIntegrator(answer_1="朱重八", answer_2="明朝第一任皇帝是朱元璋")
    integrated_answer = ai.integrate_answers("明朝第一任皇帝是谁？")
    print(integrated_answer)


