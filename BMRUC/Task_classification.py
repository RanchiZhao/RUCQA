import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
class TaskClassification:
    def __init__(self):
        self.chat = ChatOpenAI(temperature=0.0)
        self.template_string = """Now, there is a user's question Q.\n
        Your task is: Please carefully understand the what is Question Q asking? And accurately judge what type of question Q is.\n
        Here are the categories and corresponding explanations of the questions:\n
        
        Category: Addition and Subtraction Calculation Question. Explanation: The question Q requires an answer in specific numbers.\n
        Category: Single Object Question. Explanation: Question Q implicitly requires the answer to have only one object.\n
        Category: Multiple Objects Question. Explanation: Question Q implies the need to list multiple objects in the answer.\n
        Category: Counting Question. Explanation: Question Q asks for 'which number', which requires the respondent to count, like 'which one'.\n
       
        请十分注意：你需要完全服从以下要求\n
        如果问题Q中出现“哪个”，”什么“，”是谁“，”哪位“，“谁”，则这个问题Q一定是Single Object Question\n
        如果问题Q中出现“哪几个”，”哪些“，“哪几位”，则这个问题Q一定是Multiple Objects Question\n
        Here are a few examples for you to learn how to judge the types of questions:\n
        ----------------\n
        Example Question 1:中国人民大学在中国大学改革创新指数中排第几\n
        question type:Counting Question\n
        \n
        Example Question 2:中国人民大学哪四位老师获得了“庆祝香港回归祖国25周年”霍英东教育基金会第18届高等院校青年科学奖和教育教学奖？\n
        question type:Multiple Objects Question\n
        \n
        Example Question 5:中国人民大学文学院院长是谁?\n
        question type:Single Object Question\n
        ----------------\n
        Question:{query_str}\n
        question type:
        """
        self.prompt_template = ChatPromptTemplate.from_template(self.template_string)

    def return_classification(self, query_str):
        messages = self.prompt_template.format_messages(query_str=query_str)
        classification = self.chat(messages)

        return classification
if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "your_api_key"
    text_classification = TaskClassification()
    query_str = '2022年首届健康中国建设学术年会在京举办，其中上午的“开幕式和成果展示”环节由谁主持？'
    classification = text_classification.return_classification(query_str).content
    print("classification: ", classification)
