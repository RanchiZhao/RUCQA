## RUCQA - Based on LlamaIndex & LangChain 

### 1.Document_Crawler

4533 docs in total, which are crawled based on the requirement. Website: [RUC NEWS](https://news.ruc.edu.cn/)

### 2.Overall Structure

![ML Visuhttps___docs.google.com_presentation_u_0__authuser=0&usp=slides_webals_00](img-folder/ML Visuhttps___docs.google.com_presentation_u_0__authuser=0&usp=slides_webals_00.png)

### 3.VRM(Vector store index based Retrieval Module) - Powed by Llama Index

#### 3.1 Load in Documents

This phase involves loading the data. We use the `SimpleDirectoryReader` class and its `load_data` function to obtain Document objects.

#### 3.2 Index Construction

At this stage, we construct an index over these Document objects, specifically with `GPTVectorStoreIndex`. By default, OpenAI's `text-davinci-003` model is employed, with its max_tokens set at 128.

Notably, we retain the default prompt templates during the process of index construction and querying, without adding customized embeddings to our model— the defaults are adequate. For the embedding method, according to [m3e-base](https://huggingface.co/moka-ai/m3e-base), the openai-ada-002 model (which we employ) is sufficiently effective for a range of Chinese tasks, thereby negating the necessity of text2vec. We may explore the potential of m3e-base in the future.

#### 3.3 Query the index

Given our use of `VectorStoreIndex`, the `VectorIndexRetriever` should be used to construct our `QueryEngine`. At this point, we can feed the model a query to find an answer!

A useful approach here is to append the string `'\n请特别注意：你需要用中文回答，只能参考我提供的资料，不允许查看外部资料'` to the query to restrict the model.

#### 3.4 More information about Vector Store Index

The Vector Store Index holds each Node along with a corresponding embedding within a [Vector Store](https://gpt-index.readthedocs.io/en/latest/how_to/integrations/vector_stores.html#vector-store-index).

![img](https://gpt-index.readthedocs.io/en/latest/_images/vector_store.png)

Querying a Vector Store Index involves retrieving the top-k most similar Nodes (in this case, we set k to 2) and passing these to our Response Synthesis module. You can find more details about this on [ReadTheDocs](https://gpt-index.readthedocs.io/en/latest/).

![img](https://gpt-index.readthedocs.io/en/latest/_images/vector_store_query.png)

For this part, more details available on [ReadTheDocs](https://gpt-index.readthedocs.io/en/latest/)

### 4.(KRM)Keywords based Retrieval Module - Powered by LangChain 

Given the relatively straightforward nature of the provided questions, we require a more robust retrieval method, namely one based on the keywords found within queries.

Our pipeline for this process includes:

**<1> Extract keywords from the query**

**<2> Search for the most related chunks based on these keywords**

**<3> Find the sentence related to the answer within these chunks (which serve as reference materials here)**

**<4> Determine the answer based on the answer-related sentence**

**<5> Integrate the answer from this process with the answer from VRM into a single response**

**<6> Simplify the integrated answer into the required format**

As evident, we break the whole process down into manageable tasks. For each segment, we delegate these specific subdivisions of tasks to the LLM via LangChain (except for step <2>).

The prompts used in this process play a crucial role. We have designed them with care, primarily employing In-Context Learning (ICL), and the entire procedure can, to some degree, be seen as a Chain of Thought (CoT). Here, we present examples of these prompts:

- <1> Extract keywords from the query

  ```python
  """
  现在有一个用户的问题Q\n
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
  ```

- <3> Find the sentence related to the answer within these chunks (which serve as reference materials here)

  ```python
  """
  现在有一个用户的问题Q和几个参考资料片段R\n
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
  ```

- <4> Determine the answer based on the answer-related sentence

  For this aspect, we found it intriguing that the model seemed reluctant to respond when presented with few-shot learning tasks. This could potentially be because the examples we initially provided were not well-suited or perhaps too complex.

  ```python
  """
  现在有一个用户的问题Q，对于这个问题，有一个参考资料片段P\n
  你的任务是：请你根据参考资料片段P，生成正确的答案A来回答问题Q\n
  你需要十分注意的要求：你一定可以从这个资料片段P得到答案A，尽管答案A可能不是直接就能得到，因此你需要一步一步地思考。你被【禁止】无法得到能够回答问题Q的答案A\n
  ----------------\n
  问题Q：{query_str}\n
  资料片段P：{context}\n
  答案A：
  """
  ```

- <5> Integrate the answer from this process with the answer from VRM into a single response

  ```python
  """
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
  ```

- <6> Simplify the integrated answer into the required format

  ```python
  """
  现在有一个用户的问题Q，对于这个问题，有一个准确的答案A\n
  你的任务是：请你极度地简化这个答案A，也就是说，将在问题中出现的部分剔除，只保留答案部分\n
  你需要十分注意并满足的要求：简化后的答案B中不要有句号，且不要出现在问题Q中出现过的内容，但不能丢失其余的关键信息,且如果是数字类型，直接返回阿拉伯数字\n
  ---------------\n
  下面是几个示例，请你学习极度简化的要求：\n
  示例问题Q1：在大会中，主持人说太阳能是什么？\n
  简化前的答案A：大会中主持人谈到，太阳能是可持续发展的铁军、先锋、示范区\n
  简化后的答案B：可持续发展的铁军、先锋、示范区\n
  ---------------\n
  示例问题Q2：马云在参观阿里巴巴全球研发中心时，主要了解了哪两个研究方向？\n
  简化前的答案A：马云在杭州参观阿里巴巴全球研发中心，期间他深入了解了“智能科技推动商业进步，为全球经济注入新动力”和“环保创新，倡导绿色可持续发展”的两大研究方向。\n
  简化后的答案B：“智能科技推动商业进步，为全球经济注入新动力”；“环保创新，倡导绿色可持续发展”\n
  ---------------\n
  示例问题Q3：中国人民大学在2020年高考分数线中排第几？\n
  简化前的答案A：中国人民大学在2020年高考分数线中排第三。\n
  简化后的答案B：3\n
  ----------------\n
  问题Q：{query_str}\n
  简化前的回答A：{answer_str}\n
  简化后的回答B：
  """
  ```

  
