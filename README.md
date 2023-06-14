## RUCQA - Based on LlamaIndex & LangChain 

### 0.Task Description

#### Goal: 

- Build a Knowledge-based Question-Answering System - **RUCQA**

#### Requirements:

- No corpus is provided, it needs to be crawled from the archive of articles published on the People's University News website over the past year (from June 2022 to May 2023) (also the evaluation range).
- The LLM interface needs to be invoked.

#### Evaluation:

- 100 fill-in-the-blank questions, checking for correctness of the answers.
- 60 simple questions: Directly extracted from the corpus (similar to reading comprehension tasks).
- 40 difficult questions: Require summarizing and reasoning based on the corpus.
- 20 open questions, assessing the correctness and fluency of the generated answers. No standard answers, flexible generation (up to 512 characters). The basis for the generated answer needs to be given (up to 5 specific, relevant news pieces). This part is evaluated manually.

#### Examples:：

- 60 simple questions:

  - Q：中国人民大学信息学院教授王珊获得2022年中国计算机学会的什么奖项？

    A：CCF最高科学技术奖        https://news.ruc.edu.cn/archives/417168


  - Q：中国人民大学文学院院长是谁？

    A：陈剑澜 		https://news.ruc.edu.cn/archives/424830


  - Q：中国人民大学性社会学研究所所长是谁？

    A：黄盈盈       https://news.ruc.edu.cn/archives/394641


  - Q: 中国与尼加拉瓜、叙利亚、阿根廷、马拉维、巴勒斯坦等国签署共建什么合作谅解备忘录 

    A：一带一路        https://news.ruc.edu.cn/archives/415228


  - Q: 中国人民大学与哪个公司签约成立未来媒体智能联合实验室？

    A: 快手       https://news.ruc.edu.cn/archives/429314

- 15 counting questions:

  - Count the corresponding content of the article to get the result, the answer is a number.

  - Q: 中国人民大学在中国大学改革创新指数中排第几？

    A: 3

  - https://news.ruc.edu.cn/archives/396913

- 10 summarizing questions:

  - Summarize the corresponding content of the article to get the answer.

  - Q: 中国人民大学哪四位老师获得了“庆祝香港回归祖国25周年”霍英东教育基金会第18届高等院校青年科学奖和教育教学奖？

    A: 王孝松、陈璇、王润泽、张成思

  - https://news.ruc.edu.cn/archives/418092

- 10 multi-document questions:

  - Need to refer to multiple articles to get the answer.

  - Q: 哪位教授同时获得2022年“CCF最高科学技术奖”和“中国计算机学会（CCF）创建60周年杰出贡献奖”

    A: 王珊

  - https://news.ruc.edu.cn/archives/386431

  - https://news.ruc.edu.cn/archives/417168

- 5 addition and subtraction questions:

  - Calculate the answer by adding or subtracting the corresponding content of the article, the answer is a number.

  - Q: 中国人民大学哪一年建校？

    A: 1937

  - https://news.ruc.edu.cn/archives/415642


*For the convenience of evaluation, it is hoped that answers of a numeric type (such as counting questions, calculation questions, etc.) are directly output in the form of Arabic numerals. Other types of answers will be overwritten by the assistant according to the rules during the evaluation.*



### 1.Document Crawler

A total of 4,533 documents have been crawled based on the requirements from the website: [RUC NEWS](https://news.ruc.edu.cn/)

### 2.Overall Structure

![Overall Structure](https://github.com/RanchiZhao/RUCQA/raw/main/img-folder/Overall_Structure.png)

The overall strategy is to use one route based on **vector retrieval** and another based on **keyword retrieval.** The vector-based retrieval is achieved by calling the Llama Index (which performs particularly well for questions that require global information). The keyword-based approach uses Langchain to break down the overall task into a series of consecutive subtasks (which performs particularly well for simple questions). 

Finally, the answers from the two approaches are merged to produce the final answer. 

In addition, for different question categories, corresponding prompts have been designed in the keyword-based retrieval pathway to enhance the ability to perform tasks of that particular category.

### 3.VRM(Vector store index based Retrieval Module) - Powered by Llama Index

#### 3.1 Load in Documents

This phase involves loading the data. We use the `SimpleDirectoryReader` class and its `load_data` function to obtain Document objects.

#### 3.2 Index Construction

At this stage, we construct an index over these Document objects, specifically with `GPTVectorStoreIndex`. By default, OpenAI's `text-davinci-003` model is employed, with its max_tokens set at 128.

Notably, we retain the default prompt templates during the process of index construction and querying, without adding customized embeddings to our model— the defaults are adequate. For the embedding method, according to [m3e-base](https://huggingface.co/moka-ai/m3e-base), the openai-ada-002 model (which we employ) is sufficiently effective for a range of Chinese tasks, thereby negating the necessity of text2vec. We may explore the potential of m3e-base in the future.

#### 3.3 Query the index

Given our use of `VectorStoreIndex`, the `VectorIndexRetriever` should be used to construct our `QueryEngine`. At this point, we can feed the model a query to find an answer!

A useful approach here is to append the string `'\nPlease note specifically: You need to answer in Chinese, and you can only refer to the materials I provide. It is not allowed to view external materials.'` to the query to restrict the model.

#### 3.4 More information about Vector Store Index

The Vector Store Index holds each Node along with a corresponding embedding within a [Vector Store](https://gpt-index.readthedocs.io/en/latest/how_to/integrations/vector_stores.html#vector-store-index).

![img](https://gpt-index.readthedocs.io/en/latest/_images/vector_store.png)

Querying a Vector Store Index involves retrieving the top-k most similar Nodes (in this case, we set k to 2) and passing these to our Response Synthesis module. You can find more details about this on [ReadTheDocs](https://gpt-index.readthedocs.io/en/latest/).

![img](https://gpt-index.readthedocs.io/en/latest/_images/vector_store_query.png)

For this part, more details available on [ReadTheDocs](https://gpt-index.readthedocs.io/en/latest/)

### 4.(KRM)Keywords based Retrieval Module - Powered by LangChain 

Given the relatively straightforward nature of the provided questions, we require a more robust retrieval method, namely one based on the keywords found within queries.

Our pipeline for this process includes:

**<1> Extract keywords from the query & Discriminate the classification of query**

**<2> Search for the most related chunks based on these keywords**

**<3> Find the sentence related to the answer (just locate, do not answer) within these chunks (which serve as reference materials here)**

**<4> Determine the answer based on the answer-related sentence**

**<5> Integrate the answer from this process with the answer from VRM into a single response**

**<6> Simplify the integrated answer into the required format**

As evident, we break the whole process down into manageable tasks. For each segment, we delegate these specific subdivisions of tasks to the LLM via LangChain (except for step <2>).

Specifically, for different categories of queries identified by the discriminator, the specific implementation of each step (the design of the prompt) may vary, but the overall idea is as you see!

The prompts used in this process play a crucial role. We have designed them with care, primarily employing In-Context Learning (ICL), and the entire procedure can, to some degree, be seen as a Chain of Thought (CoT). 





