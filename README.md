# Design-Space-for-RAG (ongoing)

## Overview
This repository serves as a comprehensive survey of the design space for Retrieval Augmented Generation (RAG). RAG is a powerful approach that combines retrieval mechanisms with generative models to enhance the quality and relevance of generated text. 
However, why do we need to define the design space of RAG? Because the research community and practitioners need clarity on which components are necessary for specific tasks and how to select them. In other words, we need a new roadmap for RAG research—one that maps out the landscape of choices and tradeoffs in designing effective RAG systems.  This survey explores different modules within RAG, such as retrievers, refiners, rerankers, and generators, as well as the various pipelines that can be constructed using these modules.

## Table of Contents  
1. [RAG Modules](#rag-modules)  
   1. [Retriever](#retriever)  
      1. [Encoding](#encoding)  
      2. [Indexing](#indexing)  
      3. [Retrieval](#retrieval)  
   2. [Reranker](#reranker)  
   3. [Refiner](#refiner)  
   4. [Generator](#generator)  
2. [RAG Pipelines](#rag-pipelines)  
3. [Contributing](#contributing)  
4. [License](#license)  
## RAG Modules

### Encoder
In a RAG system, the core role of the encoder is to encode the **queries** from the users and **documents** from the external database into **dense vectors** for **semantic similarity calculation**. Subsequently, by computing the distances between these vectors, the system rapidly retrieves the **most relevant documents** from the knowledge base that match the user's query.

In this section, we will only discuss dense encoding models, and we will categorize common encoders into two categories: open-source and closed-source for discussion.

#### Open-source Encoder:
1. [BGE Series](https://bge-model.com/bge/index.html): BGE stands for **B**AAI **G**eneral **E**mbeddings, which is a series of BERT-based embedding models released by BAAI.

| Model Name                     | Dimension | Max Token | Parameter Scale   | Memory Usage (MB)     | Comments                                                                 |
|--------------------------------|-----------|-----------|-------------------|-----------------------|--------------------------------------------------------------------------|
| `bge-base-v1.5`                | 768       | 512       | 109M              | 390                   | Most commonly used embedding models among BGE family. It has two separate model versions for encoding Chinese and English respectively. Three embedding models of different sizes (small / base / large) are provided in version 1.5 . |
| `bge-m3`                       | 1024      | 8192      | 596M              | 1370                  | **Multi-Functionality:** simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval. <br>**Multi-Linguality:** support more than 100 working languages. <br>**Multi-Granularity:** is able to process inputs with up to 8192 tokens.  |
| `bge-en-icl`                   | 4096      | 32768     | 7.1B              | 27125                 | Adopted Mistral-7B as the backbone. By providing few-shot examples in the query, it can significantly enhance the model’s ability to handle new tasks.|

2. [E5 Series](https://github.com/microsoft/unilm/blob/master/e5/README.md): The e5-large model is a powerful text embedding model developed by Microsoft, known for its strong generalization capabilities.

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage (MB) | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|-------------------|-----------------------------------------------------------------------------|
| `e5-base-v2`                   | 1024      | 514       | 109M            | 418           | Most commonly used embedding models among e5 family. This model only works for English texts. Long texts will be truncated to at most 512 tokens. Three embedding models of different sizes (small / base / large) are providedin version 2.  |
| `multilingual-e5-base`         | 768       | 514       | 278M            | 1061          | Specifically designed for multi-lingual input. Three embedding models of different sizes (small / base / large) are provided. An instruction-tuned embedding model based on multilingual-e5-large was introduced too.                                                                    |
| `e5-mistral-7b-instruct`       | 4096      | 32768     | 7B              | 13563         | This model is initialized from Mistral-7B-v0.1 and fine-tuned on a mixture of multilingual datasets. As a result, it has some multilingual capability. However, since Mistral-7B-v0.1 is mainly trained on English data, it is recommended to use this model for English only. For multilingual use cases, please refer to multilingual-e5.  |

3. [GTE Series](https://huggingface.co/collections/Alibaba-NLP/gte-models-6680f0b13f885cb431e6d469): GTE stands for **G**eneral **T**ext **E**mbedding Models Released by Tongyi Lab of Alibaba Group. 

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage (MB) | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|-------------------|-----------------------------------------------------------------------------|
| `gte-base-v2`                  | 768       | 512       | -               | -                 | It has two separate model versions for encoding Chinese and English respectively. |
| `gte-large`                    | 1024      | 512       | -               | -                 | -                                                                           |
| `gte-Qwen2-7B-instruct`        | 1024      | 514       | 7B              | 29040 MB          | -                                                                           |

4. [Jina Series](https://huggingface.co/collections/Alibaba-NLP/gte-models-6680f0b13f885cb431e6d469).

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage  | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|---------------|-----------------------------------------------------------------------------|
| `jina-embeddings-v2-base-en`   | 768       | -         | -               | -             |                                                                             |
| `jina-embeddings-v2-base-zh`   | 768       | -         | -               | -             |                                                                             |
| `jina-embeddings-v2-small-en`  | 384       | -         | -               | -             |                                                                             |
| `jina-embeddings-v3`           | -         | -         | -               | -             |                                                                             |

5. [SFR Series](https://huggingface.co/moka-ai/m3e-base) 

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage  | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|---------------|-----------------------------------------------------------------------------|
| `SFR-Embedding-Mistral`        | 768       | -         | -               | -             |                                                                             |
| `SFR-Embedding-2_R`            | 1024      | -         | -               | -             |                                                                             |

6. [M3e Series](https://huggingface.co/moka-ai/m3e-base)

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage  | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|---------------|-----------------------------------------------------------------------------|
| `m3e-base`                     | 768       | -         | -               | -             |                                                                             |
| `m3e-large`                    | 1024      | -         | -               | -             |                                                                             |
| `m3e-small`                    | 384       | -         | -               | -             |                                                                             |

#### Close-source Encoder:

1. [OpenAI text-embeding Series](https://openai.com/index/new-embedding-models-and-api-updates/)

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage  | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|---------------|-----------------------------------------------------------------------------|
| `m3e-small`                    | 384       | -         | -               | -             |                                                                             |
| `m3e-base`                     | 768       | -         | -               | -             |                                                                             |
| `m3e-large`                    | 1024      | -         | -               | -             |                                                                             |

2. [Cohere embed Series](https://cohere.com/embed)

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage  | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|---------------|-----------------------------------------------------------------------------|
| `m3e-small`                    | 384       | -         | -               | -             |                                                                             |
| `m3e-base`                     | 768       | -         | -               | -             |                                                                             |
| `m3e-large`                    | 1024      | -         | -               | -             |                                                                             |

3. [Gemini embedding Series](https://ai.google.dev/gemini-api/docs/embeddings)

| Model Name                     | Dimension | Max Token | Parameter Scale | Memory Usage  | Comments                                                                    |
|--------------------------------|-----------|-----------|-----------------|---------------|-----------------------------------------------------------------------------|
| `m3e-small`                    | 384       | -         | -               | -             |                                                                             |
| `m3e-base`                     | 768       | -         | -               | -             |                                                                             |
| `m3e-large`                    | 1024      | -         | -               | -             |                                                                             |

### Indexing
Indexing plays a pivotal role in RAG systems. It is the process of structuring and organizing the data in a corpus to enable efficient and rapid retrieval of relevant information.
In the context of RAG, indexing is essential for the retriever module. A well - designed index allows the retriever to quickly sift through a large amount of data and identify documents that are most relevant to the input query. This is crucial as the speed and accuracy of retrieval directly impact the quality of the generated response by the generator.

### Retriever
The retriever is responsible for retrieving relevant documents from a large corpus based on the input query. It plays a crucial role in RAG by providing the generator with the necessary context to generate high-quality responses. Different types of retrievers include:
- **Dense Retrievers**: These retrievers use neural networks to represent queries and documents as dense vectors and then compute similarity scores between them.
- **Sparse Retrievers**: These retrievers use traditional information retrieval techniques, such as term frequency-inverse document frequency (TF-IDF), to represent documents and queries as sparse vectors.

### Refiner
The refiner module is used to refine the retrieved documents before passing them to the generator. It can perform tasks such as filtering, summarization, and rephrasing to improve the quality and relevance of the retrieved documents.

### Reranker
The reranker module is used to re-rank the retrieved documents based on their relevance to the input query. It can use machine learning models or other techniques to assign a score to each document and then sort them in descending order of relevance.

### Generator
The generator is responsible for generating the final response based on the input query and the retrieved documents. It can be a pre-trained language model, such as GPT-3 or BERT, or a custom-built model.

## RAG Pipelines
Different RAG systems can have different pipelines, depending on the specific requirements and use cases. 

## Relevant Works
[FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research](https://arxiv.org/abs/2405.13576)  

[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)

[Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921)

[A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2405.06211)


## Contributing
We welcome contributions to this survey! If you have any suggestions, improvements, or new findings related to the design space for RAG, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
