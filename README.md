# Design-Space-for-RAG

## Overview
This repository serves as a comprehensive survey of the design space for Retrieval Augmented Generation (RAG). RAG is a powerful approach that combines retrieval mechanisms with generative models to enhance the quality and relevance of generated text. 
However, why do we need to define the design space of RAG? Because the research community and practitioners need clarity on which components are necessary for specific tasks and how to select them. In other words, we need a new roadmap for RAG research—one that maps out the landscape of choices and tradeoffs in designing effective RAG systems.  This survey explores different modules within RAG, such as retrievers, refiners, rerankers, and generators, as well as the various pipelines that can be constructed using these modules.

## Table of Contents
1. [RAG Modules](#rag-modules)
    - [Encoder](#encoder)
    - [Indexing](#indexing)
    - [Retriever](#retriever)
    - [Refiner](#refiner)
    - [Reranker](#reranker)
    - [Generator](#generator)
3. [RAG Pipelines](#rag-pipelines)
4. [Survey Findings](#survey-findings)
5. [Contributing](#contributing)
6. [License](#license)

## RAG Modules

### Encoder
In a RAG system, the core role of the encoder is to encode the **queries** from the users and **documents** from the external database into **dense vectors** for **semantic similarity calculation**. Subsequently, by computing the distances between these vectors, the system rapidly retrieves the **most relevant documents** from the knowledge base that match the user's query.

In this section, we will only discuss dense encoding models, and we will categorize common encoders into two categories: open-source and closed-source for discussion.

#### Open-source Encoder:
1. BGE Series:
	1. [BGE-v1&v1.5](https://bge-model.com/bge/bge_v1_v1.5.html): BGE stands for BAAI General Embeddings, which is a series of BERT-based embedding models released by BAAI. The v1 version, released in August 2023, introduced multilingual (Chinese/English) and multi-scale (large/medium/small) models. The v1.5 version, released in September 2023, optimized retrieval capabilities in instruction-free scenarios, addressing issues with vector similarity distribution while balancing embedding quality and model scale. This encoder also supports [Matryoshka truncation](https://arxiv.org/pdf/2205.13147). During fine-tuning, users can specify different truncation lengths to adapt the model to various input scenarios.
	2. [BGE-M3](https://bge-model.com/bge/bge_m3.html)
 	3. 

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
