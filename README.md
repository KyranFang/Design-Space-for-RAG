# Design-Space-for-RAG

## Overview
This repository serves as a comprehensive survey of the design space for Retrieval Augmented Generation (RAG). RAG is a powerful approach that combines retrieval mechanisms with generative models to enhance the quality and relevance of generated text. 
However, why do we need to define the design space of RAG? Because the research community and practitioners need clarity on which components are necessary for specific tasks and how to select them. In other words, we need a new roadmap for RAG research—one that maps out the landscape of choices and tradeoffs in designing effective RAG systems.  This survey explores different modules within RAG, such as retrievers, refiners, rerankers, and generators, as well as the various pipelines that can be constructed using these modules.

## Table of Contents
1. [RAG Modules](#rag-modules)
    - [Retriever](#retriever)
    - [Refiner](#refiner)
    - [Reranker](#reranker)
    - [Generator](#generator)
2. [RAG Pipelines](#rag-pipelines)
3. [Survey Findings](#survey-findings)
4. [Contributing](#contributing)
5. [License](#license)

## RAG Modules

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
Flash Rag(https://arxiv.org/abs/2405.13576)

## Contributing
We welcome contributions to this survey! If you have any suggestions, improvements, or new findings related to the design space for RAG, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
