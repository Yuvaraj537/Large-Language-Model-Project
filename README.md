# 🤖 Natural Language Processing & AI Concepts  

Welcome to this **NLP & AI Concepts Repository** 🎯  
This repo contains simplified explanations of **core concepts** used in **Machine Learning, Deep Learning, and Generative AI** 🚀  

---

## 📚 Contents  
- 🔹 [LLM (Large Language Models)](#-llm-large-language-models)  
- 🔹 [NLTK](#-nltk-natural-language-toolkit)  
- 🔹 [spaCy](#-spacy)  
- 🔹 [Text Processing Types](#-text-processing-types)  
- 🔹 [N-Grams](#-n-grams)  
- 🔹 [Vectors](#-vectors)  
- 🔹 [Bag of Words (BoW)](#-bag-of-words-bow)  
- 🔹 [Embeddings](#-embeddings)  
- 🔹 [BERT](#-bert)  
- 🔹 [Transformers](#-transformers)  
- 🔹 [Prompt Engineering](#-prompt-engineering)  
- 🔹 [RAG (Retrieval-Augmented Generation)](#-rag-retrieval-augmented-generation)  
- 🔹 [Hugging Face](#-hugging-face)  
- 🔹 [MCP (Model Context Protocol)](#-mcp-model-context-protocol)  
- 🔹 [AI Agents](#-ai-agents)  
- 🔹 [FAISS](#-faiss)  
- 🔹 [Word Embeddings](#-word-embeddings)  

---

## 🧠 LLM (Large Language Models)  
**Definition:**  
LLMs are **deep learning models** trained on massive amounts of text data to understand, generate, and reason with human language.  
✅ Examples: GPT, LLaMA, Gemini.  

---

## 📘 NLTK (Natural Language Toolkit)  
- Python library for NLP.  
- Used for **tokenization, stemming, lemmatization, stopwords removal, POS tagging, parsing**.  
✅ Example: `from nltk.tokenize import word_tokenize`  

---

## ⚡ spaCy  
- **Fast NLP library** optimized for production.  
- Features: Tokenization, POS tagging, Named Entity Recognition (NER), Dependency Parsing.  
✅ Example: `import spacy; nlp = spacy.load("en_core_web_sm")`  

---

## ✂️ Text Processing Types  
- **Tokenization** – Split text into words/sentences.  
- **Stopword Removal** – Remove common words (is, the, and).  
- **Stemming** – Reduce words to root (playing → play).  
- **Lemmatization** – Context-aware root word (better → good).  
- **POS Tagging** – Identify nouns, verbs, adjectives.  

---

## 🔡 N-Grams  
An **N-Gram** is a continuous sequence of **N items (words, characters, or tokens)** from a text.  
They are widely used in **text analysis, language modeling, and NLP tasks**.  

--- 
- **Types:**  
  - Unigram: 1 word → "AI"  
  - Bigram: 2 words → "AI Agent"  
  - Trigram: 3 words → "AI is powerful"  

---

## 🧩 Vectors  
- **Mathematical representation of words/sentences** in numerical form.  
- **Types:**  
  - One-hot Encoding  
  - TF-IDF  
  - Dense Embeddings  

## 🧩 Vectors  

In NLP, **vectors** are the numerical representation of text (words, sentences, or documents).  
They allow algorithms to **process and understand language mathematically**.  

---

### 📑 Types of Vectors  

1. **🔹 One-Hot Encoding**  
   - Each word is represented as a **binary vector**.  
   - The position of `1` indicates the word, and the rest are `0`s.  
   - ✅ Example (Vocabulary = ["AI", "is", "powerful"]):  
     - "AI" → [1, 0, 0]  
     - "is" → [0, 1, 0]  
     - "powerful" → [0, 0, 1]  
   - ⚠️ Limitation: High-dimensional & does not capture meaning.  

2. **🔹 TF-IDF (Term Frequency – Inverse Document Frequency)**  
   - Represents text based on **importance of words**.  
   - Formula:  
     - **TF** = How often a word appears in a document.  
     - **IDF** = How unique the word is across all documents.  
   - ✅ Example:  
     - Common word like "the" → low score.  
     - Rare but important word like "neural" → high score.  
   - ⚠️ Still ignores context & word order.  

3. **🔹 Dense Embeddings**  
   - Words/sentences represented as **low-dimensional dense vectors**.  
   - Captures **semantic meaning** and similarity.  
   - Generated using **neural networks** (Word2Vec, GloVe, FastText, BERT, etc.).  
   - ✅ Example:  
     - "king" and "queen" will be **close** in vector space.  
     - "king" and "car" will be **far apart**.  
   - ⚡ Advantage: Captures context, meaning, and relationships.  

---

## 📦 Bag of Words (BoW)

**Definition:**  
Bag of Words (BoW) is a simple way to represent text as a **numerical vector** based on **word frequency**.  
It **ignores grammar, word order, and context**.  

---

### 🔹 How it Works
1. Identify all **unique words** in the text (vocabulary).  
2. Count **how many times each word appears**.  
3. Represent the text as a **vector or dictionary** with these counts.  

---

# 🔑 Embeddings  

Embeddings are **dense vector representations of text** that capture **semantic meaning**.  
Unlike Bag-of-Words or TF-IDF (which are sparse), embeddings place similar words closer in a **vector space**.  


### 📑 Types of Embeddings  

1. **🔹 Word2Vec** (Google, 2013)  
   - Learns word embeddings using **neural networks**.  
   - Two architectures:  
     - **CBOW (Continuous Bag of Words):** Predicts a word from its context.  
     - **Skip-Gram:** Predicts surrounding context words from a given word.  
   - ✅ Example: `"king - man + woman ≈ queen"`  
   - Usage: Semantic similarity, text clustering.  

2. **🔹 GloVe (Global Vectors for Word Representation)**  
   - Developed by **Stanford**.  
   - Uses **word co-occurrence matrix** + statistical information.  
   - Captures both **global statistics** (entire corpus) and **local context**.  
   - ✅ Example: Better at analogies and global word relationships.  
   - Usage: NLP tasks where global corpus meaning is important.  

3. **🔹 FastText (by Facebook/Meta)**  
   - Extension of Word2Vec.  
   - Breaks words into **character n-grams** → better for **rare & OOV words**.  
   - ✅ Example: `"playing"` → subwords: `"play"`, `"lay"`, `"ing"`.  
   - Usage: Morphologically rich languages (e.g., German, Finnish, Tamil).  

4. **🔹 Transformer-based Embeddings**
   - Transformers are a **deep learning architecture** based on the **Attention mechanism** ⚡.  
   - Unlike RNNs/LSTMs (which process sequentially), Transformers process input **in parallel** → much faster and more accurate for long text.  
   - Derived from **Transformer models** (BERT, RoBERTa, OpenAI, etc.).  
   - Contextual → meaning depends on **surrounding words**.  
   - ✅ Example:  
     - "bank" (river bank 🌊) vs "bank" (financial 🏦) → different embeddings.  
   - Usage: State-of-the-art NLP tasks (QA, summarization, semantic search).  

---
## 🐝 BERT  
- **Bidirectional Encoder Representations from Transformers**.  
- Reads text **both left-to-right & right-to-left**.  
- Used for **text classification, sentiment analysis, QA systems**.  

---

## 📝 Prompt Engineering  
- Crafting **effective prompts** for LLMs.  
- **Types:**  
  - Zero-shot → No example  
  - Few-shot → With examples  
  - Chain-of-thought → Step-by-step reasoning  
  - Instruction-tuning → Direct command style  

---

## 📡 RAG (Retrieval-Augmented Generation)  
- Combines **LLMs + external knowledge base**.  
- **Techniques:**  
  - Vector search (using FAISS)  
  - Retrieval pipelines  
  - Hybrid RAG (structured + unstructured data)  

---

## 🤗 Hugging Face  
- Open-source hub for **NLP models, datasets, transformers**.  
- Provides `transformers` library for easy model use.  

---

## 🔗 MCP (Model Context Protocol)  

**MCP (Model Context Protocol)** is a **standardized framework** designed to connect **AI models (LLMs) with external tools, APIs, and data sources**.  
It ensures that models can **understand, retrieve, and interact with external information** in a consistent way.  

---

### 📑 Key Features of MCP  

1. **Interoperability**  
   - Allows different AI models and systems to **communicate seamlessly**.  
   - Example: Connecting a GPT model with a database or a search engine.  

2. **Context Management**  
   - Provides a **structured context** for models to make decisions.  
   - Example: Passing user data, previous conversations, or external API responses to the model.  

3. **Tool & API Integration**  
   - Enables models to **invoke tools** or external services reliably.  
   - Example: Calling a weather API or retrieving a document from a knowledge base.  

4. **Standardized Communication**  
   - Defines **protocols** for how models exchange messages with external systems.  
   - Reduces **errors** and improves **scalability**.  

---

### 🔹 Example Use Case  

- **Scenario:** A chatbot needs to answer user queries about stock prices.  
  1. User asks: *"What is Tesla's stock price today?"*  
  2. GPT model receives the prompt.  
  3. MCP allows GPT to **query a stock API** and retrieve real-time data.  
  4. Model responds with **accurate, up-to-date information**.  

---

## 🤖 AI Agents  
- Autonomous systems powered by LLMs.  
- Can **plan, reason, and act** using tools (APIs, databases).  
✅ Example: LangChain, AutoGPT, BabyAGI.  

---

## 🗂️ FAISS (Facebook AI Similarity Search)  
- Library for **fast similarity search** in large vector datasets.  
- Used in **semantic search, RAG, recommendation systems**.  

---

## 🚀 Interactive Demo Links  
- [🔗 Hugging Face Models](https://huggingface.co/models)  
- [🔗 TensorFlow Playground](https://playground.tensorflow.org/)  
- [🔗 NLP Visualization Tools](https://explosion.ai/demos)  

---
