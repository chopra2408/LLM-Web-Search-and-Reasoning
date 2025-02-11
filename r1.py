import nltk
import asyncio
import streamlit as st
import chromadb
import sys
import tempfile
import os
from dotenv import load_dotenv
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
import ollama  # Make sure Ollama is installed and configured

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

def call_llm(input: str, with_context: bool = True, context: str | None = None, use_deepseek: bool = False):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {input}"
        }
    ]
    
    if not with_context:
        messages.pop(0)
        messages[0]["content"] = input
 
    if use_deepseek:
        response = ollama.chat(model="deepseek-r1:1.5b", stream=True, messages=messages)
        for chunk in response:
            if not chunk.get("done", True):
                yield chunk["message"]["content"]
            else:
                break
    else:
        response = ollama.chat(model="llama3.2:3b", stream=True, messages=messages)
        for chunk in response:
            if not chunk.get("done", True):
                yield chunk["message"]["content"]
            else:
                break

def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    
    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings()
    )
    
    return (
        chroma_client.get_or_create_collection(
            name="web-llm",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"}
        ),
        chroma_client,
    )
    
def normalize_url(url):
    normalized_url = (
        url.replace("https://", "")
           .replace("www.", "")
           .replace("/", "_")
           .replace("-", "_")
           .replace(".", "_")
    )
    print("Normalized URL", normalized_url)
    return normalized_url    
    
def add_to_vector_database(results: list[CrawlResult]):
    collection, _ = get_vector_collection()

    for result in results:
        documents, metadatas, ids = [], [], []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        if result.markdown_v2:
            markdown_result = result.markdown_v2.fit_markdown
        else:
            continue
        
        temp_file = tempfile.NamedTemporaryFile("w", delete=False, suffix=".md", encoding="utf-8")
        temp_file.write(markdown_result)
        temp_file.flush()
        
        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        normalized_url = normalize_url(result.url)
        
        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")
                
            print("Upsert collection: ", id(collection))
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )        
                
async def crawl_webpages(urls: list[str], query: str) -> CrawlResult:
    bm25_filter = BM25ContentFilter(user_query=query, bm25_threshold=1.2)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)
    
    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=['nav', 'footer', 'header', 'form', 'img', 'a'],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        page_timeout=20000,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, crawler_config)
        return results
    
def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []
    
    for url in urls:
        try:
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()
            
            if rp.can_fetch("*", url):
                allowed_urls.append(url)
                
        except Exception as e:
            allowed_urls.append(url)
            
    return allowed_urls

def get_web_urls(search_terms: str, num_results: int = 10) -> list[str]:
    try:
        query = search_terms
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
        for url in discard_urls:
            query += f" -site:{url}"
        
        results = DDGS().text(query, max_results=num_results)
        results = [result["href"] for result in results]
        
        st.write(results)
        return check_robots_txt(results)
    
    except Exception as e:
        st.error(f"❌ Error fetching web results: {e}")
        st.stop()
                
async def run():
    st.set_page_config(page_title="AI with Web", page_icon="🤖")
    
    st.header("🔍 LLM Web Search and Reasoning")
    user_input = st.text_area(
        label="Put your query here...",
        placeholder="Add your query here...",
        label_visibility="hidden"
    )
    is_web_search = st.toggle("Enable web search", value=False, key="enable_web_search")
    use_deepseek = st.toggle("Enable DeepSeekR1 Reasoning", value=False, key="deepseek_reasoning")
    go = st.button("Go")
    
    collection, chroma_client = get_vector_collection()
    
    if user_input and go:
        if is_web_search:
            web_urls = get_web_urls(search_terms=user_input)
            if not web_urls:
                st.write("No results found 😔")
                st.stop()
            
            results = await crawl_webpages(urls=web_urls, query=user_input)
            add_to_vector_database(results)
            
            query_results = collection.query(query_texts=[user_input], n_results=10)
            context = query_results.get("documents")[0]
            
            chroma_client.delete_collection(name="web-llm")
            
            llm_response = call_llm(
                input=user_input, 
                with_context=is_web_search, 
                context=context, 
                use_deepseek=use_deepseek
            )
            
            st.write_stream(llm_response)
        else:
            llm_response = call_llm(
                input=user_input, 
                with_context=is_web_search,
                use_deepseek=use_deepseek
            )
            st.write_stream(llm_response)
            
if __name__ == "__main__":
    asyncio.run(run())
