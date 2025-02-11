# AI-Powered Web Search and Reasoning

## Overview
This project is an AI-powered web search and reasoning system that utilizes large language models (LLMs) for answering queries based on web content. The application integrates real-time web crawling, vector database storage, and generative AI to provide well-structured, context-aware responses.

## Features
- **AI-Powered Query Processing:** Uses LLaMA3 and DeepSeek-R1 models for reasoning.
- **Web Search Integration:** Fetches real-time web content using DuckDuckGo search.
- **Web Crawling:** Extracts relevant content using `crawl4ai`.
- **Vector Database Storage:** Stores and retrieves processed content with ChromaDB.
- **Context-Aware Responses:** Generates responses based solely on retrieved web data.
- **Streamlit UI:** User-friendly web interface for interactive search and responses.

## Technologies Used
- **Python**: Backend scripting
- **Streamlit**: UI for search interface
- **ChromaDB**: Vector storage for retrieved web content
- **LangChain**: Text processing and document handling
- **Ollama**: Local LLM serving
- **DeepSeek-R1 & LLaMA3**: AI models for summarization and reasoning
- **DuckDuckGo Search API**: Web search engine
- **crawl4ai**: Intelligent web crawling

## Installation
### Prerequisites
- Python 3.9+
- `ollama` installed and running (`ollama serve`)
- CUDA (if using GPU acceleration)

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-web-search.git
   cd ai-web-search
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add the following:
     ```
     GROQ_API_KEY=your_groq_api_key
     ```
5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage
- Enter a query in the text box.
- Toggle **Enable Web Search** to fetch real-time data.
- Toggle **Enable DeepSeekR1 Reasoning** for advanced AI responses.
- Click "Go" to get AI-generated responses.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

 

## Acknowledgments
- [Meta AI](https://ai.facebook.com/research/) for LLaMA3
- [DeepSeek](https://deepseek.com/) for DeepSeek-R1
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for UI
- [DuckDuckGo Search API](https://duckduckgo.com/) for web search

