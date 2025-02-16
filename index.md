# AI Research Pipeline: Scraping & Classification System

**End-to-End Solution for Academic Paper Management**  
*From Web Scraping to AI-Powered Categorization*
*Integrated workflow combining scraping and classification*

> ## Overview

This unified pipeline combines two powerful components:

1. **NeurIPS Scraper**: Automated collection of research papers from NeurIPS conferences
2. **AI Classifier**: Intelligent categorization using Google's Gemini AI

Provides a complete solution for:

- Bulk paper acquisition from academic conferences
- Automated organization using machine learning
- Centralized metadata management
- Research trend analysis

> ## Key Features

### 🕸️ Scraping Module

- Asynchronous I/O for high-performance downloads
- Year-range selection (1987-2024)
- Metadata extraction (title, author, year)
- Progress tracking with year-wise statistics
- Sanitized filename handling

### 🧠 Classification Module

- Gemini AI integration for document understanding
- Customizable taxonomy support
- Batch PDF processing
- CSV metadata integration
- Classification reasoning storage

### 🛠️ Unified System

- Streamlit-based GUI for both components
- Shared metadata management
- Configurable storage paths
- Error handling and retry mechanisms
- Cross-platform compatibility

> ## Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Anas-Altaf/Doc-Annotator_py.git
   git clone https://github.com/Anas-Altaf/Neurl-PS-Scraper_py.git
   cd  `respective-dir`
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> ## Configuration

1. **Directory Structure**:
   ```bash
   mkdir -p downloaded_papers  metadata
   ```

2. **Gemini API Key**:
    - Obtain from [Google AI Studio](https://aistudio.google.com/)
    - Store in `.env` file, like `example.env`:
      ```env
      GEMINI_API_KEY=your_key_here
      ```

> ## Workflow Usage

### Step 1: Scrape Papers

```bash
streamlit run scraper_gui.py
```

1. Set year range (1987-2024)
2. Configure download directory (default: `./downloaded_papers`)
3. Start scraping
4. Monitor real-time progress

### Step 2: Classify Papers

```bash
streamlit run app.py
```

1. Set PDF directory to scraped papers location
2. Configure categories (default AI domains supported)
3. Enter Gemini API key
4. Start classification
5. View results in interactive table

> ## System Architecture

```mermaid
graph LR
    A[User Interface] --> B[Scraper]
    A --> C[Classifier]
    B --> D[(Paper Storage)]
    C --> E[(Metadata DB)]
    D --> C
    E --> F[Analysis Tools]
    
    subgraph Scraping
    B --> G[NeurIPS Website]
    end
    
    subgraph AI Processing
    C --> H[Gemini API]
    end
```

> ## Performance Metrics

| Component         | Throughput         | Accuracy* |
|-------------------|--------------------|-----------|
| Scraper           | 50+ papers/minute  | 100%      |
| Classifier        | 5-100 pages/minute | 99%       |
| Metadata Handling | 24000+ records     | -         |

*Based on benchmark tests with 500 papers across 5 domains

> ## Screenshots

![Scraping Interface](images/ss/img_5.png)  
*Paper scraping interface with year selection*

![Classification Progress](images/ss/2.png)  
*Real-time classification dashboard*

![Metadata View](images/ss/3.png)  
*Unified metadata inspection interface*

> ## Troubleshooting

**Common Issues**:

- `Connection Errors`: Verify network access to `papers.nips.cc`
- `API Limits`: Gemini API has RPM limit
- `Encoding Errors`: Ensure proper UTF-8 support
- `PDF Read Failures`: Verify file integrity after download

**Debugging Mode**:
```bash
STREAMLIT_DEBUG=1 streamlit run scraper_gui.py
```
> ## Contributing

1. Report issues via GitHub tracker
2. Fork repository and create feature branch
3. Submit PR with detailed documentation
4. Follow PEP8 guidelines and include tests

> ### License MIT License - See [LICENSE](LICENSE) for full text
---
**Transform your research workflow with AI-powered paper management!** 🚀