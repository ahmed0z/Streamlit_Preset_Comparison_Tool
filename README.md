# 🔍 Streamlit Preset Comparison Tool

A production-grade Streamlit application that intelligently compares user-provided input values against a reference database using advanced NLP and fuzzy matching techniques.

## 🌟 Features

### Core Functionality
- **🎯 Intelligent Matching**: Advanced fuzzy string matching with multiple algorithms
- **📊 Complex Data Support**: Handles units, conditions, multiple values, and complex formats
- **🔍 Multiple Match Types**: Exact matches, similar matches, and semantic similarity
- **📈 Interactive Results**: Real-time filtering, sorting, and analysis
- **📤 Export Options**: Excel and CSV export with detailed formatting

### Advanced Capabilities
- **🧠 Smart Value Processing**: Extracts numbers, units, conditions from complex strings
- **⚡ Performance Optimized**: Efficient processing for Streamlit Cloud
- **🎨 Modern UI**: Clean, intuitive interface with responsive design
- **📊 Analytics Dashboard**: Comprehensive insights and recommendations
- **🗄️ Database Management**: Easy database updates and backup functionality

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Streamlit_Preset_Comparison_Tool.git
cd Streamlit_Preset_Comparison_Tool

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Usage Workflow

1. **📥 Download Template**: Get the Excel template from the app
2. **✏️ Fill Your Data**: Add your input values using the template format
3. **📤 Upload & Compare**: Upload your file and start the comparison
4. **📋 Review Results**: Examine matches, filter results, and analyze insights
5. **📊 Export**: Download results in Excel or CSV format

## 📁 Project Structure

```
Streamlit_Preset_Comparison_Tool/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── utils/                          # Core business logic
│   ├── file_handler.py            # Excel file operations
│   ├── data_processor.py          # Data cleaning and normalization
│   ├── matching_engine.py         # Comparison algorithms
│   └── export_manager.py          # Results export functionality
├── components/                     # UI components
│   ├── ui_components.py           # Reusable UI elements
│   ├── sidebar.py                 # Sidebar navigation
│   └── results_display.py         # Results visualization
├── data/                          # Data files
│   ├── preset_database.xlsx       # Reference database
│   └── templates/                 # Input templates
└── tests/                         # Test suite
```

## 📊 Supported Data Formats

The tool intelligently handles various input formats:

### Simple Values
- **Plain text**: `"Protective Cap"`
- **Numbers**: `"25"`, `"3.14"`

### Values with Units
- **Single unit**: `"20 kg"`, `"0.181 inches"`
- **Multiple units**: `"20 kg, 15 cm"`

### Complex Formats
- **Conditions**: `"20 kg @ 30°C"`
- **Dimensions**: `"0.618" L x 78.740" W x 3.307" H"`
- **Technical specs**: `"(15.70mm x 200.00mm x 84.00mm)"`

### Smart Processing
- **Unit normalization**: Converts between unit variations
- **Condition extraction**: Identifies @ symbols and contextual conditions
- **Multi-value handling**: Processes comma-separated values
- **Synonym matching**: Recognizes alternative terminology

## 🔧 Configuration

### Matching Settings

```python
# config.py
MATCHING_THRESHOLD = 0.75      # Minimum similarity (75%)
EXACT_MATCH_THRESHOLD = 0.99   # Exact match threshold (99%)
MAX_RESULTS_PER_INPUT = 5      # Max matches per input
```

### Performance Settings

```python
MAX_FILE_SIZE_MB = 50         # File upload limit
CACHE_TTL = 3600             # Cache duration (1 hour)
CHUNK_SIZE = 1000            # Processing chunk size
```

## 🎯 Matching Algorithms

### 1. Exact Matching
- Direct string comparison with normalization
- Handles case, spacing, and punctuation differences

### 2. Fuzzy String Matching
- **Levenshtein Distance**: Character-level edits
- **Jaro-Winkler**: Prefix-focused similarity
- **Token Sort Ratio**: Word order independence
- **Token Set Ratio**: Partial match detection

### 3. Component-Based Matching
- **Numerical Similarity**: Value-aware comparison
- **Unit Compatibility**: Same measurement type detection
- **Condition Matching**: Context-aware evaluation

### 4. Semantic Matching (Optional)
- Uses sentence embeddings for meaning-based similarity
- Handles synonyms and paraphrasing

## 📈 Results & Analysis

### Match Types
- **Exact Match**: 100% similarity, perfect match
- **Similar Match**: ≥75% similarity, high confidence
- **Not Found**: <75% similarity, no suitable match

### Analytics Features
- **Status Distribution**: Visual breakdown of match types
- **Similarity Scores**: Histogram of matching confidence
- **Quality Metrics**: Data consistency indicators
- **Performance Stats**: Processing time and throughput

### Export Options
- **Excel**: Formatted workbook with multiple sheets
- **CSV**: Simple comma-separated format
- **Filtered Export**: Custom criteria-based export

## 🗄️ Database Management

### Updating the Database
1. Navigate to Settings page
2. Upload new Excel file with same structure
3. System automatically creates backup before update

### Required Database Columns
- **Category**: Main classification
- **Sub-Category**: Secondary classification (optional)
- **Attribute Name**: Parameter identifier
- **Preset Values**: Reference values for comparison

## 🔍 Advanced Features

### Filtering & Search
- **Status Filtering**: Filter by match type
- **Similarity Range**: Numeric threshold filtering
- **Text Search**: Search across input and matched values
- **Category/Attribute**: Hierarchical filtering

### Performance Optimization
- **Caching**: Intelligent data caching for speed
- **Chunked Processing**: Memory-efficient large file handling
- **Progressive Loading**: Real-time progress indicators

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/

# With coverage
pytest --cov=. tests/
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 📋 Requirements

### Python Dependencies
- **streamlit**: Web framework
- **pandas**: Data manipulation
- **openpyxl**: Excel file handling
- **rapidfuzz**: High-performance fuzzy matching
- **plotly**: Interactive visualizations

### System Requirements
- **Python 3.9+**
- **Memory**: 1GB+ for large datasets
- **Storage**: 500MB+ for data and cache

## 🌐 Deployment

### Streamlit Cloud
1. Push to GitHub repository
2. Connect Streamlit Cloud to repository
3. Deploy with automatic updates

### Local Deployment
```bash
# Production mode
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing web framework
- **RapidFuzz**: For high-performance string matching
- **Plotly**: For interactive visualizations
- **Pandas**: For powerful data manipulation

---

**Built with ❤️ for intelligent data comparison**