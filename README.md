# AI Resume Evaluator

A backend service that evaluates CVs and project reports against job descriptions using AI-powered analysis with RAG (Retrieval-Augmented Generation) capabilities.

## 🚀 Features

- **CV Analysis**: Extract and evaluate candidate information against job requirements
- **Project Evaluation**: Score project deliverables based on predefined rubrics
- **RAG Integration**: Vector database for contextual job descriptions and scoring rubrics
- **Advanced Error Handling**: Retry mechanisms with exponential backoff
- **Rate Limiting**: Built-in API rate limiting simulation
- **Dynamic Temperature Control**: Task-specific AI model temperature settings
- **Async Processing**: Redis-based queue system for background evaluation
- **Multi-format Support**: PDF and DOCX file processing

## 🛠️ Tech Stack

- **Backend**: Node.js with Express.js
- **AI/LLM**: Google Gemini API
- **Vector Database**: ChromaDB for RAG implementation
- **Queue System**: Bull Queue with Redis
- **File Processing**: PDF-parse, Mammoth (DOCX)
- **Error Handling**: Exponential-backoff library

## 📋 Prerequisites

- Node.js (v18 or higher)
- Redis server
- Google Gemini API key
- ChromaDB (automatically handled)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-resume
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   PORT=3000
   ```

4. **Start Redis server**:
   ```bash
   # macOS with Homebrew
   brew services start redis
   
   # Or manually
   redis-server
   ```

5. **Run the application**:
   ```bash
   node index.js
   ```

## 🔧 API Endpoints

### 1. Upload Files
**POST** `/upload`

Upload CV and project report files.

**Form Data**:
- `cv`: PDF or DOCX file
- `report`: PDF or DOCX file

**Response**:
```json
{
  "uploadId": "uuid",
  "message": "Files uploaded successfully"
}
```

### 2. Start Evaluation
**POST** `/evaluate`

Start the evaluation process for uploaded files.

**Body**:
```json
{
  "uploadId": "uuid-from-upload"
}
```

**Response**:
```json
{
  "evalId": "uuid",
  "message": "Evaluation started"
}
```

### 3. Get Results
**GET** `/result/:id`

Retrieve evaluation results.

**Response**:
```json
{
  "status": "completed",
  "evaluation": {
    "cv_match_rate": "85%",
    "cv_feedback": "Strong technical background...",
    "project_score": 8.2,
    "project_feedback": "Well-structured implementation...",
    "overall_summary": "The candidate demonstrates..."
  }
}
```

## 🏗️ Architecture & Design Choices

### 1. **RAG Implementation**
- **ChromaDB**: Used for storing and retrieving job descriptions and scoring rubrics
- **Semantic Search**: Contextual retrieval based on CV content and project analysis
- **Dynamic Context**: Relevant context is retrieved for each evaluation step

### 2. **Error Handling Strategy**
- **Exponential Backoff**: Automatic retry with increasing delays
- **Rate Limiting**: Simulated API rate limits for testing robustness
- **Graceful Degradation**: Fallback values when API calls fail
- **Comprehensive Logging**: Detailed error tracking and debugging

### 3. **Temperature Control**
- **Task-Specific Settings**:
  - Extraction: 0.1 (low creativity)
  - Comparison: 0.3 (moderate)
  - Scoring: 0.2 (objective)
  - Feedback: 0.7 (creative)
  - Summary: 0.5 (balanced)

### 4. **Async Processing**
- **Bull Queue**: Redis-based job queue for background processing
- **Non-blocking**: API responds immediately with evaluation ID
- **Scalable**: Can handle multiple concurrent evaluations

### 5. **Evaluation Pipeline**
1. **Text Extraction**: PDF/DOCX to plain text
2. **CV Analysis**: Structured data extraction with RAG context
3. **Job Matching**: Compare CV against job requirements
4. **Project Scoring**: Evaluate deliverables against rubrics
5. **Feedback Generation**: Detailed analysis and recommendations
6. **Summary Creation**: Comprehensive 3-5 sentence overview

## 🧪 Testing the System

### Basic Test Flow:

1. **Upload files**:
   ```bash
   curl -X POST http://localhost:3000/upload \
     -F "cv=@path/to/cv.pdf" \
     -F "report=@path/to/report.pdf"
   ```

2. **Start evaluation**:
   ```bash
   curl -X POST http://localhost:3000/evaluate \
     -H "Content-Type: application/json" \
     -d '{"uploadId":"your-upload-id"}'
   ```

3. **Check results**:
   ```bash
   curl http://localhost:3000/result/your-eval-id
   ```

### Error Simulation:
The system includes built-in error simulation (10% random failure rate) to test robustness.

## 📊 Evaluation Criteria

### CV Evaluation (1-5 scale):
- **Technical Skills Match** (40%)
- **Experience Level** (25%)
- **Relevant Achievements** (20%)
- **Cultural/Collaboration Fit** (15%)

### Project Evaluation (1-5 scale):
- **Correctness** (30%)
- **Code Quality & Structure** (25%)
- **Resilience & Error Handling** (20%)
- **Documentation & Explanation** (15%)
- **Creativity/Bonus** (10%)

## 🔍 Monitoring & Debugging

- **Console Logging**: Detailed step-by-step evaluation logs
- **Error Tracking**: Comprehensive error messages and stack traces
- **Performance Metrics**: API response times and success rates
- **Queue Monitoring**: Redis queue status and job processing

## 🚨 Common Issues & Solutions

### 1. **Redis Connection Error**
```bash
# Start Redis server
brew services start redis
# Or check if Redis is running
redis-cli ping
```

### 2. **Gemini API Rate Limits**
- The system includes automatic retry with exponential backoff
- Rate limiting simulation helps test robustness

### 3. **File Upload Issues**
- Ensure files are PDF or DOCX format
- Check file size limits (default: no explicit limit set)

### 4. **Vector Database Initialization**
- ChromaDB initializes automatically on startup
- Check console logs for initialization status

## 📝 License

MIT License - see LICENSE file for details.