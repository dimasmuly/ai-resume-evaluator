import express from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import pdf from 'pdf-parse';
import mammoth from 'mammoth';
import OpenAI from 'openai';
import fetch from 'node-fetch';
import Queue from 'bull';
import { backOff } from 'exponential-backoff';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

// Custom Simple Embedding Function
class SimpleEmbeddingFunction {
  async generate(texts) {
    return texts.map(text => {
      // Simple hash-based embedding
      const words = text.toLowerCase().split(/\s+/);
      const embedding = new Array(384).fill(0); // 384-dimensional vector
      
      words.forEach((word, index) => {
        for (let i = 0; i < word.length; i++) {
          const charCode = word.charCodeAt(i);
          const pos = (charCode + index) % 384;
          embedding[pos] += Math.sin(charCode * 0.1) * Math.cos(index * 0.1);
        }
      });
      
      // Normalize vector
      const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
      return magnitude > 0 ? embedding.map(val => val / magnitude) : embedding;
    });
  }
}

// Simple In-Memory Vector Store
class SimpleVectorStore {
  constructor() {
    this.collections = new Map();
  }

  async getOrCreateCollection(config) {
    const { name, embeddingFunction } = config;
    if (!this.collections.has(name)) {
      this.collections.set(name, {
        name,
        embeddingFunction,
        documents: [],
        embeddings: [],
        metadatas: [],
        ids: []
      });
    }
    return new SimpleCollection(this.collections.get(name));
  }
}

class SimpleCollection {
  constructor(collection) {
    this.collection = collection;
  }

  async count() {
    return this.collection.documents.length;
  }

  async add({ documents, metadatas, ids }) {
    const embeddings = await this.collection.embeddingFunction.generate(documents);
    
    documents.forEach((doc, i) => {
      this.collection.documents.push(doc);
      this.collection.embeddings.push(embeddings[i]);
      this.collection.metadatas.push(metadatas[i]);
      this.collection.ids.push(ids[i]);
    });
  }

  async query({ queryTexts, nResults = 3 }) {
    if (this.collection.documents.length === 0) {
      return { documents: [[]], metadatas: [[]], distances: [[]] };
    }

    const queryEmbeddings = await this.collection.embeddingFunction.generate(queryTexts);
    const results = [];

    queryEmbeddings.forEach(queryEmb => {
      const similarities = this.collection.embeddings.map((docEmb, idx) => {
        const similarity = this.cosineSimilarity(queryEmb, docEmb);
        return { similarity, idx };
      });

      similarities.sort((a, b) => b.similarity - a.similarity);
      const topResults = similarities.slice(0, nResults);

      results.push({
        documents: topResults.map(r => this.collection.documents[r.idx]),
        metadatas: topResults.map(r => this.collection.metadatas[r.idx]),
        distances: topResults.map(r => 1 - r.similarity)
      });
    });

    return {
      documents: results.map(r => r.documents),
      metadatas: results.map(r => r.metadatas),
      distances: results.map(r => r.distances)
    };
  }

  cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }
}

// Initialize In-Memory Vector Store
const client = new SimpleVectorStore();
const embedder = new SimpleEmbeddingFunction();

// Global variables for collections
let jobDescriptionsCollection;
let scoringRubricsCollection;

async function initializeVectorDB() {
  try {
    console.log('🔄 Initializing In-Memory Vector Database...');
    
    // Create or get collections
    jobDescriptionsCollection = await client.getOrCreateCollection({
      name: "job_descriptions",
      embeddingFunction: embedder
    });
    
    scoringRubricsCollection = await client.getOrCreateCollection({
      name: "scoring_rubrics", 
      embeddingFunction: embedder
    });
    
    // Check if collections are empty and populate them
    const jobCount = await jobDescriptionsCollection.count();
    const rubricCount = await scoringRubricsCollection.count();
    
    if (jobCount === 0) {
      await populateJobDescriptions();
    }
    
    if (rubricCount === 0) {
      await populateScoringRubrics();
    }
    
    console.log('✅ In-Memory Vector Database initialized successfully');
  } catch (error) {
    console.error('❌ Error initializing vector database:', error);
    console.log('⚠️  Continuing without vector database...');
  }
}

// Populate job descriptions in vector DB
async function populateJobDescriptions() {
  const jobDescriptions = [
    {
      id: "backend-engineer",
      content: "Strong backend skills (Node.js, Django, Rails). Database management (MySQL, PostgreSQL, MongoDB). REST APIs, Security, Cloud (AWS, GCP, Azure). Familiarity with LLM APIs, embeddings, vector DBs, prompt design. No strict degree requirement — focus on skills.",
      metadata: { type: "job_description", role: "backend_engineer" }
    },
    {
      id: "technical-skills",
      content: "Backend development expertise in Node.js, Python, or similar. Experience with databases, API design, cloud platforms, and modern development practices.",
      metadata: { type: "job_requirement", category: "technical" }
    },
    {
      id: "ai-ml-skills",
      content: "Experience with LLM APIs, embeddings, vector databases, prompt engineering, and AI/ML integration in backend systems.",
      metadata: { type: "job_requirement", category: "ai_ml" }
    }
  ];
  
  await jobDescriptionsCollection.add({
    ids: jobDescriptions.map(job => job.id),
    documents: jobDescriptions.map(job => job.content),
    metadatas: jobDescriptions.map(job => job.metadata)
  });
  
  console.log('Job descriptions populated in vector DB');
}

// Populate scoring rubrics in vector DB
async function populateScoringRubrics() {
  const scoringRubrics = [
    {
      id: "cv-evaluation",
      content: "CV Match Evaluation (1–5 scale per parameter): Technical Skills Match (40%), Experience Level (25%), Relevant Achievements (20%), Cultural / Collaboration Fit (15%)",
      metadata: { type: "scoring_rubric", category: "cv_evaluation" }
    },
    {
      id: "project-evaluation",
      content: "Project Deliverable Evaluation (1–5 scale per parameter): Correctness (30%), Code Quality & Structure (25%), Resilience & Error Handling (20%), Documentation & Explanation (15%), Creativity / Bonus (10%)",
      metadata: { type: "scoring_rubric", category: "project_evaluation" }
    },
    {
      id: "technical-assessment",
      content: "Technical assessment criteria focusing on backend development skills, database knowledge, API design, security practices, and cloud platform experience.",
      metadata: { type: "assessment_criteria", category: "technical" }
    },
    {
      id: "ai-assessment",
      content: "AI/ML assessment criteria for LLM API usage, embedding implementation, vector database integration, and prompt engineering capabilities.",
      metadata: { type: "assessment_criteria", category: "ai_ml" }
    }
  ];
  
  await scoringRubricsCollection.add({
    ids: scoringRubrics.map(rubric => rubric.id),
    documents: scoringRubrics.map(rubric => rubric.content),
    metadatas: scoringRubrics.map(rubric => rubric.metadata)
  });
  
  console.log('Scoring rubrics populated in vector DB');
}

// Retrieve relevant context from vector DB
async function retrieveRelevantContext(query, collectionType = 'both', limit = 3) {
  try {
    const results = [];
    
    if (collectionType === 'job' || collectionType === 'both') {
      const jobResults = await jobDescriptionsCollection.query({
        queryTexts: [query],
        nResults: limit
      });
      
      if (jobResults.documents && jobResults.documents[0]) {
        results.push(...jobResults.documents[0].map((doc, idx) => ({
          content: doc,
          metadata: jobResults.metadatas[0][idx],
          score: jobResults.distances[0][idx],
          type: 'job_description'
        })));
      }
    }
    
    if (collectionType === 'rubric' || collectionType === 'both') {
      const rubricResults = await scoringRubricsCollection.query({
        queryTexts: [query],
        nResults: limit
      });
      
      if (rubricResults.documents && rubricResults.documents[0]) {
        results.push(...rubricResults.documents[0].map((doc, idx) => ({
          content: doc,
          metadata: rubricResults.metadatas[0][idx],
          score: rubricResults.distances[0][idx],
          type: 'scoring_rubric'
        })));
      }
    }
    
    // Sort by relevance score (lower distance = higher relevance)
    return results.sort((a, b) => a.score - b.score);
    
  } catch (error) {
    console.error('Error retrieving context from vector DB:', error);
    return [];
  }
}

// Enhanced retry mechanism with exponential backoff
async function callGeminiWithRetry(prompt, maxRetries = 3) {
  const options = {
    numOfAttempts: maxRetries,
    startingDelay: 1000,
    timeMultiple: 2,
    maxDelay: 10000,
    retry: (error) => {
      console.log(`Retrying Gemini API call due to error: ${error.message}`);
      return true;
    }
  };
  
  return await backOff(async () => {
    return await callGemini(prompt);
  }, options);
}

// Dynamic temperature control based on task type
function getTemperatureForTask(taskType) {
  const temperatureMap = {
    'extraction': 0.1,      // Low creativity for data extraction
    'comparison': 0.3,      // Moderate for comparison tasks
    'scoring': 0.2,         // Low for objective scoring
    'feedback': 0.7,        // Higher for creative feedback
    'summary': 0.5          // Balanced for summaries
  };
  
  return temperatureMap[taskType] || 0.7;
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, uuidv4() + '-' + file.originalname);
  }
});
const upload = multer({ storage });

const results = {};

async function extractText(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  
  if (ext === '.pdf') {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdf(dataBuffer);
    return data.text;
  } else if (ext === '.docx') {
    const result = await mammoth.extractRawText({ path: filePath });
    return result.value;
  } else {
    throw new Error('Unsupported file format');
  }
}

// Utility function to safely parse JSON with fallback
function safeJsonParse(jsonString, fallbackValue = null) {
  try {
    // Clean the string from markdown and control characters
    let cleaned = jsonString
      .replace(/```json\s*/g, '')  // Remove ```json
      .replace(/```\s*/g, '')      // Remove ```
      .replace(/[\x00-\x1F\x7F]/g, '')  // Remove control characters
      .trim();
    
    // Find JSON object boundaries
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    
    if (start !== -1 && end !== -1 && end > start) {
      cleaned = cleaned.substring(start, end + 1);
    }
    
    return JSON.parse(cleaned);
  } catch (error) {
    console.error('JSON parsing failed:', error.message);
    console.error('Raw string:', jsonString.substring(0, 200) + '...');
    return fallbackValue;
  }
}

const jobDescription = `
Strong backend skills (Node.js, Django, Rails).
Database management (MySQL, PostgreSQL, MongoDB).
REST APIs, Security, Cloud (AWS, GCP, Azure).
Familiarity with LLM APIs, embeddings, vector DBs, prompt design.
No strict degree requirement — focus on skills.
`;

const scoringRubric = `
CV Match Evaluation (1–5 scale per parameter):
- Technical Skills Match (40%)
- Experience Level (25%)
- Relevant Achievements (20%)
- Cultural / Collaboration Fit (15%)

Project Deliverable Evaluation (1–5 scale per parameter):
- Correctness (30%)
- Code Quality & Structure (25%)
- Resilience & Error Handling (20%)
- Documentation & Explanation (15%)
- Creativity / Bonus (10%)
`;

// Gemini API Configuration
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = 'gemini-2.0-flash-001';  

if (!GEMINI_API_KEY) {
  console.error('Error: GEMINI_API_KEY is not set in .env file.');
  process.exit(1);
}

// Rate limiting simulation
class RateLimiter {
  constructor(maxRequests = 10, windowMs = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = new Map();
  }
  
  async checkLimit(identifier = 'default') {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    
    if (!this.requests.has(identifier)) {
      this.requests.set(identifier, []);
    }
    
    const userRequests = this.requests.get(identifier);
    
    // Remove old requests outside the window
    const validRequests = userRequests.filter(time => time > windowStart);
    this.requests.set(identifier, validRequests);
    
    if (validRequests.length >= this.maxRequests) {
      const oldestRequest = Math.min(...validRequests);
      const waitTime = oldestRequest + this.windowMs - now;
      throw new Error(`Rate limit exceeded. Try again in ${Math.ceil(waitTime / 1000)} seconds`);
    }
    
    validRequests.push(now);
    this.requests.set(identifier, validRequests);
    
    // Simulate random failures for testing
    if (Math.random() < 0.1) { // 10% chance of simulated failure
      throw new Error('Simulated API timeout');
    }
    
    return true;
  }
}

const rateLimiter = new RateLimiter(15, 60000); // 15 requests per minute

// Enhanced Gemini API function with advanced error handling
async function callGemini(prompt, taskType = 'general', retryCount = 0, extraConfig = {}) {
  const maxRetries = 5;  // Naikkan ke 5 untuk handle overload (503) lebih banyak
  const temperature = getTemperatureForTask(taskType);
  
  try {
    // Check rate limiting
    await rateLimiter.checkLimit('gemini-api');
    
    console.log(`Calling Gemini API (attempt ${retryCount + 1}/${maxRetries + 1}) for task: ${taskType}`);
    console.log(`Using temperature: ${temperature}`);
    
    const generationConfig = {
      temperature: temperature,
      topK: 40,
      topP: 0.95,
      maxOutputTokens: 2048,
      ...extraConfig  
    };
    
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: generationConfig,  // Pakai config yang digabung
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          }
        ]
      }),
      timeout: 30000
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    console.log('Gemini API response received successfully');
    console.log('Response structure:', JSON.stringify(data, null, 2));

    if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
      throw new Error('Invalid response structure from Gemini API');
    }

    const content = data.candidates[0].content.parts[0].text;
    console.log('Extracted content length:', content.length);
    
    return content;
    
  } catch (error) {
    console.error(`Gemini API call failed (attempt ${retryCount + 1}):`, error.message);
    
    // Determine if error is retryable
    const isRetryable = 
      error.message.includes('timeout') ||
      error.message.includes('Rate limit') ||
      error.message.includes('503') ||
      error.message.includes('502') ||
      error.message.includes('500') ||
      error.message.includes('404');  
    
    if (isRetryable && retryCount < maxRetries) {
      const delay = Math.pow(2, retryCount) * 1000 + Math.random() * 1000; // Exponential backoff with jitter
      console.log(`Retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return await callGemini(prompt, taskType, retryCount + 1);
    }
    
    throw error;
  }
}

async function performEvaluation(evalId, evalData) {
  try {
    // Validasi input
    if (!evalData || !evalData.cvPath || !evalData.reportPath) {
      throw new Error('Invalid evalData: missing cvPath or reportPath');
    }
    
    console.log('Starting evaluation for:', evalData.id);
    console.log('CV Path:', evalData.cvPath);
    console.log('Report Path:', evalData.reportPath);
    
    const jobRequirements = `
    We are looking for a Full Stack Developer with:
    - Strong experience in JavaScript, Node.js, React
    - Database management skills (SQL/NoSQL)
    - API development and integration
    - Version control with Git
    - Problem-solving and analytical thinking
    - Good communication skills
    - Experience with cloud platforms is a plus
    `;
    
    // Tambahkan definisi scoringCriteria di sini
    const scoringCriteria = `
    Evaluation criteria on a scale of 1-5:
    - Correctness (30%): Does the solution work as expected? Are all requirements met?
    - Code Quality & Structure (25%): Is the code well-organized, readable, and maintainable?
    - Resilience & Error Handling (20%): Does the solution handle errors gracefully?
    - Documentation & Explanation (15%): Is the code well-documented? Are design decisions explained?
    - Creativity/Bonus (10%): Did the candidate go beyond requirements or show innovative approaches?
    `;

    // Extract text from uploaded files
    const cvText = await extractText(evalData.cvPath);
    const reportText = await extractText(evalData.reportPath);
    
    console.log('Text extraction completed');
    console.log('CV text length:', cvText.length);
    console.log('Report text length:', reportText.length);

    // Step 1: Extract structured CV data with RAG context
    const cvContext = await retrieveRelevantContext(cvText, 'job', 2);
    const cvContextText = cvContext.map(ctx => ctx.content).join('\n\n');
    
    const extractionPrompt = `Extract key information from this CV and format as JSON.

Relevant Job Requirements:\n${cvContextText}\n\nCV Text:\n${cvText}\n\nReturn ONLY a valid JSON object with these fields:\n{\n  "name": "candidate name",\n  "technical_skills": ["skill1", "skill2"],\n  "experience_years": number,\n  "relevant_experience": "description",\n  "education": "education background",\n  "achievements": ["achievement1", "achievement2"]\n}`;

    let extractedCV;
    try {
      console.log('Step 1: Extracting CV data with RAG context...');
      const cvResponse = await callGemini(extractionPrompt, 'extraction');
      extractedCV = safeJsonParse(cvResponse, { name: "Unknown", technical_skills: [], experience_years: 0, relevant_experience: "Not specified", education: "Not specified", achievements: [] });
      console.log('CV extraction successful:', extractedCV);
    } catch (error) {
      console.error('Error in CV extraction:', error);
      extractedCV = { name: "Unknown", technical_skills: [], experience_years: 0, relevant_experience: "Error extracting CV data", education: "Not specified", achievements: [] };
    }

    const comparisonPrompt = `
Analyze the CV against the job requirements and return ONLY a valid JSON object with this exact structure:

{
  "cv_match_rate": "percentage like 75%",
  "cv_feedback": "detailed feedback about strengths and gaps"
}

IMPORTANT: Return ONLY the JSON object above, no markdown, no explanations, no additional text.

CV Content:
${cvText}

Job Requirements:
${jobRequirements}
`;

    let cvComparison;
    try {
      console.log('Step 2: Comparing CV with job requirements...');
      const cvResult = await callGemini(comparisonPrompt, 'comparison');
      cvComparison = safeJsonParse(cvResult, { cv_match_rate: "0%", cv_feedback: "Error evaluating CV match rate." });
      console.log('CV comparison successful:', cvComparison);
    } catch (error) {
      console.error('Error in CV comparison:', error);
      cvComparison = { cv_match_rate: "0%", cv_feedback: "Error evaluating CV match rate." };
    }

    // Step 3: Score project deliverables with RAG context
    const projectContext = await retrieveRelevantContext(reportText, 'rubric', 3);
    const projectContextText = projectContext.map(ctx => ctx.content).join('\n\n');
    
    const scoringPrompt = `Score this project report based on the evaluation criteria.\n\nScoring Rubric:\n${projectContextText}\n\nProject Report:\n${reportText}\n\nEvaluate each criterion (1-5 scale):\n- Correctness (30%)\n- Code Quality & Structure (25%)\n- Resilience & Error Handling (20%)\n- Documentation & Explanation (15%)\n- Creativity/Bonus (10%)\n\nReturn ONLY a valid JSON object:\n{\n  "correctness": 1-5,\n  "code_quality": 1-5,\n  "error_handling": 1-5,\n  "documentation": 1-5,\n  "creativity": 1-5\n}`;
    
    let initialScore;
    try {
      console.log('Step 3: Scoring project with RAG context...');
      const scoringResponse = await callGemini(scoringPrompt, 'scoring');
      initialScore = safeJsonParse(scoringResponse, { correctness: 1, code_quality: 1, error_handling: 1, documentation: 1, creativity: 1 });
      console.log('Project scoring successful:', initialScore);
    } catch (error) {
      console.error('Error in project scoring:', error);
      initialScore = { correctness: 1, code_quality: 1, error_handling: 1, documentation: 1, creativity: 1 };
    }
    
    const scores = {
      correctness: initialScore.correctness || 1,
      code_quality: initialScore.code_quality || 1,
      error_handling: initialScore.error_handling || 1,
      documentation: initialScore.documentation || 1,
      creativity: initialScore.creativity || 1
    };
    
    const weightedScore = (
      scores.correctness * 0.30 +
      scores.code_quality * 0.25 +
      scores.error_handling * 0.20 +
      scores.documentation * 0.15 +
      scores.creativity * 0.10
    ); 
    
    const feedbackPrompt = `
Evaluate the provided content STRICTLY as a PROJECT REPORT (assume it is the project report even if it resembles a resume). Extract and evaluate any project-related details mentioned. Return ONLY a valid JSON object with this exact structure:

{
  "project_score": integer_number_between_1_and_5 (MUST be an integer from 1 to 5, no decimals like 5.5),
  "project_feedback": "Confirmation: This evaluation is based solely on the uploaded project report file. Detailed feedback about the PROJECT quality, implementation details, code structure, error handling, documentation, creativity, and areas for improvement. If the content lacks project-specific details (e.g., seems resume-like), note that and suggest providing a dedicated project report, but still evaluate available project descriptions. Include a breakdown of scores per category (e.g., Correctness: X/5, Code Quality: Y/5, etc.) and explain why each score was given."
}

IMPORTANT: Base your evaluation strictly on the content provided below and the scoring criteria. Keep scores as integers 1-5. Return ONLY the JSON object, no markdown, no explanations, no additional text.

Provided Content (Project Report):
${reportText}

Scoring Criteria:
${scoringCriteria}
`;
    
    let projectResult;
    try {
      console.log('Step 4: Generating project feedback...');
      const feedbackResponse = await callGemini(feedbackPrompt, 'feedback', 0, { /* params */ });
      console.log('Raw project feedback response from Gemini:', feedbackResponse);
      
      projectResult = safeJsonParse(feedbackResponse, { project_score: weightedScore, project_feedback: "Error generating project feedback." });
      console.log('Project feedback successful:', projectResult);
    } catch (error) {
      console.error('Error in project feedback:', error);
      projectResult = { project_score: weightedScore, project_feedback: "Error generating project feedback." };
    }
    
    // Step 5: Generate overall summary with enhanced prompt
    const summaryPrompt = `Create a comprehensive evaluation summary based on the following data:\n\nCV Match Rate: ${cvComparison.cv_match_rate}\nCV Feedback: ${cvComparison.cv_feedback}\nProject Score: ${projectResult.project_score}/5\nProject Feedback: ${projectResult.project_feedback}\n\nProvide a 3-5 sentence summary that includes:\n1. Overall candidate fit assessment\n2. Key strengths identified\n3. Main gaps or areas for improvement\n4. Final hiring recommendation with reasoning\n\nFormat the response as a cohesive paragraph that flows naturally from strengths to gaps to recommendations.`;
    
    let overallSummary;
    try {
      console.log('Step 5: Generating overall summary...');
      overallSummary = await callGemini(summaryPrompt, 'summary');
      console.log('Summary generation successful');
    } catch (error) {
      console.error('Error generating summary:', error);
      overallSummary = "Unable to generate summary due to an error.";
    }
    
    // Store result
    evalData.status = 'completed';
    evalData.evaluation = {
      cv_match_rate: cvComparison.cv_match_rate,
      cv_feedback: cvComparison.cv_feedback,
      project_score: projectResult.project_score,
      project_feedback: projectResult.project_feedback,
      overall_summary: overallSummary
    };
    
    console.log(`Evaluation completed successfully for ID: ${evalId}`);
    
  } catch (error) {
    console.error('Evaluation error:', error);
    evalData.status = 'failed';
    evalData.error = error.message;
  }
}

// Initialize vector database on startup
initializeVectorDB().catch(console.error);

// Endpoint: POST /upload - Upload CV and Project Report
app.post('/upload', upload.fields([{ name: 'cv', maxCount: 1 }, { name: 'report', maxCount: 1 }]), (req, res) => {
  const files = req.files;
  if (!files.cv || !files.report) {
    return res.status(400).json({ error: 'Both CV and report files are required' });
  }
  const uploadId = uuidv4();
  results[uploadId] = {
    cvPath: files.cv[0].path,
    reportPath: files.report[0].path,
    status: 'uploaded'
  };
  res.json({ uploadId, message: 'Files uploaded successfully' });
});

const evaluationQueue = new Queue('evaluation-queue', 'redis://127.0.0.1:6379');

evaluationQueue.process(async (job) => {
  const { evalId } = job.data;
  await performEvaluation(evalId, results[evalId]);  // Tambahkan evalId sebagai argumen pertama
});

app.post('/evaluate', (req, res) => {
  const { uploadId } = req.body;
  if (!uploadId || !results[uploadId]) {
    return res.status(400).json({ error: 'Invalid upload ID' });
  }
  
  const evalId = uuidv4();
  results[evalId] = {
    ...results[uploadId],
    status: 'queued'
  };
  
  evaluationQueue.add({ evalId });
  res.json({ evalId, message: 'Evaluation started' });
});

app.get('/result/:id', (req, res) => {
  const { id } = req.params;
  const result = results[id];
  
  if (!result) {
    return res.status(404).json({ error: 'Result not found' });
  }
  
  res.json(result);
});

if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

app.listen(port, () => {
  console.log(`AI Resume Evaluator running on port ${port}`);
});