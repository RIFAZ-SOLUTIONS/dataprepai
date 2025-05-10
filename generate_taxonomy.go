package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// TaxonomyMapping represents the structure of clean_taxonomy file
type TaxonomyMapping struct {
	Filename string `json:"filename"`
	Taxonomy string `json:"taxonomy"`
}

// YAMLData represents the structure of the qna.yaml file
type YAMLData struct {
	Version          int                    `json:"version"`
	Domain           string                 `json:"domain"`
	CreatedBy        string                 `json:"created_by"`
	SeedExamples     []SeedExample          `json:"seed_examples"`
	DocumentOutline  string                 `json:"document_outline"`
	Document         DocumentInfo           `json:"document"`
	ProcessingStatus map[string]interface{} `json:"-"` // For tracking progress, not included in YAML
}

// SeedExample represents a single context with its Q&A pairs
type SeedExample struct {
	Context             string              `json:"context"`
	QuestionsAndAnswers []QuestionAndAnswer `json:"questions_and_answers"`
}

// QuestionAndAnswer represents a single Q&A pair
type QuestionAndAnswer struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// DocumentInfo represents document metadata
type DocumentInfo struct {
	Repo     string   `json:"repo"`
	Commit   string   `json:"commit"`
	Patterns []string `json:"patterns"`
}

// GPT4Request represents the request format for the OpenAI API
type GPT4Request struct {
	Model    string        `json:"model"`
	Messages []GPT4Message `json:"messages"`
}

// GPT4Message represents a message in the GPT-4 API request
type GPT4Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// GPT4Response represents the response from the OpenAI API
type GPT4Response struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// RateLimiter handles API rate limiting
type RateLimiter struct {
	tokensPerMinute       int
	requestsPerMinute     int
	tokenCount            int
	requestCount          int
	lastResetTime         time.Time
	mutex                 sync.Mutex
	usagePercentageTarget float64 // Target percentage of rate limit to use
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(tokensPerMinute, requestsPerMinute int, usagePercentageTarget float64) *RateLimiter {
	return &RateLimiter{
		tokensPerMinute:       tokensPerMinute,
		requestsPerMinute:     requestsPerMinute,
		tokenCount:            0,
		requestCount:          0,
		lastResetTime:         time.Now(),
		usagePercentageTarget: usagePercentageTarget,
	}
}

// CheckAndWait checks if we're within rate limits and waits if necessary
func (rl *RateLimiter) CheckAndWait(tokenCount int) {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	// Reset counters if a minute has passed
	now := time.Now()
	if now.Sub(rl.lastResetTime) >= time.Minute {
		rl.tokenCount = 0
		rl.requestCount = 0
		rl.lastResetTime = now
	}

	// Calculate target limits based on usage percentage
	targetTokensPerMinute := int(float64(rl.tokensPerMinute) * rl.usagePercentageTarget)
	targetRequestsPerMinute := int(float64(rl.requestsPerMinute) * rl.usagePercentageTarget)

	// Check if adding this request would exceed our target
	if rl.tokenCount+tokenCount > targetTokensPerMinute || rl.requestCount+1 > targetRequestsPerMinute {
		// Calculate how long to wait before the next reset
		timeToNextMinute := time.Minute - now.Sub(rl.lastResetTime)
		log.Printf("Rate limit approaching. Waiting %v seconds before continuing...", timeToNextMinute.Seconds())
		time.Sleep(timeToNextMinute)
		
		// Reset counters
		rl.tokenCount = tokenCount
		rl.requestCount = 1
		rl.lastResetTime = time.Now()
		return
	}

	// If we're within limits, just update the counters
	rl.tokenCount += tokenCount
	rl.requestCount++
}

// ProcessingState represents the current state of document processing
type ProcessingState struct {
	CompletedDocuments map[string]bool     `json:"completed_documents"`
	InProgressYAMLs    map[string]YAMLData `json:"in_progress_yamls"`
}

// CountWords counts the approximate number of words in a string
func CountWords(s string) int {
	return len(strings.Fields(s))
}

// LoadProcessingState loads the processing state from a file
func LoadProcessingState(filename string) (*ProcessingState, error) {
	state := &ProcessingState{
		CompletedDocuments: make(map[string]bool),
		InProgressYAMLs:    make(map[string]YAMLData),
	}

	data, err := ioutil.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return state, nil
		}
		return nil, err
	}

	if err := json.Unmarshal(data, state); err != nil {
		return nil, err
	}

	return state, nil
}

// SaveProcessingState saves the processing state to a file
func SaveProcessingState(state *ProcessingState, filename string) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filename, data, 0644)
}

// LoadTaxonomyMappings loads the taxonomy mappings from a file
func LoadTaxonomyMappings(filename string) ([]TaxonomyMapping, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var mappings []TaxonomyMapping
	if err := json.Unmarshal(data, &mappings); err != nil {
		return nil, err
	}

	return mappings, nil
}

// FindChunks finds all chunks for a specific document
func FindChunks(chunksDir, documentName string) ([]string, error) {
	pattern := filepath.Join(chunksDir, documentName+"_doc_*")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}

	return matches, nil
}

// CreateYAMLStructure creates the directory structure for a YAML file
func CreateYAMLStructure(baseDir, taxonomy string) (string, error) {
	// Create the full path
	dirPath := filepath.Join(baseDir, taxonomy)
	
	// Create all directories in the path
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return "", err
	}
	
	return dirPath, nil
}

// GenerateQnAPairs generates Q&A pairs using GPT-4o
func GenerateQnAPairs(context string, rateLimiter *RateLimiter, openaiAPIKey string) ([]QuestionAndAnswer, error) {
	const estimatedTokensPerRequest = 1000 // Rough estimate
	rateLimiter.CheckAndWait(estimatedTokensPerRequest)

	prompt := fmt.Sprintf(`Generate 5 insightful question and answer pairs based on the following context. 
Each question should ask about specific details or concepts in the text. 
Answers should be factual, concise (1-3 sentences), and directly based on the information provided.
Format your response as JSON with the following structure:
[
  {"question": "Question 1?", "answer": "Answer 1"},
  {"question": "Question 2?", "answer": "Answer 2"},
  {"question": "Question 3?", "answer": "Answer 3"}
]

CONTEXT:
%s`, context)

	request := GPT4Request{
		Model: "gpt-4o",
		Messages: []GPT4Message{
			{Role: "system", Content: "You are an AI assistant that helps create educational question-answer pairs for knowledge bases."},
			{Role: "user", Content: prompt},
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", "https://taxfilingagent3540458317.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+openaiAPIKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var gptResponse GPT4Response
	if err := json.NewDecoder(resp.Body).Decode(&gptResponse); err != nil {
		return nil, err
	}

	// Update token count with actual usage
	rateLimiter.mutex.Lock()
	rateLimiter.tokenCount += gptResponse.Usage.TotalTokens
	rateLimiter.mutex.Unlock()

	// Parse the response content as JSON
	var questionAnswers []QuestionAndAnswer
	if err := json.Unmarshal([]byte(gptResponse.Choices[0].Message.Content), &questionAnswers); err != nil {
		// If the JSON parsing fails, try to extract with regex as a fallback
		return extractQAPairsWithRegex(gptResponse.Choices[0].Message.Content)
	}

	return questionAnswers, nil
}

// extractQAPairsWithRegex attempts to extract question-answer pairs using regex when JSON parsing fails
func extractQAPairsWithRegex(content string) ([]QuestionAndAnswer, error) {
	// This is a simple regex pattern that might need adjustment based on actual API responses
	questionRegex := regexp.MustCompile(`"question":\s*"([^"]+)"`)
	answerRegex := regexp.MustCompile(`"answer":\s*"([^"]+)"`)

	questionMatches := questionRegex.FindAllStringSubmatch(content, -1)
	answerMatches := answerRegex.FindAllStringSubmatch(content, -1)

	if len(questionMatches) != len(answerMatches) {
		return nil, fmt.Errorf("unequal number of questions and answers found")
	}

	var pairs []QuestionAndAnswer
	for i := 0; i < len(questionMatches); i++ {
		if len(questionMatches[i]) > 1 && len(answerMatches[i]) > 1 {
			pairs = append(pairs, QuestionAndAnswer{
				Question: questionMatches[i][1],
				Answer:   answerMatches[i][1],
			})
		}
	}

	if len(pairs) == 0 {
		return nil, fmt.Errorf("failed to extract any question-answer pairs")
	}

	return pairs, nil
}

// GenerateDocumentOutline generates a document outline using GPT-4o
func GenerateDocumentOutline(fullText string, rateLimiter *RateLimiter, openaiAPIKey string) (string, error) {
	const estimatedTokensPerRequest = 1200 // Rough estimate
	rateLimiter.CheckAndWait(estimatedTokensPerRequest)

	// Create a truncated version of the text if it's too long
	textToSend := fullText
	if len(textToSend) > 4000 {
		textToSend = textToSend[:4000] + "...[document continues]"
	}

	prompt := fmt.Sprintf(`Create a brief document outline (1-2 sentences) that describes the following document content:

%s`, textToSend)

	request := GPT4Request{
		Model: "gpt-4o",
		Messages: []GPT4Message{
			{Role: "system", Content: "You are an AI assistant that helps summarize document content."},
			{Role: "user", Content: prompt},
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://taxfilingagent3540458317.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+openaiAPIKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var gptResponse GPT4Response
	if err := json.NewDecoder(resp.Body).Decode(&gptResponse); err != nil {
		return "", err
	}

	// Update token count with actual usage
	rateLimiter.mutex.Lock()
	rateLimiter.tokenCount += gptResponse.Usage.TotalTokens
	rateLimiter.mutex.Unlock()

	return gptResponse.Choices[0].Message.Content, nil
}

// WriteYAMLFile writes a YAMLData struct to a file in YAML format
func WriteYAMLFile(data YAMLData, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// Write YAML header
	writer.WriteString("version: 1\n")
	writer.WriteString(fmt.Sprintf("domain: %s\n", data.Domain))
	writer.WriteString(fmt.Sprintf("created_by: %s\n", data.CreatedBy))
	
	// Write seed examples
	writer.WriteString("seed_examples:\n")
	for _, example := range data.SeedExamples {
		writer.WriteString("  - context: |\n")
		for _, line := range strings.Split(example.Context, "\n") {
			writer.WriteString(fmt.Sprintf("      %s\n", line))
		}
		
		writer.WriteString("    questions_and_answers:\n")
		for _, qa := range example.QuestionsAndAnswers {
			writer.WriteString("      - question: |\n")
			for _, line := range strings.Split(qa.Question, "\n") {
				writer.WriteString(fmt.Sprintf("          %s\n", line))
			}
			
			writer.WriteString("        answer: |\n")
			for _, line := range strings.Split(qa.Answer, "\n") {
				writer.WriteString(fmt.Sprintf("          %s\n", line))
			}
		}
	}
	
	// Write document outline
	writer.WriteString("document_outline: |\n")
	for _, line := range strings.Split(data.DocumentOutline, "\n") {
		writer.WriteString(fmt.Sprintf("  %s\n", line))
	}
	
	// Write document info
	writer.WriteString("document:\n")
	writer.WriteString(fmt.Sprintf("  repo: %s\n", data.Document.Repo))
	writer.WriteString(fmt.Sprintf("  commit: %s\n", data.Document.Commit))
	writer.WriteString("  patterns:\n")
	for _, pattern := range data.Document.Patterns {
		writer.WriteString(fmt.Sprintf("    - %s\n", pattern))
	}
	
	return writer.Flush()
}

// WriteAttributionFile writes a basic attribution.txt file
func WriteAttributionFile(filePath string, mdFile string) error {
	content := fmt.Sprintf(`Title of work: %s
Link to work: [Repository URL]
License of the work: [License]
Creator names: [Authors]
`, mdFile)

	return ioutil.WriteFile(filePath, []byte(content), 0644)
}


func main() {
    // Configuration
    dataDir := "./data_md"
    bestChunksDir := "./taxonomy_chunks"
    taxonomyMappingFile := "./cleaned_taxonomy.json"
    outputDir := "./taxonomy_output"
    stateFile := "./processing_state.json"
    githubRepoURL := "https://github.com/RIFAZ-SOLUTIONS/dataprepai.git"
    commitSHA := "1e7334d"
    githubUsername := "RIFAZ-SOLUTIONS"
    
    // Get OpenAI API key from environment variable
    openaiAPIKey := os.Getenv("AZUREAI_API_KEY")
    if openaiAPIKey == "" {
        log.Fatal("AZUREAI_API_KEY environment variable not set")
    }
    
    // Initialize rate limiter (70% of limits)
    rateLimiter := NewRateLimiter(100000, 600, 0.7)
    
    // Load taxonomy mappings
    mappings, err := LoadTaxonomyMappings(taxonomyMappingFile)
    if err != nil {
        log.Fatalf("Error loading taxonomy mappings: %v", err)
    }
    
    // Load or initialize processing state
    state, err := LoadProcessingState(stateFile)
    if err != nil {
        log.Fatalf("Error loading processing state: %v", err)
    }
    
    // Create output directory if it doesn't exist
    if err := os.MkdirAll(outputDir, 0755); err != nil {
        log.Fatalf("Error creating output directory: %v", err)
    }
    
    // Process each document
    for _, mapping := range mappings {
        // Skip already completed documents
        if state.CompletedDocuments[mapping.Filename] {
            log.Printf("Skipping already completed document: %s", mapping.Filename)
            continue
        }
        
        log.Printf("Processing document: %s", mapping.Filename)
        
        // Load or initialize YAML data
        yamlData, exists := state.InProgressYAMLs[mapping.Filename]
        if !exists {
            yamlData = YAMLData{
                Version:         1,
                Domain:          strings.Split(mapping.Taxonomy, "/")[0], // Take the first part of taxonomy path as domain
                CreatedBy:       githubUsername,
                SeedExamples:    []SeedExample{},
                DocumentOutline: "",
                Document: DocumentInfo{
                    Repo:     githubRepoURL,
                    Commit:   commitSHA,
                    Patterns: []string{"data_md/" + mapping.Filename + ".md"},
                },
                ProcessingStatus: make(map[string]interface{}),
            }
        }
        
        // Read the full document for document outline if not already generated
        if yamlData.DocumentOutline == "" {
            fullDocPath := filepath.Join(dataDir, mapping.Filename+".md")
            fullDocContent, err := ioutil.ReadFile(fullDocPath)
            if err != nil {
                log.Printf("Error reading full document %s: %v", fullDocPath, err)
                continue
            }
            
            // Generate document outline
            outline, err := GenerateDocumentOutline(string(fullDocContent), rateLimiter, openaiAPIKey)
            if err != nil {
                log.Printf("Error generating document outline for %s: %v", mapping.Filename, err)
                continue
            }
            yamlData.DocumentOutline = outline
            
            // Save progress
            state.InProgressYAMLs[mapping.Filename] = yamlData
            if err := SaveProcessingState(state, stateFile); err != nil {
                log.Printf("Error saving processing state: %v", err)
            }
        }
        
        // Look for pre-processed chunks in the best chunks directory
        bestChunksDocDir := filepath.Join(bestChunksDir, mapping.Filename)
        if _, err := os.Stat(bestChunksDocDir); os.IsNotExist(err) {
            log.Printf("No pre-processed chunks found for %s, skipping", mapping.Filename)
            continue
        }
        
        // Find all pre-processed chunks for this document
        bestChunkFiles, err := filepath.Glob(filepath.Join(bestChunksDocDir, mapping.Filename+"_selected_*"))
        if err != nil {
            log.Printf("Error finding pre-processed chunks for %s: %v", mapping.Filename, err)
            continue
        }
        
        if len(bestChunkFiles) == 0 {
            log.Printf("No pre-processed chunks found for %s, skipping", mapping.Filename)
            continue
        }
        
        // Process each pre-processed chunk
        for _, chunkPath := range bestChunkFiles {
            // Skip if this chunk is already processed
            chunkBaseName := filepath.Base(chunkPath)
            if _, processed := yamlData.ProcessingStatus[chunkBaseName]; processed {
                continue
            }
            
            // Read chunk content
            chunkContent, err := ioutil.ReadFile(chunkPath)
            if err != nil {
                log.Printf("Error reading chunk %s: %v", chunkPath, err)
                continue
            }
            
            chunkText := string(chunkContent)
            
            // Generate Q&A pairs for this chunk
            qaPairs, err := GenerateQnAPairs(chunkText, rateLimiter, openaiAPIKey)
            if err != nil {
                log.Printf("Error generating Q&A pairs for chunk %s: %v", chunkPath, err)
                yamlData.ProcessingStatus[chunkBaseName] = "error_generating_qa"
                
                // Save progress even when there's an error
                state.InProgressYAMLs[mapping.Filename] = yamlData
                if err := SaveProcessingState(state, stateFile); err != nil {
                    log.Printf("Error saving processing state: %v", err)
                }
                
                continue
            }
            
            // Add to seed examples
            yamlData.SeedExamples = append(yamlData.SeedExamples, SeedExample{
                Context:             chunkText,
                QuestionsAndAnswers: qaPairs,
            })
            
            yamlData.ProcessingStatus[chunkBaseName] = "processed"
            
            // Save progress after each chunk
            state.InProgressYAMLs[mapping.Filename] = yamlData
            if err := SaveProcessingState(state, stateFile); err != nil {
                log.Printf("Error saving processing state: %v", err)
            }
            
            // Add delay between chunk processing to avoid rate limiting
            time.Sleep(1 * time.Second)
        }
        
        // Skip to next document if we couldn't process any chunks
        if len(yamlData.SeedExamples) == 0 {
            log.Printf("Warning: No chunks were processed for %s", mapping.Filename)
            continue
        }
        
        // Create directory structure for YAML file
        yamlDir, err := CreateYAMLStructure(outputDir, mapping.Taxonomy)
        if err != nil {
            log.Printf("Error creating directory structure for %s: %v", mapping.Taxonomy, err)
            continue
        }
        
        // Write YAML file
        qna := fmt.Sprintf("%s_qna.yaml", mapping.Filename)
        yamlPath := filepath.Join(yamlDir, qna)
        if err := WriteYAMLFile(yamlData, yamlPath); err != nil {
            log.Printf("Error writing YAML file for %s: %v", mapping.Filename, err)
            continue
        }
        
        // Write attribution file
		attr := fmt.Sprintf("%s_attribution.txt", mapping.Filename)
        attributionPath := filepath.Join(yamlDir, attr)
        if err := WriteAttributionFile(attributionPath, mapping.Filename+".md"); err != nil {
            log.Printf("Error writing attribution file for %s: %v", mapping.Filename, err)
        }
        
        // Mark document as completed
        state.CompletedDocuments[mapping.Filename] = true
        delete(state.InProgressYAMLs, mapping.Filename)
        
        // Save processing state
        if err := SaveProcessingState(state, stateFile); err != nil {
            log.Printf("Error saving processing state: %v", err)
        }
        
        log.Printf("Successfully processed document: %s", mapping.Filename)
    }
    
    log.Println("Processing completed!")
}
