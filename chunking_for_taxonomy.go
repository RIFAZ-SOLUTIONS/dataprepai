package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"encoding/json"
)

// Configuration structure
type Config struct {
	InputChunksDir      string   `json:"input_chunks_dir"`
	OutputDir           string   `json:"output_dir"`
	TaxonomyMappingFile string   `json:"taxonomy_mapping_file"`
	ChunksPerDocument   int      `json:"chunks_per_document"`
	WordLimit           int      `json:"word_limit"`
}

// TaxonomyMapping represents the structure of clean_taxonomy file
type TaxonomyMapping struct {
	Filename string `json:"filename"`
	Taxonomy string `json:"taxonomy"`
}

// ChunkInfo stores information about a chunk
type ChunkInfo struct {
	Path           string
	Score          int
	Content        string
	ProcessedContent string
	WordCount      int
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

// CountWords counts the approximate number of words in a string
func CountWords(s string) int {
	return len(strings.Fields(s))
}

// EvaluateChunkQuality assigns a score to a chunk based on its content quality
func EvaluateChunkQuality(chunkText string) int {
	wordCount := CountWords(chunkText)
	
	// Too short chunks are low quality
	if wordCount < 100 {
		return 0
	}
	
	// Calculate a basic score
	score := wordCount // Base score is the word count
	
	// Adjust score based on other heuristics
	// For example, presence of structured content like headings, code blocks, lists
	if strings.Contains(chunkText, "##") || strings.Contains(chunkText, "```") {
		score += 50
	}
	
	// Presence of explanatory phrases often indicates good content
	explanatoryPhrases := []string{"means", "is a", "refers to", "example", "such as", "defined as"}
	for _, phrase := range explanatoryPhrases {
		if strings.Contains(strings.ToLower(chunkText), phrase) {
			score += 10
		}
	}
	
	// Additional heuristics for technical content
	technicalTerms := []string{"function", "method", "class", "interface", "struct", "object", "variable", "constant"}
	for _, term := range technicalTerms {
		if strings.Contains(strings.ToLower(chunkText), term) {
			score += 5
		}
	}
	
	return score
}

// TruncateToWordLimit truncates text to specified word limit while preserving coherence
func TruncateToWordLimit(text string, wordLimit int) string {
	words := strings.Fields(text)
	if len(words) <= wordLimit {
		return text
	}
	
	// Try to find a good breaking point (end of sentence) near the word limit
	shortenedText := strings.Join(words[:wordLimit], " ")
	
	// Find the last period, question mark, or exclamation point
	lastPeriod := strings.LastIndex(shortenedText, ".")
	lastQuestion := strings.LastIndex(shortenedText, "?")
	lastExclamation := strings.LastIndex(shortenedText, "!")
	
	// Find the maximum of these positions
	lastBreak := max(lastPeriod, max(lastQuestion, lastExclamation))
	
	if lastBreak > len(shortenedText)/2 {
		// If we found a good breaking point in the second half, use it
		return shortenedText[:lastBreak+1]
	}
	
	// Otherwise just use the word limit
	return shortenedText
}

// Helper function for finding max of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ProcessDocument processes a single document
func ProcessDocument(mapping TaxonomyMapping, config Config) error {
	log.Printf("Processing document: %s", mapping.Filename)
	
	// Find all chunks for this document
	chunks, err := FindChunks(config.InputChunksDir, mapping.Filename)
	if err != nil {
		return fmt.Errorf("error finding chunks: %v", err)
	}
	
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks found for document %s", mapping.Filename)
	}
	
	// Process and score each chunk
	var chunkInfos []ChunkInfo
	for _, chunkPath := range chunks {
		// Read chunk content
		content, err := ioutil.ReadFile(chunkPath)
		if err != nil {
			log.Printf("Warning: Error reading chunk %s: %v", chunkPath, err)
			continue
		}
		
		chunkText := string(content)
		
		// Skip empty or very short chunks
		if len(strings.TrimSpace(chunkText)) < 100 {
			log.Printf("Skipping chunk %s: too short", chunkPath)
			continue
		}
		
		// Score the chunk
		score := EvaluateChunkQuality(chunkText)
		
		// Process the chunk content (truncate if necessary)
		processedContent := chunkText
		wordCount := CountWords(chunkText)
		if wordCount > config.WordLimit {
			processedContent = TruncateToWordLimit(chunkText, config.WordLimit)
		}
		
		chunkInfos = append(chunkInfos, ChunkInfo{
			Path:            chunkPath,
			Score:           score,
			Content:         chunkText,
			ProcessedContent: processedContent,
			WordCount:       wordCount,
		})
	}
	
	// Sort chunks by score in descending order
	sort.Slice(chunkInfos, func(i, j int) bool {
		return chunkInfos[i].Score > chunkInfos[j].Score
	})
	
	// Determine how many chunks to keep
	numChunksToKeep := min(config.ChunksPerDocument, len(chunkInfos))
	if numChunksToKeep == 0 {
		return fmt.Errorf("no suitable chunks found for document %s", mapping.Filename)
	}
	
	log.Printf("Selected %d best chunks out of %d total chunks", numChunksToKeep, len(chunkInfos))
	
	// Create output directory for this document
	documentOutputDir := filepath.Join(config.OutputDir, mapping.Filename)
	if err := os.MkdirAll(documentOutputDir, 0755); err != nil {
		return fmt.Errorf("error creating output directory: %v", err)
	}
	
	// Save the selected chunks
	for i := 0; i < numChunksToKeep; i++ {
		chunkInfo := chunkInfos[i]
		
		// Create a more descriptive filename
		originalBasename := filepath.Base(chunkInfo.Path)
		outputFilename := fmt.Sprintf("%s_selected_%d_score_%d_words_%d%s", 
			mapping.Filename, 
			i+1, 
			chunkInfo.Score,
			CountWords(chunkInfo.ProcessedContent),
			filepath.Ext(originalBasename))
		
		outputPath := filepath.Join(documentOutputDir, outputFilename)
		
		// Save the processed content
		if err := ioutil.WriteFile(outputPath, []byte(chunkInfo.ProcessedContent), 0644); err != nil {
			log.Printf("Warning: Error writing processed chunk %s: %v", outputPath, err)
			continue
		}
		
		log.Printf("Saved processed chunk: %s", outputFilename)
	}
	
	// Create a metadata file with information about the selection process
	metadataPath := filepath.Join(documentOutputDir, "metadata.json")
	metadata := struct {
		DocumentName          string `json:"document_name"`
		Taxonomy              string `json:"taxonomy"`
		TotalChunks           int    `json:"total_chunks"`
		SelectedChunks        int    `json:"selected_chunks"`
		ProcessingDate        string `json:"processing_date"`
		MaxWordLimit          int    `json:"max_word_limit"`
		ChunksPerDocumentLimit int   `json:"chunks_per_document_limit"`
	}{
		DocumentName:          mapping.Filename,
		Taxonomy:              mapping.Taxonomy,
		TotalChunks:           len(chunkInfos),
		SelectedChunks:        numChunksToKeep,
		ProcessingDate:        time.Now().Format(time.RFC3339),
		MaxWordLimit:          config.WordLimit,
		ChunksPerDocumentLimit: config.ChunksPerDocument,
	}
	
	metadataBytes, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		log.Printf("Warning: Error creating metadata: %v", err)
	} else {
		if err := ioutil.WriteFile(metadataPath, metadataBytes, 0644); err != nil {
			log.Printf("Warning: Error writing metadata file: %v", err)
		}
	}
	
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	// Default configuration
	config := Config{
		InputChunksDir:      "./data_refined_sentences_md",
		OutputDir:           "./taxonomy_chunks",
		TaxonomyMappingFile: "./cleaned_taxonomy.json",
		ChunksPerDocument:   5,  // Select 5 best chunks per document
		WordLimit:           300, // Limit to 500 words per chunk
	}
	
	// Load configuration from file if exists
	configData, err := ioutil.ReadFile("chunk_processor_config.json")
	if err == nil {
		if err := json.Unmarshal(configData, &config); err != nil {
			log.Printf("Warning: Error parsing config file: %v. Using default configuration.", err)
		}
	} else {
		log.Printf("Using default configuration (no config file found)")
	}
	
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}
	
	// Load taxonomy mappings
	mappings, err := LoadTaxonomyMappings(config.TaxonomyMappingFile)
	if err != nil {
		log.Fatalf("Error loading taxonomy mappings: %v", err)
	}
	
	// Process each document
	successCount := 0
	for _, mapping := range mappings {
		if err := ProcessDocument(mapping, config); err != nil {
			log.Printf("Error processing document %s: %v", mapping.Filename, err)
		} else {
			successCount++
		}
	}
	
	log.Printf("Processing completed! Successfully processed %d out of %d documents.", successCount, len(mappings))
	
	// Save configuration for reference
	configBytes, err := json.MarshalIndent(config, "", "  ")
	if err == nil {
		if err := ioutil.WriteFile(filepath.Join(config.OutputDir, "config_used.json"), configBytes, 0644); err != nil {
			log.Printf("Warning: Failed to save config record: %v", err)
		}
	}
}
