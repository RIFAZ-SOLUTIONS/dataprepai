package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/fs"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/neurosnap/sentences"
	"github.com/neurosnap/sentences/english"
)

const (
	inputDir              = "../taxlawsanddata-server"
	outputDir             = "../data_refined_sentences"
	maxWorkers            = 12  // tune this based on available CPU
	targetWordCount       = 180 // Aim for around 150-200 words per paragraph
	overlapSentenceCount  = 3   // Overlap by the last 3 sentences within a paragraph
	paragraphsPerDocument = 5   // Number of paragraphs per output document
	lineLength            = 80  // Maximum characters per line in the output chunks
)

type Chunk struct {
	Content  string
	Filename string
}

func main() {
	fmt.Println("🚀 Starting concurrent document processing with 5-paragraph documents in plain text...")

	os.MkdirAll(outputDir, os.ModePerm)
	files := collectFiles(inputDir)

	// Load the English sentence tokenizer
	tokenizer, err := english.NewSentenceTokenizer(nil)
	if err != nil {
		fmt.Printf("❌ Error loading sentence tokenizer: %v\n", err)
		return
	}

	var wg sync.WaitGroup
	fileChan := make(chan string, len(files))
	var failedFiles []string

	// Spawn initial workers
	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go worker(fileChan, &wg, &failedFiles, tokenizer, false)
	}

	// Feed files
	for _, file := range files {
		fileChan <- file
	}
	close(fileChan)
	wg.Wait()

	// Retry failed files
	if len(failedFiles) > 0 {
		fmt.Println("\n🔄 Retrying processing of failed files...")
		var retryWg sync.WaitGroup
		retryFileChan := make(chan string, len(failedFiles))
		var stillFailedFiles []string

		for i := 0; i < maxWorkers; i++ {
			retryWg.Add(1)
			go worker(retryFileChan, &retryWg, &stillFailedFiles, tokenizer, true)
		}
		for _, file := range failedFiles {
			retryFileChan <- file
		}
		close(retryFileChan)
		retryWg.Wait()

		if len(stillFailedFiles) > 0 {
			fmt.Println("\n❌ Some files still failed after retry:")
			for _, file := range stillFailedFiles {
				fmt.Println(file)
			}
		}
	}

	fmt.Println("✅ Done processing all files.")
}

func collectFiles(dir string) []string {
	var files []string
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if !d.IsDir() && (strings.HasSuffix(path, ".pdf") || strings.HasSuffix(path, ".docx") || strings.HasSuffix(path, ".doc")) {
			files = append(files, path)
		}
		return nil
	})
	return files
}

func worker(fileChan <-chan string, wg *sync.WaitGroup, failedFiles *[]string, tokenizer *sentences.DefaultSentenceTokenizer, isRetry bool) {
	defer wg.Done()
	for path := range fileChan {
		fmt.Printf("📝 Processing: %s\n", path)
		text := extractText(path)
		if text == "" {
			fmt.Printf("⚠️ Text extraction failed for %s\n", path)
			if !isRetry {
				*failedFiles = append(*failedFiles, path)
			}
			continue
		}
		base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		paragraphs := chunkText(text, base, tokenizer)
		documentCounter := 1
		for i := 0; i < len(paragraphs); i += paragraphsPerDocument {
			end := i + paragraphsPerDocument
			if end > len(paragraphs) {
				end = len(paragraphs)
			}
			documentParagraphs := paragraphs[i:end]
			if len(documentParagraphs) > 0 {
				saveDocument(documentParagraphs, base, documentCounter)
				documentCounter++
			}
		}
	}
}

func extractText(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		return extractPDFText(path)
	case ".docx":
		return extractDocxText(path)
	case ".doc":
		return convertDocToText(path)
	default:
		return ""
	}
}

func extractPDFText(path string) string {
	cmd := exec.Command("pdftotext", path, "-")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		fmt.Printf("❌ PDF extraction failed for %s: %v\n", path, err)
		return ""
	}
	return out.String()
}

func extractDocxText(path string) string {
	cmd := exec.Command("docx2txt", path, "-")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		fmt.Printf("❌ DOCX extraction failed for %s: %v\n", path, err)
		return ""
	}
	return out.String()
}

func convertDocToText(path string) string {
	cmd := exec.Command("libreoffice", "--headless", "--convert-to", "txt:Text", path, "--outdir", ".")
	_ = cmd.Run()
	txtFile := strings.TrimSuffix(filepath.Base(path), ".doc") + ".txt"
	data, err := ioutil.ReadFile(txtFile)
	if err != nil {
		return ""
	}
	os.Remove(txtFile)
	return string(data)
}

func chunkText(text, base string, tokenizer *sentences.DefaultSentenceTokenizer) []string {
	var paragraphs []string
	var currentParagraph strings.Builder
	var currentWordCount int
	var sentenceBuffer []string

	sentences := tokenizer.Tokenize(text)
	sentenceTexts := make([]string, len(sentences))
	for i, s := range sentences {
		sentenceTexts[i] = s.Text
	}

	for i, sentenceText := range sentenceTexts {
		sentenceWordCount := len(strings.Fields(sentenceText))

		if currentWordCount+sentenceWordCount <= targetWordCount && i < len(sentenceTexts) {
			if currentParagraph.Len() > 0 {
				currentParagraph.WriteString(" ")
			}
			currentParagraph.WriteString(sentenceText)
			currentWordCount += sentenceWordCount
			sentenceBuffer = append(sentenceBuffer, sentenceText)
		} else {
			// Save the current paragraph
			if currentParagraph.Len() > 0 {
				paragraphs = append(paragraphs, currentParagraph.String())
			}

			// Start a new paragraph with overlap (last few sentences from the previous paragraph)
			currentParagraph.Reset()
			currentWordCount = 0
			overlapStart := max(0, len(sentenceBuffer)-overlapSentenceCount)
			overlapSentences := sentenceBuffer[overlapStart:]

			for _, overlapSentence := range overlapSentences {
				overlapSentenceWordCount := len(strings.Fields(overlapSentence))
				if currentWordCount+overlapSentenceWordCount <= targetWordCount/2 { // Keep overlap reasonable
					if currentParagraph.Len() > 0 {
						currentParagraph.WriteString(" ")
					}
					currentParagraph.WriteString(overlapSentence)
					currentWordCount += overlapSentenceWordCount
				}
			}
			sentenceBuffer = overlapSentences

			// Add the current sentence to the new paragraph
			if currentParagraph.Len() > 0 {
				currentParagraph.WriteString(" ")
			}
			currentParagraph.WriteString(sentenceText)
			currentWordCount += sentenceWordCount
			sentenceBuffer = append(sentenceBuffer, sentenceText)
		}

		// Save the last paragraph
		if i == len(sentenceTexts)-1 && currentParagraph.Len() > 0 {
			paragraphs = append(paragraphs, currentParagraph.String())
		}
	}

	return paragraphs
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func formatText(text string) string {
    var formatted strings.Builder
    words := strings.Split(text, " ")

    const wordsPerSection = 100

    sectionNum := 1

    for i := 0; i < len(words); i += wordsPerSection {
        // Change from "%d. " to "%dps. "
        formatted.WriteString(fmt.Sprintf("%dps. ", sectionNum))
        sectionNum++

        end := i + wordsPerSection
        if end > len(words) {
            end = len(words)
        }

        // Write content for this section with line wrapping
        currentLineLength := 0
        // Account for the section number indent on first line
        firstLine := true

        for j := i; j < end; j++ {
            word := words[j]

            // Handle line wrapping
            if firstLine {
                // First line already has the section number
                // Note the increased offset to account for the longer prefix "Xps. " vs "X. "
                if currentLineLength+len(word) <= lineLength-5 { // -5 for "Xps. " format
                    formatted.WriteString(word)
                    currentLineLength = len(word)
                    firstLine = false
                } else {
                    formatted.WriteString("\n   " + word) // Indent continuation lines
                    currentLineLength = len(word) + 3
                    firstLine = false
                }
            } else if currentLineLength+len(word)+1 <= lineLength {
                formatted.WriteString(" " + word)
                currentLineLength += len(word) + 1
            } else {
                formatted.WriteString("\n   " + word) // Indent continuation lines
                currentLineLength = len(word) + 3
            }
        }

        // Add double newline between sections
        formatted.WriteString("\n\n")
    }

    return formatted.String()
}

func saveDocument(paragraphs []string, baseFilename string, documentNumber int) {
    outputFilename := filepath.Join(outputDir, fmt.Sprintf("%s_doc_%04d.txt", baseFilename, documentNumber))
    fmt.Printf("[Process %d] Saving document: %s\n", os.Getpid(), outputFilename)
    outfile, err := os.Create(outputFilename)
    if err != nil {
        fmt.Printf("❌ Error creating output file %s: %v\n", outputFilename, err)
        return
    }
    defer outfile.Close()

    writer := bufio.NewWriter(outfile)
    defer writer.Flush()

    for i, paragraph := range paragraphs {
        formattedParagraph := formatText(paragraph)
        // Change from "%d. " to "%dps. "
        _, err = writer.WriteString(fmt.Sprintf("%dps. %s\n\n", i+1, formattedParagraph))
        if err != nil {
            fmt.Printf("❌ Error writing to output file %s: %v\n", outputFilename, err)
        }
    }
}
