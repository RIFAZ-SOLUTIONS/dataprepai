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
	"strings"
	"sync"
	"time"
)

// RateLimiter controls API usage to avoid exceeding hourly/minute quotas.
type RateLimiter struct {
	tokensPerMinute   int
	requestsPerMinute int
	tokenCount        int
	requestCount      int
	lastReset         time.Time
	mutex             sync.Mutex
	targetFraction    float64
}

// NewRateLimiter sets up a limiter that uses up to targetFraction of allowed rate.
func NewRateLimiter(tokensPerMinute, requestsPerMinute int, targetFraction float64) *RateLimiter {
	return &RateLimiter{
		tokensPerMinute:   tokensPerMinute,
		requestsPerMinute: requestsPerMinute,
		tokenCount:        0,
		requestCount:      0,
		lastReset:         time.Now(),
		targetFraction:    targetFraction,
	}
}

// CheckAndWait blocks until adding tokenCount won't exceed target rate per minute.
func (rl *RateLimiter) CheckAndWait(tokenCount int) {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	now := time.Now()
	if now.Sub(rl.lastReset) >= time.Minute {
		rl.tokenCount = 0
		rl.requestCount = 0
		rl.lastReset = now
	}

	tokenLimit := int(float64(rl.tokensPerMinute) * rl.targetFraction)
	reqLimit := int(float64(rl.requestsPerMinute) * rl.targetFraction)

	if rl.tokenCount+tokenCount > tokenLimit || rl.requestCount+1 > reqLimit {
		sleep := time.Minute - now.Sub(rl.lastReset)
		log.Printf("[RateLimiter] sleeping %v to respect rate limit", sleep)
		time.Sleep(sleep)
		// reset
		rl.tokenCount = tokenCount
		rl.requestCount = 1
		rl.lastReset = time.Now()
		return
	}

	rl.tokenCount += tokenCount
	rl.requestCount++
}

// QAPair holds a generated question and answer.
type QAPair struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// CompletedChunks maps each file to the set of chunk indices fully processed.
type CompletedChunks map[string]map[int]bool

// InProgressFiles tracks partial results for each file's chunk.
type InProgressFiles map[string]map[int][]QAPair

// ProcessingState manages persistent state on disk.
type ProcessingState struct {
	StateFile       string
	CompletedChunks CompletedChunks
	InProgressFiles InProgressFiles
	mutex           sync.Mutex
}

// StateData is used for JSON marshalling of state to disk.
type StateData struct {
	CompletedChunks map[string][]int         `json:"completed_chunks"`
	InProgressFiles map[string]map[int][]QAPair `json:"in_progress_files"`
}

// NewProcessingState constructs state, loading or initializing the file.
func NewProcessingState(stateFile string) *ProcessingState {
	ps := &ProcessingState{
		StateFile:       stateFile,
		CompletedChunks: make(CompletedChunks),
		InProgressFiles: make(InProgressFiles),
	}
	LoadState(ps)
	return ps
}

// LoadState reads or initializes the JSON state file.
func LoadState(ps *ProcessingState) {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	// Ensure maps exist
	if ps.CompletedChunks == nil {
		ps.CompletedChunks = make(CompletedChunks)
	}
	if ps.InProgressFiles == nil {
		ps.InProgressFiles = make(InProgressFiles)
	}

	abs, _ := filepath.Abs(ps.StateFile)
	info, err := os.Stat(ps.StateFile)
	if os.IsNotExist(err) || (err == nil && info.Size() == 0) {
		// initialize
		if writeErr := ioutil.WriteFile(ps.StateFile, []byte("{}"), 0644); writeErr != nil {
			log.Printf("[LoadState] failed to init %s: %v", abs, writeErr)
		} else {
			log.Printf("[LoadState] created new state file %s", abs)
		}
		return
	}
	if err != nil {
		log.Printf("[LoadState] stat error %s: %v", abs, err)
		return
	}

	data, err := ioutil.ReadFile(ps.StateFile)
	if err != nil {
		log.Printf("[LoadState] read error %s: %v", abs, err)
		return
	}

	var sd StateData
	if err := json.Unmarshal(data, &sd); err != nil {
		log.Printf("[LoadState] parse error %s: %v", abs, err)
		// reset corrupted
		ioutil.WriteFile(ps.StateFile, []byte("{}"), 0644)
		return
	}

	// repopulate
	for file, arr := range sd.CompletedChunks {
		if ps.CompletedChunks[file] == nil {
			ps.CompletedChunks[file] = make(map[int]bool)
		}
		for _, idx := range arr {
			ps.CompletedChunks[file][idx] = true
		}
	}

	for file, cmap := range sd.InProgressFiles {
		ps.InProgressFiles[file] = cmap
	}

	log.Printf("[LoadState] loaded %d complete, %d in-progress", len(ps.CompletedChunks), len(ps.InProgressFiles))
}

// SaveState writes in-memory state back to disk.
func SaveState(ps *ProcessingState) error {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	abs, _ := filepath.Abs(ps.StateFile)
	log.Printf("[SaveState] writing state to %s", abs)

	sd := StateData{
		CompletedChunks: make(map[string][]int, len(ps.CompletedChunks)),
		InProgressFiles: ps.InProgressFiles,
	}
	for file, cmap := range ps.CompletedChunks {
		for idx, done := range cmap {
			if done {
				sd.CompletedChunks[file] = append(sd.CompletedChunks[file], idx)
			}
		}
	}

	b, err := json.MarshalIndent(sd, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal state error: %v", err)
	}
	if err := ioutil.WriteFile(ps.StateFile, b, 0644); err != nil {
		return fmt.Errorf("write state error: %v", err)
	}
	log.Printf("[SaveState] success")
	return nil
}

// IsChunkCompleted returns true if the chunk was already done.
func (ps *ProcessingState) IsChunkCompleted(file string, idx int) bool {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()
	base := filepath.Base(file)
	cmap, ok := ps.CompletedChunks[base]
	return ok && cmap[idx]
}

// MarkChunkCompleted flags a chunk as done and persists state.
func (ps *ProcessingState) MarkChunkCompleted(file string, idx int) {
	ps.mutex.Lock()
	if ps.CompletedChunks[filepath.Base(file)] == nil {
		ps.CompletedChunks[filepath.Base(file)] = make(map[int]bool)
	}
	ps.CompletedChunks[filepath.Base(file)][idx] = true
	ps.mutex.Unlock()
	SaveState(ps)
}

// GetInProgress retrieves cached QA pairs for a chunk, if any.
func (ps *ProcessingState) GetInProgress(file string, idx int) []QAPair {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()
	return ps.InProgressFiles[filepath.Base(file)][idx]
}

// UpdateInProgress caches QA pairs and persists.
func (ps *ProcessingState) UpdateInProgress(file string, idx int, pairs []QAPair) {
	ps.mutex.Lock()
	if ps.InProgressFiles[filepath.Base(file)] == nil {
		ps.InProgressFiles[filepath.Base(file)] = make(map[int][]QAPair)
	}
	ps.InProgressFiles[filepath.Base(file)][idx] = pairs
	ps.mutex.Unlock()
	SaveState(ps)
}

// ClearInProgress removes all in-progress entries for a file.
func (ps *ProcessingState) ClearInProgress(file string) {
	ps.mutex.Lock()
	delete(ps.InProgressFiles, filepath.Base(file))
	ps.mutex.Unlock()
	SaveState(ps)
}

func removeControlChars(s string) string {
    return strings.Map(func(r rune) rune {
        if r < 32 && r != '\n' && r != '\t' {
            return -1
        }
        return r
    }, s)
}

func normalizeWhitespace(s string) string {
    s = strings.ReplaceAll(s, "\t", " ")
    s = regexp.MustCompile(` +`).ReplaceAllString(s, " ")
    s = regexp.MustCompile(`\n+`).ReplaceAllString(s, "\n")
    return strings.TrimSpace(s)
}

func removeNonUTF8(input string) string {
    return strings.ToValidUTF8(input, "")
}

func stripHTMLTags(s string) string {
    re := regexp.MustCompile(`<[^>]*>`)
    return re.ReplaceAllString(s, "")
}

func sanitizeChunk(chunk string) string {
    chunk = removeControlChars(chunk)
    chunk = removeNonUTF8(chunk)
    chunk = stripHTMLTags(chunk)
    chunk = normalizeWhitespace(chunk)
    return chunk
}


// GenerateQAPairs reads input, chunks on markers, generates Q&A, writes JSONL.
func GenerateQAPairs(input, outputDir, apiKey string, rl *RateLimiter, ps *ProcessingState) error {
	// Prepare output
	name := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
	outPath := filepath.Join(outputDir, name+".jsonl")
	os.MkdirAll(filepath.Dir(outPath), 0755)

	inF, err := os.Open(input)
	if err != nil {
		return err
	}
	defer inF.Close()
	mode := os.O_CREATE|os.O_WRONLY|os.O_TRUNC
	if _, err := os.Stat(outPath); err == nil {
		mode = os.O_CREATE|os.O_WRONLY|os.O_APPEND
	}
	outF, err := os.OpenFile(outPath, mode, 0644)
	if err != nil {
		return err
	}
	defer outF.Close()

	scanner := bufio.NewScanner(inF)
	chunk := ""
	idx := 0

	flushChunk := func() {
		if chunk == "" {
			return
		}
		log.Printf("[Process] chunk %d start", idx)
		if !ps.IsChunkCompleted(input, idx) {
			pairs := ps.GetInProgress(input, idx)
			if pairs == nil {
				// call GPT
				rl.CheckAndWait(1000)

				cleanedChunk := sanitizeChunk(chunk)
				
				userPrompt := fmt.Sprintf(
					`Generate exactly 5 pairs using *these prefixes* on separate lines (no numbering, no extra text):
						Question: <your question>
						Answer: <your answer>
					TEXT:
					%s`, cleanedChunk)
				
				pairs, err = callGPT(apiKey, userPrompt)
				if err != nil {
					log.Printf("GPT error: %v", err)
					return
				}
				ps.UpdateInProgress(input, idx, pairs)
			}
			// write JSONL
			for _, p := range pairs {
				msg := map[string][]map[string]string{ "messages": {
					{ "role": "user", "content": chunk+"\n"+p.Question },
					{ "role": "assistant", "content": p.Answer },
				} }
				b, _ := json.Marshal(msg)
				outF.Write(b)
				outF.WriteString("\n")
			}
			ps.MarkChunkCompleted(input, idx)
		}
		chunk = ""
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, fmt.Sprintf("%dps.", idx+1)) {
			flushChunk()
			idx++
			chunk = strings.TrimPrefix(line, fmt.Sprintf("%dps.", idx))
		} else if chunk != "" && line != "" {
			chunk += "\n" + line
		}
	}
	flushChunk()

	ps.ClearInProgress(input)
	log.Printf("[Process] done file %s -> %s", input, outPath)
	return nil
}

// callGPT encapsulates the HTTP call to generate Q&A pairs.
func callGPT(apiKey, text string) ([]QAPair, error) {
	reqBody := map[string]interface{}{
		"model": "gpt-4o",
		"messages": []map[string]string{
			{ "role": "system", "content": "You are an AI assistant that WILL produce exactly 5 question-and-answer pairs." },
			{ "role": "user", "content": text },
		},
		"temperature": 0.7,
		"max_tokens": 2000,
	}
	b, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", os.Getenv("AZURE_ENDPOINT"), bytes.NewReader(b))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out struct { Choices []struct { Message struct { Content string } } };
	json.NewDecoder(resp.Body).Decode(&out)
	raw := out.Choices[0].Message.Content
	log.Printf("[DEBUG] GPT raw output:\n%s\n", raw)
	return extractPairs(raw)
}

// extractPairs parses GPT content into QAPair slices.
func extractPairs(raw string) ([]QAPair, error) {
	lines := strings.Split(raw, "\n")
	var list []QAPair
	var cur QAPair
	for _, l := range lines {
		s := strings.TrimSpace(l)
		if strings.HasPrefix(s, "Question:") {
			if cur.Question!="" && cur.Answer!="" { list = append(list, cur) }
			cur = QAPair{Question: strings.TrimSpace(strings.TrimPrefix(s, "Question:"))}
		} else if strings.HasPrefix(s, "Answer:") {
			cur.Answer = strings.TrimSpace(strings.TrimPrefix(s, "Answer:"))
		}
	}
	if cur.Question!="" && cur.Answer!="" { list = append(list, cur) }
	if len(list)==0 {
		return nil, fmt.Errorf("no QAs parsed")
	}
	return list, nil
}

func main() {
	input := "../data_refined_sentences_md"
	outputDir := "../finetuning_md_gpt4o"
	apiKey := os.Getenv("AZUREAI_API_KEY")
	if apiKey=="" {
		log.Fatal("AZUREAI_API_KEY not set")
	}
	state := NewProcessingState(filepath.Join(outputDir, "state.json"))
	rl := NewRateLimiter(100000, 600, 0.7)

	// Determine single file vs. directory
	info, err := os.Stat(input)
	if err!=nil { log.Fatal(err) }
	if info.IsDir() {
		files, _ := ioutil.ReadDir(input)
		for _, f := range files {
			if !f.IsDir() && strings.HasSuffix(f.Name(), ".txt") {
				GenerateQAPairs(filepath.Join(input, f.Name()), outputDir, apiKey, rl, state)
			}
		}
	} else {
		GenerateQAPairs(input, outputDir, apiKey, rl, state)
	}
}
