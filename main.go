package main

import (
    "encoding/json"
    "fmt"
    "io/ioutil"
    "os"
    "path/filepath"
    "strings"
)

// Entry represents the reduced JSON structure
type Entry struct {
    Taxonomy string `json:"taxonomy"`
    Filename string `json:"filename"`
}

// FullEntry represents the original JSON structure
type FullEntry struct {
    TaxonomyPath string `json:"TaxonomyPath"`
    MatchedWith  string `json:"MatchedWith"`
}

func main() {
    // Hardcoded relative paths
    const inputPathRel = "taxonomy_data.json"
    const dirPathRel = "data_md"
    const outputPathRel = "taxonomy-mapper.json"

    // Determine absolute paths based on CWD
    cwd, err := os.Getwd()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
        os.Exit(1)
    }

    inputPath, err := filepath.Abs(filepath.Join(cwd, inputPathRel))
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error resolving input path: %v\n", err)
        os.Exit(1)
    }
    dirPath, err := filepath.Abs(filepath.Join(cwd, dirPathRel))
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error resolving directory path: %v\n", err)
        os.Exit(1)
    }
    outputPath, err := filepath.Abs(filepath.Join(cwd, outputPathRel))
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error resolving output path: %v\n", err)
        os.Exit(1)
    }

    fmt.Fprintf(os.Stderr, "Working directory: %s\n", cwd)
    fmt.Fprintf(os.Stderr, "Resolved input path: %s\n", inputPath)
    fmt.Fprintf(os.Stderr, "Resolved dir path: %s\n", dirPath)
    fmt.Fprintf(os.Stderr, "Resolved output path: %s\n", outputPath)

    // Read input JSON
    data, err := ioutil.ReadFile(inputPath)
    if err != nil {
        if os.IsNotExist(err) {
            fmt.Fprintf(os.Stderr, "Input file does not exist: %s\n", inputPath)
        }
        fmt.Fprintf(os.Stderr, "Error reading input file: %v\n", err)
        os.Exit(1)
    }

    // Parse original entries
    var full []FullEntry
    if err := json.Unmarshal(data, &full); err != nil {
        fmt.Fprintf(os.Stderr, "Error parsing JSON: %v\n", err)
        os.Exit(1)
    }

    // Build map and slice of reduced entries
    entries := make([]Entry, 0, len(full))
    seen := make(map[string]bool)
    for _, fe := range full {
        tax := strings.ToLower(strings.ReplaceAll(fe.TaxonomyPath, "\\", "/"))
        tax = strings.ReplaceAll(tax, "//", "/")
        entries = append(entries, Entry{Taxonomy: tax, Filename: fe.MatchedWith})
        seen[fe.MatchedWith] = true
    }

    // Scan directory for extra files
    files, err := ioutil.ReadDir(dirPath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error reading directory '%s': %v\n", dirPath, err)
        os.Exit(1)
    }
    for _, fi := range files {
        if fi.IsDir() {
            continue
        }
        base := strings.TrimSuffix(fi.Name(), filepath.Ext(fi.Name()))
        if !seen[base] {
            entries = append(entries, Entry{Taxonomy: "tax/taxes", Filename: base})
        }
    }

    // Write output JSON
    outData, err := json.MarshalIndent(entries, "", "  ")
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error marshalling output: %v\n", err)
        os.Exit(1)
    }
    if err := ioutil.WriteFile(outputPath, outData, 0644); err != nil {
        fmt.Fprintf(os.Stderr, "Error writing output file: %v\n", err)
        os.Exit(1)
    }
    fmt.Printf("Wrote %d entries to %s\n", len(entries), outputPath)
}
