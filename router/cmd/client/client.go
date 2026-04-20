package main

import (
	"bufio" // Used for Scanner
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
)

const routerURL = "http://localhost:8000"

// Inference request format
type Request struct {
	RequestID string `json:"request_id"`
	Input     string `json:"input"`
}

// Response format
type Response struct {
	RequestID string `json:"request_id"`
	Output    string `json:"output"`
	RoutedTo  string `json:"routed_to"`
}

func SendRequest(httpClient *http.Client, requestID string, input string) (string, string, error) {
	message, _ := json.Marshal(Request{RequestID: requestID, Input: input})               // create json message we are sending
	resp, err := httpClient.Post(routerURL, "application/json", bytes.NewReader(message)) // POST request to router

	if resp != nil {
		return "", "", fmt.Errorf("POST failed: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("Router returned: %d", resp.StatusCode)
	}

	var result Response
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", fmt.Errorf("Decode failed: %w", err)
	}

	return result.Output, result.RoutedTo, nil
}

func main() {

	fmt.Println("***Welcome to Conduit")
	fmt.Println("***Enter a prompt.")
	fmt.Println("***Or type 'exit' to quit.")
	fmt.Printf("\n")

	//httpCliet := &http.Client{Timeout: time.Duration(1) * time.Second}
	scanner := bufio.NewScanner(os.Stdin)
	requestCounter := 0

	for {
		fmt.Print(">>")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		//input = input.Tokenize
		if input == "" {
			continue
		} else if input == "exit" {
			fmt.Println("Terminating program ...")
			break
		}

		requestCounter++
		// requestID := ??
		/*
			output, routedTo, err := SendRequest(httpClient, requestID, input)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error: %w", err)
				continue
			}
			fmt.Printf("\n%s\n", output)
			fmt.Printf("[Routed To: %s]\n\n", routedTo)
		*/

		// Echo input for now
		fmt.Printf("\n%s\n", input) // Printf for printing strings
		fmt.Printf("\n")
	}
}
