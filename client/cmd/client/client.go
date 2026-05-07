package main

import (
	"Conduit/postprocessor"
	"Conduit/preprocessor"
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const routerURL = "http://localhost:8081"

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

// Send to API Gateway
func SendRequest(httpClient *http.Client, requestID string, input string) (string, string, string, error) {
	message, _ := json.Marshal(Request{RequestID: requestID, Input: input})                           // create json message we are sending
	resp, err := httpClient.Post(routerURL+"/generate", "application/json", bytes.NewReader(message)) // POST request to router

	if err != nil {
		return "", "", "", fmt.Errorf("POST failed: %w\n", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", "", fmt.Errorf("Router returned: %d", resp.StatusCode)
	}

	var result Response
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", "", fmt.Errorf("Decode failed: %w", err)
	}

	return result.RequestID, result.Output, result.RoutedTo, nil
}

func main() {
	fmt.Println("***Welcome to Conduit")
	fmt.Println("***Enter a prompt.")
	fmt.Println("***Or type 'exit' to quit.")
	fmt.Printf("\n")

	httpClient := &http.Client{Timeout: time.Duration(100) * time.Second}
	scanner := bufio.NewScanner(os.Stdin)
	requestCounter := 0

	for {
		fmt.Print(">>")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" || input == "<END>" {
			continue
		} else if input == "/exit" {
			fmt.Println("Terminating program ...")
			break
		}

		if !preprocessor.ProcessInput(input) {
			break
		}

		requestCounter++
		requestID := strconv.Itoa(requestCounter)

		// Send request to Go API Gateway
		id, output, routedTo, err := SendRequest(httpClient, requestID, input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %s", err)
			continue
		}

		postprocessor.CheckResponse()

		fmt.Printf("\nRequest ID is: %s\n", id)
		fmt.Printf("Response: %s\n", output)
		fmt.Printf("[Routed To: %s]\n\n", routedTo)
	}
}
