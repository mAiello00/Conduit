package main

import (
	"bufio" // Used for Scanner
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
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

// TODO: Send to the model we have chosen using the gate
func SendRequest(httpClient *http.Client, requestID string, input string) (string, string, string, error) {
	message, _ := json.Marshal(Request{RequestID: requestID, Input: input})               // create json message we are sending
	resp, err := httpClient.Post(routerURL, "application/json", bytes.NewReader(message)) // POST request to router

	if err != nil {
		return "", "", "", fmt.Errorf("POST failed: %w", err)
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

func StartGate() (*io.WriteCloser, *io.ReadCloser) {
	fmt.Println("Setting up Gate...")
	cmd := exec.Command("python3", "../../../model/src/moe/gate/gate.py")

	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}

	err = cmd.Start()
	if err != nil {
		log.Fatal(err)
	}
	return &stdin, &stdout
}

func ChooseModel(stdin *io.WriteCloser, stdout *io.ReadCloser) int {
	return 1
}

func main() {

	// TODO: Load the gating network
	//stdin, stdout := StartGate()

	fmt.Println("***Welcome to Conduit")
	fmt.Println("***Enter a prompt.")
	fmt.Println("***Or type 'exit' to quit.")
	fmt.Printf("\n")

	httpClient := &http.Client{Timeout: time.Duration(1) * time.Second}
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

		requestCounter++
		requestID := strconv.Itoa(requestCounter)

		// TODO: Decide which LLM to use with the Gating Network
		//model := ChooseModel(stdin, stdout)

		// Send request to LLM
		id, output, routedTo, err := SendRequest(httpClient, requestID, input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %s", err)
			continue
		}
		fmt.Printf("\nRequest ID is: %s\n", id)
		fmt.Printf("Response: %s\n", output)
		fmt.Printf("[Routed To: %s]\n\n", routedTo)
	}
}
