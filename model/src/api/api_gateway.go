package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"
)

// add to config
const URL string = "localhost:8081"
const llmURL string = "http://localhost:8082"

// Incoming request format
type Request struct {
	RequestID string `json:"request_id"`
	Input     string `json:"input"`
}

// Reponse format
type Response struct {
	RequestID string `json:"request_id"`
	Output    string `json:"output"`
	RoutedTo  string `json:"routed_to"`
}

/*
Receive the request and forward to our only Python microservice
*/
func ReceiveRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read body from the buffer
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
	}
	defer r.Body.Close()

	// Parse request body to get the RequestID
	var req Request
	err = json.Unmarshal(bodyBytes, &req)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
	}

	httpClient := &http.Client{Timeout: time.Duration(60) * time.Second}

	// TODO: Implement load balancing after scaling microservices

	// Forward to the LLM microservice
	result, err := httpClient.Post(llmURL+"/generate", "application/json", bytes.NewReader(bodyBytes))

	if err != nil {
		log.Printf("Python service error: %v", err)
		http.Error(w, "LLM Service unreachable", http.StatusBadGateway)
		return
	}
	defer result.Body.Close()

	llmResult, err := io.ReadAll(result.Body)
	if err != nil {
		http.Error(w, "Failed to read the response", http.StatusInternalServerError)
		return
	}

	// Parse request body for the generated message
	var llmResponse Response
	err = json.Unmarshal(llmResult, &llmResponse)
	if err != nil {
		http.Error(w, "Faild to read LLM response body", http.StatusBadRequest)
	}

	finalResult := Response{
		RequestID: req.RequestID,
		Output:     llmResponse.Output,
		RoutedTo:  "Model-1",
	}

	// Create json response and send to client
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(finalResult)
}

/*
TODO: Shutdown gracefully
*/
func Shutdown() {

}

func main() {

	mux := http.NewServeMux()
	mux.HandleFunc("/generate", ReceiveRequest) // first parameter take a pattern string; second indicates the handler function

	log.Fatal(http.ListenAndServe(URL, mux)) // creates the server, log errors if any arise
}
