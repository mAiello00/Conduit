package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"
)

// add to config
const URL string = "localhost:8081"
const llmURL string = "http://localhost:8082"

// Used for logging
var logger *slog.Logger

type Circuit func(context.Context) (string, error)

var llmURLs = []string{
	"http://localhost:8082",
	"http://localhost:8083",
	"http://localhost:8084",
}

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

// TODO
func LogInfo() {

}

/*
Receive the request and forward to our only Python microservice
*/
func ReceiveRequest(w http.ResponseWriter, r *http.Request) {
	//TODO: have a logger function to clean up this section
	start := time.Now()
	modelRoutedTo := "Model-1"

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
	requestSize := len(bodyBytes)

	// Parse request body to get the RequestID
	var req Request
	err = json.Unmarshal(bodyBytes, &req)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
	}

	httpClient := &http.Client{Timeout: time.Duration(60) * time.Second}

	// TODO: Implement load balancing

	// Forward to the LLM microservice
	result, err := httpClient.Post(llmURL+"/generate", "application/json", bytes.NewReader(bodyBytes))
	latency := time.Since(start).Milliseconds()

	if err != nil {
		logger.Error(
			"upstream request failed",
			"request_id", req.RequestID,
			"method", r.Method,
			"path", r.URL.Path,
			"model_routed_to", modelRoutedTo,
			"upstream_status", 0,
			"latency_ms", latency,
			"error", err.Error(),
			"request_size", requestSize,
			"response_size", 0,
		)
		http.Error(w, "LLM Service unreachable", http.StatusBadGateway)
		return
	}
	defer result.Body.Close()

	llmResult, err := io.ReadAll(result.Body)
	if err != nil {
		logger.Error(
			"failed reading upstream response",
			"request_id", req.RequestID,
			"method", r.Method,
			"path", r.URL.Path,
			"model_routed_to", modelRoutedTo,
			"upstream_status", result.StatusCode,
			"latency_ms", latency,
			"error", err.Error(),
			"request_size", requestSize,
			"response_size", 0,
		)
		http.Error(w, "Failed to read the response", http.StatusInternalServerError)
		return
	}

	responseSize := len(llmResult)

	logger.Info(
		"request completed",
		"request_id", req.RequestID,
		"method", r.Method,
		"path", r.URL.Path,
		"model_routed_to", modelRoutedTo,
		"upstream_status", result.StatusCode,
		"latency_ms", latency,
		"error", "",
		"request_size", requestSize,
		"response_size", responseSize,
	)

	// Parse request body for the generated message
	var llmResponse Response
	err = json.Unmarshal(llmResult, &llmResponse)
	if err != nil {
		logger.Error(
			"failed to read llm response body",
			"request_id", req.RequestID,
			"method", r.Method,
			"path", r.URL.Path,
			"model_routed_to", modelRoutedTo,
			"upstream_status", result.StatusCode,
			"latency_ms", latency,
			"error", "",
			"request_size", requestSize,
			"response_size", responseSize,
		)
		http.Error(w, "Faild to read LLM response body", http.StatusBadRequest)
		return
	}

	finalResult := Response{
		RequestID: req.RequestID,
		Output:    llmResponse.Output,
		RoutedTo:  "Model-1",
	}

	// Create json response and send to client
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(finalResult)
}

func Breaker(circuit Circuit, failureThreshold uint) Circuit {
	var consecutiveFailures int = 0
	var lastAttempt = time.Now()
	var m sync.RWMutex

	return func(ctx context.Context) (string, error) {
		m.RLock()
		d := consecutiveFailures - int(failureThreshold)

		if d >= 0 {
			shouldRetryAt := lastAttempt.Add(time.Second * 2 << d)
			if !time.Now().After(shouldRetryAt) {
				m.RUnlock()
				return "", errors.New("service unreachable")
			}
		}

		m.RUnlock()

		response, err := circuit(ctx) // Issue request proper

		m.Lock()
		defer m.Unlock()

		lastAttempt = time.Now()
		if err != nil {
			consecutiveFailures++
			return response, err
		}

		consecutiveFailures = 0

		return response, nil
	}
}

/*
TODO: Shutdown gracefully
*/
func Shutdown() {

}

func main() {

	logger = slog.New(slog.NewJSONHandler(os.Stdout, nil))

	mux := http.NewServeMux()
	mux.HandleFunc("/generate", ReceiveRequest) // first parameter take a pattern string; second indicates the handler function

	log.Fatal(http.ListenAndServe(URL, mux)) // creates the server, log errors if any arise
}
