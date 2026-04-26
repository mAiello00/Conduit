package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"sync"
)

// add to Dockerfile config
const URL string = "http://localhost:8080"
const N int = 1 // fixed number of workers in the pool - make this larger once you implemented a queue

// Each worker will be an instance of the LLM
type Worker struct {
	stdin   io.WriteCloser
	scanner bufio.Scanner
	busy    bool
	mu      sync.Mutex
}

type Pool struct {
	workers chan *Worker
}

// Incoming request format
type Request struct {
	RequestID string `json:"request_id"`
	Input     string `json:"input"`
}

// Reponse format
type Response struct {
	RequestID string `json:"request_id"`
	Input     string `json:"input"`
	RoutedTo  string `json:"routed_to"`
}

/*
Format of prompts is <START>|{ID}|{CONTENT}|<END>
*/
func BuildPrompt(id string, input string) string {
	return fmt.Sprintf("<START>|{%s}|{%s}|<END>", id, input)
}

/*
Pass the prompt to stdin and then read the response from Python
*/
func (pool *Pool) ReceiveRequest(w http.ResponseWriter, r *http.Request) {

	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	var data Request
	err := json.NewDecoder(r.Body).Decode(&data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	worker := <-pool.workers                  // get an available worker
	defer func() { pool.workers <- worker }() // put the worker back in the pool

	worker.mu.Lock()
	defer worker.mu.Unlock()

	// Format prompt and send it to stdin for the LLM
	prompt := BuildPrompt(r.FormValue("request_id"), r.FormValue("input"))
	fmt.Println(worker.stdin, prompt)

	// Read response of LLM from stdin
	var result strings.Builder
	for worker.scanner.Scan() {
		line := worker.scanner.Text()
		if line == "<END>" {
			break
		}
		result.WriteString(line)
	}

	// Create json response and send to client
	response, _ := json.Marshal(Response{RequestID: r.FormValue("request_id"), Input: result.String(), RoutedTo: "1"})
	w.Header().Set("Content-type", "application/json")
	if err := json.NewEncoder(w).Encode(&response); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

/*
Initialize experts and return pipes
*/
func InitializeExpert() (io.WriteCloser, io.ReadCloser) {
	fmt.Println("Launching Expert")

	cmd := exec.Command("python3", "expert.py")

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

	return stdin, stdout
}

/*
TODO: Close all pipes and terminate processes gracefully
*/
func Shutdown() {

}

func main() {

	fmt.Println("Initializing Expert Model...")

	// Designate the number of workers
	pool := &Pool{
		workers: make(chan *Worker, N),
	}

	// Initlalize our pool of workers
	for i := 0; i < N; i++ {
		stdin, stdout := InitializeExpert()

		w := &Worker{
			stdin:   stdin,
			scanner: *bufio.NewScanner(stdout),
		}

		pool.workers <- w
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/generate", pool.ReceiveRequest) // first parameter take a pattern string; second indicates the handler function

	fmt.Println("Server listening to :8000")
	log.Fatal(http.ListenAndServe(URL, mux)) // creates the server, log errors if any arise
}
