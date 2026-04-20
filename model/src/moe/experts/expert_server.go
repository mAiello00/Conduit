package main

import (
	"fmt"
	"net/http"
	"os/exec"
)

const URL = "http://localhost:8000"

func ReceiveRequest(w http.ResponseWriter, r *http.Request) {

}

func Initialize_Expert() {
	fmt.Println("Launching Expert")
	exec.Command("expert.py").Run()
}

func main() {
	Initialize_Expert()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /user", ReceiveRequest)

	fmt.Println("Server listening to :8000")
	http.ListenAndServe(URL, mux) // What creates the server
}
