package main

import (
	"fmt"
	"regexp"
)

func Tokenize(input string) []string {
	reg := regexp.MustCompile(`^[a-z]+\[[0-9]+\]$`)
	tokens := reg.Split(input, -1)
	fmt.Printf("\n%s\n", tokens)

	return tokens
}
