build-client:
	cd router && go build -o bin/moe-client cmd/client/client.go

start-client:
	./router/bin/moe-client

# Compile and launch in one step - gets rid of binary afterwards 
run-client:
	cd router && go run cmd/client/client.go

clean:
	cd router && rm -rf bin/