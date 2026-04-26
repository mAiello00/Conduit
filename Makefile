build-client:
	go build -o bin/moe-client router/cmd/client/client.go

build-expert-server:
	go build -o bin/expert_server model/src/moe/experts/expert_server.go

start-client:
	./bin/moe-client

# Compile and launch in one step - gets rid of binary afterwards 
run-client:
	cd router && go run cmd/client/client.go

clean:
	rm -rf bin/