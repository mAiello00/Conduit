build-client:
	go build -o bin/moe-client client/cmd/client/client.go

build-moe-api:
	go build -o bin/moe-api model/src/api/api_gateway.go

start-client:
	./bin/moe-client

start-moe-api:
	./bin/moe-api

# Compile and launch in one step - gets rid of binary afterwards 
run-client:
	cd client && go run cmd/client/client.go

clean:
	rm -rf bin/