# amele build targets. CI runs exactly these targets so local and CI results
# never diverge.

# Budgets enforced by `make budget` (docs/engineering.md §8).
SIZE_BUDGET_BYTES := 10485760  # 10 MB hard ceiling
COVER_BUDGET      := 80        # minimum % statement coverage over internal/

GOFLAGS := -trimpath
LDFLAGS := -s -w -X main.version=$(shell git describe --tags --always --dirty) -X main.commit=$(shell git rev-parse --short HEAD) -X main.date=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: build test race cover lint fmt budget clean all

all: fmt lint test race cover budget

build:
	CGO_ENABLED=0 go build $(GOFLAGS) -ldflags="$(LDFLAGS)" -o amele ./cmd/amele

test:
	go test ./...

race:
	go test -race ./...

# -coverpkg pins the denominator to internal/ (the docs/engineering.md §6 scope): a
# plain ./... total would let well-covered cmd/ wiring mask a coverage drop
# inside the packages that matter.
cover:
	go test -coverprofile=coverage.out -coverpkg=./internal/... ./... > /dev/null
	@go tool cover -func=coverage.out | tail -1
	@total=$$(go tool cover -func=coverage.out | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1); \
	ok=$$(echo "$$total >= $(COVER_BUDGET)" | bc); \
	if [ "$$ok" != "1" ]; then echo "FAIL: coverage $$total% < $(COVER_BUDGET)%"; exit 1; fi

# fmt fails (rather than rewrites) in CI so unformatted code cannot merge.
fmt:
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then echo "gofmt needed:"; echo "$$unformatted"; exit 1; fi

lint:
	go vet ./...
	@command -v golangci-lint >/dev/null && golangci-lint run ./... || echo "golangci-lint not installed locally; CI enforces it"

# budget builds the release binary and enforces the size ceiling.
budget: build
	@size=$$(stat -c %s amele 2>/dev/null || stat -f %z amele); \
	echo "binary size: $$size bytes (budget $(SIZE_BUDGET_BYTES))"; \
	if [ "$$size" -gt "$(SIZE_BUDGET_BYTES)" ]; then echo "FAIL: binary exceeds size budget"; exit 1; fi

clean:
	rm -f amele coverage.out
