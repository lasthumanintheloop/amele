# amele build targets. CI runs exactly these targets so local and CI results
# never diverge.

# Budgets enforced by `make budget` (docs/engineering.md §8).
SIZE_BUDGET_BYTES := 10485760  # 10 MB hard ceiling
COVER_BUDGET      := 80        # minimum % statement coverage over internal/

GOFLAGS := -trimpath
LDFLAGS := -s -w -X main.version=$(shell git describe --tags --always --dirty) -X main.commit=$(shell git rev-parse --short HEAD) -X main.date=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: build test race cover lint fmt budget dist clean all

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

# dist cross-compiles the release archives into dist/: one archive per
# platform (binary + LICENSE + README.md) and a SHA256SUMS file over them.
# VERSION defaults to the tag; the release workflow runs exactly this target
# so a local `make dist` reproduces what a tag ships (up to the build date).
VERSION   ?= $(shell git describe --tags --always --dirty | sed 's/^v//')
PLATFORMS := linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64
dist:
	rm -rf dist && mkdir -p dist
	@set -e; for p in $(PLATFORMS); do \
	  os=$${p%/*}; arch=$${p#*/}; ext=; [ "$$os" = windows ] && ext=.exe; \
	  stage=dist/stage/amele_$(VERSION)_$${os}_$${arch}; mkdir -p $$stage; \
	  CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build $(GOFLAGS) -ldflags="$(LDFLAGS)" -o $$stage/amele$$ext ./cmd/amele; \
	  cp LICENSE README.md $$stage/; \
	  if [ "$$os" = windows ]; then (cd $$stage && zip -q -X ../../$$(basename $$stage).zip amele.exe LICENSE README.md); \
	  else tar -C $$stage -czf dist/$$(basename $$stage).tar.gz amele LICENSE README.md; fi; \
	  echo "built $$(basename $$stage)"; \
	done
	rm -rf dist/stage
	cd dist && sha256sum *.tar.gz *.zip > SHA256SUMS && cat SHA256SUMS

clean:
	rm -rf amele coverage.out dist
