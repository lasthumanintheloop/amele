# amele build targets. CI runs exactly these targets so local and CI results
# never diverge.

# Budgets enforced by `make budget` (docs/engineering.md §8).
SIZE_BUDGET_BYTES := 14680064  # 14 MB hard ceiling (raised from 10 MB for the MCP SDK, 2026-08-19)
COVER_BUDGET      := 80        # minimum % statement coverage over internal/

GOFLAGS := -trimpath
LDFLAGS := -s -w -X main.version=$(shell git describe --tags --always --dirty) -X main.commit=$(shell git rev-parse --short HEAD) -X main.date=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: build test race cover lint lint-clean fmt budget dist clean all

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
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	elif [ -x "$$(go env GOPATH)/bin/golangci-lint" ]; then \
		"$$(go env GOPATH)/bin/golangci-lint" run ./...; \
	else \
		echo "golangci-lint not installed locally (checked PATH and $$(go env GOPATH)/bin); CI enforces it"; \
	fi

# lint-clean drops golangci-lint's result cache before linting. The cache is
# keyed by package path, so after a worktree under this checkout is removed
# (or moved) a plain `make lint` can replay that worktree's stale findings
# under paths that no longer exist; this target is the fix for that.
lint-clean:
	@if command -v golangci-lint >/dev/null 2>&1; then golangci-lint cache clean; \
	elif [ -x "$$(go env GOPATH)/bin/golangci-lint" ]; then "$$(go env GOPATH)/bin/golangci-lint" cache clean; fi
	@$(MAKE) --no-print-directory lint

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
# Every target that builds and whose unix/windows code paths are exercised
# (README "Runs everywhere" lists the tested tier vs the built tier).
PLATFORMS := linux/amd64 linux/arm64 linux/arm linux/386 linux/riscv64 linux/ppc64le linux/s390x linux/mips64le \
             darwin/amd64 darwin/arm64 \
             windows/amd64 windows/arm64 windows/386 \
             freebsd/amd64 freebsd/arm64 openbsd/amd64 openbsd/arm64 netbsd/amd64 dragonfly/amd64 \
             illumos/amd64 android/arm64
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
