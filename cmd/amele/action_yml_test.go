package main

import (
	"os"
	"testing"

	goyaml "gopkg.in/yaml.v3"
)

// actionSpec mirrors the subset of the GitHub composite-action schema that
// the repo-root action.yml must satisfy. The test pins the action's contract
// (inputs, output, composite shape) without executing it, because composite
// actions only run inside the Actions runner.
type actionSpec struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Inputs      map[string]struct {
		Description string `yaml:"description"`
		Required    bool   `yaml:"required"`
	} `yaml:"inputs"`
	Outputs map[string]struct {
		Description string `yaml:"description"`
		Value       string `yaml:"value"`
	} `yaml:"outputs"`
	Runs struct {
		Using string `yaml:"using"`
		Steps []struct {
			Name  string `yaml:"name"`
			Uses  string `yaml:"uses"`
			Run   string `yaml:"run"`
			Shell string `yaml:"shell"`
		} `yaml:"steps"`
	} `yaml:"runs"`
}

// loadActionSpec reads and parses the repo-root action.yml, failing the test
// on any I/O or YAML error.
func loadActionSpec(t *testing.T) actionSpec {
	t.Helper()
	raw, err := os.ReadFile("../../action.yml")
	if err != nil {
		t.Fatalf("reading action.yml: %v", err)
	}
	var spec actionSpec
	if err := goyaml.Unmarshal(raw, &spec); err != nil {
		t.Fatalf("parsing action.yml: %v", err)
	}
	return spec
}

// TestActionYAMLMetadata asserts the action declares a name and description
// (both are mandatory in the actions metadata schema).
func TestActionYAMLMetadata(t *testing.T) {
	spec := loadActionSpec(t)
	if spec.Name == "" {
		t.Error("action.yml: name is empty")
	}
	if spec.Description == "" {
		t.Error("action.yml: description is empty")
	}
}

// TestActionYAMLInputs asserts the inputs are exactly {config, task, model}
// with only config required. A fourth input (or a dropped one) is a contract
// change.
func TestActionYAMLInputs(t *testing.T) {
	spec := loadActionSpec(t)
	wantInputs := map[string]bool{ // name -> required
		"config": true,
		"task":   false,
		"model":  false,
	}
	for name, required := range wantInputs {
		in, ok := spec.Inputs[name]
		if !ok {
			t.Errorf("action.yml: input %q missing", name)
			continue
		}
		if in.Required != required {
			t.Errorf("action.yml: input %q required = %v, want %v", name, in.Required, required)
		}
		if in.Description == "" {
			t.Errorf("action.yml: input %q has no description", name)
		}
	}
	for name := range spec.Inputs {
		if _, ok := wantInputs[name]; !ok {
			t.Errorf("action.yml: unexpected input %q", name)
		}
	}
}

// TestActionYAMLCompositeSteps asserts the composite-action requirements
// GitHub enforces only at runtime: runs.using is "composite" and every step
// with a run: script names its shell. Catching them here beats catching them
// at first workflow use.
func TestActionYAMLCompositeSteps(t *testing.T) {
	spec := loadActionSpec(t)
	if spec.Runs.Using != "composite" {
		t.Errorf("action.yml: runs.using = %q, want %q", spec.Runs.Using, "composite")
	}
	if len(spec.Runs.Steps) == 0 {
		t.Error("action.yml: runs.steps is empty")
	}
	for i, step := range spec.Runs.Steps {
		if step.Run != "" && step.Shell != "bash" {
			t.Errorf("action.yml: step %d (%q) has run: but shell = %q, want %q", i, step.Name, step.Shell, "bash")
		}
	}
}

// TestActionYAMLAnswerOutput asserts the answer output workflows consume is
// declared and wired to a step output.
func TestActionYAMLAnswerOutput(t *testing.T) {
	spec := loadActionSpec(t)
	out, ok := spec.Outputs["answer"]
	if !ok {
		t.Fatal("action.yml: output \"answer\" missing")
	}
	if out.Value == "" {
		t.Error("action.yml: output \"answer\" has no value mapping")
	}
	if out.Description == "" {
		t.Error("action.yml: output \"answer\" has no description")
	}
}
