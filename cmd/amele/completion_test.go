package main

import (
	"os/exec"
	"strings"
	"testing"
)

// completionShellsUnderTest is every shell `amele completion` must answer
// for, in the order the help page's EXAMPLES should mention them.
var completionShellsUnderTest = []string{"bash", "zsh", "fish"}

// TestCompletionCommand pins the happy path: one static script per shell, on
// stdout, exit 0, nothing on stderr.
func TestCompletionCommand(t *testing.T) {
	for _, shell := range completionShellsUnderTest {
		t.Run(shell, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, []string{"completion", shell}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if stderr != "" {
				t.Errorf("unexpected stderr: %q", stderr)
			}
			if strings.TrimSpace(stdout) == "" {
				t.Fatal("stdout is empty")
			}
			if !strings.HasSuffix(stdout, "\n") {
				t.Error("script is not newline-terminated")
			}
			for _, name := range helpCommands {
				if name == "completion" {
					continue // the script names the OTHER commands as completable arguments
				}
				if !strings.Contains(stdout, name) {
					t.Errorf("%s script never mentions command %q", shell, name)
				}
			}
		})
	}
}

// TestCompletionNoArg pins `amele completion` with no shell name as a usage
// error, not a default guess: a script that forgets the argument must fail
// loudly instead of silently getting the wrong shell's syntax.
func TestCompletionNoArg(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"completion"}, "")
	if code != ExitConfigError {
		t.Errorf("exit %d, want %d", code, ExitConfigError)
	}
	if stdout != "" {
		t.Errorf("unexpected stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "usage") {
		t.Errorf("stderr = %q, want a usage message", stderr)
	}
}

// TestCompletionUnknownShell pins an unrecognized shell name as a usage
// error naming the shells that ARE supported, rather than a silent no-op.
func TestCompletionUnknownShell(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"completion", "tcsh"}, "")
	if code != ExitConfigError {
		t.Errorf("exit %d, want %d", code, ExitConfigError)
	}
	if stdout != "" {
		t.Errorf("unexpected stdout: %q", stdout)
	}
	if !strings.Contains(stderr, `"tcsh"`) {
		t.Errorf("stderr = %q, want the bad shell name quoted", stderr)
	}
	for _, shell := range completionShellsUnderTest {
		if !strings.Contains(stderr, shell) {
			t.Errorf("stderr = %q, want it to name the supported shell %q", stderr, shell)
		}
	}
}

// TestCompletionExtraArgs pins that `completion` takes exactly one argument:
// a second one is a usage error, the same stance `schema`/`version` take.
func TestCompletionExtraArgs(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"completion", "bash", "zsh"}, "")
	if code != ExitConfigError {
		t.Errorf("exit %d, want %d", code, ExitConfigError)
	}
	if stdout != "" {
		t.Errorf("unexpected stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "usage") {
		t.Errorf("stderr = %q, want a usage message", stderr)
	}
}

// TestCompletionBashSyntax pipes the generated bash script through `bash -n`
// (parse-only, does not execute it) so a hand-written script cannot ship a
// typo that would only surface the first time a user sources it. Skips
// gracefully when bash is not on PATH (e.g. a minimal CI image), per the plan.
func TestCompletionBashSyntax(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash not found on PATH")
	}
	_, stdout, _ := execCLI(t, []string{"completion", "bash"}, "")
	cmd := exec.Command(bash, "-n", "-") //nolint:gosec // G204: bash is resolved via exec.LookPath above; the rest is constant.
	cmd.Stdin = strings.NewReader(stdout)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("bash -n rejected the completion script: %v\n%s", err, out)
	}
}

// TestCompletionZshFishStatic is the "best-effort" check the plan calls for
// on the two shells without a syntax checker guaranteed to be on PATH: the
// script is non-empty and names the real subcommands.
func TestCompletionZshFishStatic(t *testing.T) {
	for _, shell := range []string{"zsh", "fish"} {
		t.Run(shell, func(t *testing.T) {
			_, stdout, _ := execCLI(t, []string{"completion", shell}, "")
			if strings.TrimSpace(stdout) == "" {
				t.Fatalf("%s script is empty", shell)
			}
			for _, name := range []string{"run", "chat", "validate", "explain", "schema", "init", "version", "completion", "mcp", "help"} {
				if !strings.Contains(stdout, name) {
					t.Errorf("%s script never mentions command %q", shell, name)
				}
			}
		})
	}
}

// TestCompletionZshConfigGlobs pins how the zsh script asks for YAML files.
// Regression: it passed '*.yaml *.yml' as ONE quoted argument to _files, so
// the embedded space was part of the pattern rather than a separator between
// two patterns. Each extension must be its own -g pattern.
func TestCompletionZshConfigGlobs(t *testing.T) {
	_, stdout, _ := execCLI(t, []string{"completion", "zsh"}, "")

	if strings.Contains(stdout, `'*.yaml *.yml'`) {
		t.Error("zsh script still passes both globs as one space-joined pattern")
	}
	for _, want := range []string{`-g '*.yaml'`, `-g '*.yml'`} {
		if !strings.Contains(stdout, want) {
			t.Errorf("zsh script is missing the separate pattern %s", want)
		}
	}
	// Every config-taking command group (run|chat, validate|explain and the
	// mcp subcommands) must carry the fix, not just the first one.
	if got := strings.Count(stdout, `_files -g '*.yaml' -g '*.yml'`); got != 3 {
		t.Errorf("config-file completion appears %d times, want 3", got)
	}
}

// TestCompletionZshSyntax syntax-checks the zsh script when a zsh is around,
// mirroring TestCompletionBashSyntax. Skipped otherwise: zsh is not a build
// dependency of this project.
func TestCompletionZshSyntax(t *testing.T) {
	zsh, err := exec.LookPath("zsh")
	if err != nil {
		t.Skip("zsh not found on PATH")
	}
	_, stdout, _ := execCLI(t, []string{"completion", "zsh"}, "")
	cmd := exec.Command(zsh, "-n") //nolint:gosec // G204: zsh is resolved via exec.LookPath above; the rest is constant.
	cmd.Stdin = strings.NewReader(stdout)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("zsh -n rejected the completion script: %v\n%s", err, out)
	}
}

// TestCompletionScriptsAreDistinct guards against a copy-paste mistake where
// two shells end up serving the same script - each has a distinct syntax, so
// that would be silently broken for at least one of them.
func TestCompletionScriptsAreDistinct(t *testing.T) {
	seen := make(map[string]string, len(completionShellsUnderTest))
	for _, shell := range completionShellsUnderTest {
		_, stdout, _ := execCLI(t, []string{"completion", shell}, "")
		for otherShell, otherScript := range seen {
			if otherScript == stdout {
				t.Errorf("%s and %s produced the identical script", shell, otherShell)
			}
		}
		seen[shell] = stdout
	}
}

// TestFishCompletionOffersDirectories: the config slot accepts directories
// (pack shorthand), so fish must not filter completions down to .yaml/.yml.
func TestFishCompletionOffersDirectories(t *testing.T) {
	if !strings.Contains(completionFish, "__fish_complete_directories") {
		t.Error("fish completion does not offer directories in the config-path slot")
	}
}

// TestFishConfigSlotIsPositionScoped: the YAML/directory completions belong
// to the config-path slot only (the first argument after the subcommand).
// Regression: with a bare __fish_seen_subcommand_from condition they kept
// firing while completing task text and flag values.
func TestFishConfigSlotIsPositionScoped(t *testing.T) {
	if !strings.Contains(completionFish, "__amele_config_slot") {
		t.Error("fish completion does not scope config-path completions to the config argument position")
	}
}

// TestCompletionResumeIsRunOnly: --resume is registered on both agent
// commands so `chat --resume x` can be answered with what is actually wrong
// (cmdChat refuses it), but completing it under `chat` would advertise that
// error as a feature. Every shell must therefore offer it for `run` alone.
//
// The assertion is per LINE rather than on a whole script: a completion rule
// names its command on the same line as the flag it offers, in all three
// shells, so "a line that mentions resume must not mention chat" is the shape
// of the claim. Comments are skipped - they explain the rule, they are not it.
func TestCompletionResumeIsRunOnly(t *testing.T) {
	for _, shell := range completionShellsUnderTest {
		t.Run(shell, func(t *testing.T) {
			_, script, _ := execCLI(t, []string{"completion", shell}, "")
			var offered bool
			for _, line := range strings.Split(script, "\n") {
				if strings.HasPrefix(strings.TrimSpace(line), "#") || !strings.Contains(line, "resume") {
					continue
				}
				offered = true
				if strings.Contains(line, "chat") {
					t.Errorf("%s offers --resume where chat is completed too: %s", shell, line)
				}
			}
			if !offered {
				t.Errorf("%s script never offers --resume", shell)
			}
		})
	}
}

// TestCompletionResumeTakesAnArgument: the flag's value is a session log
// path. A completer that offered the flag list again after --resume would
// suggest another flag where a file belongs.
func TestCompletionResumeTakesAnArgument(t *testing.T) {
	_, bashScript, _ := execCLI(t, []string{"completion", "bash"}, "")
	if !strings.Contains(bashScript, "--resume)") {
		t.Error("bash script has no case arm for the value that follows --resume")
	}
	if !strings.Contains(completionFish, `-l resume -r `) {
		t.Error("fish script does not mark --resume as requiring an argument (-r)")
	}
	// zsh has no equivalent of fish's -r, so the value slot is a rule of its
	// own: one line that hands the previous-word check to _files. Asserted at
	// string level, like the bash arm above - the scripts are hand-written and
	// there is no shell to run them in here.
	var zshArm bool
	for _, line := range strings.Split(completionZsh, "\n") {
		if strings.Contains(line, "--resume") && strings.Contains(line, "_files") {
			zshArm = true
		}
	}
	if !zshArm {
		t.Error("zsh script does not complete a file where --resume's value goes")
	}
}
