package main

// The three shell completion scripts `amele completion` prints. Each is a
// plain string const, hand-written against that shell's own builtins
// (bash's compgen/complete, zsh's compsys, fish's complete) - no generator,
// no shared template, no third-party completion framework, so a static
// binary keeps completions without carrying a rendering layer for them
// (docs/engineering.md §2.1/2.2: single binary, no runtime dependency).
//
// CONTRACT: every script must stay in sync BY HAND with the command and flag
// inventory in docs/contracts/cli.md - there is no code generation step that
// would catch a drift automatically. A command or flag added to the CLI
// should get a matching line here in the same change.
const (
	completionBash = `# amele bash completion
#
# Install (pick one):
#   amele completion bash > /etc/bash_completion.d/amele
#   amele completion bash > "$(brew --prefix)/etc/bash_completion.d/amele"   # Homebrew bash-completion
#   amele completion bash >> ~/.bashrc                                      # per-user, sourced on login
#
# Hand-written against bash's own compgen/complete builtins - no
# bash-completion package required.

_amele_complete() {
	local cur prev cmd
	COMPREPLY=()
	cur="${COMP_WORDS[COMP_CWORD]}"
	prev="${COMP_WORDS[COMP_CWORD-1]}"

	local commands="run chat validate explain schema init version completion help"
	local shells="bash zsh fish"
	local agent_flags="--model --set -w --workspace -q --quiet -v --verbose -h --help"
	local inspect_flags="--set -w --workspace -h --help"

	if [[ $COMP_CWORD -eq 1 ]]; then
		COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
		return 0
	fi

	cmd="${COMP_WORDS[1]}"

	# A --set pair, --model or -w/--workspace takes a free-form argument
	# (key=value, a model name, a directory) that no static list can predict.
	case "$prev" in
		--set|--model|-w|--workspace)
			return 0
			;;
	esac

	case "$cmd" in
		run|chat)
			if [[ $COMP_CWORD -eq 2 ]]; then
				COMPREPLY=( $(_amele_yaml_files "$cur") )
				return 0
			fi
			COMPREPLY=( $(compgen -W "$agent_flags" -- "$cur") )
			;;
		validate|explain)
			if [[ $COMP_CWORD -eq 2 ]]; then
				COMPREPLY=( $(_amele_yaml_files "$cur") )
				return 0
			fi
			COMPREPLY=( $(compgen -W "$inspect_flags" -- "$cur") )
			;;
		init)
			if [[ $COMP_CWORD -eq 2 ]]; then
				COMPREPLY=( $(compgen -f -- "$cur") )
			fi
			;;
		completion)
			if [[ $COMP_CWORD -eq 2 ]]; then
				COMPREPLY=( $(compgen -W "$shells" -- "$cur") )
			fi
			;;
		help)
			if [[ $COMP_CWORD -eq 2 ]]; then
				COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
			fi
			;;
	esac
}

# _amele_yaml_files completes to YAML files (and directories, so a user can
# keep descending into them) in the config-path slot. Written by hand instead
# of an extglob compgen -X pattern so it works with extglob left at its bash
# default (off).
_amele_yaml_files() {
	local cur="$1" f
	while IFS= read -r f; do
		if [[ -d "$f" || "$f" == *.yaml || "$f" == *.yml ]]; then
			printf '%s\n' "$f"
		fi
	done < <(compgen -f -- "$cur")
}

complete -F _amele_complete amele
`

	completionZsh = `#compdef amele
# amele zsh completion
#
# Install (pick one):
#   amele completion zsh > "${fpath[1]}/_amele"
#   mkdir -p ~/.zsh/completions && amele completion zsh > ~/.zsh/completions/_amele
#       (then add ~/.zsh/completions to fpath, before compinit runs)
#
# Hand-written against zsh's own _describe/_files/_values builtins - no
# completion framework required.

_amele() {
	local -a commands
	commands=(
		'run:Run the agent once on a task and print the answer'
		'chat:Talk to the same agent interactively'
		'validate:Check a config and report every violation'
		'explain:Dry-run report of tools, permissions and budgets'
		'schema:Print the config JSON Schema'
		'init:Write an annotated starter config'
		'version:Print version, commit and build date'
		'completion:Print a shell completion script'
		'help:Print this text, or the detailed page for one command'
	)

	if (( CURRENT == 2 )); then
		_describe 'command' commands
		return
	fi

	local cmd="${words[2]}"
	local -a agent_flags inspect_flags
	agent_flags=('--model' '--set' '-w' '--workspace' '-q' '--quiet' '-v' '--verbose' '-h' '--help')
	inspect_flags=('--set' '-w' '--workspace' '-h' '--help')

	case "$cmd" in
		run|chat)
			if (( CURRENT == 3 )); then
				# One -g per extension: passing both globs inside a single
				# quoted argument makes the space part of the pattern in older
				# zsh, so no config file matched. Repeated -g accumulate.
				_files -g '*.yaml' -g '*.yml'
			else
				_describe 'flag' agent_flags
			fi
			;;
		validate|explain)
			if (( CURRENT == 3 )); then
				# One -g per extension: passing both globs inside a single
				# quoted argument makes the space part of the pattern in older
				# zsh, so no config file matched. Repeated -g accumulate.
				_files -g '*.yaml' -g '*.yml'
			else
				_describe 'flag' inspect_flags
			fi
			;;
		init)
			_files
			;;
		completion)
			if (( CURRENT == 3 )); then
				_values 'shell' bash zsh fish
			fi
			;;
		help)
			if (( CURRENT == 3 )); then
				_describe 'command' commands
			fi
			;;
	esac
}

_amele "$@"
`

	completionFish = `# amele fish completion
#
# Install:
#   amele completion fish > ~/.config/fish/completions/amele.fish
#
# Hand-written against fish's own complete builtin - no completion framework
# required.

set -l amele_commands run chat validate explain schema init version completion help

complete -c amele -f

# The config-path slot is exactly the first argument after the subcommand.
# Without this guard the YAML/directory completions below would keep firing
# while the user completes task text or flag values (the subcommand check
# alone stays true for the rest of the command line).
function __amele_config_slot
    test (count (commandline -opc)) -eq 2
end

complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a run -d "Run the agent once on a task and print the answer"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a chat -d "Talk to the same agent interactively"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a validate -d "Check a config and report every violation"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a explain -d "Dry-run report of tools, permissions and budgets"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a schema -d "Print the config JSON Schema"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a init -d "Write an annotated starter config"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a version -d "Print version, commit and build date"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a completion -d "Print a shell completion script"
complete -c amele -n "not __fish_seen_subcommand_from $amele_commands" -a help -d "Print this text, or the detailed page for one command"

# Config-path slot: YAML files and directories (pack shorthand).
complete -c amele -n "__fish_seen_subcommand_from run chat validate explain; and __amele_config_slot" -k -a "(__fish_complete_directories)"
complete -c amele -n "__fish_seen_subcommand_from run chat validate explain; and __amele_config_slot" -k -a "(__fish_complete_suffix .yaml)"
complete -c amele -n "__fish_seen_subcommand_from run chat validate explain; and __amele_config_slot" -k -a "(__fish_complete_suffix .yml)"

complete -c amele -n "__fish_seen_subcommand_from run chat" -l model -d "Shortcut for --set model=MODEL"
complete -c amele -n "__fish_seen_subcommand_from run chat validate explain" -l set -d "Override one config field: key=value"
complete -c amele -n "__fish_seen_subcommand_from run chat validate explain" -s w -l workspace -d "Shortcut for --set workspace=DIR"
complete -c amele -n "__fish_seen_subcommand_from run chat" -s q -l quiet -d "Suppress the summary line and non-error notes"
complete -c amele -n "__fish_seen_subcommand_from run chat" -s v -l verbose -d "Print a progress line per loop event to stderr"
complete -c amele -n "__fish_seen_subcommand_from $amele_commands" -s h -l help -d "Print the detailed help page"

complete -c amele -n "__fish_seen_subcommand_from init" -a "(__fish_complete_suffix .yaml)"

complete -c amele -n "__fish_seen_subcommand_from completion" -a "bash zsh fish" -d "Shell"
complete -c amele -n "__fish_seen_subcommand_from help" -a "$amele_commands"
`
)

// completionScripts maps a shell name to its script. A map (like helpPages)
// so `amele completion <name>` and its error path share one source of truth
// for which shells are supported.
var completionScripts = map[string]string{
	"bash": completionBash,
	"zsh":  completionZsh,
	"fish": completionFish,
}
