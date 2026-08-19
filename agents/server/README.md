# Server Agent

The Server Agent has a read-only Linux and service diagnostic baseline. Changes
are deliberately not autonomous: every `sudo` or state-changing action requires
human approval with the exact command, impact, rollback, and validation plan.

Remote diagnostics are disabled by default because
[allowed-ssh-hosts.example](allowed-ssh-hosts.example) contains no hosts. Copy
it to the ignored local `allowed-ssh-hosts` file and add SSH config aliases only
after human approval. See [instructions.md](instructions.md) and
[evaluation-tasks.md](evaluation-tasks.md).