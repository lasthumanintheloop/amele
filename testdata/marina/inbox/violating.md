# Release notes: Backup Agent 2.4

We are shipping a revolutionary backup agent this week!

The scheduler now retries a failed upload three times before giving up. The
retry interval is two minutes and is not configurable in this release.

Our support team has verified the new restore path on Linux and Windows.

Restores of archives larger than 2 TB remain unsupported.
