# Release notes: Backup Agent 2.5

The scheduler retries a failed upload three times before giving up. The retry
interval is two minutes and is not configurable in this release.

The restore path has been verified on Linux and Windows.

Restores of archives larger than 2 TB remain unsupported.
