# Code Review

All code changes require at least one approval from a code owner before merging. For changes touching authentication, payments, or core data models, two approvals are required.

Reviewer SLA: respond within one business day. If you cannot review within that window, leave a comment indicating when you will or reassign to another code owner.

PRs should be under 400 lines of changed code where possible. Larger PRs are accepted but require a description that explains the reviewing strategy (e.g., commit-by-commit, file-by-file).

Author responsibilities: include context in the PR description, link the relevant ticket, ensure CI is green before requesting review, and respond to review comments within one business day.
