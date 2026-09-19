# XGrammar Governance

XGrammar is an open-source project maintained by its contributor community. This document describes who is responsible for the project, how decisions are made, and how maintainers join or leave.

## Roles and responsibilities

### Contributors

Anyone can contribute code, tests, documentation, bug reports, ideas, or reviews. Contributors do not need a formal role to participate in public technical discussions. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution and review process.

### Core maintainers

Core maintainers, also called **Community Committers**, share responsibility for technical direction, reviewing contributions, helping contributors, project infrastructure, and releases. Maintainers with the relevant repository or package permissions carry out merges, releases, and access changes.

The current core maintainers are:

- [@Ubospica](https://github.com/Ubospica)
- [@Seven-Streams](https://github.com/Seven-Streams)
- [@izhuhaoran](https://github.com/izhuhaoran)
- [@ushiromiya-lion](https://github.com/ushiromiya-lion)
- [@DarkSharpness](https://github.com/DarkSharpness)
- [@CharlieFRuan](https://github.com/CharlieFRuan)

This is the project-wide maintainer roster. [CODEOWNERS](CODEOWNERS) records review responsibilities for specific files and directories. Maintainers coordinate work across those areas and keep both records up to date when responsibilities change.

## Technical decisions

Routine changes follow [CONTRIBUTING.md](CONTRIBUTING.md): a pull request receives at least one approval from a designated code owner or project lead, automated checks pass, and a Community Committer merges it.

Significant changes to architecture, public APIs, compatibility, or project direction should be discussed in a public GitHub issue or pull request. The discussion should explain the problem, proposed approach, alternatives, and effects on users. Everyone may participate. Maintainers seek consensus, address substantive objections, and record the decision and its rationale in the discussion.

When a technical disagreement cannot be resolved by the relevant code owners, a participant may ask the core maintainers to review it. If discussion still does not produce consensus, a maintainer may call a public vote. The vote remains open for at least seven calendar days, and approval requires more than half of the current core maintainers. Abstentions and nonresponses do not count as approvals; a tie leaves the proposal unaccepted. The outcome is recorded in the same issue or pull request.

Urgent fixes and reversions follow the existing contribution and stability process. The responsible maintainer explains the action in the associated issue or pull request.

## Maintainer changes

### Becoming a maintainer

A maintainer may nominate a contributor through a public pull request to this document. Nominations should describe sustained contributions, review experience, collaboration with others, and the responsibilities the nominee is willing to take on. The nominee must agree to the role.

The nomination remains open for review for at least seven calendar days. Approval requires more than half of the current core maintainers. Once approved, the roster and any affected entries in CODEOWNERS are updated, and repository administrators arrange the access needed for the role.

### Stepping down or changing responsibilities

A maintainer may step down at any time by notifying the other maintainers. The maintainers coordinate a handover, update this roster and CODEOWNERS, record the retirement below, and remove access that is no longer needed. A voluntary retirement does not require a vote.

If a maintainer is inactive or unable to fulfill their responsibilities, another maintainer may propose changing or ending the role. The affected maintainer must be contacted directly and given at least seven calendar days to respond before a decision. Approval requires more than half of the other current core maintainers. The affected maintainer does not vote on their own removal. Sensitive personal matters are discussed privately with uninvolved maintainers; personal details are not included in public role-change records.

A former maintainer may return through the nomination process above.

### Retired committers

Retirements are recorded in this section with the maintainer's GitHub username, effective date, and a link to the public role-change record. No retired committers are recorded in this version.

## Releases

The maintainers coordinate releases and designate a maintainer to prepare each release. That maintainer checks release readiness, prepares release notes, and coordinates publication according to the [release policy in the README](README.md#releases).

## Changes to this document

Governance changes are proposed through a public pull request. The proposal remains open for review for at least seven calendar days and requires approval from more than half of the current core maintainers. Approved changes take effect when merged; the pull request and Git history preserve the decision and its date.
