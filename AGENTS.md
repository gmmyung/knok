# Agent Guidelines

These rules apply to Codex and other automated agents working in this
repository.

## Document Map

- [README.md](README.md): user-facing API, feature modes, examples, limits, and
  benchmark notes.
- [CONTRIBUTING.md](CONTRIBUTING.md): contribution workflow, validation
  commands, branch naming, and versioning policy.
- [DEVELOPERS.md](DEVELOPERS.md): crate layout, compile/runtime flow, release
  flow, and platform notes.
- [CHANGELOG.md](CHANGELOG.md): user-facing changes that must be maintained for
  public API, behavior, feature, or release changes.

## Working Rules

- Read the relevant crate code before editing. Do not infer the architecture
  from README examples alone.
- Keep changes tightly scoped. Backwards compatibility is not a default goal
  unless the user explicitly asks for it.
- Do not overwrite user changes. Check `git status --short --branch` before
  committing or preparing a PR.
- Use `apply_patch` for manual edits. Avoid shell write tricks for repo files.
- Prefer existing helpers and local patterns over new abstractions.
- Add dependencies only when they are necessary. Optional type support should
  remain feature-gated.

## Validation

Use focused checks while iterating, then run the release check when feasible:

```sh
scripts/release-check.sh
```

The release check covers formatting, core/macro tests, no-std checks, docs, and
hosted runtime tests. If it cannot be run, report exactly which narrower checks
were run and why the full check was skipped.

## API Changes

For public API changes, update all relevant user surfaces:

- README examples
- compile-pass or runtime tests
- trybuild tests and `.stderr` snapshots for diagnostics
- `CHANGELOG.md`

For macro diagnostics, keep error text direct and stable. Do not update
trybuild snapshots just to hide an unrelated regression.

## Releases

- Do not create or push release tags unless the user explicitly asks.
- Release PRs must bump all crate versions in lockstep.
- Also update workspace dependency versions for internal crates.
- Move changelog entries from `Unreleased` into `## X.Y.Z - YYYY-MM-DD` before
  tagging.
- Use `scripts/verify-release.sh vX.Y.Z` to check release metadata locally.

## Project-Specific Cautions

- `knok-core` owns parsing and type checking. Keep semantic validation there
  when possible.
- `knok-compile` owns MLIR lowering, IREE compilation, and macro codegen.
- `knok` owns public tensor containers, runtime wrappers, and feature gates.
- Keep `no_std + alloc` compiling when touching public tensor APIs or error
  types.
- Bool tensors are real `i1` graph values, not numeric masks.
- Hosted runtime tests may depend on the IREE compiler/runtime libraries from
  the Nix shell.
