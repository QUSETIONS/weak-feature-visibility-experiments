# Remote Upload Instructions

Do not push this repository from a personal or identifiable GitHub account.

## Recommended Anonymous GitHub Flow

1. Create a new anonymous GitHub account in a clean browser profile.
2. Create an empty public repository, for example:

   `weak-feature-visibility-experiments`

3. In this local repository, run:

```bash
git remote add origin https://github.com/<anonymous-account>/weak-feature-visibility-experiments.git
git branch -M main
git push -u origin main
```

4. Optional: pass the public repository URL through an anonymization service such as Anonymous GitHub / 4open.science if required by the venue.

## Token-Based Push

If using a personal access token from the anonymous account, use HTTPS credential prompts or a temporary environment variable in a fresh shell. Do not store the token in the repository.

## Current Local State

The local repository has one anonymous commit:

```bash
git log --oneline -1
```

The commit author is configured as:

```text
Anonymous Authors <anonymous@example.com>
```

Large model weights, activation caches, SAE weights, Python caches, logs, local paths, and private notes are excluded.
