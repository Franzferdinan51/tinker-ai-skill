# Anonymization: credentials by value, identity per session

## What must go

1. **Credentials.** API keys, tokens, JWTs, Bearer headers, passwords inside connection
   URLs, private-key blocks, and `SOMETHING_KEY=...` / `password: ...` assignments with
   high-entropy values. Transcripts contain them because people `cat .env`, paste
   curl commands, or the agent prints a config file.
2. **Identity.** Home-directory usernames (`/Users/jdoe`, `/home/jdoe`, project slugs
   `-Users-jdoe-...`), full names, emails, GitHub and chat handles, employer names and
   domains, and identity-bearing top-level folders (`/Users/jdoe/acme-work/...`).
3. **Nothing else.** Repository names, file names, code, generic folders (`src`,
   `scripts`, `Desktop`) and ordinary technical content stay. Over-scrubbing makes
   the data less realistic and the model worse.

## How `anonymize_sessions.py` does it

**Credentials, corpus-wide, by value.** Candidates are collected from every session
first, then every occurrence is replaced by `REDACTED_CREDENTIAL_<sha256 prefix>`.
One value maps to one marker everywhere, so a key seen in `.env` and later in a
command stays consistent. Obvious placeholders (`example`, `your-key`, `xxxx`) are
ignored. Values are never printed; the manifest stores only the markers.

**Identity, per session, randomized.** For each session a fake identity (given name,
surname, username, handle, org, email, domain) is drawn from a seeded RNG, and the
top-level folders under the home directory that carry identity get random
container names. Ordered rules rewrite:

1. `/Users/<real>/<folder>/` and `~/<folder>/` -> `/Users/<fake>/<container>/`
   (same for `/home`); the project slug `-Users-<real>-<folder>` the same way;
2. `/Users/<real>` -> `/Users/<fake>`;
3. emails, full names (and their given/surname tokens in lower, Capitalized and
   UPPER spellings), handles, orgs (with and without hyphens, title-cased), domains;
4. bare usernames, with alphanumeric boundaries so `sam` does not hit `sample`.

The header's `cwd` is rewritten with the same rules and the source path is reduced
to the file name.

**Verification.** After rewriting, every session is scanned for any remaining
identity string (case-insensitive; name tokens as whole words) and for credential
candidates. The run exits 1 if anything survives. `--allow-token` whitelists a
literal that is legitimately present (a surname that is also a common word).

## Why per-session randomization

Our first release anonymized every session to the same constant world
(`/home/user/agents/...`). It looked clean and the loss was fine. In use, the model
treated that world as ground truth: it probed for fake paths that did not exist on
the user's machine, addressed the user by the wrong name, and invented a
repository fork, because a byte-constant string that appears in every training row
is the strongest feature in the dataset. Randomizing per session (coherent inside a
session, different between sessions) removed the confabulation. Randomize names
and containers; keep repository names real.

## Checklist with the user

- Fill `identity.json` together: usernames on every machine the sessions came from,
  full name, all emails, handles, employer and client names, domains.
- Read the detected username list; `--keep-username` for CI or service accounts.
- Exclude whole projects that must not leave the machine at the collect step
  (`--exclude-slug`), not by scrubbing afterwards.
- Open three anonymized sessions and read them. Secrets have shapes the regexes
  do not know (internal URLs, customer names in tickets).
- Use a new `--salt` for every public release so identities differ between releases.
