# ua-tts-eval design

Status: the package and the path changes are implemented. `ua-tts-arena` is implemented and runs locally. It is not deployed yet.

## Overview

The project evaluates Ukrainian text-to-speech models on a fixed dataset, using automatic metrics and human ratings. It has three parts:

| Part                   | Where                   | Role                                                     |
| ---------------------- | ----------------------- | -------------------------------------------------------- |
| `ua-tts-eval` package  | this repo               | Ships the dataset, generates audio, computes metrics     |
| `ua-tts-bench`         | separate repo           | Adapter scripts, one for each model                      |
| `ua-tts-bench` dataset | Hugging Face            | Stores generated audio and metrics, one folder per model |
| `ua-tts-arena`         | separate repo           | Human voting and the leaderboard                         |

The three repos are in one `ua-tts-benchmark` folder, next to each other.

```
ua-tts-bench adapter script
   ├─ generate() ─▶ audio + manifest.json
   ├─ evaluate() ─▶ metrics.json
   ├─ export_mp3() ─▶ MP3 copies
   └─ hf upload  ─▶ HF dataset ─▶ ua-tts-arena ─▶ votes in Supabase
```

Model code never lives in this repo. Only the maintainer adds models to the leaderboard, in these steps:

1. Add an adapter script to `ua-tts-bench` that subclasses `TtsModel`. The script declares the model's packages and this package as its dependencies.
2. Run the script, which generates audio, computes metrics and makes MP3 copies.
3. Upload the resulting folder to the HF dataset.
4. Add the model on the arena's Models admin page. Until then, it isn't available to raters.

## v1 scope

**In scope**

- Generation from text only, one variant per item
- WER and CER through a swappable ASR, plus RTF with the hardware recorded
- Metrics for each file
- MP3 copies with aligned loudness, for the arena to play

**Left out of v1**

- Voice cloning and reference audio
- UTMOS and speaker similarity
- Aggregated metrics (overall, per tag, confidence intervals)
- A hidden dataset split
- Submissions from model authors

## Package

- Python `>=3.10`
- Created with `uv init --package`, which also picks the build backend
- Runtime dependencies are light and have no strict version pins, because the package is installed into each model's own environment
- No optional extras in v1: the ASR runs behind HTTP, so torch is not needed
- `export_mp3` needs `ffmpeg` on `PATH`. `generate` and `evaluate` don't.

```bash
uv pip install git+https://github.com/maksym-halych/ua-tts-eval
```

### Public API

The public API is four names: `TtsModel`, `generate`, `evaluate` and `export_mp3`.

```python
class TtsModel(ABC):
    name: str
    device: str
    params: dict = {}

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (samples, sample_rate), with float samples in [-1.0, 1.0]."""


def generate(model: TtsModel) -> Path: ...

def evaluate(folder: Path, asr_base_url: str, asr_model: str) -> None: ...

def export_mp3(folder: Path) -> None: ...
```

| Attribute | Meaning                                                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`    | Name of the output folder, so it also identifies the model on the leaderboard. Allowed characters: ASCII letters, digits, `.`, `_`, `-`. It must start with a letter or digit, which rules out `.` and `..`. |
| `device`  | `cuda`, `cpu` or `api`                                                                                                                                                                                       |
| `params`  | Free-form settings that differ per model, such as checkpoint, voice, temperature and seed. Recorded in the manifest. No code reads the keys inside.                                                          |

Choosing a voice is up to the adapter. A fixed voice can be recorded in `params`. The voice used for each item is not recorded.

### `generate(model)`

1. Deletes `.cache/ua-tts-eval/<name>/` relative to the current directory, if it exists, and creates it again. Every generation starts clean.
2. Copies the script that called it to `adapter.py`.
3. Calls `synthesize(item.text)` for every dataset item and times each call.
4. Writes each result to `audio/<id>.wav`.
5. Writes `manifest.json` last and returns the folder path.

The model receives `text`, never `text_normalized`. Items tagged `numbers`, `abbreviation`, `symbol` and `units` test whether the model handles normalization itself.

**Post-processing.** Audio keeps the model's native sample rate and is not otherwise changed, apart from conversion to WAV PCM16 mono. Multi-channel audio is mixed down to mono. Samples outside ±1 are clipped, and this is logged.

**Failures.** An exception, empty audio, NaN or infinite samples, and non-float samples count as a failure. The failure is logged, no file is written, and generation continues with the next item. Completing a folder after a failure means running the whole generation again.

`params` that can't be written as JSON raise an error before synthesis starts.

**Invariant.** `manifest.json` exists only after generation has finished, and its `items` list exactly the files in `audio/`.

### `evaluate(folder, asr_base_url, asr_model)`

1. Reads `manifest.json`. A folder without a manifest is an error.
2. Resamples each file to 16 kHz mono in memory and sends it to the ASR with `language="uk"`. The files on disk are not changed.
3. Normalizes the transcript and the item's `text_normalized`, then computes WER and CER.
4. Rewrites `metrics.json` completely. Earlier contents are not merged.

An item whose transcription fails is logged and left out of `metrics.json`. If no item succeeds, for example because the ASR is down, `evaluate` raises an error and keeps the existing `metrics.json`.

### `export_mp3(folder)`

Makes the copies the arena plays. The adapter script calls it after `evaluate`, and it can also run on an existing folder without generating again.

1. Reads `manifest.json`. A folder without a manifest is an error, and so is `ffmpeg` missing from `PATH`. Both are checked before any work starts.
2. Measures each WAV file's loudness and true peak with `ffmpeg`.
3. Applies one constant gain per clip, aiming for -23 LUFS. The gain is capped so the true peak stays at or below -1 dBTP, so a clip can end up quieter than the target but is never distorted.
4. Encodes the result at 44.1 kHz and 192 kbps, so every model has the same format.
5. Replaces `mp3/` completely. Earlier contents are not merged.

A clip shorter than the 400 ms that the measurement needs is measured by repeating it for 3 seconds, and the copy keeps its real length. A short clip can be what the model really produced, so leaving it out would hide a failure from raters. A silent clip is logged and gets no copy. Its WER and CER still count. An item whose conversion fails is logged and left out. If no item succeeds, `export_mp3` raises an error and keeps the existing `mp3/`.

### ASR

Any server with the OpenAI transcription API works, called through the `openai` client. Switching ASR means changing `asr_base_url` and `asr_model`. The API key comes from `OPENAI_API_KEY` if it is set; local servers don't need one.

The first ASR is [achetronic/parakeet](https://github.com/achetronic/parakeet), which serves `parakeet-tdt-0.6b-v3` at `http://127.0.0.1:5092/v1`. Whisper is the fallback if Parakeet doesn't work out.

Its image should be pinned to a specific version instead of `latest`. All models need to be evaluated again whenever the ASR or its image changes, otherwise their WER values aren't comparable.

### Text normalization

These rules apply to both the reference and the transcript before WER and CER:

- Apply Unicode NFC and convert to lowercase
- Replace the apostrophe variants `ʼ` and `’` with `'`
- Replace hyphens and dashes with spaces, so `Docker-контейнер` counts as two words
- Remove punctuation, including an apostrophe that isn't between two word characters, since that is a quotation mark
- Collapse repeated whitespace

CER counts the spaces in the normalized text.

Digits are not converted to words, and Latin script is not converted to Cyrillic. Those cases count as errors for now, and the decision will be revisited after seeing ASR output on real audio.

## Dataset

- The file moves to `src/ua_tts_eval/data/dataset.jsonl` and ships inside the package.
- An item's `id` is the first 12 hex characters of the md5 of its text.
- The dataset version is the md5 of the file. There are no `v1`/`v2` labels.
- `manifest.json` and `metrics.json` each record `dataset_md5`. They can differ, because `evaluate` may run later with a newer dataset.

## Output folder

Fields are kept to the minimum. Adding a field later is easy, while removing one breaks whatever reads it, so a field needs a concrete current use.

```
.cache/ua-tts-eval/             relative to cwd; maps to the root of the HF dataset
└── holos/                      one folder per model, named after TtsModel.name
    ├── manifest.json           how the audio was generated (written by generate)
    ├── metrics.json            metrics (fully rewritten by each evaluate)
    ├── adapter.py              copy of the script that ran generate
    ├── bench.log               shared log of generate and evaluate
    ├── audio/
    │   ├── a4f076a8e050.wav    WAV PCM16, mono, model's native sample rate
    │   ├── 6c33a2c31bc5.wav
    │   └── ...
    └── mp3/                    copies for the arena (written by export_mp3)
        ├── a4f076a8e050.mp3
        ├── 6c33a2c31bc5.mp3
        └── ...
```

The adapter scripts in `ua-tts-bench` all write to the same folder, so it can contain several models. The HF dataset collects all of them.

Values in the examples below are illustrative.

### `manifest.json`

```json
{
  "format_version": 1,
  "generated_at": "2026-11-02T14:03:10Z",
  "dataset_md5": "5d41402abc4b2a76b9719d911017c592",
  "ua-tts-eval": { "commit": "c99d2f2" },
  "model": {
    "repo": "https://github.com/maksym-halych/ua-tts-bench",
    "commit": "9f2c1ab",
    "params": { "checkpoint": "holos-1.2", "voice": "anna", "temperature": 0.7, "seed": 42 }
  },
  "device": "cuda",
  "hardware": {
    "cpu": "AMD Ryzen 7 7840HS",
    "gpu": "NVIDIA RTX 4070",
    "ram_gb": 32,
    "os": "Linux 7.2.9"
  },
  "items": {
    "a4f076a8e050": { "synthesis_time_s": 0.81 },
    "6c33a2c31bc5": { "synthesis_time_s": 0.92 }
  }
}
```

| Field                         | Meaning                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `generated_at`                | When generation started                                                                                             |
| `ua-tts-eval.commit`          | Commit of the installed package, `null` if unknown                                                                  |
| `model.repo`, `model.commit`  | Read from git in the adapter script's folder, `null` outside a git repo. Credentials in the remote URL are dropped. |
| `model.params`                | Copy of `TtsModel.params`                                                                                           |
| `device`                      | Copy of `TtsModel.device`. `hardware` only describes the machine, not what the model ran on.                        |
| `hardware`                    | Collected automatically                                                                                             |
| `items.<id>.synthesis_time_s` | Time spent in `synthesize()`, excluding model loading                                                               |

There is no model version field. The commit identifies the code, a checkpoint goes in `params`, and an API model's id goes in its `name`, such as `elevenlabs-multilingual-v2`. There is no model name field either, because the folder name is the name.

### `metrics.json`

```json
{
  "format_version": 1,
  "evaluated_at": "2026-11-02T18:40:00Z",
  "dataset_md5": "5d41402abc4b2a76b9719d911017c592",
  "ua-tts-eval": { "commit": "c99d2f2" },
  "asr": "parakeet-tdt-0.6b-v3",
  "items": {
    "a4f076a8e050": {
      "transcript": "Увечері над містом пройшла коротка злива.",
      "wer": 0.0,
      "cer": 0.0,
      "duration_s": 3.42,
      "rtf": 0.237
    },
    "6c33a2c31bc5": {
      "transcript": "Бібліотека тимчасово зачинена через ремонт дому.",
      "wer": 0.1667,
      "cer": 0.0426,
      "duration_s": 3.87,
      "rtf": 0.238
    }
  }
}
```

| Field        | Meaning                                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| `asr`        | Value of `asr_model`                                                                                       |
| `transcript` | Raw ASR output, before normalization. Stored so metrics can be recalculated without running the ASR again. |
| `duration_s` | Length of the WAV file                                                                                     |
| `rtf`        | `synthesis_time_s / duration_s`                                                                            |

### `adapter.py`

Lives in the `ua-tts-bench` repo and is copied into the folder by `generate`, so the result stays reproducible if the script changes later.

```python
from holos import load_model
from ua_tts_eval import TtsModel, evaluate, export_mp3, generate


class HolosTtsModel(TtsModel):
    name = "holos"
    device = "cuda"
    params = {"checkpoint": "holos-1.2", "voice": "anna", "temperature": 0.7, "seed": 42}

    def __init__(self):
        self.model = load_model(self.params["checkpoint"], device=self.device)

    def synthesize(self, text):
        samples, sample_rate = self.model.tts(
            text,
            voice=self.params["voice"],
            temperature=self.params["temperature"],
            seed=self.params["seed"],
        )
        return samples, sample_rate


folder = generate(HolosTtsModel())
evaluate(
    folder,
    asr_base_url="http://127.0.0.1:5092/v1",
    asr_model="parakeet-tdt-0.6b-v3",
)
export_mp3(folder)
```

### `bench.log`

`generate`, `evaluate` and `export_mp3` share one log. `generate` starts it fresh along with the folder, and the other two append to it.

```
2026-11-02 14:03:11 INFO  generate a4f076a8e050 ok 0.81s
2026-11-02 14:03:12 ERROR generate 6c33a2c31bc5 RuntimeError: CUDA out of memory
2026-11-02 14:03:40 INFO  generate done: 199 ok, 1 failed
2026-11-02 14:03:41 INFO  evaluate a4f076a8e050 wer=0.000 cer=0.000
2026-11-02 14:07:19 INFO  evaluate done: 199 ok, 0 failed
2026-11-02 14:07:20 WARNING export_mp3 947ec8a6bed1 silent, skipped
2026-11-02 14:07:31 INFO  export_mp3 done: 198 ok, 1 skipped, 0 failed
```

## Publishing to Hugging Face

The upload is a plain CLI command, with no code in the package:

```bash
hf upload <user>/ua-tts-bench .cache/ua-tts-eval/holos holos --repo-type dataset
```

- Upload a folder only after every item was generated. The log shows the count of failed items.
- The log is uploaded too. Check logs of API models first, because a client can put a key into an error message.
- A new model version gets a new `name`, such as `holos-2`, and so a new folder.
- Once voting has started on a folder, don't upload different audio to it. Votes refer to specific audio, so old votes would appear to rate audio nobody heard. Metrics can still be updated.

## ua-tts-arena

The arena is a separate repo, in the `ua-tts-arena` folder next to this one. It runs locally for now. It shares no code with this package. The format of the HF dataset is the contract between them.

**Reference: TTS Arena**

[TTS Arena](https://github.com/TTS-AGI/TTS-Arena) (Apache 2.0, read at commit `7a0d13a`) is the reference for this app. We don't fork it. We build a much smaller app and reuse what fits, keeping the license notice on anything copied.

Reused from it: Next.js, Tailwind, the wavesurfer.js player, Zod, neutral audio URLs tied to a server-side session, and the leaderboard's `±` display.

| Part              | TTS Arena                                                  | This app                                        |
| ----------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| Audio             | Synthesized on request, through a Hono router and adapters | Pre-generated, so no router or adapters         |
| Text              | About 500k prompts plus custom text                        | The dataset's 200 items, with no custom text    |
| Vote limit        | Each prompt moves the ranking once                         | One vote per rater for each item and model pair |
| Database          | Postgres with Drizzle                                      | Supabase with `supabase-js`                     |
| Sign-in           | Hugging Face, with its own OAuth code                      | Google, through Supabase Auth                   |
| Abuse protection  | hCaptcha, FingerprintJS, risk scoring, background sweep    | See "Decided" below                             |
| Listening check   | In the browser                                             | On the server                                   |
| Leaderboard       | Votes only                                                 | Votes and automatic metrics                     |
| Admin             | Eight sections, including synthesis monitoring             | Six pages, see "Admin pages" below              |
| Tables and charts | TanStack Query, TanStack Table, Recharts                   | Plain Next.js pages and tables                  |
| Docs              | Separate site                                              | None                                            |
| Processes         | Two services under `supervisord` in one container          | One service, no `supervisord`                   |

**Decided**

- Reads model folders from one configured source: a local folder in development, or the HF dataset when deployed. Both have the same layout, so only the root differs. An optional `HF_TOKEN` on the server allows a private dataset.
- The set of models isn't fixed, and a new version or checkpoint gets a new folder. Models are registered in the database. Uploading a folder doesn't make a model available: an admin has to add it on the Models page, and the app checks that the folder is complete.
- Pulls `dataset.jsonl` from this repo's git. Ids are hashes of the text, so a lookup returns either the right text or nothing, and items not found are skipped.
- Built with Next.js and Tailwind, using Bun. App code avoids Bun-only APIs, so it can also run on Node.js. Audio plays through wavesurfer.js.
- Pages for raters are available in English and Ukrainian, and admin pages are in English only. Raters' Ukrainian isn't checked.
- Checks requests to the server and files from the model folders with Zod.
- Stores votes in Supabase, queried with `supabase-js`, with no ORM. The app's server holds the key, and the browser has no direct database access.
- Raters sign in with a Google account through Supabase Auth.
- No captcha or device fingerprinting in v1. Protection comes from sign-in, a unique constraint on rater, item and model pair, a server-side listening check, a per-rater rate limit, and the Raters admin page.
- No separate docs site.
- Offers A/B comparisons, where raters must choose which sample sounds more natural. There's no tie option. MOS, if added later, would be a separate tab.
- Picks a random item and two random active models for each comparison, using only items where both models have an MP3 copy and skipping combinations the rater has already voted on.
- Ranks models with Bradley–Terry, sorted by the lower confidence bound, as in TTS Arena. Paused models' votes stay in the fit, so pausing a model doesn't shift the others' ratings.
- The leaderboard opens in rank order. A rater can sort by rating, win rate, WER, CER or RTF with the column headings, and the best value comes first. Models with no value come last. The sort key is in the URL (`?sort=wer`). The rank number stays with the model in each order.
- The public leaderboard lists every active model regardless of vote count, with its voting results and average WER, CER and RTF. RTF depends on the hardware, which the note under the table says. Paused models appear only in admin. The app averages `metrics.json` itself, because the package doesn't aggregate metrics.
- Serves audio through its server under neutral names, so the browser never sees a path containing the model's name.
- Plays the MP3 copies made by `export_mp3`, because a louder sample tends to win A/B comparisons. Nothing is processed per request. On the two current models, a target of -16 LUFS would push over half the clips above -1 dBTP, and -23 LUFS pushes none. TTS Arena's single-pass `loudnorm` isn't used, because it adjusts the level dynamically, which alters the audio being judged.

**Configuration**

- `MODELS_SOURCE` is the one configured source. Its value is a local path or `hf://datasets/<user>/<name>`.
- `DATASET_SOURCE` is the location of `dataset.jsonl`, as a local path or a URL. The value for a deployment is the raw GitHub URL of the file in this repo.
- The app reads `metrics.json` and the dataset again after a maximum of five minutes.

**Stack details**

- Bun is the package manager, the test runner and the default runtime. The app was tested on the Bun runtime in development and in production, and on Node.js in development.
- TypeScript stays at version 6, because `typescript-eslint` doesn't support version 7 yet.
- The translations are a small dictionary in the code, with no library. A cookie stores the language. With no cookie, the app uses Ukrainian if the browser prefers it, and otherwise English.
- The auth server of Supabase runs through the app's server. The browser has no Supabase key, so the session cookies are `httpOnly`.

**Local sign-in**

A local form signs in with only an email address, so local work doesn't need Google credentials. It is available only if `LOCAL_SIGN_IN=true` and `SUPABASE_URL` points to this computer. It can't turn on with a hosted database.

An admin address must be confirmed, because an address that isn't confirmed could belong to someone else. The Email provider of a hosted project should be off.

**Database schema**

| Table      | Purpose                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| `models`   | Registered models. `name` is the folder name. `item_ids` lists the items with an MP3 copy when the model was added. `paused_at` is set for a paused model. |
| `raters`   | One row per rater who opened a comparison: Supabase Auth id, email address and exclusion data. There is no foreign key to `auth.users`, so votes stay if an account is deleted. |
| `rounds`   | The open comparison of each rater. `rater_id` is unique, so a rater has at most one. The row is the only link between the neutral audio URLs and the model names. It is deleted after the vote. |
| `votes`    | `rater_id`, `item_id`, `winner`, `loser`, `chosen_side`, `decision_ms`, `models_shown`, and exclusion data. A unique index on rater, item and both models in either order allows one vote per combination. |
| `settings` | One row with the two admin settings.                                                                                     |

- The views `counted_votes`, `pair_tallies`, `item_tallies` and `daily_tallies` hold only the votes that count. They run with the rights of the caller.
- Excluding a rater sets a field on `raters`, not on each vote. So later votes of that rater are also excluded, and restoring a rater doesn't restore a vote that was excluded separately.
- Every table has row level security with no policies, and the `anon` and `authenticated` roles have no grants. Only the secret key reaches the data.

**Voting flow**

1. The page asks the server for a round. If the rater has an open round, the server returns the same round. A reload does not change the samples.
2. A new round is one random choice among all combinations of item and active model pair that are left for the rater. The side of each model is also random. If nothing is left, the page says that the rater rated every comparison.
3. The server sends two audio URLs of the form `/api/rounds/<id>/audio/a`. Only the rater of the round can get them. The server records when it first sent each clip.
4. The page plays A, then B the first time. A rater can replay or pause each clip. Starting a clip stops the other.
5. The vote buttons turn on when each clip has played for 3 seconds, or for its full length if it is shorter. This value comes from TTS Arena.
6. The server repeats the check. It accepts a vote only if it sent both clips, and if the required time passed since it sent the later clip. The required time is the sum of the two values from step 5, from `duration_s` in `metrics.json`.
7. The server allows 15 votes per rater per minute. It records the vote, deletes the round, and returns the model names if the setting allows it.

- Before the vote, a rater can skip the sentence. The server deletes the round and starts a new one with a different item. If no other item is left, the server shows the skipped item again. The server records no vote and does not store the skip, and the skipped comparison can appear again later. There is no limit on skips.
- A skip lets a rater avoid a comparison, so a rater can select the comparisons that get a vote. The model names stay hidden, but a rater can recognize a voice. The app can not measure this effect, because it does not store skips.
- An open round expires after one hour. The server also replaces a round if one of its models is paused.
- If a clip is missing from the source, the server deletes the round, and the page offers a new one. Otherwise the rater would be stuck, because the listening check could never pass.
- The listening check is a lower limit on time. It can't prove that a rater listened.

**Results**

- A model with no votes goes last on the leaderboard and shows no rating. Its zero-width interval at the center would otherwise place it above models with real results.
- The fit keeps the small prior from TTS Arena: two virtual games per model, half of them won. So one win doesn't give an infinite rating.
- Agreement on the Raters page compares each vote with the majority of the other raters on the same item and pair. A tie or no other votes gives no result for that vote.
- The CSV has rater ids and no email addresses, so the file can be shared.

**Admin pages**

Access is limited to an `ADMIN_USERS` list of email addresses, checked on the server for every page and action. Admins can also vote.

| Page     | Shows                                                                                     | Actions                                            |
| -------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Overview | Total votes and raters, votes per day, model pairs and items that still need votes        | None                                               |
| Models   | Registered models and their status, plus folders not added yet                            | Add a model, pause or resume a model               |
| Results  | Each model's rating with `±`, win rate and automatic metrics, and a head-to-head grid     | None                                               |
| Raters   | Each rater's vote count, typical decision time, A/B split and agreement with the majority | Exclude or restore all of a rater's votes          |
| Votes    | Votes, filterable by rater, model and item                                                | Exclude or restore a vote, export all votes as CSV |
| Settings | Current settings                                                                          | Turn each setting on or off                        |

- The settings are whether raters see the models after each vote and whether the leaderboard is public. Both are on by default.
- Each vote records whether its rater was seeing the models after each vote, because that can't be worked out later and it affects how far the vote can be trusted.
- Excluding a vote stores a reason and date, and the leaderboard leaves the vote out. Nothing is deleted. The reasons are included in the CSV.
- Pausing a model stops it appearing in new comparisons and keeps its past votes.
- Left for later: results by item or tag, rating history, and editing model details.

**Still to decide**

- Whether the HF dataset is public or private. This doesn't affect the code.
- Where to deploy: Vercel is the likely choice, with an HF Space or a server as alternatives. The app should stay portable until then, so it shouldn't rely on a long-running process, such as loading `dataset.jsonl` once at startup or running `ffmpeg` on each request.
- Free-plan limits to recheck before deploying, as of 2026-11. Supabase has a 500 MB database, pauses projects after a week of inactivity and has no automatic backups, so votes should be exported regularly. Vercel's Hobby plan is for non-commercial, personal use and limits responses to 4.5 MB. The largest current clip is 1.67 MB as a WAV.
- A stricter voting round for a paper: models hidden, leaderboard hidden, exclusion rules written beforehand, raters' Ukrainian checked, and analysis limited to that round's votes. Votes collected under the default settings aren't blind, and the main rater is also the admin.

## Changes to the existing repo

- Move `dataset.jsonl` into the package and update its path in `Makefile`, `.github/workflows/dataset-site.yml` and the default of `scripts/validate_dataset_ids.py`.
- `site/` stays on GitHub Pages, as the arena doesn't replace it. The page loads `dataset.jsonl` relative to itself, and the build still copies the file next to it, so the page doesn't change.

## Known limitations

- **WER misses pronunciation errors.** 55 of the 200 items target stress, homographs, Russian-accent patterns and similar. A wrong stress usually transcribes to the same text, so these need human ratings.
- **Parakeet ignores `language`.** It detects the language itself. In a test with clips synthesized by espeak-ng, `uk` and `en` gave identical output, in a mix of Latin-script and Polish-like text. espeak-ng is very robotic and Parakeet works well on recorded Ukrainian lectures, but short sentences from a weak TTS model may get a misleading WER. To be checked on real audio during development.
- **The output can contain digits.** This happened in the same test, while references contain none, so correctly spoken numbers can count as errors.
- **Sample rate affects Parakeet's output.** The same clip gave different text at 22 kHz and 16 kHz, which is why `evaluate` resamples before sending.
- **RTF is only comparable on the same hardware.** For API models it includes the network and the provider's hardware.
