# ua-tts-bench design

Status: the package and the path changes are implemented. `ua-tts-arena` has not been started.

## Overview

The project evaluates Ukrainian text-to-speech models on a fixed dataset, using automatic metrics and human ratings. It has three parts:

| Part                   | Where                   | Role                                                     |
| ---------------------- | ----------------------- | -------------------------------------------------------- |
| `ua-tts-bench` package | this repo               | Ships the dataset, generates audio, computes metrics     |
| `ua-tts-bench` dataset | Hugging Face            | Stores generated audio and metrics, one folder per model |
| `ua-tts-arena`         | separate repo, HF Space | Human voting and the leaderboard                         |

```
model repo + adapter script
   ├─ generate() ─▶ audio + manifest.json
   ├─ evaluate() ─▶ metrics.json
   └─ hf upload  ─▶ HF dataset ─▶ ua-tts-arena ─▶ votes in Supabase
```

Model code never lives in this repo. Only the maintainer adds models to the leaderboard, in these steps:

1. Clone the model's repo.
2. Install this package there and add an adapter script that subclasses `TtsModel`.
3. Run the script, which generates audio and computes metrics.
4. Upload the resulting folder to the HF dataset.

## v1 scope

**In scope**

- Generation from text only, one variant per item
- WER and CER through a swappable ASR, plus RTF with the hardware recorded
- Metrics for each file

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

```bash
uv pip install git+https://github.com/maksym-halych/ua-tts-bench
```

### Public API

The public API is three names: `TtsModel`, `generate` and `evaluate`.

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
```

| Attribute | Meaning                                                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`    | Name of the output folder, so it also identifies the model on the leaderboard. Allowed characters: ASCII letters, digits, `.`, `_`, `-`. It must start with a letter or digit, which rules out `.` and `..`. |
| `device`  | `cuda`, `cpu` or `api`                                                                                                                                                                                       |
| `params`  | Free-form settings that differ per model, such as checkpoint, voice, temperature and seed. Recorded in the manifest. No code reads the keys inside.                                                          |

Choosing a voice is up to the adapter. A fixed voice can be recorded in `params`. The voice used for each item is not recorded.

### `generate(model)`

1. Deletes `.cache/ua-tts-bench/<name>/` relative to the current directory, if it exists, and creates it again. Every generation starts clean.
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

- The file moves to `src/ua_tts_bench/data/dataset.jsonl` and ships inside the package.
- An item's `id` is the first 12 hex characters of the md5 of its text.
- The dataset version is the md5 of the file. There are no `v1`/`v2` labels.
- `manifest.json` and `metrics.json` each record `dataset_md5`. They can differ, because `evaluate` may run later with a newer dataset.

## Output folder

Fields are kept to the minimum. Adding a field later is easy, while removing one breaks whatever reads it, so a field needs a concrete current use.

```
.cache/ua-tts-bench/            relative to cwd; maps to the root of the HF dataset
└── holos/                      one folder per model, named after TtsModel.name
    ├── manifest.json           how the audio was generated (written by generate)
    ├── metrics.json            metrics (fully rewritten by each evaluate)
    ├── adapter.py              copy of the script that ran generate
    ├── bench.log               shared log of generate and evaluate
    └── audio/
        ├── a4f076a8e050.wav    WAV PCM16, mono, model's native sample rate
        ├── 6c33a2c31bc5.wav
        └── ...
```

Each model is generated in its own clone, so locally there is usually one folder. The HF dataset collects all of them.

Values in the examples below are illustrative.

### `manifest.json`

```json
{
  "format_version": 1,
  "generated_at": "2026-11-02T14:03:10Z",
  "dataset_md5": "5d41402abc4b2a76b9719d911017c592",
  "ua-tts-bench": { "commit": "c99d2f2" },
  "model": {
    "repo": "https://github.com/example/holos",
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
| `ua-tts-bench.commit`         | Commit of the installed package, `null` if unknown                                                                  |
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
  "ua-tts-bench": { "commit": "c99d2f2" },
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

Lives in the model's repo and is copied into the folder by `generate`, so the result stays reproducible after the clone is gone.

```python
from holos import load_model
from ua_tts_bench import TtsModel, evaluate, generate


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
```

### `bench.log`

`generate` and `evaluate` share one log. `generate` starts it fresh along with the folder, and each `evaluate` appends to it.

```
2026-11-02 14:03:11 INFO  generate a4f076a8e050 ok 0.81s
2026-11-02 14:03:12 ERROR generate 6c33a2c31bc5 RuntimeError: CUDA out of memory
2026-11-02 14:03:40 INFO  generate done: 199 ok, 1 failed
2026-11-02 14:03:41 INFO  evaluate a4f076a8e050 wer=0.000 cer=0.000
2026-11-02 14:07:19 INFO  evaluate done: 199 ok, 0 failed
```

## Publishing to Hugging Face

The upload is a plain CLI command, with no code in the package:

```bash
hf upload <user>/ua-tts-bench .cache/ua-tts-bench/holos holos --repo-type dataset
```

- Upload a folder only after every item was generated. The log shows the count of failed items.
- The log is uploaded too. Check logs of API models first, because a client can put a key into an error message.
- A new model version gets a new `name`, such as `holos-2`, and so a new folder.
- Once voting has started on a folder, don't upload different audio to it. Votes refer to specific audio, so old votes would appear to rate audio nobody heard. Metrics can still be updated.

## ua-tts-arena

The arena is a separate repo, deployed as an HF Space. It shares no code with this package. The format of the HF dataset is the contract between them.

**Decided**

- Reads model folders from the HF dataset.
- Pulls `dataset.jsonl` from this repo's git when the Space starts. Ids are hashes of the text, so a lookup returns either the right text or nothing, and items not found are skipped.
- Stores votes in a database, most likely Supabase. The Space's server holds the key, and the browser has no direct database access.
- Offers blind MOS ratings and blind A/B comparisons.
- Shows a leaderboard with both automatic metrics and voting results.
- Serves audio through its server under neutral names, so the browser never sees a path containing the model's name.
- Aligns loudness when playing audio to raters, because a louder sample tends to win A/B comparisons. Stored audio is not changed.

**Still to design**

- The rating aspects. Speaker similarity doesn't apply without cloning, and only 6 items are tagged `emotion`.
- The ranking method for A/B votes
- Quality control for raters
- The database schema and the voting interface
- How metrics are aggregated
- Whether the HF dataset is public or private

## Changes to the existing repo

- Move `dataset.jsonl` into the package and update its path in `Makefile`, `.github/workflows/dataset-site.yml` and the default of `scripts/validate_dataset_ids.py`.
- `site/` stays on GitHub Pages, as the arena doesn't replace it. The page loads `dataset.jsonl` relative to itself, and the build still copies the file next to it, so the page doesn't change.

## Known limitations

- **WER misses pronunciation errors.** 55 of the 200 items target stress, homographs, Russian-accent patterns and similar. A wrong stress usually transcribes to the same text, so these need human ratings.
- **Parakeet ignores `language`.** It detects the language itself. In a test with clips synthesized by espeak-ng, `uk` and `en` gave identical output, in a mix of Latin-script and Polish-like text. espeak-ng is very robotic and Parakeet works well on recorded Ukrainian lectures, but short sentences from a weak TTS model may get a misleading WER. To be checked on real audio during development.
- **The output can contain digits.** This happened in the same test, while references contain none, so correctly spoken numbers can count as errors.
- **Sample rate affects Parakeet's output.** The same clip gave different text at 22 kHz and 16 kHz, which is why `evaluate` resamples before sending.
- **RTF is only comparable on the same hardware.** For API models it includes the network and the provider's hardware.
