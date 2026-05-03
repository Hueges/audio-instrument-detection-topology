---
title: Musical Classifier
emoji: 🏆
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Musical Instrument Classifier

A multi-label audio classifier that detects which instruments are present in an audio clip. Built with VGGish audio embeddings and a Random Forest classifier trained on the OpenMIC-2018 dataset.

## Detected instruments

accordion, banjo, bass, cello, clarinet, cymbals, drums, flute, guitar, mallet percussion, mandolin, organ, piano, saxophone, synthesizer, trombone, trumpet, ukulele, violin, voice

## API

The API is deployed on Hugging Face Spaces: `https://hueges-musical-classifier.hf.space`

### Endpoints

**GET /** — health check

**POST /klasifikuj** — upload an audio file and get detected instruments with confidence scores

```bash
curl -X POST https://hueges-musical-classifier.hf.space/klasifikuj \
  -F "audio=@your_audio_file.wav"
```

Response:
```json
{
  "instrumenti": {
    "piano": 0.47,
    "guitar": 0.38
  }
}
```

Supported formats: `.wav`, `.mp3`, `.ogg`, `.flac`

## Run locally with Docker

```bash
docker pull musical-classifier
docker run -p 8000:8000 musical-classifier
```

Then send requests to `http://localhost:8000/klasifikuj`

## Train your own model

1. Download the [OpenMIC-2018 dataset](https://zenodo.org/record/1432913) and extract to `openmic-2018/`
2. Run `python Train.py` — saves `model_vggish.pkl`
3. Run `python api.py` or use Docker

## How it works

1. Audio is loaded and resampled to 16kHz
2. VGGish (Google's audio CNN) extracts 128-dimensional embeddings per 0.96s segment
3. Embeddings are averaged across all segments into a single vector
4. A Random Forest classifier predicts the probability for each of the 20 instruments
5. Instruments above the confidence threshold (0.35) are returned

The model was trained on 20,000 clips from OpenMIC-2018 with weak crowd-sourced labels.

## Files

- `api.py` — FastAPI REST API
- `Klasifikuj.py` — inference pipeline
- `Train.py` — model training on OpenMIC-2018 VGGish features
- `Podaci.py` — TDA-based feature extraction pipeline (original approach)
- `Vizualizacija.py` — topology visualization
- `Dockerfile` — container definition
