# therionis

A local backend that runs pretrained models over chest X-rays and returns what they find, with a heatmap showing where the model looked. A FastAPI service, everything on your own machine.

Built in 2025. A learning project, not a medical tool. It does not diagnose anyone.

## How it works

- Models come from [TorchXRayVision](https://github.com/mlmed/torchxrayvision), pretrained on public chest X-ray datasets.
- You post an image to `/analyze`. The service runs the model and returns the findings.
- It builds a Grad-CAM heatmap over the X-ray, so you can see which part of the image drove the result.
- Images and results are stored in SQLite and the local filesystem, so it works offline.

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/analyze` | run a model on an image, default model `chexpert` |
| GET | `/cases/{id}` | fetch one stored case |
| GET | `/cases` | list cases |
| GET | `/models` | list available models |
| GET | `/health` | health check |

## Run it

You need Python 3.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The API then runs on `http://localhost:8000`. There is a worked example in `scripts/simple_example.py` and more detail in `docs/API.md`.
