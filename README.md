## Demo for image-matching with [LoFTR](https://github.com/zju3dv/LoFTR): Detector-Free Local Feature Matching with Transformers 

### Run with pipenv

##### Requirements
* Python
* pipenv

Install

```bash
pipenv shell --python 3.8
pip install -r requirements.txt
```

Run
```bash
python test.py
```

### Run with conda

##### Requirements
* Python
* Conda

Install

```bash
conda env create -f environment.yaml
conda activate loftr
```

Run
```bash
python test.py
```

### Run with Docker
##### Requirements
* Docker

Build image

```bash
docker build -t loftr_demo .
```

Run image

```bash
docker run --name loftr_demo_run loftr_demo
docker cp loftr_demo_run:/output .
```

Remove image

```bash
docker rm loftr_demo_run
docker image rm loftr_demo
```

