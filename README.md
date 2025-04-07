# PaliGemma Batch Inference

This repository contains code for running batch inference with the PaliGemma model using Anyscale and Ray.

## Overview

The project uses the PaliGemma model to generate captions for images in batch. It leverages Ray for distributed processing and Anyscale for cloud deployment.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jenniferbae/paligemma_batch_inference.git
   cd paligemma_batch_inference
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token:
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Running the Job

To run the job on Anyscale:

```bash
anyscale job submit job.yaml
```

## Configuration

The job configuration is defined in `job.yaml`. It uses:
- Anyscale's base image: `anyscale/ray:2.44.1-slim-py312-cu128`
- Git repository for code: `https://github.com/jenniferbae/paligemma_batch_inference.git`
- Environment variables for Hugging Face token

## License

MIT 