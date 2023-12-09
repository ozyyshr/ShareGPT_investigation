# ShareGPT_investigation
Code and annotated data for "The Shifted and The Overlooked: A Task-oriented Investigation of User-GPT Interactions"

### Download ShareGPT Data
The ShareGPT collection we used for annotation is publically available [here](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json).

### Self-demonstrated Annotation
By first configuring GPT API and running `python annotation.py --data_file $DATA$ --demo_file $DEMO$ --output_file $OUTPUT$`, the annotation starts with default engine `GPT-4`.

### Annotated ShareGPT Data
Due to the high demand for computing resources, we provide our annotated version [here](https://drive.google.com/file/d/1zAU3uWhNSN6NvMl85cmVi2mj-uvswE_O/view?usp=drive_link), where for each sample, we have the following attributes:

```

  "id": the ID for each user query in the ShareGPT collection.
  "domain": annotated results for domain/topics.
  "summary": one sentence summarization of the user query.
  "task_type": the specific task generated in a free-form manner by GPT-4.

```
