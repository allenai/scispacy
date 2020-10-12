
### Creating a release

Scispacy has two components:

- The scispacy pip package
- The scispacy models

The scispacy pip package is published automatically using the `.github/actions/publish.yml` github action. It happens whenever a release is published (with an associated tag) in the github releases UI.

In order to create a new release, the following should happen:

#### Updating `scispacy/version.py`
Update the version in version.py.

#### Training new models

For the release, new models should be trained using the `scripts/pipeline.sh` and `scripts/ner_pipeline.sh` scripts, for the small, medium and large models, and specialized NER models. Remember to export the `ONTONOTES_PATH` and `ONTONOTES_PERCENT` environment variables to mix in the ontonotes training data.

```
bash scripts/pipeline.sh small
bash scripts/pipeline.sh medium
bash scripts/pipeline.sh large
bash scripts/ner_pipeline.sh
```

these should then be uploaded to the `https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/{VERSION}` S3 bucket, and references to previous models (e.g in the readme and in the docs) should be updated.

#### Merge a PR with the above changes
Merge a PR with the above changes, and publish a release with a tag corresponding to the commit from the merged PR. This should trigger the publish github action, which will create the `scispacy` package and publish it to pypi.

