
### Creating a release

Scispacy has two components:

- The scispacy pip package
- The scispacy models

The scispacy pip package is published automatically using the `.github/actions/publish.yml` github action. It happens whenever a release is published (with an associated tag) in the github releases UI.

In order to create a new release, the following should happen:

#### Updating `scispacy/version.py`
Update the version in version.py.

#### Training new models

The entire pipeline can be run using `spacy project run all`. This will train and package all the models.

The packages should then be uploaded to the `https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/{VERSION}` S3 bucket, and references to previous models (e.g in the readme and in the docs) should be updated. You can find all these places using `git grep <previous version>`.

The scripts `install_local_packages.py`, `instal_remote_packages.py`, `print_out_metrics.py`, `smoke_test.py`, and `uninstall_local_packages.py` are useful for testing at each step of the process. Before uploading, `install_local_packages.py` and `smoke_test.py` can be used to make sure the packages are installable and do a quick check of output. `print_out_metrics.py` can then be used to easily get the metrics that need to be update in the README. Once the packages have been uploaded, `uninstall_local_packages.py`, `install_remote_packages.py`, and `smoke_test.py` can be used to ensure everything was uploaded correctly.

#### Merge a PR with the above changes
Merge a PR with the above changes, and publish a release with a tag corresponding to the commit from the merged PR. This should trigger the publish github action, which will create the `scispacy` package and publish it to pypi.

