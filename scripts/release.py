from typing import List, Optional
import os
import pathlib
import shutil
import subprocess
import sys
import re
import uuid

def run(command: List[str], confirm: bool = True, capture_output: bool = False) -> Optional[str]:
    """
    Shell out to run a command. Optionally ask for confirmation and capture the output.
    """
    if confirm:
        while True:
            print(f"about to run {command}")
            response = input("type YES to confirm: ")
            if response.lower().strip() == "yes":
                break
            print("ok, not running it, type ctrl-c to exit")
            print()

    if capture_output:
        captured = subprocess.run(command, stdout=subprocess.PIPE)
        return captured.stdout.decode('utf-8')
    else:
        subprocess.run(command)


def run_prerelease_checks(release: str):

    # Check for a good release name
    if release is None:
        raise RuntimeError("RELEASE environment variable must be specified e.g. RELEASE=v0.1.0")

    if not re.match(r"^v([0-9]+)\.([0-9])+\.([0-9])+$", release):
        raise RuntimeError("RELEASE must be of the form v0.1.2 (any numbers will do)")


    # Make sure we're in the right place.
    if not all([os.path.exists(x) for x in ('scispacy', 'setup.py', '.git')]):
        raise RuntimeError("You must run this script from the root of the SciSpaCy repo")

    # Make sure we're in a fork.
    remote_output = run(['git', 'remote', '-v'], confirm=False, capture_output=True)
    for line in remote_output.split("\n"):
        if line.strip():
            upstream_name, url, fetch_push = line.split()
            if fetch_push == "(push)" and (("allenai/scispacy" in url) and ("github.com" in url)):
                    break
    else:
        # Never found a remote pointing at the right repo
        raise RuntimeError("could not find remote pointing at allenai/SciSpaCy")

    if upstream_name != "upstream":
        raise RuntimeError("you should have your upstream remote pointing at allenai/SciSpaCy. "
                f"(currently {upstream_name})")


def change_version_and_maybe_retrain_models(RELEASE: str):

    print("\nYOU go update the versions\n")
    print(f"Please update the version in scispacy/version.py (and everywhere else) to {RELEASE}, then press ENTER")
    print(f"(Please use `git grep` with the previous version to find areas that need to be updated)")

    CURRENT_VERSION=""
    while CURRENT_VERSION != RELEASE:
        input()

        _VERSION_RUN = {}
        with open("scispacy/version.py", "r") as version_file:
            exec(version_file.read(), _VERSION_RUN)
        CURRENT_VERSION = f"v{_VERSION_RUN['VERSION']}"

        if CURRENT_VERSION != RELEASE:
            print(f"current version {CURRENT_VERSION} does not match the release version {RELEASE}")
            print("please update scispacy/version.py for real, then press ENTER")


    print("Do you need to retrain the scispacy models for this new version? (YES/NO)")
    while True:
        response = input()
        response = response.strip().lower()
        if response in ["yes", "no"]:
            if response == "yes":
                print("\nPlease retrain the models now, making sure to change the version in scispacy/version.py BEFORE retraining.")
                print("See the README to find instructions on how to retrain the models. This process takes roughly 2 hours.")
                print("Once you have retrained the models, run this script again and answer no to this question.")
                sys.exit()
            else:
                break


def main():

    RELEASE = os.environ.get('RELEASE')
    RELEASE_BRANCH = f'release-{RELEASE}'

    # Checks before we start.
    run_prerelease_checks(RELEASE)

    # Check model version and ask user if they need to retrain the models.
    change_version_and_maybe_retrain_models(RELEASE)

    # Start actually doing the release.
    print("\nAbout to create a new branch and check it out\n")
    run(['git', 'branch', RELEASE_BRANCH])
    run(['git', 'checkout', RELEASE_BRANCH])

    print("\nAbout to create a release commit\n")
    run(['git', 'commit', '-a', '-m', f"bump version number to {RELEASE}"])

    print("\nAbout to git tag the previous commit\n")
    run(['git', 'tag', '-a', RELEASE])

    print("\nAbout to checkout the tagged commit\n")
    run(['git', 'checkout', RELEASE])

    print("\nAbout to create the distribution\n")

    dist = pathlib.Path('dist/')
    run(['python', 'setup.py', 'bdist_wheel'])
    run(['python', 'setup.py', 'sdist'])


    # Find wheel file
    wheels = [filename for filename in os.listdir(dist) if filename.endswith('.whl')]
    if len(wheels) != 1:
        raise RuntimeError(f"There should be exactly one .whl file in the dist/ directory; found {len(wheels)}")
    wheel = dist / wheels[0]
    wheelname = str(wheel.name)

    print("\nAbout to checkout release branch\n")
    run(['git', 'checkout', RELEASE_BRANCH])

    print("\nYOU go update the version again\n")
    print(f"Please update the version in scispacy/version.py to {RELEASE}-unreleased (or whatever) then press ENTER")
    input()

    NEW_VERSION = RELEASE

    while NEW_VERSION == RELEASE:
        _VERSION_RUN = {}
        with open("scispacy/version.py", "r") as version_file:
            exec(version_file.read(), _VERSION_RUN)
        NEW_VERSION = f"v{_VERSION_RUN['VERSION']}"

        if NEW_VERSION == RELEASE:
            print(f"new version {NEW_VERSION} is the same as release version {RELEASE}")
            print("please update scispacy/version.py for real")



    run(['git', 'add', 'scispacy/version.py'])
    run(['git', 'commit', '-m', f"Bump version numbers to {NEW_VERSION}"])

    print("\nAbout to push the release branch and tags to GitHub\n")
    run(['git', 'push', 'upstream', RELEASE_BRANCH])
    run(['git', 'push', 'upstream', '--tags'])

    print(f"\nCreate a pull request from {RELEASE_BRANCH} onto master (but do not merge it), and ....\n")
    print("\nNOW GET THAT PR CODE REVIEWED\n")
    while True:
        yes = input("type YES once the PR has been approved ")
        if yes.strip().lower() == "yes":
            break

    print("\nManually merging to master\n")
    run(['git', 'checkout', 'master'])
    run(['git', 'pull'])
    run(['git', 'merge', RELEASE_BRANCH])
    run(['git', 'push', 'upstream', 'master'])



    print("\nALMOST DONE!\n")
    print("please complete the following last steps:")
    print("1. Upload the package distributions to pypi. You can do this with:")
    print("\t\ttwine upload dist/<release>. The user and password can be found in LastPass.")
    print("2. Upload any new models you have trained to the s3 scispacy bucket.")
    print("3. Update README.md and docs/INDEX.md to reference the latest models and release.")
    print(f"4. cut a release from the tag {RELEASE} using the GitHub UI, and add release notes")
    print("git log `git describe --always --tags --abbrev=0 HEAD^`..HEAD --oneline")


if __name__ == "__main__":
    main()

