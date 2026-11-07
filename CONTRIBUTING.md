# Contributing to KineticAugment

First off, thank you for considering contributing to KineticAugment! We welcome any and all contributions, from simple bug reports to new feature implementations. Your help is essential for making this framework better for everyone.

This document provides a set of guidelines for contributing. These are mostly guidelines, not strict rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents
* [Code of Conduct](#code-of-conduct)
* [How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

## Code of Conduct

This project and everyone participating in it is governed by the [KineticAugment Code of Conduct](./CODE_OF_CONDUCT.md). 

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://github.com/imics-lab/KineticAugment/issues). Before creating a bug report, please check the existing issues to see if someone has already reported the problem.

When you create a bug report, please **use the Bug Report template** and include as many details as possible:
*   **A clear and descriptive title.**
*   **A detailed description of the problem.** Explain the behavior you saw and what you expected to see.
*   **Steps to reproduce the behavior.** Provide a minimal code snippet that demonstrates the problem.
*   **Your environment.** Include your operating system, Python version, and versions of relevant libraries.

### Suggesting Enhancements

Enhancements include new features, new augmentations, or improvements to existing functionality or documentation.

To suggest an enhancement, please **use the Feature Request template** on the [GitHub issues](https://github.com/imics-lab/KineticAugment/issues) page. Provide the following:
*   **A clear and descriptive title.**
*   **A step-by-step description of the suggested enhancement.**
*   **A clear motivation.** Explain why this enhancement would be useful to other KineticAugment users.
*   **Example code or usage.** Show how the new feature might be used.

### Your First Code Contribution

Unsure where to begin? You can start by looking through `good-first-issue` and `help-wanted` issues:
*   [Good first issues](https://github.com/imics-lab/KineticAugment/labels/good-first-issue) - issues which should only require a few lines of code, and a test or two.
*   [Help wanted issues](https://github.com/imics-lab/KineticAugment/labels/help-wanted) - issues which should be a bit more involved than `good-first-issue` issues.

### Pull Requests

The process for submitting a code contribution is as follows:

1.  **Fork the repository** by clicking the 'Fork' button on the repository's page. This creates a copy of the repository in your own GitHub account.
2.  **Create a new branch** from `main` in your forked repository. Use a descriptive name, like `feature/add-new-augmentation` or `fix/crash-on-validation`.
    ```bash
    git checkout -b your-branch-name
    ```
3.  **Make your changes.** Add your new feature or fix the bug.
4.  **Add tests.** If you are adding a new feature, please include tests that cover it. If you are fixing a bug, add a test that demonstrates the bug and shows that your fix works.
5.  **Run the tests.** Install the dev extras and run the suite from the repository root:
    ```bash
    pip install -e '.[dev]'
    pytest tests/
    ```
    The older print-based smoke scripts in `scripts/legacy/` are not part of the test suite.
6.  **Ensure your code is clean and well-documented.** Follow the existing coding style and add comments where necessary.
7.  **Push your branch** to your forked repository on GitHub.
    ```bash
    git push origin your-branch-name
    ```
8.  **Open a Pull Request** to the `main` branch of the original `KineticAugment` repository.
9.  **Provide a clear title and description** for your pull request, explaining the changes you have made and referencing any relevant issues.

Your pull request will be reviewed by the maintainers. We may ask for changes before it is merged. We appreciate your contribution!