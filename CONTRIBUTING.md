# Contributing to XGrammar

We welcome contributions of all kinds, including new features, bug fixes, documentation improvements, and more. To ensure a smooth process, here is a general guide to contributing.

For significant changes, such as adding a major new feature or refactoring core code, it's often a good idea to open a GitHub issue first to discuss your proposal. This step is optional, but can be very helpful as it allows the maintainers and the community to provide feedback and helps ensure your work aligns with the project's goals.

The general workflow for submitting a change is:

1. **Fork the repository** and create a new branch for your work.
2. Make your changes, including adding tests if applicable. Please refer to the [`README.md`](README.md) for project-specific setup and testing instructions.
3. Format the code. Please run the commands below first:
```bash
pre-commit install
pre-commit run -a
ruff check
```
4. Push your changes to your fork, and **open a pull request** to the main repository. Please provide a clear description of your changes and link to the relevant issue if one exists.
5. **Iterate on the pull request** by responding to feedback from reviewers until the change is ready to be merged.

## **Pull Request Naming Convention**

To maintain consistency and clarity, please follow this naming convention for your pull requests:

```
<type>: <brief description>
```

**Available types:**

* `feat`: A new feature or enhancement
* `fix`: A bug fix
* `perf`: Performance improvement
* `refactor`: Code refactoring without changing functionality
* `test`: Adding or updating tests
* `docs`: Documentation changes
* `style`: Code style changes (formatting, whitespace, etc.)
* `build`: Changes to build system or dependencies
* `ci`: Changes to CI/CD configuration
* `chore`: Maintenance tasks and other changes

**Examples:**

* `feat: support XGrammar with FP16 logits`
* `fix: correct token mask generation with Unicode characters`
* `perf: optimize the generation of token masks`
* `docs: update installation guide`
* `test: add unit tests for structural tags`

## **Review**

Once you've opened a pull request, the review process begins:

* **Community Review:** We encourage everyone to participate in the review process. All feedback on pull requests is welcome and valued.
* **Approval:** For a pull request to be merged, it must receive at least **one approval** from designated code owners for the files you've changed, or from a project lead. The [`CODEOWNERS`](./CODEOWNERS) file in the repository lists the members responsible for different parts of the codebase.

We hope this collaborative approach will maintain high code quality and ensure knowledge is shared effectively among contributors.

## **Merge**

After your pull request has been approved and all automated checks (CI) have passed, a **Community Committer** will merge it into the main branch.

### **Performance and Stability**

Maintaining high performance and stability are key goals for XGrammar. If a performance regression or functional issue occurs after a merge, we encourage anyone to report it. The process is as follows:

1. **File an Issue:** Anyone can file a high-priority issue in the repository. It is helpful to tag the original pull request and notify the author.
2. **Collaboration:** A project committer or lead will collaborate with the community to address the issue.
3. **Resolution:** If a quick fix is not available, the change may be reverted to maintain project stability.

## **Community Committer Role**

XGrammar is maintained by a group of **Community Committers**. These are core contributors who have earned the role by providing frequent and valuable contributions to the project. They are responsible for reviewing and merging pull requests, maintaining the project's standards, and guiding new contributors.
