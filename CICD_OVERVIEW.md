# Godot Stat Math: CI/CD Development Flow

This document outlines the automated CI/CD pipeline for the Godot Stat Math addon, detailing the process from pull request to final release. The flow is designed to ensure code quality, stability, and automated delivery.

## Core Branches
- `develop`: The main integration branch for new features and bugfixes.
- `release`: The staging branch for preparing and deploying new releases.

---

## 1. The Pull Request (PR) Phase: Verification

All new code enters the `develop` or `release` branches via Pull Requests. We have two distinct verification workflows to guard these branches.

### PR to `develop` (`verify-develop.yaml`)
- **Trigger**: Opening or updating a PR targeting the `develop` branch.
- **Goal**: Quickly verify that the changes are structurally sound and don't break core functionality.
- **Actions**:
    1.  **Validate Structure**: Checks that essential addon files like `plugin.cfg` are present and correctly configured.
    2.  **Run Core Tests**: Executes a subset of the test suite (`/tests/core` and `stat_math_test.gd`). This provides a fast feedback loop by focusing on critical unit and integration tests, excluding slower performance tests.
    3.  **Upload Results**: Test reports are uploaded as artifacts for inspection.

### PR to `release` (`verify-release.yaml`)
- **Trigger**: Opening or updating a PR targeting the `release` branch.
- **Goal**: Perform a comprehensive, final check before a new version is released.
- **Actions**:
    1.  **Check Version Bump**: Ensures the project version in `project.godot` has been semantically versioned upwards.
    2.  **Validate Structure**: Same as the `develop` PR check.
    3.  **Trigger AWS Runner**: Spins up the AWS test runner.
    4.  **Run Full Test Suite**: Executes the *entire* test suite, including core tests, integration tests, and performance tests (`/tests`). This is a thorough validation to catch any regressions.
    5.  **Upload Results**: Full test reports are uploaded as artifacts.

---

## 2. The `develop` Branch: Continuous Integration

- **Trigger**: Pushing a commit to the `develop` branch (typically after a PR is merged).
- **Workflow**: `build-develop.yaml`
- **Goal**: To create a continuously updated, testable development build of the addon.
- **Actions**:
    1.  **Trigger AWS Runner**: Spins up the AWS test runner.
    2.  **Run Full Test Suite**: Ensures the integrity of the `develop` branch by running all tests.
    3.  **Create Dev Build**: Packages the `addons/godot-stat-math` directory into a zip file named `godot-stat-math-<version>-dev.zip`.
    4.  **Upload Artifact**: The development build zip is uploaded as a GitHub artifact. This allows developers and testers to easily download and try out the latest "nightly" version without waiting for an official release.

---

## 3. The `release` Branch: Automated Release

- **Trigger**: Pushing a commit to the `release` branch.
- **Workflow**: `build-release.yaml`
- **Goal**: To automatically build, test, and publish a new pre-release on GitHub.
- **Actions**:
    1.  **Trigger AWS Runner**: Spins up the AWS test runner.
    2.  **Run Full Test Suite**: A final, comprehensive test run is performed as a safeguard before release.
    3.  **Extract Release Notes**: Pulls the relevant release notes for the new version from `CHANGELOG.md`.
    4.  **Create Release Build**: Packages the addon into a clean, versioned-release zip file.
    5.  **Create GitHub Pre-Release**: Automatically creates a new pre-release on the project's GitHub Releases page, tagging it with the version number, attaching the addon zip, and populating the description with the extracted release notes.

This automated flow ensures that every change is rigorously tested and that development and release builds are created consistently, reducing manual effort and improving the reliability of the project. 