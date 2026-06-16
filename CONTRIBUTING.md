# Contributing to ML Engineering Pipeline for Satellite Imagery

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in Issues
- Create a new issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (OS, Python version, etc.)

### Suggesting Features
- Open an issue describing your feature idea
- Explain why it would be valuable
- Discuss implementation approach if you have ideas

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**: `git commit -m "Add: brief description"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ml_project.git
cd ml_project/ml-engine

# Install dependencies
pip install -r requirements.txt

# Make your changes
# ...

# Test your changes
python test_model_forward.py
```

## Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

## Commit Message Format

Use clear, descriptive commit messages:
- `Add: new feature or file`
- `Fix: bug fix description`
- `Update: improvements or changes`
- `Remove: deleted files or features`

## Questions?

Feel free to open an issue for any questions about contributing!
