# Project Setup Guide

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd banking-regulatory-intelligence-platform
```

---

## 2. Install Poetry

If Poetry is not installed:

### macOS / Linux

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:

```bash
poetry --version
```

---

## 3. Install Python 3.12

The project uses **Python 3.12**.

Check your installed version:

```bash
python3.12 --version
```

If Python 3.12 is not installed, install it first.

---

## 4. Create the Poetry Virtual Environment

Tell Poetry to use Python 3.12:

```bash
poetry env use python3.12
```

---

## 5. Install Project Dependencies

```bash
poetry install
```

Poetry will:

* create a virtual environment,
* install all dependencies from `poetry.lock`.

---

## 6. Activate the Virtual Environment

```bash
poetry shell
```

Your terminal should now look similar to:

```text
(reguaz-py3.12)
```

---

## 7. Verify the Installation

Check the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.x
```

List installed packages:

```bash
poetry show
```

---

## 8. Deactivate the Environment

When finished:

```bash
exit
```

or press:

```text
Ctrl + D
```

---

## Daily Workflow

Whenever you start working on the project:

```bash
cd banking-regulatory-intelligence-platform
poetry shell or poetry env activate => and copy paste the output 
git pull origin main
```

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push origin <your-branch>
```

---

## Useful Poetry Commands

Show installed packages:

```bash
poetry show
```

Add a dependency:

```bash
poetry add package_name
```

Remove a dependency:

```bash
poetry remove package_name
```

Install dependencies after pulling new changes:

```bash
poetry install
```

Update the lock file:

```bash
poetry lock
```
