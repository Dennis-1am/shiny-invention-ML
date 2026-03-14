# 🚀 Machine Learning & Python Mini-Projects
A collection of small-scale machine learning experiments, utility scripts, and deep learning models.

## 📂 Project Directory

| Project Name | Description | Status |
| :--- | :--- | :--- |
| **[MNIST_DIGITAL_CLASSIFIER](./MNIST_DIGITAL_CLASSIFIER)** | CNN-based handwritten digit recognizer with 98% accuracy. | IN PROGRESS  |
| **Future Project** | *Coming soon...* | ⏳ Pending |

---

## 🛠️ Common Workflow: Dehydrate & Rehydrate
To keep this repository lightweight and avoid uploading gigabytes of library files, each project uses a "Dehydration" workflow.

## 🏜️ Dehydrate (Save space when done)
Run the dehydrate.sh script inside the project folder to update your requirements and wipe the .venv:

```bash
./dehydrate.sh
```

### 💧 Rehydrate (Bring a project to life)
When you want to work on a specific project, navigate into its folder and run this one-liner to rebuild the environment:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

