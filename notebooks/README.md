# Notebooks

Use this directory for Colab or local notebook experiments. A typical Colab flow is:

```python
!pip install -r requirements.txt
!python main.py --config configs/default.yaml
```

For a Colab-friendly cross-validation run:

```python
!python main.py --cross-validate --folds 3 --no-flower
```
