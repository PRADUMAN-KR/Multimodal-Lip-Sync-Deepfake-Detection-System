# Model weights

Place your trained lip-sync model checkpoint here so the service can load it at startup.

- **Expected file:** `best_model.pth`
- **Format:** PyTorch `state_dict` (or a dict with a `"state_dict"` key), compatible with `LipSyncModel`.

To train a model, use your training script and save with:

```python
torch.save(model.state_dict(), "weights/best_model.pth")
# or
torch.save({"state_dict": model.state_dict(), "epoch": epoch}, "weights/best_model.pth")
```

Until `best_model.pth` exists, the app will start but `/api/lip-sync` will return **503** with a message to load weights.
