"""Checkpoint loading and image enhancement utilities."""
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF
from .model import LowLightUNet
from .data import IMAGE_EXTENSIONS


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto": return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str | Path, device: str = "auto", base_channels: int | None = None):
    device_obj = choose_device(device)
    # Checkpoints should be from a trusted source; weights_only=False may be needed for older PyTorch versions.
    try:
        ckpt = torch.load(str(checkpoint), map_location=device_obj, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location=device_obj)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        saved_args = ckpt.get("args", {})
        channels = base_channels or saved_args.get("channels", saved_args.get("base_channels", 32))
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]; channels = base_channels or 32
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt):
        state = ckpt; channels = base_channels or 32
    else:
        raise ValueError("Unsupported checkpoint format. Expected a model checkpoint or state_dict.")
    model = LowLightUNet(base_channels=int(channels)).to(device_obj)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError("Checkpoint architecture mismatch. Verify base_channels (32 or 64) and checkpoint format.") from exc
    model.eval()
    return model, device_obj


def enhance_image(model, image: Image.Image, device: torch.device) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    padded_w = ((width + 15) // 16) * 16
    padded_h = ((height + 15) // 16) * 16
    padded = TF.pad(image, (0, 0, padded_w - width, padded_h - height))
    tensor = ToTensor()(padded).unsqueeze(0).to(device)
    with torch.inference_mode(): output = model(tensor)
    arr = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr[:height, :width])


def enhance_path(input_path: str | Path, output_dir: str | Path, checkpoint: str | Path,
                 device: str = "auto", base_channels: int | None = None) -> list[Path]:
    input_path = Path(input_path); output_dir = Path(output_dir)
    if not input_path.exists(): raise FileNotFoundError(f"Input path not found: {input_path}")
    model, device_obj = load_model(checkpoint, device, base_channels)
    images = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if input_path.is_dir() else [input_path]
    if not images: raise FileNotFoundError(f"No supported images found in {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True); outputs = []
    for path in images:
        with Image.open(path) as im: result = enhance_image(model, im, device_obj)
        destination = output_dir / f"enhanced_{path.stem}.png"
        result.save(destination); outputs.append(destination)
        print(f"Saved {destination}")
    return outputs
