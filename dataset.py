import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import librosa
from transformers import AutoTokenizer

def _evenly_spaced_indices(n, seq_len, is_train, clip_idx=0, clip_count=1):
    if n <= 0:
        return [0] * seq_len
    step = n / float(seq_len)
    if step <= 1.0:
        return [min(int(i), n - 1) for i in range(seq_len)]
    if is_train:
        offset = np.random.uniform(0, step)
    else:
        # evenly spaced offsets for multi-clip eval
        clip_count = max(1, int(clip_count))
        clip_idx = min(max(0, int(clip_idx)), clip_count - 1)
        offset = (clip_idx + 0.5) * (step / clip_count)
    return [min(int(np.floor(offset + i * step)), n - 1) for i in range(seq_len)]

class MultimodalPersonalityDataset(Dataset):
    """
    
     - evenly spaced temporal sampling (random offset for train, K even offsets for val)
     - text via your local BERT path
    """
    def __init__(
        self,
        frame_dirs,
        audio_paths,
        transcriptions,
        annotations,
        transform=None,
        seq_len=100,
        device='cpu',
        is_train=True,
        bert_path='/home/su/SIJ/waqas/models/bert-base-uncased'
    ):
        self.frame_dirs = frame_dirs
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions or {}
        self.annotations = annotations
        self.transform = transform
        self.seq_len = seq_len
        self.is_train = is_train
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)

        # for multi-clip validation
        self.eval_clip_idx = 0
        self.eval_clip_count = 1

    def set_eval_clip(self, idx, count):
        """Call before each evaluation pass to pick an even-offset clip."""
        self.eval_clip_idx = int(idx)
        self.eval_clip_count = int(max(1, count))

    def __len__(self):
        return len(self.frame_dirs)

    def _load_audio_features(self, audio_path):
        try:
            if not os.path.exists(audio_path):
                return torch.zeros(40, device=self.device)
            y, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            logmel = librosa.power_to_db(mel, ref=np.max)
            feat = np.mean(logmel, axis=1).astype(np.float32)
            return torch.tensor(feat, device=self.device)
        except Exception as e:
            print(f"[AUDIO] {audio_path}: {e}", flush=True)
            return torch.zeros(40, device=self.device)

    def _load_frames(self, frame_dir):
        files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        n = len(files)
        if n == 0:
            return torch.stack([torch.zeros(3, 224, 224, device=self.device) for _ in range(self.seq_len)], dim=0)
        idxs = _evenly_spaced_indices(n, self.seq_len, self.is_train, self.eval_clip_idx, self.eval_clip_count)
        frames = []
        for idx in idxs:
            fpath = os.path.join(frame_dir, files[idx])
            try:
                with Image.open(fpath) as img:
                    img = img.convert('RGB')
                img_t = self.transform(img) if self.transform else transforms.ToTensor()(img)
                frames.append(img_t.to(self.device, non_blocking=True))
            except Exception as e:
                print(f"[FRAME] {fpath}: {e}", flush=True)
                frames.append(torch.zeros(3, 224, 224, device=self.device))
        return torch.stack(frames, dim=0)

    def _load_text(self, video_id_mp4):
        try:
            text = self.transcriptions.get(video_id_mp4, "")
            toks = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            return toks["input_ids"].squeeze(0).to(self.device), toks["attention_mask"].squeeze(0).to(self.device)
        except Exception as e:
            print(f"[TEXT] {video_id_mp4}: {e}", flush=True)
            toks = self.tokenizer("", return_tensors="pt", max_length=512, truncation=True, padding=True)
            return toks["input_ids"].squeeze(0).to(self.device), toks["attention_mask"].squeeze(0).to(self.device)

    def __getitem__(self, idx):
        frame_dir = self.frame_dirs[idx]
        video_id = os.path.basename(frame_dir) + ".mp4"
        if video_id not in self.annotations:
            raise KeyError(f"Annotation not found for {video_id}")
        label = torch.tensor([
            self.annotations[video_id]["extraversion"],
            self.annotations[video_id]["neuroticism"],
            self.annotations[video_id]["agreeableness"],
            self.annotations[video_id]["conscientiousness"],
            self.annotations[video_id]["openness"]
        ], dtype=torch.float32, device=self.device)

        frames = self._load_frames(frame_dir)
        audio = self._load_audio_features(self.audio_paths[idx])
        input_ids, attention_mask = self._load_text(video_id)
        return frames, audio, input_ids, attention_mask, label

def collate_fn(batch):
    frames, audio, input_ids, attention_mask, labels = zip(*batch)
    frames = torch.stack(frames, 0)
    audio = torch.stack(audio, 0)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels, 0)
    return frames, audio, input_ids, attention_mask, labels

# default transform (train & val share; val gets centered offsets via is_train=False)
frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
