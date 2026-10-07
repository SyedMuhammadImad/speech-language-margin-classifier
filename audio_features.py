"""Shared validated local WAV processing with content-based caching."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
LABELS = {'eng': 'English', 'en': 'English', 'english': 'English', 'ur': 'Urdu', 'urdu': 'Urdu', 'mix': 'Mixed', 'mixed': 'Mixed', 'ue': 'Mixed', 'ar': 'Arabic', 'arabic': 'Arabic'}

def discover(root, manifest=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Dataset root must be a directory')
    records = []
    seen = {}
    if manifest:
        with Path(manifest).open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows or any((not {'path', 'label', 'speaker'} <= row.keys() for row in rows)):
            raise ValueError('Manifest needs path,label,speaker columns')
    else:
        rows = []
        for path in sorted(root.rglob('*.wav')):
            match = re.fullmatch('(.+?)[ _-](eng|english|en|urdu|ur|mix|mixed|ue|arabic|ar)', path.parent.name.lower())
            if not match:
                raise ValueError('Unrecognized folder; provide an explicit CSV manifest')
            rows.append({'path': str(path.relative_to(root)), 'label': LABELS[match[2]], 'speaker': match[1]})
    for row in rows:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Manifest paths must be relative to data root')
        path = root / rel
        if not path.resolve().is_relative_to(root) or any(((root / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1))):
            raise ValueError('Dataset links or traversal are unsupported')
        if not path.is_file() or path.suffix.lower() != '.wav' or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('Expected a WAV file under 32 MB')
        label, speaker = (row['label'].strip(), row['speaker'].strip())
        if not label or not speaker:
            raise ValueError('Empty label or speaker')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != (label, speaker):
                raise ValueError('Duplicate recording has conflicting labels or speaker')
            continue
        seen[digest] = (label, speaker)
        records.append((path, label, speaker, digest))
    if len(records) < 6:
        raise ValueError('At least six distinct recordings required')
    return records

def features(path):
    import librosa, soundfile as sf
    try:
        info = sf.info(path)
    except sf.LibsndfileError as error:
        raise ValueError('Invalid or unsupported WAV input') from error
    if info.duration < 0.15 or info.duration > 60 or info.channels > 2:
        raise ValueError('Audio must be 0.15–60 seconds and mono/stereo')
    audio, sr = librosa.load(path, sr=16000, mono=True, duration=30)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-07:
        raise ValueError('Silent or invalid audio')
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256).T
    d1 = librosa.feature.delta(mfcc.T, width=9, mode='nearest').T
    d2 = librosa.feature.delta(mfcc.T, width=9, order=2, mode='nearest').T
    flat = np.concatenate([v for a in (mfcc, d1, d2) for v in (a.mean(axis=0), a.std(axis=0))])
    if not np.isfinite(flat).all():
        raise ValueError('Invalid audio features')
    return (flat.astype(np.float32), mfcc.astype(np.float32))

def load_features(records, cache=None):
    key = hashlib.sha256(json.dumps([(r[1], r[2], r[3]) for r in records], ensure_ascii=False).encode() + b'mfcc13-16k-nfft512-hop256-first30s-v1').hexdigest()
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
    flat = []
    sequences = []
    for i, (path, _label, _speaker, digest) in enumerate(records):
        file = Path(cache) / (digest + '.npz') if cache else None
        if file and file.exists():
            with np.load(file, allow_pickle=False) as data:
                a, b = (data['flat'], data['sequence'])
        else:
            a, b = features(path)
            if file:
                np.savez_compressed(file, flat=a, sequence=b)
        if a.shape != (78,) or b.ndim != 2 or b.shape[1] != 13 or (not len(b)) or (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
            raise ValueError('Invalid feature cache')
        flat.append(a)
        sequences.append(b)
    return (np.stack(flat), sequences, key)

def measure(target, prediction, classes):
    p, r, f, _ = precision_recall_fscore_support(target, prediction, labels=list(range(classes)), average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(target, prediction)), 'macro_precision': float(p), 'macro_recall': float(r), 'macro_f1': float(f), 'confusion_matrix': confusion_matrix(target, prediction, labels=list(range(classes))).tolist()}

def speaker_folds(y, groups):
    speakers = np.unique(groups)
    if len(speakers) < 3:
        raise ValueError('At least three speakers required for separate train/validation/test speakers')
    for i, speaker in enumerate(speakers):
        test = np.flatnonzero(groups == speaker)
        val = np.flatnonzero(groups == speakers[(i + 1) % len(speakers)])
        train = np.flatnonzero((groups != speaker) & (groups != speakers[(i + 1) % len(speakers)]))
        if set(y[train]) != set(y) or set(y[val]) != set(y) or set(y[test]) != set(y):
            raise ValueError('Every speaker split must contain all classes')
        assert set(groups[train]).isdisjoint(groups[val]) and set(groups[train]).isdisjoint(groups[test])
        yield (train, val, test)
