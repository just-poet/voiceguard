from pathlib import Path
import os
from download_benchmarks import download
root=Path(os.environ.get('VOICEGUARD_DATA','D:/VoiceGuardData'))
for name in ('asv_train_parquet','asv_dev_parquet','asv_test_parquet'):
    download(name,root,24)
