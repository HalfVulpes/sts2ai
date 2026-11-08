from dataclasses import asdict
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from sts2ai.model import ModelConfig,Sts2Net


@pytest.mark.parametrize('training_config,expected',[(None,True),({'coop_prob':0.0},False)])
def test_weights_only_metadata_survives_cli_export(tmp_path,training_config,expected):
    config=ModelConfig(vocab_size=16,d_model=16,n_layers=1,n_heads=2,d_ff=32,n_memory=2)
    model=Sts2Net(config)
    checkpoint=dict(model=model.state_dict(),ema=model.state_dict(),mcfg=asdict(config),
                    inference_metadata={'encode_allies':True},weights_only_import=True)
    if training_config is not None:checkpoint['cfg']=training_config
    source=tmp_path/'parameters.pt';destination=tmp_path/'policy.bin';torch.save(checkpoint,source)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'train'),OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-m','sts2ai.export','--ckpt',str(source),'--ema','--out',str(destination)],
                   env=env,capture_output=True,text=True,check=True,timeout=60)
    with destination.open('rb') as f:
        assert f.read(8)==b'STS2NET1'
        metadata=json.loads(f.read(struct.unpack('<i',f.read(4))[0]))
    assert metadata['encode_allies'] is expected
