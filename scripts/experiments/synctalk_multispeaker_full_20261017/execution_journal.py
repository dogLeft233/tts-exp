"""Exclusive coordinator ownership and atomic, resumable external journal writes."""
import contextlib,json,os
from pathlib import Path

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temporary.replace(path)

@contextlib.contextmanager
def exclusive_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0,os.SEEK_END)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
