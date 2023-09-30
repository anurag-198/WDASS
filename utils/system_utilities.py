from runx.logx import logx
import os
import sys
import zipfile

# Import autoresume module
sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)

# Test Mode run two epochs with a few iterations of training and val
def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes. /ptmp/andas/project/semantic-segmentation/logs/train_deepv3x71/full_synbn/code/train.py
    return port

def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        if (base == "./.git"):
            continue
        for file in files:
            fn = os.path.join(base, file)
            if "logs" not in fn :
                zipobj.write(fn, fn[rootlen:])

def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.max_iter),
                    epoch,
                    args.max_iter
                )
                AutoResume.request_resume(
                    user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                               "TENSORBOARD_DIR": args.result_dir,
                               "EPOCH": str(epoch)
                               }, message=progress)
                return 1
            else:
                return 1
    return 0