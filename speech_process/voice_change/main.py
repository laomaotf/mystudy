import pytsmod as tsm
import soundfile as sf 
import librosa as ra
import os,tqdm
import argparse
def parse_args():
    ap = argparse.ArgumentParser(description="change voice by tune shift")
    ap.add_argument("input",help="input file or folder")
    ap.add_argument("-steps",help="n_steps to call pitch_shift()",default=5,type=int)
    ap.add_argument("-s",help="s to call wsola()",default=1.2,type=float)
    return ap.parse_args()

def convert(input,output,args):
    samplerate = ra.get_samplerate(input)
    x, sr = ra.load(input,sr=samplerate)
    assert(sr == samplerate)
    x = ra.effects.pitch_shift(x,sr=samplerate,n_steps=args.steps)
    x = tsm.wsola(x, args.s)
    sf.write(output,x,samplerate)

def main(args): 
    for filename in tqdm.tqdm(os.listdir(args.input)):
        _,ext = os.path.splitext(filename) 
        if filename.split('_')[-1] == "update.wav":
            continue
        if ext.lower() in {".mp3",".wav",".m4a"}:
            input = os.path.join(args.input,filename)
            output = os.path.splitext(input)[0] + "_update.wav"
            convert(input,output,args)
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)