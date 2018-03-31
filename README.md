# event-detector

This software annotates `.c3d` gait trajectories with Heel-Strike and Foot-Off events. It uses neural networks (more precisely Long Short Term Memory networks) through `keras` and `tensorflow` packages.

## Requirements

You only need `python`, all the dependencies will be installed automatically.

### Linux, Mac OS

Good news! Linux and Mac OS come with python distributions ready to use. You can go to the Installation step

### Windows

You need to install python, for example from here
https://www.python.org/downloads/windows/

## Instalation

Once python is installed, open terminal/command prompt and type

```bash
pip install eventdetector
```

This will install `eventdetector` scripts with all required dependencies

## Running

Navigate Terminal or Command Prompt to a directory with `.c3d` files you wish to annotate with Heel-Strike and Foot-Off events. Then type

```bash
event-detector [file-in.c3d] [file-out.c3d]
```

where `[file-in.c3d]` is the name of the file to annotate and `[file-out.c3d]` is the name of the new file in which you want to store annotation. 

Enjoy!

## Credits 

This research was sponsored by the Mobilize Center, a National Institutes of Health Big Data to Knowledge (BD2K) Center of Excellence supported through Grant U54EB020405. The model is trained on the data from Gillette Children's Specialty Healthcare, in accordance with the data sharing agreement. For the training scripts refer to https://github.com/kidzik/event-detector-train
