
# Variational Autoencoders
Variational autoencoders are pretty nice and in my experience a lot better then denoising encoders. They can be a bit tricky to train though. I made this not as a tutorial on variational autoencoders but as a troubleshooting guild. To make the code a little more fun I used a dataset of bouncing ball images instead of MNIST. The code to generate bouncing ball images is included and was originally from Ilya Sutskever's Recurrent Temporal Restricted Boltzmann Machine. 

# Problems With training

Here is a short list of the problems I had with training

## Getting NANS
Check to make sure that the output of your network is going through a sigmoid layer. The loss on the reconstruction will NAN if there are negatives because of the logs.

## Not NANing but not converging well either
Ok so this is the whole reason I made this github


