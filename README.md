
# Variational Autoencoders
Variational autoencoders are pretty nice and, in my experience, a lot better then denoising encoders. They can be a bit tricky to train though so I made a small troubleshooting guild. To make the code a little more fun I used a dataset of bouncing ball images instead of MNIST. The code to generate bouncing ball images is included and was originally from Ilya Sutskever's [Recurrent Temporal Restricted Boltzmann Machine](http://www.uoguelph.ca/~gwtaylor/publications/nips2008/rtrbm.pdf). I converted the ball bouncing to 32, 32, 3 images where the second and third color depict the x and y velocity.

# Troubleshooting

Here is a short list of the problems I had getting training to work. Hopefully this will save someone a little time.

## Getting NANS a few steps after training
Check to make sure that the output of your network is going through a sigmoid layer. The loss on the reconstruction will NAN if there are negatives because of the logs.

## Not NANing but not converging well either
Ok, so this is really the whole reason I made this github. I had a problem for a while where my loss and network appeared to be set up just fine however it was not training beyond the average. After a bit of digging I found it was how I init my layers. Because I was just using a network from another problem I was working on, I init the layers very small (around .001 for both conv and fully connected). This caused problems because when first run the autoencoder produced small values for the mean and stddev encoder part causing the vae loss to be small. It seems that it gets stuck in this minimum and the reconstruction loss never really falls. In most tutorials they seem to have no problem with this for 2 reasons. They do mini batch normalization or they use xavier initialization. When I was first looking at this I didn't really want to do batch normalization and didn't think xavier initialization really mattered. Then I proceeded to waste several hours. 

For the example code I just set the fully connected layer to init to .1 std and that fixed it. I put it as a flag so you can see that if its set to .001 it will not converge.

# Pictures!!!
true image
![alt text](https://github.com/loliverhennigh/Variational-autoencoder-tricks-and-tips/blob/master/real_balls.jpg)
generated image
![alt text](https://github.com/loliverhennigh/Variational-autoencoder-tricks-and-tips/blob/master/generated_balls.jpg)
. This is only after like 10 mins on a cpu though. With the same training time and .001 std of the fully connected it does this,
![alt text](https://github.com/loliverhennigh/Variational-autoencoder-tricks-and-tips/blob/master/bad_generated_balls.jpg)
when the true is 
![alt text](https://github.com/loliverhennigh/Variational-autoencoder-tricks-and-tips/blob/master/bad_real_balls.jpg)
and the same amount of training time

