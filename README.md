## Install

`pip install -r requirements.txt`

Can do that in a venv if you want.

## Running

Just running `python main.py` should be sufficient, assuming you have the mode and correct image locally. It should work with any image format but give it a go and see what breaks. This isnt perfect.


## Code

I split it up into a class and some runner code. The runner code is inside the if __name__ == "__main__". 

I made it a class so each function can be reused elsewhere. If you have an idea and want to just upscale images for example it should work.

To intrgrate this with your bot id do the following. In your __init__  create an instance of this class. This will load the model into memory which is one of the bottle necks currently. Then when you want to use it just call `self.bot.convert_to_png()` or whichever function it is. 

The canvas size and padding around the object are hardcoded in the __init__ of the ImageProcessor class atm. These can be changed or added to a config file (my preference, but ive left it up to you).

The overlay code is insane and I used Chad for a lot of it. But hopefully the comments help somewhat.

## n.b.

I didnt spend long on this, an hour or so. Shit will break. Just ask me any questions you have, either when shit breaks or if something in the code isnt clear. Hopefully this is an ok example of code. Not the best ever but does the job.