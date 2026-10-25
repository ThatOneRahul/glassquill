# Local explanations (XGBoost + SHAP)

## true_ai (P(AI) = 1.00, source: finance)

- Perplexity (log) = 1.98 pushed the score towards AI
- Unevenness of surprise = 2.05 pushed the score towards AI
- Share of top-10 guessable words = 0.89 pushed the score towards AI

> A market order is an order to buy or sell a security at the best available price, and it is generally used when you want to execute a trade immediately. This type of order is often used when the trader has a strong conviction about the direction of the market and wants to get the trade done as quickly as possible.On the other hand, a limit order is an order to buy or sell a security at a specific price or better. This type of order is used when the trader wants to set a certain price at which th...

## true_human (P(AI) = 0.00, source: reddit_eli5)

- Perplexity (log) = 3.64 pushed the score towards human
- Share of top-10 guessable words = 0.68 pushed the score towards human
- Unevenness of surprise = 2.99 pushed the score towards human

> Your computer takes the data you " delete " and just sort of flags it. The flag basically means " Hey, I'm going to go hang out in the Recycle Bin. I'm still around, but overwrite me if you need to write something else. " (When you choose to " Recover " something from your recycle bin, you're basically clearing that flag.) This is also how some data recovery applications function. They scan the disk for any flagged data and clear the flag (or " recover " the data)....

## false_positive (P(AI) = 0.53, source: reddit_eli5)

- GPT-2 uncertainty = 3.65 pushed the score towards AI
- Perplexity (log) = 2.95 pushed the score towards human
- Burstiness = 0.14 pushed the score towards AI

> Assuming that you're using a digital camera, the camera's sensor has a grid of pixels, and a lens that projects an image onto the sensor. If the distance between the stripes in the projected image is less than the distance between pixels in the sensor, the sensor is physically unable to capture the detail in the stripes. The data format used to store photos (JPEG) also has limitations that are more apparent when the photo has fine detail, like stripes....

## false_negative (P(AI) = 0.07, source: reddit_eli5)

- GPT-2 uncertainty = 3.68 pushed the score towards AI
- Share of top-1 guessable words = 0.35 pushed the score towards human
- Punctuation per word = 0.21 pushed the score towards human

> The small speakers you see in the picture are called "tweeters." They are used to produce high frequency sounds, such as the s's in "sizzle" or the high notes in a piano. The larger speakers are called "woofers" and they produce low frequency sounds, like the bass in music. By combining the sounds from both the tweeters and the woofers, we can get a full and balanced sound. Is that helpful?...
