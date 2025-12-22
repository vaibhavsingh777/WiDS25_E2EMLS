# Week 1 Learnings: Getting My Hands Dirty with Data

When I started this week, I thought I was just looking at leaf pictures. But after digging into the **PlantVillage** dataset, I realized that "cleaning" the data is where the real work happens. Here is what I did and what I found:

### **1. Tackling the Class Imbalance**

One of the first things I did was plot the distribution of images. I immediately noticed a massive gap: the dataset is heavily skewed toward **Citrus Greening (Orange)**, which has over 5,000 samples, while **Healthy Potatoes** only have about 150.

* **Why I care**: I realized that if I don't fix this, my model will just become "lazy" and guess the majority class to get a high score.
* **My Plan**: For Week 2, I'm going to use data augmentation or weighted loss functions to give those rare classes more "voice".

### **2. The Hunt for Blurry Images**

I didn't want my model learning from "trash," so I ran an OpenCV script to check for blurriness.

* **The Result**: I found 6 images that were significantly out of focus.
* **The "Human" Test**: When I looked at them, I could barely see the disease spots myself. If I can't see it, I can't expect the AI to learn it. I've flagged these for removal to keep the training set high-quality.

### **3. Understanding Visual Similarity**

I wanted to see if I could distinguish between similar diseases, like the different types of **Tomato Blight**.

* **Observation**: In my sample grid, I noticed that while some leaves are distinct, others look nearly identical in color and texture.
* **Finding**: My unsupervised clustering confirmed this—certain disease classes overlap almost perfectly in the feature space, meaning the model is going to need a very fine-tuned "eye" to tell them apart.

---

## 🧬 What exactly is t-SNE?

In your "Unsupervised Cluster Map," you used a technique called **t-SNE (t-Distributed Stochastic Neighbor Embedding)**. Here is the plain-English explanation of what happened behind the scenes:

1. **High-Dimensional Chaos**: When we passed your images through the ResNet50 model, it converted each image into a list of **2,048 different numbers** (features). Humans cannot visualize 2,048 dimensions.
2. **The "Neighbor" Logic**: t-SNE looks at those 2,048 numbers for every image and asks: *"Which images are 'neighbors' because they look similar?"*.
3. **Squashing the Data**: It then tries to "squash" those 2,048 dimensions down to just **2 dimensions (X and Y)** while trying its best to keep the "neighbors" close together.
4. **The Map**: The resulting plot is essentially a **Similarity Map**.
* If points are clustered together, it means the computer thinks those images look very similar in terms of color, shape, and texture.
* If a cluster is messy or overlapping (like your Apple clusters), it tells us that even a powerful model like ResNet50 has a hard time tellling those specific classes apart.



### **Patterns I found in the Map:**

* **Consistent Quality**: The dense clusters (like the yellow ones in your plot) show where the images are very consistent in how they were photographed.
* **Noisy Variation**: The "scattered" points are the outliers—images with weird lighting or unusual leaf positions that didn't fit neatly into a group.

---

**Would you like me to help you write the specific slide text for your t-SNE results so you can explain this clearly in your 2-minute presentation?**
