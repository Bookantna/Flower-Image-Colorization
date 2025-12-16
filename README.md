# 🎨 Flower Image Colorization using Conditional Generative Adversarial Networks (CGAN)

## Project Overview

This project focuses on the image-to-image translation task of colorizing grayscale flower images. We implemented a **Conditional Generative Adversarial Network (CGAN)** to generate realistic color versions of input images, a great demonstration of advanced Deep Learning and Neural Network skills.

The goal is to transform a grayscale flower image into a plausible, colorized image.

## 🎯 Methodology

### 1. Model Architecture: Conditional GAN (CGAN)
We used a Conditional Generative Adversarial Network (CGAN) for this image-to-image translation task.

The network consists of two main components:
* **Generator:** Takes a Grayscale Image as input and outputs a Generated Image (Colorized).
* **Discriminator:** Takes the Generated Image and the Real RGB Image (along with the Grayscale input, as implied by CGAN) to determine if the generated image is "Real" or "Fake".

### 2. Generator Architecture: U-NET
The Generator employs the **U-NET architecture**, which is effective for image segmentation and generation tasks due to its skip connections that help retain fine-grained spatial information during the generation process.

### 3. Discriminator Architecture: Patch Discriminator
The Discriminator uses a **Patch Discriminator**, which classifies whether overlapping patches of an image are real or fake, rather than classifying the entire image at once.

## 📊 Dataset & Data Collection

* **Total Data Collection Size (Calculated):** The data collection method was structured to produce a potential 21,600 images based on 6 stems, 5 flowers, and 360 variations multiplied by 2.
* **Data Used (Actual):** **5,000 images** were used for the actual training of the model.

## 📈 Training and Results

The model was trained over 10 epochs (as shown in the training loss table).

| Metric | Epoch 1 | Epoch 5 | Epoch 10 | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Discriminator Loss** (D Loss) | 0.6916 | 0.5581 | 0.2655 | **Decreased** (Discriminator is improving at distinguishing real/fake) |
| **Generator Loss** (G Loss) | 0.7175 | 1.1287 | 1.6807 | **Increased** (Generator struggled to fool the Discriminator effectively) |

## 🖼️ Visual Results

The project successfully generates colorized output images from grayscale inputs, providing a visual comparison between the Input (Gray), the Generated (Colorized) image, and the Real Image.

## 💻 Repository Structure (Inferred)

* `[project].ipynb`: Main Jupyter Notebook containing model definition, training loop, and evaluation.
* `presentation.pdf`: Project documentation and presentation slides.

## 🛠️ Technologies Used
* **Deep Learning Framework:** PyTorch (Inferred from companion notebook name and Deep Learning context)
* **Architecture:** Conditional GAN, U-NET
* **Language:** Python
