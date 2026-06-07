# Progress Report — Face Authentication Project
**Nour Ebid | M.Sc. Angewandte Informatik | Seminar zu aktuellen Entwicklungen**
**Betreuer: Frau Prof. Knaut | HTW Berlin | May 2026**

---

## What I Set Out to Do

The idea for this project came from a practical question I kept thinking about: when you register your face on a phone or a door lock, you're basically handing over your biometric data permanently. That felt like a problem worth looking into. What if instead of storing real photos of someone, you could train the system on AI-generated images that look like that person — same face, but not technically a real photograph? That way the sensitive biometric data doesn't need to exist in the system at all.

So the core question became: can a face authentication system work just as well if it was trained on synthetic versions of your face instead of real photos? And to make that a proper research question with a measurable answer, I framed it around the NIST FRTE framework — specifically False Accept Rate (FAR) and False Reject Rate (FRR), which are the standard metrics for evaluating biometric systems.

---

## First Attempt: Face Swapping (And Why It Failed)

My first instinct was to do something that felt creative — take a large pool of AI-generated faces (I downloaded about 1,860 faces from thispersondoesnotexist.com, which uses StyleGAN2), and then swap my friends' faces onto those AI bodies using Poisson blending (the seamlessClone function in OpenCV). The idea was that the resulting images would look like my friends but on a completely synthetic background, making them feel more "AI-generated."

I ran the experiment. The False Reject Rate (FRR) came back at 75%. That means 3 out of every 4 times a registered friend tried to authenticate, the system rejected them. That was obviously wrong.

I spent some time trying to figure out why. It turned out that roughly half the AI base images I was using depicted women, and all four of my registered friends are men. FaceNet — the model I'm using — encodes a lot of facial geometry in its 128-dimensional embeddings: jaw structure, brow ridge, overall face shape. When you blend a male face onto a female AI base, the resulting embedding is somewhere in between, and it ends up being too far from the original genuine embedding for the system to accept it.

---

## Second Attempt: Filtering by Gender

The obvious fix seemed to be: just filter the AI faces to only use male ones. I used InsightFace's gender classification to keep only male-presenting faces from the pool and redid the experiment.

It still didn't work. The FRR stayed very high.

This time I had to dig deeper. What I found was that even with male-only bases, the Poisson blending process itself was creating a problem. The blending algorithm creates very smooth transitions at the face boundary, but those transitions introduce small low-frequency artifacts in the image that FaceNet picks up on. The genuine L2 scores (the distance between the test embedding and the stored reference) were consistently shifted upward by 1 to 2 units compared to what they should be. With my threshold set at 10.0, that shift was enough to push many genuine scores above the threshold, causing rejections.

So the issue wasn't just the gender mismatch. The blending itself was adding a systematic bias to the embeddings.

---

## Stepping Back and Rethinking

At this point I had two failed approaches and a better understanding of the actual problem. The core issue with face swapping is that you're introducing a foreign base face — even after blending, the embedding still carries information from that base face. Any variable you don't control in that base face (gender, bone structure, image quality, blending artifacts) ends up affecting the embedding.

The solution I landed on was to abandon the face swap idea entirely and instead work directly with my friends' real photos — but transform them in a controlled way. Instead of pasting a face onto an AI background, I would simulate the kinds of variation that a camera naturally produces: different lighting, different colour temperatures, slight pose differences, camera noise. The key insight was that these are all imaging variables, not identity variables. They change how the photo looks, but they shouldn't fundamentally change whose face it is.

I built a five-stage pipeline:
1. Detect the face and align it to a standard 256×256 grid using five keypoints (eyes, nose, mouth corners) — so every output has the same pose
2. Apply random gamma correction to simulate different lighting conditions
3. Shift the colour temperature randomly (warmer or cooler) to simulate different cameras and environments
4. Apply CLAHE in LAB colour space to the luminance channel only, to simulate skin tone variation under different light
5. Add Gaussian blur, noise, and a small rotation, to simulate different camera quality and natural head tilt

The result is 15 synthetic images per person, all clearly that person, but each looking like it was taken under different conditions.

---

## Addressing the 100% Accuracy Problem

This is something I want to address directly, because the original version of the experiment gave 100% accuracy on the synthetic data, and that was wrong — and the professor was right to flag it.

The reason it happened was a mistake in how I split the data. I had 15 photos per person. I used all 15 to generate synthetic training images, and then I was testing on those same 15 real photos. So the system had already "seen" those faces during the generation step, even if not directly during training. That's a circular evaluation — you can't claim good results when the test set was involved in building the model.

The fix was simple but important: I committed to a strict split. The first 10 photos per person are used for training (or for synthetic generation). The last 5 photos per person are held out completely — they're never touched during registration or generation in either experiment. Both experiments use the same held-out 5 photos per person for testing.

After applying this fix, the results became realistic and comparable between experiments.

---

## What the Two Experiments Actually Show

**Experiment 1 (Real photos for registration):**
- All 4 friends verified correctly across their 5 held-out test photos — 20/20
- 2 out of 20 AI impostors were incorrectly accepted (scores 6.60 and 5.56, both well below the threshold of 10)
- FAR = 10%, FRR = 0%, Accuracy = 95%

**Experiment 2 (Synthetic photos for registration):**
- 19 out of 20 genuine friend queries verified — one of otto's photos got a score of 11.47, just above the threshold
- All 20 AI impostors correctly rejected
- FAR = 0%, FRR = 5%, Accuracy = 97.5%

The FAR difference between the two experiments is −10 percentage points, which is within the 15 percentage point threshold I set in the hypothesis. So the hypothesis holds.

The reason the synthetic model has lower FAR makes sense in hindsight: because the pipeline controls all the imaging variables, the resulting embeddings cluster more tightly around the true identity centre. That makes it harder for an impostor to accidentally fall within the acceptance boundary. The cost of that tighter clustering is slightly less tolerance for natural variation in test photos — which is why otto's one photo got rejected.

---

## What I Learned and What Comes Next

The biggest lesson from this project was how much the synthesis method matters. Two reasonable-sounding approaches (face swapping with random bases, then gender-filtered bases) both failed for reasons I didn't anticipate — one due to demographic distribution in the base pool, one due to blending artifacts in the pixel space. Neither failure was obvious from the start; I had to run the experiments and diagnose the embedding-level effects to understand what was going wrong.

The second lesson was about experimental design. The 100% accuracy result from the early version looked impressive but was meaningless. Having a proper held-out test set that was never involved in any part of the training or generation process is essential — and both experiments need to share that same test set for the comparison to be fair.

For the final paper, I plan to expand on the analysis side — looking at full DET curves across different threshold values, and potentially trying a GAN-based approach where the registered users' faces are entirely AI-generated without a real photo as seed. That would be a stronger test of the core question: can synthetic data completely replace real biometric data in face authentication?

---

*Submitted as part of the Zwischenbericht for Seminar zu aktuellen Entwicklungen, HTW Berlin*
